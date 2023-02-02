// Copyright 2021 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORSTORE_DRIVER_IMAGE_DRIVER_IMPL_H_
#define TENSORSTORE_DRIVER_IMAGE_DRIVER_IMPL_H_

#include <assert.h>
#include <stddef.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/async_initialized_cache_mixin.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"  // IWYU pragma: keep
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_image_driver {
namespace {

template <typename Specialization>
class ImageDriverSpec
    : public internal::RegisteredDriverSpec<ImageDriverSpec<Specialization>,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  using SpecType = ImageDriverSpec<Specialization>;

  static constexpr const auto& id = Specialization::id;

  kvstore::Spec store;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  Context::Resource<internal::CachePoolResource> cache_pool;
  StalenessBound data_staleness;
  Specialization specialization;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.store,
             x.data_copy_concurrency, x.cache_pool, x.data_staleness,
             x.specialization);
  };

  static absl::Status ValidateSchema(Schema& schema) {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(dtype_v<uint8_t>));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(RankConstraint{3}));
    if (schema.codec().valid()) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("codec not supported by \"", id, "\" driver"));
    }
    if (schema.fill_value().valid()) {
      return absl::InvalidArgumentError(
          "fill_value not supported by image driver");
    }
    if (schema.dimension_units().valid()) {
      return absl::InvalidArgumentError(
          "dimension_units not supported by image driver");
    }
    if (auto domain = schema.domain(); domain.valid()) {
      if (!std::all_of(domain.origin().begin(), domain.origin().end(),
                       [](auto x) { return x == 0; })) {
        return absl::InvalidArgumentError("image domain must have 0-origin");
      }
    } else {
      TENSORSTORE_RETURN_IF_ERROR(schema.Set(
          IndexDomainBuilder<3>().origin({0, 0, 0}).Finalize().value()));
    }

    // TODO: validate schema fields:
    // schema.fill_value
    // schema.chunk_layout
    return absl::OkStatus();
  }

  constexpr static auto default_json_binder =
      tensorstore::internal_json_binding::Sequence(
          tensorstore::internal_json_binding::Initialize(
              [](auto* obj) -> absl::Status {
                return ValidateSchema(obj->schema);
              }),
          tensorstore::internal_json_binding::Member(
              internal::DataCopyConcurrencyResource::id,
              tensorstore::internal_json_binding::Projection<
                  &SpecType::data_copy_concurrency>()),
          tensorstore::internal_json_binding::Member(
              internal::CachePoolResource::id,
              tensorstore::internal_json_binding::Projection<
                  &SpecType::cache_pool>()),
          tensorstore::internal_json_binding::Projection<&SpecType::store>(
              tensorstore::internal_json_binding::KvStoreSpecAndPathJsonBinder),
          tensorstore::internal_json_binding::Member(
              "recheck_cached_data",
              tensorstore::internal_json_binding::Projection<
                  &SpecType::data_staleness>(
                  tensorstore::internal_json_binding::DefaultValue(
                      [](auto* obj) { obj->bounded_by_open_time = true; }))),
          tensorstore::internal_json_binding::Projection<
              &SpecType::specialization>()  //
      );

  absl::Status ApplyOptions(SpecOptions&& options) override {
    // An image file contains both the data and the metadata, so set the
    // staleness bound to the maximum of requested data and metadata staleness.
    if (options.recheck_cached_data.specified()) {
      data_staleness = StalenessBound(options.recheck_cached_data);
    }
    if (options.recheck_cached_metadata.specified()) {
      StalenessBound bound(options.recheck_cached_metadata);
      if (!options.recheck_cached_data.specified() ||
          bound.time > data_staleness.time) {
        data_staleness = std::move(bound);
      }
    }
    if (options.kvstore.valid()) {
      if (store.valid()) {
        return absl::InvalidArgumentError("\"kvstore\" is already specified");
      }
      store = std::move(options.kvstore);
    }
    return ValidateSchema(options);
  }

  kvstore::Spec GetKvstore() const override { return store; }

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

template <typename Specialization>
class ImageCache : public internal::KvsBackedCache<ImageCache<Specialization>,
                                                   internal::AsyncCache>,
                   public internal::AsyncInitializedCacheMixin {
  using Base = typename internal::KvsBackedCache<ImageCache<Specialization>,
                                                 internal::AsyncCache>;

 public:
  using ReadData = tensorstore::SharedArray<const uint8_t, 3>;
  using CacheType = ImageCache<Specialization>;
  using LockType = internal::AsyncCache::ReadLock<typename CacheType::ReadData>;

  ImageCache() : Base(kvstore::DriverPtr()) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = ImageCache;

    using DecodeReceiver = typename Base::Entry::DecodeReceiver;
    using EncodeReceiver = typename Base::Entry::EncodeReceiver;

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      if (!value) {
        // The KvStore entry for this key was not found.
        execution::set_error(receiver, absl::NotFoundError(""));
        return;
      }
      auto options = GetOwningCache(*this).specialization_;
      GetOwningCache(*this).executor()(
          [value = std::move(value), receiver = std::move(receiver),
           options = std::move(options)]() mutable {
            auto decode_result = options.DecodeImage(std::move(*value));
            if (!decode_result.ok()) {
              execution::set_error(receiver, decode_result.status());
            } else {
              execution::set_value(receiver, std::make_shared<ReadData>(
                                                 std::move(*decode_result)));
            }
          });
    }

    void DoEncode(std::shared_ptr<const ReadData> data,
                  EncodeReceiver receiver) override {
      auto encode_result =
          GetOwningCache(*this).specialization_.EncodeImage(*data);
      if (!encode_result.ok()) {
        execution::set_error(receiver, encode_result.status());
      } else {
        execution::set_value(receiver, std::move(*encode_result));
      }
    }
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = ImageCache;
    using Base::TransactionNode::TransactionNode;

    using ApplyOptions = typename Base::TransactionNode::ApplyOptions;
    using ApplyReceiver = typename Base::TransactionNode::ApplyReceiver;

    void DoApply(ApplyOptions options, ApplyReceiver receiver) override {
      execution::set_error(receiver, absl::UnimplementedError(
                                         Specialization::kTransactionError));
    }
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(
      internal::AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  const Executor& executor() { return data_copy_concurrency_->executor; }

  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
  Specialization specialization_;
};

template <typename Specialization>
class ImageDriver
    : public internal::RegisteredDriver<ImageDriver<Specialization>,
                                        /*Parent=*/internal::Driver> {
 public:
  using SpecType = ImageDriverSpec<Specialization>;
  using DriverType = ImageDriver<Specialization>;
  using CacheType = ImageCache<Specialization>;
  using LockType = internal::AsyncCache::ReadLock<typename CacheType::ReadData>;

  KvStore GetKvstore() override {
    auto& cache = GetOwningCache(*cache_entry_);
    return KvStore(kvstore::DriverPtr(cache.kvstore_driver()),
                   std::string(cache_entry_->key()));
  }

  // FIXME: Current image formats are restricted to rank3 (y,x,channels) and
  // uint8_t data, but there are image types which support a much wider array
  // of dtype(), and some which support multiple z levels encoded as one image.
  DataType dtype() override { return dtype_v<uint8_t>; }
  DimensionIndex rank() override { return 3; }  // COV_NF_LINE

  Executor data_copy_executor() override {
    return GetOwningCache(*cache_entry_).executor();
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    ChunkLayout layout;
    layout.Set(RankConstraint{3}).IgnoreError();
    return layout | transform;
  }

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  Future<IndexTransform<>> ResolveBounds(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      ResolveBoundsOptions options) override;

  void Read(internal::OpenTransactionPtr transaction,
            IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>
                receiver) override;

  internal::PinnedCacheEntry<CacheType> cache_entry_;
  StalenessBound data_staleness_;
};

template <typename Specialization>
Future<internal::DriverHandle> ImageDriverSpec<Specialization>::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  using DriverType = ImageDriver<Specialization>;
  using CacheType = ImageCache<Specialization>;
  using LockType = internal::AsyncCache::ReadLock<typename CacheType::ReadData>;

  if ((read_write_mode & ReadWriteMode::write) == ReadWriteMode::write) {
    return absl::InvalidArgumentError("only reading is supported");
  }
  read_write_mode = ReadWriteMode::read;
  if (!store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }
  std::string cache_identifier;
  auto request_time = absl::Now();
  internal::EncodeCacheKey(&cache_identifier, store.driver,
                           data_copy_concurrency, store.path);
  auto cache = internal::GetOrCreateAsyncInitializedCache<CacheType>(
      **cache_pool, cache_identifier,
      [&] {
        auto cache = std::make_unique<CacheType>();
        cache->data_copy_concurrency_ = data_copy_concurrency;
        cache->cache_pool_ = cache_pool;
        cache->specialization_ = specialization;
        return cache;
      },
      [&](Promise<void> initialize_promise,
          internal::CachePtr<CacheType> cache) {
        // The cache didn't previously exist.  Open the KeyValueStore.
        LinkValue(
            [cache = std::move(cache)](Promise<void> cache_promise,
                                       ReadyFuture<kvstore::DriverPtr> future) {
              auto kv = std::move(*future.result());
              cache->SetKvStoreDriver(std::move(kv));
            },
            initialize_promise, kvstore::Open(store.driver));
      });

  // Once the cache is initialized, pin the entry for the image path.
  return PromiseFuturePair<internal::DriverHandle>::LinkValue(
             [this, cache, request_time, transaction = std::move(transaction)](
                 Promise<internal::DriverHandle> p, AnyFuture f) {
               internal::ReadWritePtr<DriverType> driver =
                   internal::MakeReadWritePtr<DriverType>(ReadWriteMode::read);
               driver->cache_entry_ = GetCacheEntry(cache, store.path);
               driver->data_staleness_ =
                   data_staleness.BoundAtOpen(request_time);

               IndexDomain<> schema_domain = this->schema.domain();
               // Since writing is not allowed, resolve the cache entry
               // upon opening.
               LinkValue(
                   [driver, transaction = std::move(transaction),
                    schema_domain = std::move(schema_domain)](
                       Promise<internal::DriverHandle> p, AnyFuture f) {
                     LockType lock{*driver->cache_entry_};
                     assert(lock.data());
                     auto transform = IdentityTransform(lock.data()->domain());

                     // Validate the schema.domain constraint, if any.
                     if (schema_domain.valid() &&
                         !MergeIndexDomains(schema_domain, transform.domain())
                              .ok()) {
                       p.SetResult(absl::InvalidArgumentError(
                           tensorstore::StrCat("Schema domain ", schema_domain,
                                               " does not match image domain ",
                                               transform.domain())));
                       return;
                     }

                     p.SetResult(internal::DriverHandle{
                         std::move(driver), std::move(transform),
                         internal::TransactionState::ToTransaction(
                             std::move(transaction))});
                   },
                   std::move(p),
                   driver->cache_entry_->Read(driver->data_staleness_.time));
             },
             cache->initialized_)
      .future;
}

template <typename Specialization>
Result<internal::TransformedDriverSpec>
ImageDriver<Specialization>::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  if (transaction) {
    return absl::UnimplementedError(Specialization::kTransactionError);
  }
  auto driver_spec = internal::DriverSpec::Make<SpecType>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  auto& cache = GetOwningCache(*cache_entry_);
  TENSORSTORE_ASSIGN_OR_RETURN(driver_spec->store.driver,
                               cache.kvstore_driver()->GetBoundSpec());
  driver_spec->store.path = cache_entry_->key();
  driver_spec->data_copy_concurrency = cache.data_copy_concurrency_;
  driver_spec->cache_pool = cache.cache_pool_;
  /// TODO: Fill from pinned entry.
  driver_spec->data_staleness = data_staleness_;
  driver_spec->schema.Set(RankConstraint{3}).IgnoreError();
  driver_spec->schema.Set(dtype_v<uint8_t>).IgnoreError();
  internal::TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  spec.transform = std::move(transform);
  return spec;
}

template <typename Specialization>
Future<IndexTransform<>> ImageDriver<Specialization>::ResolveBounds(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    ResolveBoundsOptions /*options*/) {
  if (transaction) {
    return absl::UnimplementedError(Specialization::kTransactionError);
  }
  return MapFuture(
      data_copy_executor(),
      [self = internal::IntrusivePtr<DriverType>(this),
       transform = std::move(transform)](
          const Result<void>& result) -> Result<IndexTransform<>> {
        if (!result.ok()) {
          return result.status();
        }
        LockType lock{*self->cache_entry_};
        assert(lock.data());
        return PropagateExplicitBoundsToTransform(lock.data()->domain(),
                                                  std::move(transform));
      },
      cache_entry_->Read(data_staleness_.time));
}

// Summary of how read works: The driver yields a single value of type
// ReadChunk, or an error. ReadChunk exposes an interface to zero or more
// "iterables". tensorstore read operations call
//  * `LockCollection` which installs any required locks into the provided
//    LockCollection, then
//  * `BeginRead`, which returns an NDIterable::Ptr actually yielding the data.
//
// In this case, the iterable is over the cache entry data. Any
// `NDIterable::Ptr`s are destroyed before the LockCollection releases the read
// locks.

// Non-transactional `tensorstore::internal::ReadChunk::Impl` Poly interface.
template <typename Specialization>
struct ReadChunkImpl {
  using DriverType = ImageDriver<Specialization>;
  using CacheType = ImageCache<Specialization>;
  using LockType = internal::AsyncCache::ReadLock<typename CacheType::ReadData>;

  internal::IntrusivePtr<DriverType> self;
  internal::PinnedCacheEntry<CacheType> entry;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    return absl::OkStatus();
  }

  Result<internal::NDIterable::Ptr> operator()(internal::ReadChunk::BeginRead,
                                               IndexTransform<> chunk_transform,
                                               internal::Arena* arena) const {
    LockType lock{*entry};
    assert(lock.data());
    return internal::GetTransformedArrayNDIterable(*lock.data(),
                                                   chunk_transform, arena);
  }
};

template <typename Specialization>
void ImageDriver<Specialization>::Read(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>
        receiver) {
  if (transaction) {
    execution::set_starting(receiver, [] {});
    execution::set_error(
        receiver, absl::UnimplementedError(Specialization::kTransactionError));
    execution::set_stopping(receiver);
    return;
  }

  internal::ReadChunk chunk;
  chunk.impl = ReadChunkImpl<Specialization>{
      internal::IntrusivePtr<ImageDriver>(this), cache_entry_};
  chunk.transform = std::move(transform);

  // TODO: Wire in execution::set_cancel correctly.
  execution::set_starting(receiver, [] {});
  auto read_future = cache_entry_->Read(data_staleness_.time);
  read_future.ExecuteWhenReady([chunk = std::move(chunk),
                                receiver = std::move(receiver)](
                                   ReadyFuture<const void> future) mutable {
    auto& r = future.result();
    if (!r.ok()) {
      execution::set_error(receiver, r.status());
    } else {
      auto cell_transform = IdentityTransform(chunk.transform.input_domain());
      execution::set_value(receiver, std::move(chunk),
                           std::move(cell_transform));
      execution::set_done(receiver);
    }
    execution::set_stopping(receiver);
  });
}

}  // namespace
}  // namespace internal_image_driver

// Disable garbage collection.
namespace garbage_collection {
template <typename T>
struct GarbageCollection<internal_image_driver::ImageDriver<T>> {
  static constexpr bool required() { return false; }
};
}  // namespace garbage_collection
}  // namespace tensorstore

#endif
