// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/driver/driver.h"

#include <stddef.h>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include <nlohmann/json.hpp>
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/json/json_change_map.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/async_initialized_cache_mixin.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"
#include "tensorstore/internal/json_pointer.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {
namespace {

namespace jb = tensorstore::internal_json_binding;

Result<::nlohmann::json> DecodeJson(const std::optional<absl::Cord>& data) {
  if (!data) {
    return ::nlohmann::json(::nlohmann::json::value_t::discarded);
  }
  nlohmann::json raw_data =
      nlohmann::json::parse(absl::Cord(*data).Flatten(), nullptr,
                            /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  return raw_data;
}

class JsonCache
    : public internal::KvsBackedCache<JsonCache, internal::AsyncCache>,
      public AsyncInitializedCacheMixin {
  using Base = internal::KvsBackedCache<JsonCache, internal::AsyncCache>;

 public:
  using ReadData = ::nlohmann::json;

  JsonCache() : Base(kvstore::DriverPtr()) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = JsonCache;
    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [value = std::move(value), receiver = std::move(receiver)]() mutable {
            auto decode_result = DecodeJson(value);
            if (!decode_result.ok()) {
              execution::set_error(receiver, decode_result.status());
              return;
            }
            execution::set_value(receiver, std::make_shared<::nlohmann::json>(
                                               std::move(*decode_result)));
          });
    }
    void DoEncode(std::shared_ptr<const ReadData> data,
                  EncodeReceiver receiver) override {
      const auto& json_value = *data;
      if (json_value.is_discarded()) {
        execution::set_value(receiver, std::nullopt);
        return;
      }
      execution::set_value(receiver, absl::Cord(json_value.dump()));
    }
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = JsonCache;
    using Base::TransactionNode::TransactionNode;
    void DoApply(ApplyOptions options, ApplyReceiver receiver) override {
      // Determine whether a read is required to compute the updated state
      // (i.e. whether this transaction node completely overwrites the state).
      const bool unconditional = changes_.CanApplyUnconditionally({});
      const bool unmodified = changes_.underlying_map().empty();

      // Asynchronous continuation run once the read of the existing state, if
      // needed, has completed.
      auto continuation =
          [this, receiver = std::move(receiver), unconditional, unmodified,
           specify_unchanged =
               (options.apply_mode == ApplyOptions::kSpecifyUnchanged)](
              ReadyFuture<const void> future) mutable {
            if (!future.result().ok()) {
              // Propagate read error.
              execution::set_error(receiver, future.result().status());
              return;
            }

            AsyncCache::ReadState read_state;
            if (unconditional || (unmodified && !specify_unchanged)) {
              read_state.stamp = TimestampedStorageGeneration::Unconditional();
            } else {
              read_state = AsyncCache::ReadLock<void>(*this).read_state();
            }

            if (!unmodified) {
              auto* existing_json =
                  static_cast<const ::nlohmann::json*>(read_state.data.get());
              ::nlohmann::json new_json;
              // Apply changes.  If `existing_state` is non-null (equivalent to
              // `unconditional == false`), provide it to `Apply`.  Otherwise,
              // pass in a placeholder value (which won't be used).
              auto result = changes_.Apply(
                  existing_json
                      ? *existing_json
                      : ::nlohmann::json(::nlohmann::json::value_t::discarded));
              if (result.ok()) {
                new_json = std::move(*result);
              } else {
                execution::set_error(receiver, std::move(result).status());
                return;
              }
              // For conditional states, only mark dirty if it differs from the
              // existing state, since otherwise the writeback can be skipped
              // (and instead the state can just be verified).
              if (!existing_json ||
                  !internal_json::JsonSame(new_json, *existing_json)) {
                read_state.stamp.generation.MarkDirty();
                read_state.data =
                    std::make_shared<::nlohmann::json>(std::move(new_json));
              }
            }
            execution::set_value(receiver, std::move(read_state));
          };
      auto future = ((unconditional ||
                      (unmodified &&
                       options.apply_mode != ApplyOptions::kSpecifyUnchanged))
                         ? MakeReadyFuture()
                         : this->Read({options.staleness_bound}));
      future.Force();
      std::move(future).ExecuteWhenReady(WithExecutor(
          GetOwningCache(*this).executor(), std::move(continuation)));
    }
    internal_json_driver::JsonChangeMap changes_;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  const Executor& executor() { return data_copy_concurrency_->executor; }

  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
};

class JsonDriverSpec
    : public RegisteredDriverSpec<JsonDriverSpec,
                                  /*Parent=*/internal::DriverSpec> {
 public:
  constexpr static char id[] = "json";

  kvstore::Spec store;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;
  Context::Resource<CachePoolResource> cache_pool;
  StalenessBound data_staleness;
  std::string json_pointer;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.store,
             x.data_copy_concurrency, x.cache_pool, x.data_staleness,
             x.json_pointer);
  };

  static absl::Status ValidateSchema(Schema& schema) {
    TENSORSTORE_RETURN_IF_ERROR(
        schema.Set(dtype_v<::tensorstore::dtypes::json_t>));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(RankConstraint{0}));
    if (schema.codec().valid()) {
      return absl::InvalidArgumentError("codec not supported by json driver");
    }
    if (schema.fill_value().valid()) {
      return absl::InvalidArgumentError(
          "fill_value not supported by json driver");
    }
    return absl::OkStatus();
  }

  constexpr static auto default_json_binder = jb::Sequence(
      jb::Initialize([](auto* obj) -> absl::Status {
        return ValidateSchema(obj->schema);
      }),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<&JsonDriverSpec::data_copy_concurrency>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&JsonDriverSpec::cache_pool>()),
      jb::Projection<&JsonDriverSpec::store>(jb::KvStoreSpecAndPathJsonBinder),
      jb::Member(
          "recheck_cached_data",
          jb::Projection<&JsonDriverSpec::data_staleness>(jb::DefaultValue(
              [](auto* obj) { obj->bounded_by_open_time = true; }))),
      jb::Member("json_pointer",
                 jb::Projection<&JsonDriverSpec::json_pointer>(jb::Validate(
                     [](const auto& options, auto* obj) {
                       return tensorstore::json_pointer::Validate(*obj);
                     },
                     jb::DefaultInitializedValue()))));

  OpenMode open_mode() const override {
    // Since opening itself has no side effects, we return `open`.  A missing
    // json file is not actually created until the first write operation.
    return OpenMode::open;
  }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    // A json driver contains both the data and the metadata, so set the
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

  Result<IndexDomain<>> GetDomain() const override { return IndexDomain<>(0); }

  Result<ChunkLayout> GetChunkLayout() const override {
    ChunkLayout layout;
    layout.Set(RankConstraint{0}).IgnoreError();
    return layout;
  }

  kvstore::Spec GetKvstore() const override { return store; }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

class JsonDriver : public RegisteredDriver<JsonDriver,
                                           /*Parent=*/internal::Driver> {
 public:
  KvStore GetKvstore(const Transaction& transaction) override;

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    ChunkLayout layout;
    layout.Set(RankConstraint{0}).IgnoreError();
    return layout | transform;
  }

  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  DataType dtype() override { return dtype_v<::tensorstore::dtypes::json_t>; }
  DimensionIndex rank() override { return 0; }  // COV_NF_LINE

  Executor data_copy_executor() override {
    return GetOwningCache(*cache_entry_).executor();
  }

  void Read(ReadRequest request,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;
  void Write(WriteRequest request,
             AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>
                 receiver) override;

  PinnedCacheEntry<JsonCache> cache_entry_;
  StalenessBound data_staleness_;
  std::string json_pointer_;
};

Future<internal::Driver::Handle> JsonDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  if (request.read_write_mode == ReadWriteMode::dynamic) {
    request.read_write_mode = ReadWriteMode::read_write;
  }
  if (!store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }
  std::string cache_identifier;
  auto request_time = absl::Now();
  internal::EncodeCacheKey(&cache_identifier, store.driver,
                           data_copy_concurrency);
  auto cache = internal::GetOrCreateAsyncInitializedCache<JsonCache>(
      cache_pool->get(), cache_identifier,
      [&] {
        auto cache = std::make_unique<JsonCache>();
        cache->data_copy_concurrency_ = data_copy_concurrency;
        cache->cache_pool_ = cache_pool;
        return cache;
      },
      [&](Promise<void> initialize_promise, CachePtr<JsonCache> cache) {
        // The cache didn't previously exist.  Open the KeyValueStore.
        LinkValue(
            [cache = std::move(cache)](Promise<void> cache_promise,
                                       ReadyFuture<kvstore::DriverPtr> future) {
              cache->SetKvStoreDriver(std::move(*future.result()));
            },
            initialize_promise, kvstore::Open(store.driver));
      });
  auto driver = internal::MakeReadWritePtr<JsonDriver>(request.read_write_mode);
  driver->cache_entry_ = GetCacheEntry(cache, store.path);
  driver->json_pointer_ = json_pointer;
  driver->data_staleness_ = data_staleness.BoundAtOpen(request_time);
  return PromiseFuturePair<internal::Driver::Handle>::LinkError(
             internal::Driver::Handle{std::move(driver), IdentityTransform(0),
                                      internal::TransactionState::ToTransaction(
                                          std::move(request.transaction))},
             cache->initialized_)
      .future;
}

Result<TransformedDriverSpec> JsonDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  auto driver_spec = DriverSpec::Make<JsonDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  auto& cache = GetOwningCache(*cache_entry_);
  TENSORSTORE_ASSIGN_OR_RETURN(driver_spec->store.driver,
                               cache.kvstore_driver()->GetBoundSpec());
  driver_spec->store.path = cache_entry_->key();
  driver_spec->data_copy_concurrency = cache.data_copy_concurrency_;
  driver_spec->cache_pool = cache.cache_pool_;
  driver_spec->data_staleness = data_staleness_;
  driver_spec->json_pointer = json_pointer_;
  driver_spec->schema.Set(RankConstraint{0}).IgnoreError();
  driver_spec->schema.Set(dtype_v<::tensorstore::dtypes::json_t>).IgnoreError();
  TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  spec.transform = std::move(transform);
  return spec;
}

KvStore JsonDriver::GetKvstore(const Transaction& transaction) {
  auto& cache = GetOwningCache(*cache_entry_);
  return KvStore(kvstore::DriverPtr(cache.kvstore_driver()),
                 std::string(cache_entry_->key()), transaction);
}

/// TensorStore Driver ReadChunk implementation for the case of a
/// non-transactional read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly
/// interface.
struct ReadChunkImpl {
  PinnedCacheEntry<JsonCache> entry;
  IntrusivePtr<JsonDriver> driver;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    // No locks need to be held throughout read operation.  A temporary lock is
    // held in the `BeginRead` method below only while copying the shared_ptr to
    // the immutable data.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    // Note that this acquires a lock on the entry, not the node, and
    // therefore does not conflict with the lock registered with the
    // `LockCollection`.
    std::shared_ptr<const ::nlohmann::json> read_value =
        AsyncCache::ReadLock<JsonCache::ReadData>(*entry).shared_data();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto* sub_value,
        json_pointer::Dereference(*read_value, driver->json_pointer_),
        entry->AnnotateError(_, /*reading=*/true));
    return GetTransformedArrayNDIterable(
        std::shared_ptr<const ::nlohmann::json>(std::move(read_value),
                                                sub_value),
        std::move(chunk_transform), arena);
  }
};

/// TensorStore Driver ReadChunk implementation for the case of a transactional
/// read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly
/// interface.
struct ReadChunkTransactionImpl {
  OpenTransactionNodePtr<JsonCache::TransactionNode> node;
  IntrusivePtr<JsonDriver> driver;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    // No locks need to be held throughout read operation.  A temporary lock is
    // held in the `BeginRead` method below only while copying the value.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) {
    std::shared_ptr<const ::nlohmann::json> existing_value;
    StorageGeneration read_generation;
    {
      AsyncCache::ReadLock<JsonCache::ReadData> lock(*node);
      existing_value = lock.shared_data();
      read_generation = lock.stamp().generation;
    }
    auto value = std::allocate_shared<::nlohmann::json>(
        ArenaAllocator<::nlohmann::json>(arena));
    {
      UniqueWriterLock lock(*node);
      if ((node->transaction()->mode() & repeatable_read) &&
          !node->changes_.CanApplyUnconditionally(driver->json_pointer_)) {
        TENSORSTORE_RETURN_IF_ERROR(
            node->RequireRepeatableRead(read_generation));
      }
      assert(existing_value ||
             node->changes_.CanApplyUnconditionally(driver->json_pointer_));
      TENSORSTORE_ASSIGN_OR_RETURN(
          *value,
          node->changes_.Apply(
              existing_value
                  ? *existing_value
                  : ::nlohmann::json(::nlohmann::json::value_t::discarded),
              driver->json_pointer_),
          GetOwningEntry(*node).AnnotateError(_, /*reading=*/true));
    }
    return GetTransformedArrayNDIterable(std::move(value), chunk_transform,
                                         arena);
  }
};

void JsonDriver::Read(
    ReadRequest request,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  ReadChunk chunk;
  chunk.transform = std::move(request.transform);
  auto read_future = [&]() -> Future<const void> {
    const auto get_cache_read_request = [&] {
      AsyncCache::AsyncCacheReadRequest cache_read_request;
      cache_read_request.staleness_bound = data_staleness_.time;
      cache_read_request.batch = request.batch;
      return cache_read_request;
    };
    if (request.transaction) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto node, GetTransactionNode(*cache_entry_, request.transaction));
      const bool unconditional = [&] {
        UniqueWriterLock<AsyncCache::TransactionNode> lock(*node);
        return node->changes_.CanApplyUnconditionally(json_pointer_);
      }();
      auto read_future = unconditional ? MakeReadyFuture()
                                       : node->Read(get_cache_read_request());
      chunk.impl = ReadChunkTransactionImpl{std::move(node),
                                            IntrusivePtr<JsonDriver>(this)};
      return read_future;
    } else {
      chunk.impl = ReadChunkImpl{cache_entry_, IntrusivePtr<JsonDriver>(this)};
      return cache_entry_->Read(get_cache_read_request());
    }
  }();
  read_future.ExecuteWhenReady(
      [chunk = std::move(chunk),
       single_receiver = FlowSingleReceiver{std::move(receiver)}](
          ReadyFuture<const void> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          execution::set_error(single_receiver, r.status());
          return;
        }
        auto cell_transform = IdentityTransform(chunk.transform.domain());
        execution::set_value(single_receiver, std::move(chunk),
                             std::move(cell_transform));
      });
}

/// TensorStore Driver WriteChunk implementation.
///
/// This implements the `tensorstore::internal::WriteChunk::Impl` Poly
/// interface.
struct WriteChunkImpl {
  PinnedCacheEntry<JsonCache> entry;
  OpenTransactionPtr transaction;
  IntrusivePtr<JsonDriver> driver;
  // Temporary value that will be modified by writer.
  ::nlohmann::json value;

  absl::Status operator()(internal::LockCollection& lock_collection) {
    // No locks need to be held throughout write operation.  A temporary lock is
    // held in the `BeginWrite` method below only while copying the value.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) {
    // Return NDIterable that references `this->value`.  The write is not
    // recorded in the `JsonCache::TransactionNode` until `EndWrite` is called.
    return GetTransformedArrayNDIterable(UnownedToShared(&value),
                                         chunk_transform, arena);
  }

  WriteChunk::EndWriteResult operator()(WriteChunk::EndWrite,
                                        IndexTransformView<> chunk_transform,
                                        bool success, Arena* arena) {
    if (chunk_transform.domain().box().is_empty()) {
      return {};
    }
    if (!success) return {};
    const auto convert_error = [&](const absl::Status& error) {
      return WriteChunk::EndWriteResult{
          /*.copy_status=*/entry->AnnotateError(error, /*reading=*/false),
          /*.commit_future=*/{}};
    };
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node, GetWriteLockedTransactionNode(*entry, transaction),
        convert_error(_));
    TENSORSTORE_RETURN_IF_ERROR(
        node->changes_.AddChange(driver->json_pointer_, std::move(value)),
        convert_error(_));
    return {/*.copy_status=*/{},
            /*.commit_future=*/node->transaction()->future()};
  }
};

void JsonDriver::Write(
    WriteRequest request,
    AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>> receiver) {
  auto cell_transform = IdentityTransform(request.transform.domain());
  execution::set_value(
      FlowSingleReceiver{std::move(receiver)},
      WriteChunk{WriteChunkImpl{cache_entry_, std::move(request.transaction),
                                IntrusivePtr<JsonDriver>(this)},
                 std::move(request.transform)},
      std::move(cell_transform));
}

}  // namespace
}  // namespace internal

namespace garbage_collection {
template <>
struct GarbageCollection<internal::JsonDriver> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const internal::JsonDriver& value) {
    auto& cache = GetOwningCache(*value.cache_entry_);
    return garbage_collection::GarbageCollectionVisit(visitor,
                                                      *cache.kvstore_driver());
  }
};
}  // namespace garbage_collection

namespace internal {
namespace {

const internal::DriverRegistration<JsonDriverSpec> driver_registration;

}  // namespace

}  // namespace internal
}  // namespace tensorstore
