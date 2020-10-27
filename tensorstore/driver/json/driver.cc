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

#include "tensorstore/driver/json/json_change_map.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/internal/async_cache.h"
#include "tensorstore/internal/async_initialized_cache_mixin.h"
#include "tensorstore/internal/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_pointer.h"
#include "tensorstore/internal/kvs_backed_cache.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/staleness_bound_json_binder.h"
#include "tensorstore/util/sender.h"

namespace tensorstore {
namespace internal {
namespace {

namespace jb = tensorstore::internal::json_binding;

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

class JsonCache;
using JsonCacheBase = internal::AsyncCacheBase<
    JsonCache, internal::KvsBackedCache<JsonCache, internal::AsyncCache>>;
class JsonCache : public JsonCacheBase, public AsyncInitializedCacheMixin {
  using Base = JsonCacheBase;

 public:
  using ReadData = ::nlohmann::json;

  JsonCache() : JsonCacheBase(KeyValueStore::Ptr()) {}

  class Entry : public Base::Entry {
   public:
    using Cache = JsonCache;
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
                  UniqueWriterLock<AsyncCache::TransactionNode> lock,
                  EncodeReceiver receiver) override {
      lock.unlock();
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
    using Cache = JsonCache;
    using Base::TransactionNode::TransactionNode;
    void DoApply(ApplyOptions options, ApplyReceiver receiver) override {
      // Determine whether a read is required to compute the updated state
      // (i.e. whether this transaction node completely overwrites the state).
      const bool unconditional = [&] {
        UniqueWriterLock<AsyncCache::TransactionNode> lock(*this);
        return changes_.CanApplyUnconditionally({});
      }();
      // Asynchronous continuation run once the read of the existing state, if
      // needed, has completed.
      auto continuation = [this, receiver = std::move(receiver), unconditional](
                              ReadyFuture<const void> future) mutable {
        if (!future.result().ok()) {
          // Propagate read error.
          execution::set_error(receiver, future.result().status());
          return;
        }
        AsyncCache::ReadState read_state;
        if (!unconditional) {
          read_state =
              AsyncCache::ReadLock<::nlohmann::json>(*this).read_state();
        } else {
          read_state.stamp = TimestampedStorageGeneration::Unconditional();
        }
        auto* existing_json =
            static_cast<const ::nlohmann::json*>(read_state.data.get());
        ::nlohmann::json new_json;
        {
          auto result = [&] {
            UniqueWriterLock<AsyncCache::TransactionNode> lock(*this);
            // Apply changes.  If `existing_state` is non-null (equivalent to
            // `unconditional == false`), provide it to `Apply`.  Otherwise,
            // pass in a dummy value (which won't be used).
            return changes_.Apply(
                existing_json
                    ? *existing_json
                    : ::nlohmann::json(::nlohmann::json::value_t::discarded));
          }();
          if (result.ok()) {
            new_json = std::move(*result);
          } else {
            execution::set_error(receiver, std::move(result).status());
            return;
          }
        }
        // For conditional states, only mark dirty if it differs from the
        // existing state, since otherwise the writeback can be skipped (and
        // instead the state can just be verified).
        if (!existing_json ||
            !internal_json::JsonSame(new_json, *existing_json)) {
          read_state.stamp.generation.MarkDirty();
          read_state.data =
              std::make_shared<::nlohmann::json>(std::move(new_json));
        }
        execution::set_value(receiver, std::move(read_state),
                             UniqueWriterLock<AsyncCache::TransactionNode>{});
      };
      (unconditional ? MakeReadyFuture() : this->Read(options.staleness_bound))
          .ExecuteWhenReady(WithExecutor(GetOwningCache(*this).executor(),
                                         std::move(continuation)));
    }
    internal_json_driver::JsonChangeMap changes_;
  };

  const Executor& executor() { return data_copy_concurrency_->executor; }

  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
};

class JsonDriver
    : public RegisteredDriver<JsonDriver, /*Parent=*/internal::Driver> {
 public:
  constexpr static char id[] = "json";

  template <template <typename> class MaybeBound = internal::ContextUnbound>
  struct SpecT : public internal::DriverConstraints {
    MaybeBound<KeyValueStore::Spec::Ptr> store;
    std::string path;
    MaybeBound<Context::ResourceSpec<DataCopyConcurrencyResource>>
        data_copy_concurrency;
    MaybeBound<Context::ResourceSpec<CachePoolResource>> cache_pool;
    StalenessBound data_staleness;
    std::string json_pointer;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<internal::DriverConstraints>(x), x.store,
               x.path, x.data_copy_concurrency, x.cache_pool, x.data_staleness,
               x.json_pointer);
    };
  };

  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  constexpr static auto json_binder = jb::Sequence(
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection(&SpecT<>::data_copy_concurrency)),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection(&SpecT<>::cache_pool)),
      jb::Member("kvstore", jb::Projection(&SpecT<>::store)),
      jb::Member("path", jb::Projection(&SpecT<>::path)),
      jb::Member("recheck_cached_data",
                 jb::Projection(&SpecT<>::data_staleness,
                                jb::DefaultValue([](auto* obj) {
                                  obj->bounded_by_open_time = true;
                                }))),
      jb::Member(
          "json_pointer",
          jb::Projection(&SpecT<>::json_pointer,
                         jb::Validate(
                             [](const auto& options, auto* obj) {
                               return tensorstore::json_pointer::Validate(*obj);
                             },
                             jb::DefaultInitializedValue()))));

  static absl::Status ConvertSpec(SpecT<ContextUnbound>* spec,
                                  const SpecRequestOptions& options) {
    return absl::OkStatus();
  }

  static Future<internal::Driver::ReadWriteHandle> Open(
      internal::OpenTransactionPtr transaction,
      internal::RegisteredDriverOpener<BoundSpecData> spec,
      ReadWriteMode read_write_mode);

  Result<IndexTransformSpec> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, SpecT<ContextBound>* spec,
      IndexTransformView<> transform) const;

  DataType data_type() override { return DataTypeOf<json_t>(); }
  DimensionIndex rank() override { return 0; }  // COV_NF_LINE

  Executor data_copy_executor() override {
    return GetOwningCache(cache_entry_)->executor();
  }

  void Read(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) override;
  void Write(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) override;

  PinnedCacheEntry<JsonCache> cache_entry_;
  StalenessBound data_staleness_;
  std::string json_pointer_;
};

Future<internal::Driver::ReadWriteHandle> JsonDriver::Open(
    internal::OpenTransactionPtr transaction,
    internal::RegisteredDriverOpener<BoundSpecData> spec,
    ReadWriteMode read_write_mode) {
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = ReadWriteMode::read_write;
  }
  std::string cache_identifier;
  auto request_time = absl::Now();
  internal::EncodeCacheKey(&cache_identifier, spec->store,
                           spec->data_copy_concurrency);
  auto cache = internal::GetOrCreateAsyncInitializedCache<JsonCache>(
      **spec->cache_pool, cache_identifier,
      [&] {
        auto cache = std::make_unique<JsonCache>();
        cache->data_copy_concurrency_ = spec->data_copy_concurrency;
        cache->cache_pool_ = spec->cache_pool;
        return cache;
      },
      [&](Promise<void> initialize_promise, CachePtr<JsonCache> cache) {
        // The cache didn't previously exist.  Open the KeyValueStore.
        LinkValue(
            [cache = std::move(cache)](Promise<void> cache_promise,
                                       ReadyFuture<KeyValueStore::Ptr> future) {
              cache->SetKeyValueStore(std::move(*future.result()));
            },
            initialize_promise, spec->store->Open());
      });
  internal::Driver::PtrT<JsonDriver> driver(new JsonDriver);
  driver->cache_entry_ = GetCacheEntry(cache, spec->path);
  driver->json_pointer_ = spec->json_pointer;
  driver->data_staleness_ = spec->data_staleness.BoundAtOpen(request_time);
  return PromiseFuturePair<internal::Driver::ReadWriteHandle>::LinkError(
             internal::Driver::ReadWriteHandle{
                 {std::move(driver), IdentityTransform(0),
                  internal::TransactionState::ToTransaction(
                      std::move(transaction))},
                 read_write_mode},
             cache->initialized_)
      .future;
}

Result<IndexTransformSpec> JsonDriver::GetBoundSpecData(
    internal::OpenTransactionPtr transaction, SpecT<ContextBound>* spec,
    IndexTransformView<> transform) const {
  auto* cache = GetOwningCache(cache_entry_);
  TENSORSTORE_ASSIGN_OR_RETURN(spec->store, cache->kvstore()->GetBoundSpec());
  spec->path = std::string(cache_entry_->key());
  spec->data_copy_concurrency = cache->data_copy_concurrency_;
  spec->cache_pool = cache->cache_pool_;
  spec->data_staleness = data_staleness_;
  spec->json_pointer = json_pointer_;
  spec->rank = 0;
  spec->data_type = DataTypeOf<json_t>();
  return IndexTransformSpec(transform);
}

/// TensorStore Driver ReadChunk implementation for the case of a
/// non-transactional read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly interface.
struct ReadChunkImpl {
  PinnedCacheEntry<JsonCache> entry;
  internal::Driver::PtrT<JsonDriver> driver;

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
        {SharedArrayView<const void>(std::shared_ptr<const ::nlohmann::json>(
             std::move(read_value), sub_value)),
         std::move(chunk_transform)},
        arena);
  }
};

/// TensorStore Driver ReadChunk implementation for the case of a transactional
/// read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly interface.
struct ReadChunkTransactionImpl {
  OpenTransactionNodePtr<JsonCache::TransactionNode> node;
  internal::Driver::PtrT<JsonDriver> driver;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    // No locks need to be held throughout read operation.  A temporary lock is
    // held in the `BeginRead` method below only while copying the value.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) {
    auto existing_value =
        AsyncCache::ReadLock<JsonCache::ReadData>(*node).shared_data();
    auto value = std::allocate_shared<::nlohmann::json>(
        ArenaAllocator<::nlohmann::json>(arena));
    {
      UniqueWriterLock lock(*node);
      TENSORSTORE_ASSIGN_OR_RETURN(
          *value, node->changes_.Apply(*existing_value, driver->json_pointer_),
          GetOwningEntry(*node).AnnotateError(_, /*reading=*/true));
    }
    return GetTransformedArrayNDIterable(
        {SharedOffsetArrayView<const void>(std::move(value)),
         std::move(chunk_transform)},
        arena);
  }
};

void JsonDriver::Read(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  ReadChunk chunk;
  chunk.transform = std::move(transform);
  auto read_future = [&]() -> Future<const void> {
    if (transaction) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto node, GetTransactionNode(*cache_entry_, transaction));
      auto read_future = node->changes_.CanApplyUnconditionally(json_pointer_)
                             ? MakeReadyFuture()
                             : node->Read(data_staleness_);
      chunk.impl = ReadChunkTransactionImpl{
          std::move(node), internal::Driver::PtrT<JsonDriver>(this)};
      return read_future;
    } else {
      chunk.impl =
          ReadChunkImpl{cache_entry_, internal::Driver::PtrT<JsonDriver>(this)};
      return cache_entry_->Read(data_staleness_);
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
  internal::Driver::PtrT<JsonDriver> driver;
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
    return GetTransformedArrayNDIterable(
        {SharedOffsetArrayView<const void>(UnownedToShared(&value)),
         std::move(chunk_transform)},
        arena);
  }

  WriteChunk::EndWriteResult operator()(WriteChunk::EndWrite,
                                        IndexTransformView<> chunk_transform,
                                        NDIterable::IterationLayoutView layout,
                                        span<const Index> write_end_position,
                                        Arena* arena) {
    // There is only 1 element, so any non-zero index in write_end_position
    // means it was written.
    const bool modified =
        std::any_of(write_end_position.begin(), write_end_position.end(),
                    [](Index x) { return x != 0; });
    if (!modified) return {};
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
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  auto cell_transform = IdentityTransform(transform.domain());
  execution::set_value(
      FlowSingleReceiver{std::move(receiver)},
      WriteChunk{WriteChunkImpl{cache_entry_, std::move(transaction),
                                internal::Driver::PtrT<JsonDriver>(this)},
                 std::move(transform)},
      std::move(cell_transform));
}

const internal::DriverRegistration<JsonDriver> driver_registration;

}  // namespace

}  // namespace internal
}  // namespace tensorstore
