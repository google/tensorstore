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

#include "tensorstore/kvstore/kvstore.h"

#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/collecting_sender.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/sync_flow_sender.h"

using ::tensorstore::internal::IntrusivePtr;

namespace tensorstore {
namespace kvstore {

void intrusive_ptr_increment(const DriverSpec* p) {
  intrusive_ptr_increment(
      static_cast<const internal::AtomicReferenceCount<DriverSpec>*>(p));
}

void intrusive_ptr_decrement(const DriverSpec* p) {
  intrusive_ptr_decrement(
      static_cast<const internal::AtomicReferenceCount<DriverSpec>*>(p));
}

void intrusive_ptr_increment(Driver* p) {
  p->reference_count_.fetch_add(1, std::memory_order_relaxed);
}

void intrusive_ptr_decrement(Driver* p) {
  if (!internal::DecrementReferenceCountIfGreaterThanOne(p->reference_count_)) {
    p->DestroyLastReference();
  }
}

DriverSpec::~DriverSpec() = default;

absl::Status DriverSpec::NormalizeSpec(std::string& path) {
  return absl::OkStatus();
}

Result<std::string> DriverSpec::ToUrl(std::string_view path) const {
  return absl::UnimplementedError("URL representation not supported");
}

ContextBindingState DriverSpecPtr::context_binding_state() const {
  return get()->context_binding_state_;
}

void EncodeCacheKeyAdl(std::string* out, const DriverSpecPtr& ptr) {
  return ptr->EncodeCacheKey(out);
}

void EncodeCacheKeyAdl(std::string* out, const DriverPtr& ptr) {
  return ptr->EncodeCacheKey(out);
}

Result<Spec> KvStore::spec(SpecRequestOptions&& options) const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto driver_spec,
                               driver->spec(std::move(options)));
  return Spec(std::move(driver_spec), path);
}

Result<std::string> KvStore::ToUrl() const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, this->spec());
  return spec.ToUrl();
}

Result<DriverSpecPtr> Driver::spec(SpecRequestOptions&& options) const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, GetBoundSpec());
  internal::ApplyContextBindingMode(spec, options.context_binding_mode,
                                    /*default_mode=*/ContextBindingMode::strip);
  return spec;
}

Result<DriverSpecPtr> Driver::GetBoundSpec() const {
  return absl::UnimplementedError(
      "KeyValueStore does not support JSON representation");
}

void Driver::EncodeCacheKey(std::string* out) const {
  internal::EncodeCacheKey(out, reinterpret_cast<std::uintptr_t>(this));
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Spec, [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) {
  if constexpr (is_loading) {
    if (auto* s = j->template get_ptr<const std::string*>()) {
      TENSORSTORE_ASSIGN_OR_RETURN(*obj, Spec::FromUrl(*s));
      return absl::OkStatus();
    }
  }
  namespace jb = tensorstore::internal_json_binding;
  auto& registry = internal_kvstore::GetDriverRegistry();
  return jb::NestedContextJsonBinder(jb::Object(
      jb::Member("driver", jb::Projection<&Spec::driver>(registry.KeyBinder())),
      jb::Initialize([](Spec* p) {
        const_cast<DriverSpec&>(*p->driver).context_binding_state_ =
            ContextBindingState::unbound;
      }),
      jb::Member("context", jb::Projection(
                                [](const Spec& p) -> Context::Spec& {
                                  return const_cast<Context::Spec&>(
                                      p.driver->context_spec_);
                                },
                                internal::ContextSpecDefaultableJsonBinder)),
      jb::Member("path", jb::Projection(
                             [](auto& p) -> decltype(auto) { return (p.path); },
                             jb::DefaultInitializedValue())),
      [&](auto is_loading, const auto& options, auto* obj, auto* j) {
        if constexpr (is_loading) {
          TENSORSTORE_RETURN_IF_ERROR(registry.RegisteredObjectBinder()(
              is_loading, {options, obj->path}, &obj->driver, j));
          return const_cast<DriverSpec&>(*obj->driver).NormalizeSpec(obj->path);
        } else {
          return registry.RegisteredObjectBinder()(is_loading, options,
                                                   &obj->driver, j);
        }
      }))(is_loading, options, obj, j);
})

std::ostream& operator<<(std::ostream& os, ReadResult::State state) {
  switch (state) {
    case ReadResult::kUnspecified:
      os << "<unspecified>";
      break;
    case ReadResult::kMissing:
      os << "<missing>";
      break;
    case ReadResult::kValue:
      os << "<value>";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ReadResult& x) {
  std::string value;
  switch (x.state) {
    case ReadResult::kUnspecified:
      value = "<unspecified>";
      break;
    case ReadResult::kMissing:
      value = "<missing>";
      break;
    case ReadResult::kValue:
      value = tensorstore::QuoteString(absl::Cord(x.value).Flatten());
      break;
  }
  return os << "{value=" << value << ", stamp=" << x.stamp << "}";
}
}  // namespace kvstore

namespace internal_kvstore {

DriverRegistry& GetDriverRegistry() {
  static internal::NoDestructor<DriverRegistry> registry;
  return *registry;
}

}  // namespace internal_kvstore

template serialization::Registry&
serialization::GetRegistry<internal::IntrusivePtr<const kvstore::DriverSpec>>();

namespace kvstore {

Driver::~Driver() = default;

Future<KvStore> Open(Spec spec, OpenOptions&& options) {
  return MapFutureValue(
      InlineExecutor{},
      [path = std::move(spec.path),
       transaction =
           std::move(options.transaction)](DriverPtr& driver) mutable {
        return KvStore(std::move(driver), std::move(path),
                       std::move(transaction));
      },
      kvstore::Open(std::move(spec.driver),
                    static_cast<DriverOpenOptions&&>(options)));
}

Future<KvStore> Open(::nlohmann::json json_spec, OpenOptions&& options) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, Spec::FromJson(std::move(json_spec)));
  return Open(std::move(spec), std::move(options));
}

namespace {
struct OpenDriverCache {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, Driver*> map ABSL_GUARDED_BY(mutex);
};

OpenDriverCache& GetOpenDriverCache() {
  static internal::NoDestructor<OpenDriverCache> cache_;
  return *cache_;
}
}  // namespace

Future<DriverPtr> Open(DriverSpecPtr spec, DriverOpenOptions&& options) {
  TENSORSTORE_RETURN_IF_ERROR(spec.BindContext(options.context));
  return MapFutureValue(
      InlineExecutor{},
      [](DriverPtr driver) {
        std::string cache_key;
        driver->EncodeCacheKey(&cache_key);
        auto& open_cache = GetOpenDriverCache();
        absl::MutexLock lock(&open_cache.mutex);
        auto p = open_cache.map.emplace(cache_key, driver.get());
        if (p.second) {
          driver->cache_identifier_ = std::move(cache_key);
        }
#ifdef TENSORSTORE_KVSTORE_OPEN_CACHE_DEBUG
        if (p.second) {
          TENSORSTORE_LOG("Inserted kvstore into cache: ",
                          QuoteString(cache_key));
        } else {
          TENSORSTORE_LOG("Reusing cached kvstore: ", QuoteString(cache_key));
        }
#endif
        return DriverPtr(p.first->second);
      },
      spec->DoOpen());
}

void Driver::DestroyLastReference() {
  auto& open_cache = GetOpenDriverCache();
  if (!cache_identifier_.empty()) {
    // Hold `open_cache.mutex` while decrementing the count to zero, to ensure
    // that it does not concurrently increase due to being retrieved from the
    // cache.
    absl::MutexLock lock(&open_cache.mutex);
    if (reference_count_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
      // Another reference was added concurrently.  Don't destroy.
      return;
    }
    auto it = open_cache.map.find(cache_identifier_);
    if (it != open_cache.map.end()) {
      assert(it->second == this);
      open_cache.map.erase(it);
#ifdef TENSORSTORE_KVSTORE_OPEN_CACHE_DEBUG
      TENSORSTORE_LOG("Removed kvstore from open cache: ",
                      QuoteString(cache_identifier_));
#endif
    }
  } else {
    // Not stored in the open kvstore cache.  We can just decrement the
    // reference count to 0.
    if (reference_count_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
      // Another reference was added concurrently.  Don't destroy.
      return;
    }
  }
  delete this;
}

Future<ReadResult> Driver::Read(Key key, ReadOptions options) {
  return absl::UnimplementedError("KeyValueStore does not support reading");
}

Future<TimestampedStorageGeneration> Driver::Write(Key key,
                                                   std::optional<Value> value,
                                                   WriteOptions options) {
  return absl::UnimplementedError("KeyValueStore does not support writing");
}

Future<void> Driver::DeleteRange(KeyRange range) {
  return absl::UnimplementedError(
      "KeyValueStore does not support deleting by range");
}

void Driver::ListImpl(ListOptions options,
                      AnyFlowReceiver<absl::Status, Key> receiver) {
  execution::submit(FlowSingleSender{ErrorSender{absl::UnimplementedError(
                        "KeyValueStore does not support listing")}},
                    std::move(receiver));
}

AnyFlowSender<absl::Status, Key> Driver::List(ListOptions options) {
  struct ListSender {
    IntrusivePtr<Driver> self;
    ListOptions options;
    void submit(AnyFlowReceiver<absl::Status, Key> receiver) {
      self->ListImpl(options, std::move(receiver));
    }
  };
  return ListSender{IntrusivePtr<Driver>(this), std::move(options)};
}

std::string Driver::DescribeKey(std::string_view key) {
  return tensorstore::QuoteString(key);
}

absl::Status Driver::AnnotateError(std::string_view key,
                                   std::string_view action,
                                   const absl::Status& error) {
  return AnnotateErrorWithKeyDescription(DescribeKey(key), action, error);
}

absl::Status Driver::AnnotateErrorWithKeyDescription(
    std::string_view key_description, std::string_view action,
    const absl::Status& error) {
  if (absl::StrContains(error.message(), key_description)) {
    return error;
  }
  return tensorstore::MaybeAnnotateStatus(
      error, tensorstore::StrCat("Error ", action, " ", key_description));
}

Future<std::vector<Key>> ListFuture(Driver* driver, ListOptions options) {
  return tensorstore::MakeSenderFuture<std::vector<Key>>(
      tensorstore::internal::MakeCollectingSender<std::vector<Key>>(
          tensorstore::MakeSyncFlowSender(driver->List(options))));
}

Future<std::vector<Key>> ListFuture(const KvStore& store, ListOptions options) {
  return tensorstore::MakeSenderFuture<std::vector<Key>>(
      tensorstore::internal::MakeCollectingSender<std::vector<Key>>(
          tensorstore::MakeSyncFlowSender(
              kvstore::List(store, std::move(options)))));
}

absl::Status DriverSpecPtr::Set(SpecConvertOptions&& options) {
  internal::ApplyContextBindingMode(
      *this, options.context_binding_mode,
      /*default_mode=*/ContextBindingMode::retain);
  if (options.context) {
    TENSORSTORE_RETURN_IF_ERROR(BindContext(options.context));
  }
  return absl::OkStatus();
}

absl::Status DriverSpecPtr::BindContext(const Context& context) {
  return internal::BindContextCopyOnWriteWithNestedContext(*this, context);
}

absl::Status Spec::Set(SpecConvertOptions&& options) {
  return driver.Set(std::move(options));
}

void DriverSpecPtr::UnbindContext(
    const internal::ContextSpecBuilder& context_builder) {
  internal::UnbindContextCopyOnWriteWithNestedContext(*this, context_builder);
}

void DriverSpecPtr::StripContext() {
  internal::StripContextCopyOnWriteWithNestedContext(*this);
}

absl::Status Spec::BindContext(const Context& context) {
  return driver.BindContext(context);
}

void Spec::UnbindContext(const internal::ContextSpecBuilder& context_builder) {
  driver.UnbindContext(context_builder);
}

void Spec::StripContext() { driver.StripContext(); }

Result<std::string> Spec::ToUrl() const {
  if (!driver) {
    return absl::InvalidArgumentError("Invalid kvstore spec");
  }
  return driver->ToUrl(path);
}

Future<ReadResult> Read(const KvStore& store, std::string_view key,
                        ReadOptions options) {
  auto full_key = tensorstore::StrCat(store.path, key);
  if (store.transaction == no_transaction) {
    // Regular non-transactional read.
    return store.driver->Read(std::move(full_key), std::move(options));
  }
  if (!StorageGeneration::IsUnknown(options.if_equal)) {
    return absl::UnimplementedError(
        "if_equal condition not supported for transactional reads");
  }
  if (options.byte_range.inclusive_min || options.byte_range.exclusive_max) {
    return absl::UnimplementedError(
        "byte_range restriction not supported for transactional reads");
  }
  TransactionalReadOptions transactional_read_options;
  transactional_read_options.if_not_equal = std::move(options.if_not_equal);
  transactional_read_options.staleness_bound = options.staleness_bound;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  size_t phase;
  return internal_kvstore::ReadViaExistingTransaction(
      store.driver.get(), open_transaction, phase, std::move(full_key),
      std::move(transactional_read_options));
}

Future<TimestampedStorageGeneration> Write(const KvStore& store,
                                           std::string_view key,
                                           std::optional<Value> value,
                                           WriteOptions options) {
  auto full_key = tensorstore::StrCat(store.path, key);
  if (store.transaction == no_transaction) {
    // Regular non-transactional write.
    return store.driver->Write(std::move(full_key), std::move(value),
                               std::move(options));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  size_t phase;
  // Drop the write future; the transactional write completes as soon as the
  // write is applied to the transaction.
  auto future = internal_kvstore::WriteViaExistingTransaction(
      store.driver.get(), open_transaction, phase, std::move(full_key),
      std::move(value), std::move(options));
  if (future.ready()) {
    // An error must have occurred, since a successful write can complete until
    // the transaction is committed, and the transaction cannot commit while we
    // hold an open transaction reference.
    assert(!future.result().ok());
    return future;
  }
  // Just return a dummy stamp; the actual write won't complete until the
  // transaction is committed.
  return TimestampedStorageGeneration();
}

Future<TimestampedStorageGeneration> Delete(const KvStore& store,
                                            std::string_view key,
                                            WriteOptions options) {
  return Write(store, key, std::nullopt, std::move(options));
}

Future<void> DeleteRange(const KvStore& store, KeyRange range) {
  range = KeyRange::AddPrefix(store.path, std::move(range));
  if (store.transaction == no_transaction) {
    return store.driver->DeleteRange(std::move(range));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(store.transaction));
  return store.driver->TransactionalDeleteRange(open_transaction,
                                                std::move(range));
}

namespace {
void AddListOptionsPrefix(ListOptions& options, std::string_view path) {
  options.range = KeyRange::AddPrefix(path, std::move(options.range));
  options.strip_prefix_length += path.size();
}
}  // namespace

void List(const KvStore& store, ListOptions options,
          AnyFlowReceiver<absl::Status, Key> receiver) {
  if (store.transaction != no_transaction) {
    execution::submit(ErrorSender{absl::UnimplementedError(
                          "transactional list not supported")},
                      FlowSingleReceiver{std::move(receiver)});
    return;
  }
  AddListOptionsPrefix(options, store.path);
  store.driver->ListImpl(std::move(options), std::move(receiver));
}

AnyFlowSender<absl::Status, Key> List(const KvStore& store,
                                      ListOptions options) {
  if (store.transaction != no_transaction) {
    return ErrorSender{
        absl::UnimplementedError("transactional list not supported")};
  }
  AddListOptionsPrefix(options, store.path);
  return store.driver->List(std::move(options));
}

bool operator==(const KvStore& a, const KvStore& b) {
  return a.driver == b.driver && a.path == b.path &&
         a.transaction == b.transaction;
}

}  // namespace kvstore

namespace serialization {

namespace {

using DriverSpecPtrNonNullDirectSerializer =
    RegistrySerializer<internal::IntrusivePtr<const kvstore::DriverSpec>>;

using DriverSpecPtrSerializer =
    IndirectPointerSerializer<internal::IntrusivePtr<const kvstore::DriverSpec>,
                              DriverSpecPtrNonNullDirectSerializer>;

using DriverSpecPtrNonNullSerializer = NonNullIndirectPointerSerializer<
    internal::IntrusivePtr<const kvstore::DriverSpec>,
    DriverSpecPtrNonNullDirectSerializer>;

struct DriverPtrNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink,
                                   const kvstore::DriverPtr& value) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto driver_spec, value->spec(retain_context),
                                 (sink.Fail(_), false));
    return DriverSpecPtrNonNullSerializer().Encode(sink, driver_spec);
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   kvstore::DriverPtr& value) {
    kvstore::DriverSpecPtr driver_spec;
    if (!DriverSpecPtrNonNullSerializer().Decode(source, driver_spec)) {
      return false;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(value,
                                 kvstore::Open(std::move(driver_spec)).result(),
                                 (source.Fail(_), false));
    return true;
  }
};

using DriverPtrSerializer =
    IndirectPointerSerializer<kvstore::DriverPtr,
                              DriverPtrNonNullDirectSerializer>;

}  // namespace

}  // namespace serialization

namespace internal_json_binding {
TENSORSTORE_DEFINE_JSON_BINDER(
    KvStoreSpecAndPathJsonBinder,
    Sequence(Member("kvstore", DefaultInitializedPredicate([](auto* obj) {
                      return !obj->valid();
                    })),
             // DEPRECATED: "path" is supported for backward compatibility only.
             LoadSave(OptionalMember(
                 "path",
                 Compose<std::string>([](auto is_loading, const auto& options,
                                         auto* obj, std::string* j) {
                   if (!obj->valid()) {
                     return absl::InvalidArgumentError(
                         "\"path\" must be specified in conjunction with "
                         "\"kvstore\"");
                   }
                   obj->AppendPathComponent(*j);
                   return absl::OkStatus();
                 })))))
}  // namespace internal_json_binding

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr,
    tensorstore::serialization::DriverSpecPtrSerializer())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::DriverPtr,
    tensorstore::serialization::DriverPtrSerializer())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::Spec,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::kvstore::Spec>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::KvStore,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::kvstore::KvStore>())

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Driver,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::kvstore::Driver>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpec,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::kvstore::DriverSpec>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Spec,
    tensorstore::garbage_collection::ApplyMembersGarbageCollection<
        tensorstore::kvstore::Spec>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr,
    tensorstore::garbage_collection::IndirectPointerGarbageCollection<
        tensorstore::kvstore::DriverSpecPtr>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::KvStore,
    tensorstore::garbage_collection::ApplyMembersGarbageCollection<
        tensorstore::kvstore::KvStore>)
