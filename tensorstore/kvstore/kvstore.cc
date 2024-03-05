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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/registry.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal::IntrusivePtr;

namespace tensorstore {
namespace kvstore {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag kvstore_cache_logging(
    "kvstore_cache");
}

void intrusive_ptr_increment(Driver* p) {
  p->reference_count_.fetch_add(1, std::memory_order_relaxed);
}

void intrusive_ptr_decrement(Driver* p) {
  if (!internal::DecrementReferenceCountIfGreaterThanOne(p->reference_count_)) {
    p->DestroyLastReference();
  }
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

Result<KvStore> KvStore::base() const {
  return driver->GetBase(path, transaction);
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

SupportedFeatures Driver::GetSupportedFeatures(
    const KeyRange& key_range) const {
  return SupportedFeatures::kNone;
}

void Driver::EncodeCacheKey(std::string* out) const {
  internal::EncodeCacheKey(out, reinterpret_cast<uintptr_t>(this));
}

Result<KvStore> Driver::GetBase(std::string_view path,
                                const Transaction& transaction) const {
  return {std::in_place};
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
  if (!spec.valid()) {
    return absl::InvalidArgumentError("Cannot open null kvstore spec");
  }
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
          ABSL_LOG_IF(INFO, kvstore_cache_logging)
              << "Inserted kvstore into cache: "
              << QuoteString(driver->cache_identifier_);
        } else {
          ABSL_LOG_IF(INFO, kvstore_cache_logging)
              << "Reusing cached kvstore: " << QuoteString(cache_key);
        }
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
      ABSL_LOG_IF(INFO, kvstore_cache_logging)
          << "Removed kvstore from open cache: "
          << QuoteString(cache_identifier_);
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

#if 0  // Default CopyRange implementation disabled currently.
namespace {
struct CopyRangeListReceiver
    : public internal::AtomicReferenceCount<CopyRangeListReceiver> {
  using Ptr = internal::IntrusivePtr<CopyRangeListReceiver>;
  internal::OpenTransactionPtr target_transaction;
  DriverPtr source_driver;
  absl::Time source_staleness_bound;
  DriverPtr target_driver;
  size_t source_prefix_length;
  std::string target_prefix;
  Promise<void> promise;
  FutureCallbackRegistration cancel_registration;

  template <typename Cancel>
  friend void set_starting(const Ptr& self, Cancel&& cancel) {
    self->cancel_registration =
        self->promise.ExecuteWhenNotNeeded(std::forward<Cancel>(cancel));
  }

  friend void set_stopping(const Ptr& self) {
    self->cancel_registration.Unregister();
  }

  friend void set_error(const Ptr& self, absl::Status&& error) {
    SetDeferredResult(self->promise, std::move(error));
  }

  friend void set_done(const Ptr& self) {}

  friend void set_value(const Ptr& self, std::string&& key) {
    ReadOptions options;
    options.staleness_bound = self->source_staleness_bound;
    std::string target_key = absl::StrCat(
        self->target_prefix, std::string_view(key).substr(std::min(
                                 self->source_prefix_length, key.size())));
    auto read_future =
        self->source_driver->Read(std::move(key), std::move(options));
    Link(
        [self, target_key = std::move(target_key)](
            Promise<void> promise, ReadyFuture<ReadResult> future) {
          TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                       std::move(future.result()),
                                       SetDeferredResult(self->promise, _));
          if (!read_result.has_value()) return;
          Link(
              [](Promise<void> promise,
                 ReadyFuture<TimestampedStorageGeneration> future) {
                TENSORSTORE_RETURN_IF_ERROR(future.result(),
                                            SetDeferredResult(promise, _));
              },
              std::move(promise),
              kvstore::Write(KvStore(self->target_driver, std::move(target_key),
                                     internal::TransactionState::ToTransaction(
                                         self->target_transaction)),
                             "", read_result.value));
        },
        self->promise, std::move(read_future));
  }
};
}  // namespace
#endif

Future<const void> Driver::ExperimentalCopyRangeFrom(
    const internal::OpenTransactionPtr& transaction, const KvStore& source,
    Key target_prefix, CopyRangeOptions options) {
  return absl::UnimplementedError("CopyRange not supported");
#if 0
  auto receiver = internal::MakeIntrusivePtr<CopyRangeListReceiver>();
  if (source.transaction != no_transaction) {
    return absl::UnimplementedError(
        "CopyRange does not support a source KvStore with a transaction");
  }
  receiver->target_transaction = transaction;
  receiver->target_driver.reset(this);
  receiver->source_driver = source.driver;
  receiver->source_staleness_bound = options.source_staleness_bound;
  receiver->source_prefix_length = source.path.size();
  receiver->target_prefix = std::move(target_prefix);
  auto [promise, future] = PromiseFuturePair<void>::Make(std::in_place);
  receiver->promise = std::move(promise);
  ListOptions list_options;
  list_options.staleness_bound = options.source_staleness_bound;
  list_options.range = KeyRange::AddPrefix(source.path, options.source_range);
  source.driver->ListImpl(std::move(list_options), std::move(receiver));
  return std::move(future);
#endif
}

Future<const void> Driver::DeleteRange(KeyRange range) {
  return absl::UnimplementedError(
      "KeyValueStore does not support deleting by range");
}

void Driver::ListImpl(ListOptions options, ListReceiver receiver) {
  execution::submit(FlowSingleSender{ErrorSender{absl::UnimplementedError(
                        "KeyValueStore does not support listing")}},
                    std::move(receiver));
}

ListSender Driver::List(ListOptions options) {
  struct ListSender {
    IntrusivePtr<Driver> self;
    ListOptions options;
    void submit(ListReceiver receiver) {
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

bool operator==(const KvStore& a, const KvStore& b) {
  return a.driver == b.driver && a.path == b.path &&
         a.transaction == b.transaction;
}

}  // namespace kvstore

namespace serialization {

namespace {

using DriverSpecPtrNonNullDirectSerializer =
    RegistrySerializer<internal::IntrusivePtr<const kvstore::DriverSpec>>;

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
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::DriverPtr,
    tensorstore::serialization::DriverPtrSerializer())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::KvStore,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::kvstore::KvStore>())

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Driver,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::kvstore::Driver>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::KvStore,
    tensorstore::garbage_collection::ApplyMembersGarbageCollection<
        tensorstore::kvstore::KvStore>)
