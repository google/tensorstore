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
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/collecting_sender.h"
#include "tensorstore/util/execution/future_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/execution/sync_flow_sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal::IntrusivePtr;

namespace tensorstore {
namespace kvstore {

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
        }
#ifdef TENSORSTORE_KVSTORE_OPEN_CACHE_DEBUG
        ABSL_LOG(INFO) << (p.second ? "Inserted kvstore into cache: "
                                    : "Reusing cached kvstore: ")
                       << QuoteString(cache_key);
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
      ABSL_LOG(INFO) << "Removed kvstore from open cache: "
                     << QuoteString(cache_identifier_);
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

Future<const void> Driver::DeleteRange(KeyRange range) {
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
