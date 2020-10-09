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

#include "tensorstore/kvstore/memory/memory_key_value_store.h"

#include <atomic>
#include <deque>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal::json_binding;

TimestampedStorageGeneration GenerationNow(StorageGeneration generation) {
  return TimestampedStorageGeneration{std::move(generation), absl::Now()};
}

/// The actual data for a memory-based KeyValueStore.
///
/// This is a separate reference-counted object where: `MemoryKeyValueStore` ->
/// `Context::Resource<MemoryKeyValueStoreResource>` -> `StoredKeyValuePairs`.
/// This allows the `MemoryKeyValueStore` to retain a reference to the
/// `MemoryKeyValueStoreResource`, while also allowing an equivalent
/// `MemoryKeyValueStore` to be constructed from the
/// `MemoryKeyValueStoreResource`.
struct StoredKeyValuePairs
    : public internal::AtomicReferenceCount<StoredKeyValuePairs> {
  using Ptr = internal::IntrusivePtr<StoredKeyValuePairs>;
  struct ValueWithGenerationNumber {
    absl::Cord value;
    uint64_t generation_number;
    StorageGeneration generation() const {
      return StorageGeneration::FromUint64(generation_number);
    }
  };

  using Map = absl::btree_map<std::string, ValueWithGenerationNumber>;
  std::pair<Map::iterator, Map::iterator> Find(const std::string& inclusive_min,
                                               const std::string& exclusive_max)
      ABSL_SHARED_LOCKS_REQUIRED(mutex) {
    return {values.lower_bound(inclusive_min),
            exclusive_max.empty() ? values.end()
                                  : values.lower_bound(exclusive_max)};
  }

  std::pair<Map::iterator, Map::iterator> Find(const KeyRange& range)
      ABSL_SHARED_LOCKS_REQUIRED(mutex) {
    return Find(range.inclusive_min, range.exclusive_max);
  }

  absl::Mutex mutex;
  /// Next generation number to use when updating the value associated with a
  /// key.  Using a single per-store counter rather than a per-key counter
  /// ensures that creating a key, deleting it, then creating it again does
  /// not result in the same generation number being reused for a given key.
  uint64_t next_generation_number ABSL_GUARDED_BY(mutex) = 0;
  Map values ABSL_GUARDED_BY(mutex);
};

/// Defines the context resource (see `tensorstore/context.h`) that actually
/// owns the stored key/value pairs.
struct MemoryKeyValueStoreResource
    : public internal::ContextResourceTraits<MemoryKeyValueStoreResource> {
  constexpr static char id[] = "memory_key_value_store";
  struct Spec {};
  using Resource = StoredKeyValuePairs::Ptr;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() { return jb::Object(); }
  static Result<Resource> Create(
      Spec, internal::ContextResourceCreationContext context) {
    return StoredKeyValuePairs::Ptr(new StoredKeyValuePairs);
  }
  static Spec GetSpec(const Resource&,
                      const internal::ContextSpecBuilder& builder) {
    return {};
  }
};

const internal::ContextResourceRegistration<MemoryKeyValueStoreResource>
    resource_registration;

/// Defines the "memory" KeyValueStore driver.
///
/// This also serves as documentation of how to implement a KeyValueStore
/// driver.
class MemoryKeyValueStore
    : public internal::RegisteredKeyValueStore<MemoryKeyValueStore> {
 public:
  /// Specifies the string identifier under which the driver will be registered.
  static constexpr char id[] = "memory";

  /// KeyValueStore types must define a `SpecT` class template, where the
  /// `MaybeBound` parameter will be either `ContextBound` or `ContextUnbound`
  /// (defined in `tensorstore/internal/context_binding.h`).
  ///
  /// `SpecT<internal::ContextUnbound>` and `SpecT<internal::ContextBound>` are
  /// stored in the derived `KeyValueStore::Spec` and `KeyValueStore::BoundSpec`
  /// types that are defined automatically for this driver.
  template <template <typename T> class MaybeBound>
  struct SpecT {
    MaybeBound<Context::ResourceSpec<MemoryKeyValueStoreResource>>
        memory_key_value_store;

    bool atomic = true;

    /// Make this type compatible with `ContextBindingTraits`.
    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      // `x` is a reference to a `SpecT` object.  This function must invoke `f`
      // with a reference to each member of `x`.
      return f(x.memory_key_value_store, x.atomic);
    };
  };

  /// Convenience aliases used in the definitions below.  These are not required
  /// to define a `KeyValueStore`, but are recommended to reduce verbosity of
  /// the required method definitions.
  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  /// Must specify a JSON binder for the `SpecData` type.
  constexpr static auto json_binder = jb::Object(
      jb::Member(MemoryKeyValueStoreResource::id,
                 jb::Projection(&SpecData::memory_key_value_store)),
      jb::Member("atomic",
                 jb::Projection(&SpecData::atomic,
                                jb::DefaultValue([](auto* y) { *y = true; }))));

  /// Encodes the `BoundSpecData` as a cache key.  Typically this is defined by
  /// calling `internal::EncodeCacheKey` with the members of `BoundSpecData`
  /// that are relevant to caching.  Members that only affect creation but not
  /// opening should normally be skipped.
  static void EncodeCacheKey(std::string* out, const BoundSpecData& spec) {
    internal::EncodeCacheKey(out, spec.memory_key_value_store, spec.atomic);
  }

  /// Converts a `SpecData` representation in place.
  ///
  /// Currently no options are supported, making this a no-op.
  static Status ConvertSpec(SpecData* spec,
                            const KeyValueStore::SpecRequestOptions& options) {
    return absl::OkStatus();
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<void> DeleteRange(KeyRange range) override;

  void ListImpl(const ListOptions& options,
                AnyFlowReceiver<Status, Key> receiver) override;

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override;

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range) override;

  class TransactionNode;

  /// Returns a reference to the stored key value pairs.  The stored data is
  /// owned by the `Context::Resource` rather than directly by
  /// `MemoryKeyValueStore` in order to allow it to live as long as the
  /// `Context` from which the `MemoryKeyValueStore` was opened, and thereby
  /// allow an equivalent `MemoryKeyValueStore` to be re-opened from the
  /// `Context`.
  StoredKeyValuePairs& data() { return **spec_.memory_key_value_store; }

  /// Initiates opening a driver.
  static void Open(
      internal::KeyValueStoreOpenState<MemoryKeyValueStore> state) {
    // For the "memory" driver, this simply involves copying
    state.driver().spec_ = state.spec();
    // For drivers implementations for which opening is asynchronous, operations
    // may be linked (via the `Link` function in `future.h`) to
    // `state.promise()` in order to propagate cancellation.
  }

  /// Obtains a `BoundSpec` representation from an open `Driver`.
  Status GetBoundSpecData(BoundSpecData* spec) const {
    // `spec` is returned via an out parameter rather than returned via a
    // `Result`, as that simplifies use cases involving composition via
    // inheritance.
    *spec = spec_;
    return absl::Status();
  }

  /// In simple cases, such as the "memory" driver, the `Driver` can simply
  /// store a copy of the `BoundSpecData` as a member.
  BoundSpecData spec_;
};

using BufferedReadModifyWriteEntry =
    internal_kvs::AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry;
using internal_kvs::DeleteRangeEntry;
using internal_kvs::kReadModifyWrite;

class MemoryKeyValueStore::TransactionNode
    : public internal_kvs::AtomicTransactionNode {
  using Base = internal_kvs::AtomicTransactionNode;

 public:
  using Base::Base;

  /// Commits a (possibly multi-key) transaction atomically.
  ///
  /// The commit involves two steps, both while holding a lock on the entire
  /// KeyValueStore:
  ///
  /// 1. Without making any modifications, validates that the underlying
  ///    KeyValueStore data matches the generation constraints specified in the
  ///    transaction.  If validation fails, the commit is retried, which
  ///    normally results in any modifications being "rebased" on top of any
  ///    modified values.
  ///
  /// 2. If validation succeeds, applies the modifications.
  void AllEntriesDone(internal_kvs::SinglePhaseMutation& single_phase_mutation)
      override ABSL_NO_THREAD_SAFETY_ANALYSIS {
    if (!single_phase_mutation.remaining_entries_.HasError()) {
      auto& data = static_cast<MemoryKeyValueStore&>(*this->kvstore()).data();
      TimestampedStorageGeneration generation;
      UniqueWriterLock lock(data.mutex);
      absl::Time commit_time = absl::Now();
      if (!ValidateEntryConditions(data, single_phase_mutation, commit_time)) {
        lock.unlock();
        internal_kvs::RetryAtomicWriteback(single_phase_mutation, commit_time);
        return;
      }
      ApplyMutation(data, single_phase_mutation, commit_time);
      lock.unlock();
      internal_kvs::AtomicCommitWritebackSuccess(single_phase_mutation);
    } else {
      internal_kvs::WritebackError(single_phase_mutation);
    }
    MultiPhaseMutation::AllEntriesDone(single_phase_mutation);
  }

  /// Validates that the underlying `data` matches the generation constraints
  /// specified in the transaction.  No changes are made to the `data`.
  static bool ValidateEntryConditions(
      StoredKeyValuePairs& data,
      internal_kvs::SinglePhaseMutation& single_phase_mutation,
      const absl::Time& commit_time) ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    bool validated = true;
    for (auto& entry : single_phase_mutation.entries_) {
      if (!ValidateEntryConditions(data, entry, commit_time)) {
        validated = false;
      }
    }
    return validated;
  }

  static bool ValidateEntryConditions(StoredKeyValuePairs& data,
                                      internal_kvs::MutationEntry& entry,
                                      const absl::Time& commit_time)
      ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    if (entry.entry_type() == kReadModifyWrite) {
      return ValidateEntryConditions(
          data, static_cast<BufferedReadModifyWriteEntry&>(entry), commit_time);
    }
    auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
    // `DeleteRangeEntry` imposes no constraints itself, but the superseded
    // `ReadModifyWriteEntry` nodes may have constraints.
    bool validated = true;
    for (auto& deleted_entry : dr_entry.superseded_) {
      if (!ValidateEntryConditions(
              data, static_cast<BufferedReadModifyWriteEntry&>(deleted_entry),
              commit_time)) {
        validated = false;
      }
    }
    return validated;
  }

  static bool ValidateEntryConditions(StoredKeyValuePairs& data,
                                      BufferedReadModifyWriteEntry& entry,
                                      const absl::Time& commit_time)
      ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    auto& stamp = entry.read_result_.stamp;
    auto if_equal = StorageGeneration::Clean(stamp.generation);
    if (StorageGeneration::IsUnknown(if_equal)) {
      assert(stamp.time == absl::InfiniteFuture());
      return true;
    }
    auto it = data.values.find(entry.key_);
    if (it == data.values.end()) {
      if (StorageGeneration::IsNoValue(if_equal)) {
        entry.read_result_.stamp.time = commit_time;
        return true;
      }
    } else if (if_equal == it->second.generation()) {
      entry.read_result_.stamp.time = commit_time;
      return true;
    }
    return false;
  }

  /// Applies the changes in the transaction to the stored `data`.
  ///
  /// It is assumed that the constraints have already been validated by
  /// `ValidateConditions`.
  static void ApplyMutation(
      StoredKeyValuePairs& data,
      internal_kvs::SinglePhaseMutation& single_phase_mutation,
      const absl::Time& commit_time) ABSL_EXCLUSIVE_LOCKS_REQUIRED(data.mutex) {
    for (auto& entry : single_phase_mutation.entries_) {
      if (entry.entry_type() == kReadModifyWrite) {
        auto& rmw_entry = static_cast<BufferedReadModifyWriteEntry&>(entry);
        auto& stamp = rmw_entry.read_result_.stamp;
        stamp.time = commit_time;
        if (!StorageGeneration::IsDirty(
                rmw_entry.read_result_.stamp.generation)) {
          // Do nothing
        } else if (rmw_entry.read_result_.state ==
                   KeyValueStore::ReadResult::kMissing) {
          data.values.erase(rmw_entry.key_);
          stamp.generation = StorageGeneration::NoValue();
        } else {
          assert(rmw_entry.read_result_.state ==
                 KeyValueStore::ReadResult::kValue);
          auto& v = data.values[rmw_entry.key_];
          v.generation_number = data.next_generation_number++;
          v.value = std::move(rmw_entry.read_result_.value);
          stamp.generation = v.generation();
        }
      } else {
        auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
        auto it_range = data.Find(dr_entry.key_, dr_entry.exclusive_max_);
        data.values.erase(it_range.first, it_range.second);
      }
    }
  }
};

Future<KeyValueStore::ReadResult> MemoryKeyValueStore::Read(
    Key key, ReadOptions options) {
  auto& data = this->data();
  absl::ReaderMutexLock lock(&data.mutex);
  ReadResult result;
  auto& values = data.values;
  auto it = values.find(key);
  if (it == values.end()) {
    // Key not found.
    result.stamp = GenerationNow(StorageGeneration::NoValue());
    result.state = KeyValueStore::ReadResult::kMissing;
    return result;
  }
  // Key found.
  result.stamp = GenerationNow(it->second.generation());
  if (options.if_not_equal == it->second.generation() ||
      (!StorageGeneration::IsUnknown(options.if_equal) &&
       options.if_equal != it->second.generation())) {
    // Generation associated with `key` matches `if_not_equal`.  Abort.
    return result;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto byte_range, options.byte_range.Validate(it->second.value.size()));
  result.state = KeyValueStore::ReadResult::kValue;
  result.value = internal::GetSubCord(it->second.value, byte_range);
  return result;
}

Future<TimestampedStorageGeneration> MemoryKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  using ValueWithGenerationNumber =
      StoredKeyValuePairs::ValueWithGenerationNumber;
  auto& data = this->data();
  absl::WriterMutexLock lock(&data.mutex);
  auto& values = data.values;
  auto it = values.find(key);
  if (it == values.end()) {
    // Key does not already exist.
    if (!StorageGeneration::IsUnknown(options.if_equal) &&
        !StorageGeneration::IsNoValue(options.if_equal)) {
      // Write is conditioned on there being an existing key with a
      // generation of `if_equal`.  Abort.
      return GenerationNow(StorageGeneration::Unknown());
    }
    if (!value) {
      // Delete was requested, but key already doesn't exist.
      return GenerationNow(StorageGeneration::NoValue());
    }
    // Insert the value it into the hash table with the next unused
    // generation number.
    it = values
             .emplace(std::move(key),
                      ValueWithGenerationNumber{std::move(*value),
                                                data.next_generation_number++})
             .first;
    return GenerationNow(it->second.generation());
  }
  // Key already exists.
  if (!StorageGeneration::IsUnknown(options.if_equal) &&
      options.if_equal != it->second.generation()) {
    // Write is conditioned on an existing generation of `if_equal`,
    // which does not match the current generation.  Abort.
    return GenerationNow(StorageGeneration::Unknown());
  }
  if (!value) {
    // Delete request.
    values.erase(it);
    return GenerationNow(StorageGeneration::NoValue());
  }
  // Set the generation number to the next unused generation number.
  it->second.generation_number = data.next_generation_number++;
  // Update the value.
  it->second.value = std::move(*value);
  return GenerationNow(it->second.generation());
}

Future<void> MemoryKeyValueStore::DeleteRange(KeyRange range) {
  auto& data = this->data();
  absl::WriterMutexLock lock(&data.mutex);
  if (!range.empty()) {
    auto it_range = data.Find(range);
    data.values.erase(it_range.first, it_range.second);
  }
  return MakeResult();
}

void MemoryKeyValueStore::ListImpl(const ListOptions& options,
                                   AnyFlowReceiver<Status, Key> receiver) {
  auto& data = this->data();
  std::atomic<bool> cancelled{false};
  execution::set_starting(receiver, [&cancelled] {
    cancelled.store(true, std::memory_order_relaxed);
  });

  // Collect the keys.
  std::deque<Key> keys;
  {
    absl::ReaderMutexLock lock(&data.mutex);
    auto it_range = data.Find(options.range);
    for (auto it = it_range.first; it != it_range.second; ++it) {
      if (cancelled.load(std::memory_order_relaxed)) break;
      keys.push_back(it->first);
    }
  }

  // Send the keys.
  while (!keys.empty() && !cancelled.load(std::memory_order_relaxed)) {
    execution::set_value(receiver, std::move(keys.back()));
    keys.pop_back();
  }
  execution::set_done(receiver);
  execution::set_stopping(receiver);
}

absl::Status MemoryKeyValueStore::ReadModifyWrite(
    internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
    ReadModifyWriteSource& source) {
  if (!spec_.atomic) {
    return KeyValueStore::ReadModifyWrite(transaction, phase, std::move(key),
                                          source);
  }
  return internal_kvs::AddReadModifyWrite<TransactionNode>(
      this, transaction, phase, std::move(key), source);
}

absl::Status MemoryKeyValueStore::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  if (!spec_.atomic) {
    return KeyValueStore::TransactionalDeleteRange(transaction,
                                                   std::move(range));
  }
  return internal_kvs::AddDeleteRange<TransactionNode>(this, transaction,
                                                       std::move(range));
}

// Registers the driver.
const internal::KeyValueStoreDriverRegistration<MemoryKeyValueStore>
    registration;

}  // namespace

KeyValueStore::Ptr GetMemoryKeyValueStore(bool atomic) {
  KeyValueStore::PtrT<MemoryKeyValueStore> ptr(new MemoryKeyValueStore);
  ptr->spec_.memory_key_value_store =
      Context::Default()
          .GetResource(
              Context::ResourceSpec<MemoryKeyValueStoreResource>::Default())
          .value();
  ptr->spec_.atomic = atomic;
  return ptr;
}

}  // namespace tensorstore
