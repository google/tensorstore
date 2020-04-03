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
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
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
    std::string value;
    std::size_t generation_number;
    absl::string_view generation() const {
      return absl::string_view(
          reinterpret_cast<const char*>(&generation_number),
          sizeof(generation_number));
    }
  };

  absl::Mutex mutex;
  /// Next generation number to use when updating the value associated with a
  /// key.  Using a single per-store counter rather than a per-key counter
  /// ensures that creating a key, deleting it, then creating it again does
  /// not result in the same generation number being reused for a given key.
  std::size_t next_generation_number ABSL_GUARDED_BY(mutex) = 0;
  absl::flat_hash_map<std::string, ValueWithGenerationNumber> values
      ABSL_GUARDED_BY(mutex);
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

    /// Make this type compatible with `ContextBindingTraits`.
    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      // `x` is a reference to a `SpecT` object.  This function must invoke `f`
      // with a reference to each member of `x`.
      return f(x.memory_key_value_store);
    };
  };

  /// Convenience aliases used in the definitions below.  These are not required
  /// to define a `KeyValueStore`, but are recommended to reduce verbosity of
  /// the required method definitions.
  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  /// Must specify a JSON binder for the `SpecData` type.
  constexpr static auto json_binder =
      jb::Object(jb::Member(MemoryKeyValueStoreResource::id,
                            jb::Projection(&SpecData::memory_key_value_store)));

  /// Encodes the `BoundSpecData` as a cache key.  Typically this is defined by
  /// calling `internal::EncodeCacheKey` with the members of `BoundSpecData`
  /// that are relevant to caching.  Members that only affect creation but not
  /// opening should normally be skipped.
  static void EncodeCacheKey(std::string* out, const BoundSpecData& spec) {
    internal::EncodeCacheKey(out, spec.memory_key_value_store);
  }

  /// Converts a `SpecData` representation in place.
  ///
  /// Currently no options are supported, making this a no-op.
  static Status ConvertSpec(SpecData* spec,
                            const KeyValueStore::SpecRequestOptions& options) {
    return absl::OkStatus();
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key, Value value,
                                             WriteOptions options) override;

  Future<TimestampedStorageGeneration> Delete(Key key,
                                              DeleteOptions options) override;

  Future<std::int64_t> DeletePrefix(Key prefix) override;

  void ListImpl(const ListOptions& options,
                AnyFlowReceiver<Status, Key> receiver) override;

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

Future<KeyValueStore::ReadResult> MemoryKeyValueStore::Read(
    Key key, ReadOptions options) {
  auto& data = this->data();
  absl::ReaderMutexLock lock(&data.mutex);
  ReadResult result;
  auto& values = data.values;
  auto it = values.find(key);
  if (it == values.end()) {
    // Key not found.
    result.generation = GenerationNow(StorageGeneration::NoValue());
    return result;
  }
  // Key found.
  if (options.if_not_equal == it->second.generation() ||
      (!StorageGeneration::IsUnknown(options.if_equal) &&
       options.if_equal != it->second.generation())) {
    // Generation associated with `key` matches `if_not_equal`.  Abort.
    result.generation = GenerationNow(StorageGeneration::Unknown());
    return result;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto byte_range, options.byte_range.Validate(it->second.value.size()));
  result.value =
      std::string(internal::GetSubStringView(it->second.value, byte_range));
  result.generation =
      GenerationNow(StorageGeneration{std::string(it->second.generation())});
  return result;
}

Future<TimestampedStorageGeneration> MemoryKeyValueStore::Write(
    Key key, Value value, WriteOptions options) {
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
    // Insert the value it into the hash table with the next unused
    // generation number.
    it = values
             .emplace(std::move(key),
                      ValueWithGenerationNumber{std::move(value),
                                                data.next_generation_number++})
             .first;
    return GenerationNow({std::string(it->second.generation())});
  }
  // Key already exists.
  if (!StorageGeneration::IsUnknown(options.if_equal) &&
      options.if_equal != it->second.generation()) {
    // Write is conditioned on an existing generation of `if_equal`,
    // which does not match the current generation.  Abort.
    return GenerationNow(StorageGeneration::Unknown());
  }
  // Set the generation number to the next unused generation number.
  it->second.generation_number = data.next_generation_number++;
  // Update the value.
  it->second.value = std::move(value);
  return GenerationNow({std::string(it->second.generation())});
}

Future<TimestampedStorageGeneration> MemoryKeyValueStore::Delete(
    Key key, DeleteOptions options) {
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
    return GenerationNow(StorageGeneration::NoValue());
  }
  // Key already exists.
  if (!StorageGeneration::IsUnknown(options.if_equal) &&
      options.if_equal != it->second.generation()) {
    // Write is conditioned on an existing generation of `value.generation`,
    // which does not match the current generation.  Abort.
    return GenerationNow(StorageGeneration::Unknown());
  }
  values.erase(it);
  return GenerationNow(StorageGeneration::NoValue());
}

Future<std::int64_t> MemoryKeyValueStore::DeletePrefix(Key prefix) {
  auto& data = this->data();
  std::int64_t count = 0;
  absl::WriterMutexLock lock(&data.mutex);
  auto& values = data.values;
  for (auto it = values.begin(), end = values.end(); it != end;) {
    auto next = std::next(it);
    if (absl::StartsWith(it->first, prefix)) {
      values.erase(it);
      ++count;
    }
    it = next;
  }
  return count;
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
    absl::WriterMutexLock lock(&data.mutex);
    auto& values = data.values;
    for (const auto& kv : values) {
      if (cancelled.load(std::memory_order_relaxed)) break;
      if (absl::StartsWith(kv.first, options.prefix)) {
        keys.push_back(kv.first);
      }
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

// Registers the driver.
const internal::KeyValueStoreDriverRegistration<MemoryKeyValueStore>
    registration;

}  // namespace

KeyValueStore::Ptr GetMemoryKeyValueStore() {
  KeyValueStore::PtrT<MemoryKeyValueStore> ptr(new MemoryKeyValueStore);
  ptr->spec_.memory_key_value_store =
      Context::Default()
          .GetResource(
              Context::ResourceSpec<MemoryKeyValueStoreResource>::Default())
          .value();
  return ptr;
}

}  // namespace tensorstore
