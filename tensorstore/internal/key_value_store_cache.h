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

#ifndef TENSORSTORE_INTERNAL_KEY_VALUE_STORE_CACHE_H_
#define TENSORSTORE_INTERNAL_KEY_VALUE_STORE_CACHE_H_

/// \file
///
/// Integrates `AsyncStorageBackedCache` with `KeyValueStore`.

#include <type_traits>

#include "absl/time/time.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

/// Base class that integrates an `AsyncStorageBackedCache` with a
/// `KeyValueStore`.
///
/// Each cache entry is assumed to correspond one-to-one with a key in a
/// `KeyValueStore`, defined by the `GetKeyValueStoreKey` method.
///
/// To use this class, define a `Derived` class that inherits from
/// `KeyValueStoreCache<Parent>`, where `Parent` is the desired base class.  The
/// derived class is responsible for defining:
///
/// 1. the decoding of an `std::optional<std::string>` read from the
///   `KeyValueStore` into the logical "read state" of an entry (see
///   `DoDecode`).
///
/// 2. (if writing is supported) a `DoWriteback` implementation that encodes the
///    "write state" into an `std::optional<std::string>`, and then calls
///    `Writeback` to write it back to the `KeyValueStore`.
///
/// 3. overrides `GetKeyValueStoreKey` if necessary.
///
/// This class takes care of reading from and writing to the `KeyValueStore`,
/// and handling the timestamps and `StorageGeneration` values.
///
/// \tparam Parent Parent class, must inherit from (or equal)
///     `AsyncStorageBackedCache`.
template <typename Parent>
class KeyValueStoreCache : public Parent {
  static_assert(std::is_base_of_v<AsyncStorageBackedCache, Parent>);

 public:
  /// Constructs a `KeyValueStoreCache`.
  ///
  /// \param kvstore The `KeyValueStore` to use.  If `nullptr`,
  ///     `SetKeyValueStore` must be called before any read or write operations
  ///     are performed.
  /// \param executor The executor to use for decoding and writeback.
  /// \param args Arguments to forward to the `Parent` constructor.
  template <typename... U>
  explicit KeyValueStoreCache(KeyValueStore::Ptr kvstore, Executor executor,
                              U&&... args)
      : kvstore_(std::move(kvstore)),
        executor_(std::move(executor)),
        Parent(std::forward<U>(args)...) {}

  class Entry : public Parent::Entry {
   public:
    using Cache = KeyValueStoreCache;

    /// The generation associated with the current "read state".  This is part
    /// of the "read state".
    StorageGeneration last_read_generation;

    /// Used while the (possibly asynchronous) decoding process initiated by a
    /// call to `DoDecode` is in progress.  Specifies the generation and
    /// timestamp of the data being decoded.  This is used by
    /// `NotifyReadSuccess` to update `last_read_generation` and
    /// `last_read_time`.
    TimestampedStorageGeneration current_read_generation;
  };

  /// Defines the mapping from a cache entry to a `KeyValueStore` key.
  ///
  /// By default the cache entry key is used, but derived classes may override
  /// this behavior.
  virtual std::string GetKeyValueStoreKey(Cache::Entry* entry) {
    return std::string{entry->key()};
  }

  /// Implements reading for the `AsyncStorageBackedCache` interface.
  ///
  /// Reads from the `KeyValueStore` and invokes `DoDecode` with the result.
  ///
  /// If an error occurs, calls `NotifyReadError` directly without invoking
  /// `DoDecode`.
  void DoRead(Cache::PinnedEntry base_entry, absl::Time staleness_bound) final {
    auto* entry = static_cast<Entry*>(base_entry.get());
    KeyValueStore::ReadOptions options;
    options.staleness_bound = staleness_bound;
    {
      auto lock = entry->AcquireReadStateReaderLock();
      options.if_not_equal = entry->last_read_generation;
    }
    auto future =
        kvstore_->Read(this->GetKeyValueStoreKey(entry), std::move(options));
    future.ExecuteWhenReady(WithExecutor(
        executor_, [entry = static_pointer_cast<Entry>(std::move(base_entry))](
                       ReadyFuture<KeyValueStore::ReadResult> future) mutable {
          auto& result = future.result();
          if (!result) {
            GetOwningCache(entry)->NotifyReadError(entry.get(),
                                                   std::move(result.status()));
            return;
          }
          if (result->aborted()) {
            // Value has not changed.
            auto lock = entry->AcquireReadStateWriterLock();
            entry->current_read_generation.generation =
                entry->last_read_generation;
            entry->current_read_generation.time = result->generation.time;
            GetOwningCache(entry)->NotifyReadSuccess(entry.get(),
                                                     std::move(lock));
            return;
          }
          entry->current_read_generation = std::move(result->generation);
          GetOwningCache(entry)->DoDecode(std::move(entry),
                                          std::move(result->value));
        }));
  }

  /// Decodes the data read from the `KeyValueStore`.
  ///
  /// This is invoked asynchronously on `executor()` by `DoRead` and is
  /// responsible for completing the read operation.  Derived classes must
  /// define this method.
  ///
  /// It must either:
  ///
  /// 1. Acquire a `ReadStateWriterLock` on `entry`, update the "read state",
  ///    and then call `NotifyReadSuccess`; or
  ///
  /// 2. Call `NotifyReadError` if an error occurs decoding (a value of
  /// `std::nullopt` should not necessarily lead to `NotifyReadError` being
  /// called, because a missing key may be a valid state).
  ///
  /// \param entry The entry being read.
  /// \param value The value read, or `std::nullopt` if the key was not found.
  virtual void DoDecode(Cache::PinnedEntry entry,
                        std::optional<std::string> value) = 0;

  /// Extends `NotifyReadError` to annotate errors with the `KeyValueStore` key.
  void NotifyReadError(Cache::Entry* entry, absl::Status error) override {
    AsyncStorageBackedCache::NotifyReadError(
        entry, tensorstore::MaybeAnnotateStatus(
                   error, tensorstore::StrCat(
                              "Error reading ",
                              kvstore_->DescribeKey(this->GetKeyValueStoreKey(
                                  static_cast<Entry*>(entry))))));
  }

  /// Include any additional `NotifyReadSuccess` overloads `Parent` may have
  /// defined (e.g. as done by `ChunkCache`).
  using Parent::NotifyReadSuccess;

  /// Extends `NotifyReadSuccess` to set `last_read_time` and
  /// `last_read_generation`.
  void NotifyReadSuccess(
      Cache::Entry* base_entry,
      AsyncStorageBackedCache::ReadStateWriterLock lock) override {
    auto* entry = static_cast<Entry*>(base_entry);
    auto& generation = entry->current_read_generation;
    entry->last_read_time = generation.time;
    entry->last_read_generation = std::move(generation.generation);
    this->Parent::NotifyReadSuccess(entry, std::move(lock));
  }

  /// Writes the specified value back to the `KeyValueStore`.
  ///
  /// Must be called while a writeback operation is in progress (i.e. from a
  /// derived class `DoWriteback` implementation, possibly asynchronously),
  /// after calling `NotifyWritebackStarted`; this function takes care of
  /// completing the writeback.
  ///
  /// If writeback is successful, this updates `last_read_generation` and
  /// `last_read_time` to reflect the newly-written state and calls
  /// `NotifyWritebackSuccess`.  In the case of a generation mismatch, this
  /// calls `NotifyWritebackNeedsRead`.  In the case of an error, this calls
  /// `NotifyWritebackError`.  All of the notify methods are called from the
  /// associated `executor()`.
  ///
  /// Derived classes should extend the `NotifyWriteback*` methods as needed to
  /// handle writeback completion.
  ///
  /// \param base_entry The entry being written back.
  /// \param value The value to write, or `std::nullopt` to delete the key.
  /// \param unconditional If `false`, the write or delete will be conditioned
  ///     on the generation of the last read.
  void Writeback(Cache::PinnedEntry base_entry,
                 std::optional<std::string> value, bool unconditional) {
    auto* entry = static_cast<Entry*>(base_entry.get());
    KeyValueStore::WriteOptions options;
    if (!unconditional) {
      auto lock = entry->AcquireReadStateReaderLock();
      options.if_equal = entry->last_read_generation;
      if (StorageGeneration::IsUnknown(options.if_equal)) {
        options.if_equal = StorageGeneration::NoValue();
      }
    }
    auto key = this->GetKeyValueStoreKey(entry);
    Future<TimestampedStorageGeneration> future =
        kvstore_->Write(std::move(key), std::move(value), std::move(options));
    future.Force();
    std::move(future).ExecuteWhenReady(WithExecutor(
        executor_,
        [entry = static_pointer_cast<Entry>(base_entry)](
            ReadyFuture<TimestampedStorageGeneration> future) mutable {
          auto& r = future.result();
          if (!r) {
            GetOwningCache(entry)->NotifyWritebackError(
                entry.get(), entry->AcquireWriteStateLock(), r.status());
            return;
          }
          if (StorageGeneration::IsUnknown(r->generation)) {
            GetOwningCache(entry)->NotifyWritebackNeedsRead(
                entry.get(), entry->AcquireWriteStateLock(), r->time);
            return;
          }
          auto lock = entry->AcquireWriteAndReadStateLock();
          entry->last_read_time = r->time;
          static_cast<Entry*>(entry.get())->last_read_generation =
              std::move(r->generation);
          GetOwningCache(entry)->NotifyWritebackSuccess(entry.get(),
                                                        std::move(lock));
        }));
  }

  /// Extends `NotifyWritebackError` to annotate the error with the
  /// `KeyValueStore` key.
  void NotifyWritebackError(Cache::Entry* entry,
                            AsyncStorageBackedCache::WriteStateLock lock,
                            Status error) override {
    this->Parent::NotifyWritebackError(
        entry, std::move(lock),
        tensorstore::MaybeAnnotateStatus(
            error, tensorstore::StrCat("Error writing ",
                                       this->kvstore_->DescribeKey(
                                           this->GetKeyValueStoreKey(entry)))));
  }

  /// Returns the associated `KeyValueStore`.
  KeyValueStore* kvstore() { return kvstore_.get(); }

  /// Sets the KeyValueStore.  The caller is responsible for ensuring there are
  /// no concurrent read or write operations.
  void SetKeyValueStore(KeyValueStore::Ptr kvstore) {
    kvstore_ = std::move(kvstore);
  }

  /// Returns the executor used for decoding and writeback.
  const Executor& executor() { return executor_; }

  KeyValueStore::Ptr kvstore_;
  Executor executor_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_KEY_VALUE_STORE_CACHE_H_
