// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_HANDLE_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_HANDLE_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Abstract interface used by operation implementations to read the OCDBT data
/// structures for a single database.
class ReadonlyIoHandle
    : public internal::AtomicReferenceCount<ReadonlyIoHandle> {
 public:
  using Ptr = internal::IntrusivePtr<const ReadonlyIoHandle>;

  /// Reads the B+tree node at the specified location.
  virtual Future<const std::shared_ptr<const BtreeNode>> GetBtreeNode(
      const IndirectDataReference& ref) const = 0;

  /// Reads the version tree node at the specified location.
  virtual Future<const std::shared_ptr<const VersionTreeNode>>
  GetVersionTreeNode(const IndirectDataReference& ref) const = 0;

  /// Fetches the manifest with the specified staleness bound.
  virtual Future<const ManifestWithTime> GetManifest(
      absl::Time staleness_bound) const = 0;

  /// Reads indirect data at the specified location.
  virtual Future<kvstore::ReadResult> ReadIndirectData(
      const IndirectDataReference& ref,
      kvstore::ReadOptions read_options) const = 0;

  ConfigStatePtr config_state;
  Executor executor;

  virtual ~ReadonlyIoHandle();
};

struct TryUpdateManifestResult {
  absl::Time time;
  bool success;
};

/// Abstract interface used by operation implementations to write the OCDBT data
/// structures for a single database.
class IoHandle : public ReadonlyIoHandle {
 public:
  using Ptr = internal::IntrusivePtr<const IoHandle>;

  /// Performs an atomic update operation on the manifest.
  ///
  /// The specified `time` is a staleness bound (normally set to `absl::Now()`).
  /// In the particular case that `old_manifest` equals `new_manifest`
  /// (i.e. just confirming that `new_manifest` is up to date), then `time`
  /// serves as a `staleness_bound` for that check.
  virtual Future<TryUpdateManifestResult> TryUpdateManifest(
      std::shared_ptr<const Manifest> old_manifest,
      std::shared_ptr<const Manifest> new_manifest, absl::Time time) const = 0;

  /// Writes data for later retrieval via an `IndirectDataReference`, populating
  /// `ref` with its location.
  ///
  /// The data is not guaranteed to be persisted (or readable via any of the
  /// read methods) until the returned `Future` completes successfully, and
  /// writing may not start until `Future::Force` is called on the returned
  /// future.
  virtual Future<const void> WriteData(absl::Cord data,
                                       IndirectDataReference& ref) const = 0;

  /// Returns a description of the storage location,
  /// e.g. ``"\"gs://bucket/path/\""``.
  virtual std::string DescribeLocation() const = 0;
};

/// Wrapper around `Promise` that allows the same `Future` to be repeatedly
/// linked without allocating and registering redundant callbacks.
///
/// This is intended to be used with the futures returned by
/// `IoHandle::Write`.
class FlushPromise {
 public:
  FlushPromise() = default;

  FlushPromise(FlushPromise&&) noexcept;
  FlushPromise& operator=(FlushPromise&&) noexcept;

  /// Links this promise to `future`, equivalent to `LinkError`: this promise
  /// won't become ready until all linked futures become ready, and forcing this
  /// promise forces all linked futures.
  ///
  /// If this is called multiple times in succession with futures with the same
  /// shared state, subsequent calls are no-ops.  In contrast, each call to
  /// `LinkError` allocates additional memory and imposes additional overhead
  /// to handle promise/future events.
  ///
  /// Thread-safety: safe to call concurrently from multiple threads.
  void Link(Future<const void> future);

  /// Equivalent to `Link(other.future())`, but is more efficient in some cases.
  ///
  /// Thread-safety: safe to call on `*this` concurrently from multiple threads,
  /// but no other threads may access `other` concurrently.
  void Link(FlushPromise&& other);

  /// Returns the associated future, and resets this object to a
  /// default-constructed state.
  Future<const void> future() && {
    auto future =
        future_.null() ? std::move(prev_linked_future_) : std::move(future_);
    prev_linked_future_ = {};
    promise_ = {};
    return future;
  }

 private:
  Future<const void> prev_linked_future_;
  Promise<void> promise_;
  Future<const void> future_;
  absl::Mutex mutex_;
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_HANDLE_H_
