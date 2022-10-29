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

#ifndef TENSORSTORE_PROGRESS_H_
#define TENSORSTORE_PROGRESS_H_

#include <iosfwd>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/future.h"

namespace tensorstore {

/// Specifies progress statistics for `Read` operations.
///
/// \relates Read[TensorStore, Array]
struct ReadProgress {
  /// Total number of elements to be read.
  Index total_elements;

  /// Number of elements that have been copied.
  Index copied_elements;

  /// Compares two progress states for equality.
  friend bool operator==(const ReadProgress& a, const ReadProgress& b);
  friend bool operator!=(const ReadProgress& a, const ReadProgress& b);

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const ReadProgress& a);
};

/// Specifies progress statistics for `Write` operations.
///
/// \relates Write[Array, TensorStore]
struct WriteProgress {
  /// Total number of elements to be written.
  Index total_elements;

  /// Number of elements that have been copied.
  Index copied_elements;

  /// Number of elements that have been committed.
  Index committed_elements;

  /// Compares two progress states for equality.
  friend bool operator==(const WriteProgress& a, const WriteProgress& b);
  friend bool operator!=(const WriteProgress& a, const WriteProgress& b);

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const WriteProgress& a);
};

/// Specifies progress statistics for `Copy` operations.
///
/// \relates Copy[TensorStore, TensorStore]
struct CopyProgress {
  /// Total number of elements to be copied.
  Index total_elements;

  /// Number of elements that are ready for reading.
  Index read_elements;

  /// Number of elements that have been completed.
  Index copied_elements;

  /// Number of elements that have been committed.
  Index committed_elements;

  /// Compares two progress states for equality.
  friend bool operator==(const CopyProgress& a, const CopyProgress& b);
  friend bool operator!=(const CopyProgress& a, const CopyProgress& b);

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const CopyProgress& a);
};

/// Handle for consuming the result of an asynchronous write operation.
///
/// This holds two futures:
///
/// - The `copy_future` indicates when reading has completed, after which the
///   source is no longer accessed.
///
/// - The `commit_future` indicates when the write is guaranteed to be reflected
///   in subsequent reads.  For non-transactional writes, the `commit_future`
///   completes successfully only once durability of the write is guaranteed
///   (subject to the limitations of the underlying storage mechanism).  For
///   transactional writes, the `commit_future` merely indicates when the write
///   is reflected in subsequent reads using the same transaction.  Durability
///   is *not* guaranteed until the transaction itself is committed
///   successfully.
///
/// In addition, this class also provides a subset of the interface as `Future`,
/// which simply forwards to the corresponding operation on `commit_future`.
///
/// \ingroup async
struct [[nodiscard]] WriteFutures {
  /// Constructs a null handle.
  ///
  /// \id default
  WriteFutures() = default;

  /// Constructs from a `copy_future` and `commit_future`.
  ///
  /// \id copy_future, commit_future
  WriteFutures(Future<void> copy_future, Future<void> commit_future)
      : copy_future(std::move(copy_future)),
        commit_future(std::move(commit_future)) {}

  /// Constructs from an `absl::Status`.
  ///
  /// \id status
  WriteFutures(absl::Status status)
      : copy_future(status), commit_future(copy_future) {}

  /// Unwraps a `Result<WriteFutures>`.
  ///
  /// \id result
  WriteFutures(Result<WriteFutures> result) {
    if (result) {
      *this = *result;
    } else {
      *this = WriteFutures(result.status());
    }
  }

  /// Returns the `Future::result` of the `commit_future`.
  ///
  /// This implies `Force()`.
  Result<void>& result() const { return commit_future.result(); }

  /// Returns the `Future::status` of the `commit_future`.
  ///
  /// This implies `Force()`.
  absl::Status status() const { return commit_future.status(); }

  /// Returns the `Future::value` of the `commit_future`.
  ///
  /// This implies `Force()`.
  void value() const { return commit_future.value(); }

  /// Requests that writeback begins immediately.
  void Force() const { commit_future.Force(); }

  /// Becomes ready when the source is no longer needed.
  Future<void> copy_future;

  /// Becomes ready when the write has been committed (or failed).
  Future<void> commit_future;
};

/// Waits for `future.commit_future` to be ready and returns the status.
///
/// \relates WriteFutures
/// \id WriteFutures
inline absl::Status GetStatus(const WriteFutures& future) {
  return future.status();
}

/// Type-erased movable function with signature `void (ReadProgress)`.
///
/// \relates ReadProgress
using ReadProgressFunction =
    poly::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(ReadProgress)>;

/// Type-erased movable function with signature `void (WriteProgress)`.
///
/// \relates WriteProgress
using WriteProgressFunction =
    poly::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(WriteProgress)>;

/// Type-erased movable function with signature `void (CopyProgress)`.
///
/// \relates CopyProgress
using CopyProgressFunction =
    poly::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(CopyProgress)>;

}  // namespace tensorstore

#endif  // TENSORSTORE_PROGRESS_H_
