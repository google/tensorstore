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

#include "tensorstore/index.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/future.h"

namespace tensorstore {

struct ReadProgress {
  /// Total number of elements to be read.
  Index total_elements;

  /// Number of elements that have been copied.
  Index copied_elements;

  friend bool operator==(const ReadProgress& a, const ReadProgress& b);
  friend bool operator!=(const ReadProgress& a, const ReadProgress& b);
  friend std::ostream& operator<<(std::ostream& os, const ReadProgress& a);
};

struct WriteProgress {
  /// Total number of elements to be written.
  Index total_elements;

  /// Number of elements that have been copied.
  Index copied_elements;

  /// Number of elements that have been committed.
  Index committed_elements;

  friend bool operator==(const WriteProgress& a, const WriteProgress& b);
  friend bool operator!=(const WriteProgress& a, const WriteProgress& b);
  friend std::ostream& operator<<(std::ostream& os, const WriteProgress& a);
};

struct CopyProgress {
  /// Total number of elements to be copied.
  Index total_elements;

  /// Number of elements that are ready for reading.
  Index read_elements;

  /// Number of elements that have been completed.
  Index copied_elements;

  /// Number of elements that have been committed.
  Index committed_elements;

  friend bool operator==(const CopyProgress& a, const CopyProgress& b);
  friend bool operator!=(const CopyProgress& a, const CopyProgress& b);
  friend std::ostream& operator<<(std::ostream& os, const CopyProgress& a);
};

struct [[nodiscard]] WriteFutures {
  WriteFutures() = default;

  WriteFutures(Future<void> copy_future, Future<void> commit_future)
      : copy_future(std::move(copy_future)),
        commit_future(std::move(commit_future)) {}

  WriteFutures(Status status)
      : copy_future(status), commit_future(copy_future) {}

  WriteFutures(Result<WriteFutures> result) {
    if (result) {
      *this = *result;
    } else {
      *this = WriteFutures(result.status());
    }
  }

  Result<void>& result() const { return commit_future.result(); }
  void value() const { return commit_future.value(); }
  void Force() const { commit_future.Force(); }

  /// Becomes ready when the source is no longer needed.
  Future<void> copy_future;

  /// Becomes ready when the write has been committed (or failed).
  Future<void> commit_future;
};

using ReadProgressFunction =
    internal::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(ReadProgress)>;

using WriteProgressFunction =
    internal::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(WriteProgress)>;

using CopyProgressFunction =
    internal::Poly<sizeof(void*) * 2, /*Copyable=*/false, void(CopyProgress)>;

}  // namespace tensorstore

#endif  // TENSORSTORE_PROGRESS_H_
