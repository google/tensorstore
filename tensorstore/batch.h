// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_BATCH_H_
#define TENSORSTORE_BATCH_H_

#include <stddef.h>

#include <atomic>

#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {

/// `Batch` provides a mechanism for defining a batch of asynchronous operations
/// that will potentially be deferred until all references to the batch are
/// released.
///
/// For some operations, there are batch implementations there are batch
/// implementations that reduce the number of separate I/O requests performed.
class Batch {
  class ImplBase {
   public:
    // LSB indicates if the batch has been submitted: 0 -> deferred, 1 ->
    // submitted.
    //
    // All other bits indicate the reference count.
    std::atomic<size_t> reference_count_{2};
  };

 public:
  /// Tag type used for `no_batch`.
  struct no_batch_t {
    constexpr explicit no_batch_t() = default;
  };

  /// Special value that indicates not to use a batch.
  constexpr static no_batch_t no_batch = no_batch_t{};

  class Impl;
  class Entry;

  /// Unowned reference to an optional batch.
  class View {
   public:
    /// Constructs a view that refers to `no_batch`.
    constexpr View() = default;
    constexpr View(no_batch_t) {}

    /// Constructs a view that refers to an existing batch.
    constexpr View(const Batch& batch) : impl_(batch.impl_.get()) {}

    /// Returns `true` if this refers to a batch (as opposed to `no_batch`).
    constexpr explicit operator bool() const {
      return static_cast<bool>(impl_);
    }

    /// Returns `true` if this refers to a batch that has not yet been
    /// submitted.
    bool deferred() const {
      return impl_ &&
             !(impl_->reference_count_.load(std::memory_order_relaxed) & 1);
    }

    ImplBase* impl_ = nullptr;
    friend class Batch;
  };

  /// Creates a new batch.
  static Batch New();

  constexpr Batch(no_batch_t) {}
  Batch(View batch) : impl_(batch.impl_) {}

  /// Returns `true` if this refers to a batch (as opposed to `no_batch`).
  constexpr explicit operator bool() const { return static_cast<bool>(impl_); }

  /// Returns `true` if this refers to a batch that has not yet been submitted.
  bool deferred() const {
    return impl_ &&
           !(impl_->reference_count_.load(std::memory_order_relaxed) & 1);
  }

  /// Releases this reference to the batch.
  ///
  /// The batch is submitted when the last reference is released.
  void Release() { impl_.reset(); }

 private:
  Batch() = default;

  friend void intrusive_ptr_increment(ImplBase* p) {
    p->reference_count_.fetch_add(2, std::memory_order_relaxed);
  }

  friend void intrusive_ptr_decrement(ImplBase* p) {
    if (p->reference_count_.fetch_sub(2, std::memory_order_acq_rel) <= 3) {
      SubmitBatch(p);
    }
  }
  static void SubmitBatch(ImplBase* impl_base);
  internal::IntrusivePtr<ImplBase> impl_;
};

constexpr inline Batch::no_batch_t no_batch = Batch::no_batch_t{};

}  // namespace tensorstore

#endif  // TENSORSTORE_BATCH_H_
