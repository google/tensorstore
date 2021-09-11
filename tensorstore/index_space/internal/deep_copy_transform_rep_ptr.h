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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DEEP_COPY_TRANSFORM_REP_PTR_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DEEP_COPY_TRANSFORM_REP_PTR_H_

#include <utility>

#include "tensorstore/index_space/internal/transform_rep.h"

namespace tensorstore {
namespace internal_index_space {

/// Deep copying pointer to `TransformRep`, used by `IndexTransformBuilder`.
class DeepCopyTransformRepPtr {
 public:
  /// Constructs a null pointer.
  DeepCopyTransformRepPtr(std::nullptr_t = nullptr) : ptr_(nullptr) {}

  /// Takes ownership of an existing TransformRep.
  ///
  /// \param ptr Pointer to existing `TransformRep`.  May be null.
  explicit DeepCopyTransformRepPtr(TransformRep* ptr,
                                   internal::adopt_object_ref_t)
      : ptr_(ptr) {
    assert(ptr == nullptr ||
           (ptr->input_rank_capacity == 0 && ptr->output_rank_capacity == 0) ||
           ptr->reference_count == 1);
  }

  /// Initializes from a copy of an existing TransformRep.
  ///
  /// \param ptr Pointer to existing `TransformRep`.  May be null.
  explicit DeepCopyTransformRepPtr(TransformRep* ptr,
                                   internal::acquire_object_ref_t) {
    if (ptr) {
      ptr_ =
          TransformRep::Allocate(ptr->input_rank, ptr->output_rank).release();
      CopyTransformRep(ptr, ptr_);
    } else {
      ptr_ = nullptr;
    }
  }

  DeepCopyTransformRepPtr(DeepCopyTransformRepPtr&& other)
      : ptr_(std::exchange(other.ptr_, nullptr)) {}

  DeepCopyTransformRepPtr(const DeepCopyTransformRepPtr& other)
      : DeepCopyTransformRepPtr(other.ptr_, internal::acquire_object_ref) {}

  DeepCopyTransformRepPtr& operator=(DeepCopyTransformRepPtr&& other) {
    if (ptr_) Free();
    ptr_ = std::exchange(other.ptr_, nullptr);
    return *this;
  }

  DeepCopyTransformRepPtr& operator=(const DeepCopyTransformRepPtr& other) {
    return *this = DeepCopyTransformRepPtr(other.ptr_,
                                           internal::acquire_object_ref);
  }

  DeepCopyTransformRepPtr& operator=(std::nullptr_t) {
    if (ptr_) Free();
    ptr_ = nullptr;
    return *this;
  }

  ~DeepCopyTransformRepPtr() {
    if (ptr_) Free();
  }

  explicit operator bool() const { return static_cast<bool>(ptr_); }
  TransformRep* get() const { return ptr_; }
  TransformRep* operator->() const { return ptr_; }
  TransformRep& operator*() const { return *ptr_; }
  TransformRep* release() { return std::exchange(ptr_, nullptr); }

 private:
  void Free() {
    // Transfer ownership to a temporary `TransformRep::Ptr<>`.  This has the
    //  effect of decrementing the reference count and then calling
    //  `TransformRep::Free` (which expects the reference count to be 0).
    TransformRep::Ptr<>(ptr_, internal::adopt_object_ref);
  }
  TransformRep* ptr_;
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DEEP_COPY_TRANSFORM_REP_PTR_H_
