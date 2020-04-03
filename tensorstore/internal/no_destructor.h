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

#ifndef TENSORSTORE_INTERNAL_NO_DESTRUCTOR_H_
#define TENSORSTORE_INTERNAL_NO_DESTRUCTOR_H_

#include <new>
#include <utility>

namespace tensorstore {
namespace internal {

/// Stores a value of type `T` that is never destroyed, and provides a
/// pointer-like interface to the contained value.
///
/// The primary use case is to avoid destruction order problems for objects with
/// static storage duration.
template <typename T>
class NoDestructor {
 public:
  template <typename... U>
  explicit NoDestructor(U&&... args) {
    new (data_) T(std::forward<U>(args)...);
  }
  NoDestructor(const NoDestructor&) = delete;

  const T* get() const { return reinterpret_cast<const T*>(data_); }
  T* get() { return reinterpret_cast<T*>(data_); }
  const T& operator*() const { return *get(); }
  T& operator*() { return *get(); }
  const T* operator->() const { return get(); }
  T* operator->() { return get(); }

 private:
  alignas(T) char data_[sizeof(T)];
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NO_DESTRUCTOR_H_
