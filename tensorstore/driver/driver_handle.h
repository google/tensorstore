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

#ifndef TENSORSTORE_DRIVER_DRIVER_HANDLE_H_
#define TENSORSTORE_DRIVER_DRIVER_HANDLE_H_

#include <cstdint>

#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal {

struct ReadWritePtrTraits
    : public tensorstore::internal::DefaultIntrusivePtrTraits {
  template <typename U>
  using pointer = TaggedPtr<U, 2>;
};

/// ReadWritePtr is an intrusive tagged pointer where the tag carries the
/// ReadWriteMode flag.
template <typename T>
class ReadWritePtr : public IntrusivePtr<T, ReadWritePtrTraits> {
  using Base = IntrusivePtr<T, ReadWritePtrTraits>;

 public:
  using element_type = T;
  using traits_type = ReadWritePtrTraits;
  using pointer = typename ReadWritePtrTraits::template pointer<T>;

  constexpr ReadWritePtr() noexcept : Base() {}
  constexpr ReadWritePtr(std::nullptr_t) noexcept : Base(nullptr) {}

  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode) noexcept
      : ReadWritePtr(ptr, read_write_mode, acquire_object_ref) {}

  explicit ReadWritePtr(pointer ptr, acquire_object_ref_t) noexcept
      : Base(ptr, acquire_object_ref) {}

  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                        acquire_object_ref_t) noexcept
      : ReadWritePtr({ptr, static_cast<uintptr_t>(read_write_mode)},
                     acquire_object_ref) {}

  constexpr explicit ReadWritePtr(pointer ptr, adopt_object_ref_t) noexcept
      : Base(ptr, adopt_object_ref) {}

  constexpr explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                                  adopt_object_ref_t) noexcept
      : ReadWritePtr({ptr, static_cast<uintptr_t>(read_write_mode)},
                     adopt_object_ref) {}

  /// Default copy and move constructors.
  ReadWritePtr(const ReadWritePtr& rhs) noexcept = default;
  ReadWritePtr& operator=(const ReadWritePtr& rhs) noexcept = default;
  constexpr ReadWritePtr(ReadWritePtr&& rhs) noexcept = default;
  constexpr ReadWritePtr& operator=(ReadWritePtr&& rhs) noexcept = default;

  /// Copy constructs from `rhs`.  If `rhs` is not null, acquires a new
  /// reference to `rhs.get()` by calling `R::increment(rhs.get())`.
  template <typename U,
            std::enable_if_t<std::is_convertible_v<
                typename traits_type::template pointer<U>, pointer>>* = nullptr>
  ReadWritePtr(const ReadWritePtr<U>& rhs) noexcept
      : Base(rhs.get(), acquire_object_ref) {}

  template <typename U,
            std::enable_if_t<std::is_convertible_v<
                typename traits_type::template pointer<U>, pointer>>* = nullptr>
  ReadWritePtr& operator=(const ReadWritePtr<U>& rhs) {
    ReadWritePtr(rhs).swap(*this);
    return *this;
  }

  /// Move constructs from `rhs`.  If `rhs` is not null, transfers ownership of
  /// a reference from `rhs` to `*this`.
  template <typename U, typename = std::enable_if_t<std::is_convertible_v<
                            traits_type::template pointer<U>, pointer>>>
  constexpr ReadWritePtr(ReadWritePtr<U>&& rhs) noexcept
      : Base(rhs.release(), adopt_object_ref) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<
                            traits_type::template pointer<U>, pointer>>>
  constexpr ReadWritePtr& operator=(ReadWritePtr<U>&& rhs) noexcept {
    ReadWritePtr(std::move(rhs)).swap(*this);
    return *this;
  }

  // Methods inherited from IntrusivePtr<T>:
  // reset()
  // release()
  // swap()
  // operator bool()
  // get()
  // operator->()
  // operator*()
  // operator==()
  // operator!=()

  ReadWriteMode read_write_mode() const {
    return static_cast<ReadWriteMode>(this->get().tag());
  }
  void set_read_write_mode(ReadWriteMode read_write_mode) {
    *this = ReadWritePtr(this->release(), read_write_mode, adopt_object_ref);
  }
};

/// Creates an `ReadWritePtr<T>` while avoiding issues creating temporaries
/// during the construction process, a shorthand for
/// `ReadWritePtr<T>(new T(...), mode, acquire_object_ref)`.
///
///  mode is one of ReadWriteMode::dynamic, ReadWriteMode::read,
///  ReadWriteMode::write, ReadWriteMode::read_write.
///
/// Example:
///   auto p = MakeReadWritePtr<X>(ReadWriteMode::read, args...);
///   // 'p' is an ReadWritePtr<X>
///   EXPECT_EQ(p.read_write_mode(), ReadWriteMode::read);
///
template <typename T, typename... Args>
inline ReadWritePtr<T> MakeReadWritePtr(ReadWriteMode mode, Args&&... args) {
  return ReadWritePtr<T>(new T(std::forward<Args>(args)...), mode,
                         acquire_object_ref);
}

template <typename T, typename U>
inline ReadWritePtr<T> static_pointer_cast(ReadWritePtr<U> p) {
  return ReadWritePtr<T>(static_pointer_cast<T>(p.release()), adopt_object_ref);
}

class Driver;
using DriverPtr = ReadWritePtr<Driver>;

/// Pairs a `ReadWritePtr<Driver>` with an `IndexTransform<>` to apply to the
/// driver and a transaction to use.
struct DriverHandle {
  bool valid() const { return static_cast<bool>(driver); }
  ReadWritePtr<Driver> driver;

  /// Transform to apply to `driver`.  Note that read and write operations do
  /// not use this transform directly, but rather use the transform obtained by
  /// from `driver->ResolveBounds(transform)`.
  IndexTransform<> transform;

  /// Transaction to use.
  Transaction transaction{no_transaction};
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_HANDLE_H_
