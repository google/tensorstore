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

template <typename T>
class ReadWritePtr : public IntrusivePtr<T, ReadWritePtrTraits> {
  using Base = IntrusivePtr<T, ReadWritePtrTraits>;

 public:
  using Base::Base;
  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                        acquire_object_ref_t = acquire_object_ref) noexcept
      : Base({ptr, static_cast<uintptr_t>(read_write_mode)},
             acquire_object_ref) {}
  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                        adopt_object_ref_t) noexcept
      : Base({ptr, static_cast<uintptr_t>(read_write_mode)}, adopt_object_ref) {
  }
  ReadWriteMode read_write_mode() const {
    return static_cast<ReadWriteMode>(this->get().tag());
  }
  void set_read_write_mode(ReadWriteMode read_write_mode) {
    *this = ReadWritePtr(this->release(), read_write_mode, adopt_object_ref);
  }
};

template <typename T, typename U>
inline ReadWritePtr<T> static_pointer_cast(ReadWritePtr<U> p) {
  return ReadWritePtr<T>(static_pointer_cast<T>(p.release()), adopt_object_ref);
}

class Driver;
using DriverPtr = ReadWritePtr<Driver>;

template <typename Driver>
struct HandleBase {
  bool valid() const { return static_cast<bool>(driver); }
  ReadWritePtr<Driver> driver;

  /// Transform to apply to `driver`.  Note that read and write operations do
  /// not use this transform directly, but rather use the transform obtained by
  /// from `driver->ResolveBounds(transform)`.
  IndexTransform<> transform;

  /// Transaction to use.
  Transaction transaction{no_transaction};
};

/// Pairs a `Driver::Ptr` with an `IndexTransform<>` to apply to the driver and
/// a transaction to use.
using DriverHandle = HandleBase<Driver>;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_HANDLE_H_
