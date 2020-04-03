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

#ifndef TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_
#define TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_

#include <array>

#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

template <std::size_t ElementSize, std::size_t NumElements = 1>
struct SwapEndianUnalignedInplaceLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<1, Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer pointer,
                    Status* /*status*/) {
    // Type used as a placeholder for a value of size `ElementSize*NumElements`
    // without an alignment requirement.  To avoid running afoul of C++ strict
    // aliasing rules, this type should not actually be used to read or write
    // data.
    using UnalignedValue =
        std::array<std::array<unsigned char, ElementSize>, NumElements>;
    static_assert(sizeof(UnalignedValue) == ElementSize * NumElements, "");
    static_assert(alignof(UnalignedValue) == 1, "");
    // TODO(jbms): check if this loop gets auto-vectorized properly, and if not,
    // consider manually creating a vectorized implementation.
    for (Index i = 0; i < count; ++i) {
      UnalignedValue& v =
          *ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(pointer,
                                                                      i);
      for (size_t j = 0; j < NumElements; ++j) {
        SwapEndianUnalignedInplace<ElementSize>(&v[j]);
      }
    }
    return count;
  }
};

template <std::size_t ElementSize, std::size_t NumElements = 1>
struct SwapEndianUnalignedLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<2, Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    IterationBufferPointer dest, Status* /*status*/) {
    // Type used as a placeholder for a value of size `ElementSize*NumElements`
    // without an alignment requirement.  To avoid running afoul of C++ strict
    // aliasing rules, this type should not actually be used to read or write
    // data.
    using UnalignedValue =
        std::array<std::array<unsigned char, ElementSize>, NumElements>;
    static_assert(sizeof(UnalignedValue) == ElementSize * NumElements, "");
    static_assert(alignof(UnalignedValue) == 1, "");
    // TODO(jbms): check if this loop gets auto-vectorized properly, and if not,
    // consider manually creating a vectorized implementation.
    for (Index i = 0; i < count; ++i) {
      UnalignedValue& source_v =
          *ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(source,
                                                                      i);
      UnalignedValue& dest_v =
          *ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(dest, i);
      for (size_t j = 0; j < NumElements; ++j) {
        SwapEndianUnaligned<ElementSize>(&source_v[j], &dest_v[j]);
      }
    }
    return count;
  }
};

template <std::size_t ElementSize>
struct CopyUnalignedLoopTemplate {
  using ElementwiseFunctionType = ElementwiseFunction<2, Status*>;
  template <typename ArrayAccessor>
  static Index Loop(void* context, Index count, IterationBufferPointer source,
                    IterationBufferPointer dest, Status* /*status*/) {
    // Type used as a placeholder for a value of size `ElementSize` without an
    // alignment requirement.  To avoid running afoul of C++ strict aliasing
    // rules, this type should not actually be used to read or write data.
    using UnalignedValue = std::array<unsigned char, ElementSize>;
    static_assert(sizeof(UnalignedValue) == ElementSize, "");
    static_assert(alignof(UnalignedValue) == 1, "");
    for (Index i = 0; i < count; ++i) {
      std::memcpy(
          ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(dest, i),
          ArrayAccessor::template GetPointerAtOffset<UnalignedValue>(source, i),
          ElementSize);
    }
    return count;
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ENDIAN_ELEMENTWISE_CONVERSION_H_
