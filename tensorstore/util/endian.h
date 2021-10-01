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

#ifndef TENSORSTORE_UTIL_ENDIAN_H_
#define TENSORSTORE_UTIL_ENDIAN_H_

#include <cstring>
#include <ostream>

#include "absl/base/attributes.h"
#include "absl/base/internal/endian.h"
#include "tensorstore/index.h"

namespace tensorstore {

/// Indicates platform endianness, as added in C++20.
///
/// See https://en.cppreference.com/w/cpp/types/endian.
enum class endian {
#ifdef _WIN32
  little = 0,
  big = 1,
  native = little
#else
  little = __ORDER_LITTLE_ENDIAN__,
  big = __ORDER_BIG_ENDIAN__,
  native = __BYTE_ORDER__
#endif
};

inline std::ostream& operator<<(std::ostream& os, endian e) {
  return os << (e == endian::little ? '<' : '>');
}

namespace internal {

/// Swaps endianness of a single 1-, 2-, 4-, or 8-byte value.
///
/// There is no alignment requirement on `source` or `dest`.
///
/// \tparam ElementSize Size in bytes of the value.
/// \param source Pointer to source element of `ElementSize` bytes.
/// \param dest Pointer to destination element of `ElementSize` bytes.
template <size_t ElementSize>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SwapEndianUnaligned(const void* source,
                                                             void* dest) {
  static_assert((ElementSize == 1 || ElementSize == 2 || ElementSize == 4 ||
                 ElementSize == 8),
                "ElementSize must be 1, 2, 4, or 8.");
  if constexpr (ElementSize == 1) {
    std::memcpy(dest, source, 1);
  } else if constexpr (ElementSize == 2) {
    std::uint16_t temp;
    std::memcpy(&temp, source, ElementSize);
    temp = absl::gbswap_16(temp);
    std::memcpy(dest, &temp, ElementSize);
  } else if constexpr (ElementSize == 4) {
    std::uint32_t temp;
    std::memcpy(&temp, source, ElementSize);
    temp = absl::gbswap_32(temp);
    std::memcpy(dest, &temp, ElementSize);
  } else if constexpr (ElementSize == 8) {
    std::uint64_t temp;
    std::memcpy(&temp, source, ElementSize);
    temp = absl::gbswap_64(temp);
    std::memcpy(dest, &temp, ElementSize);
  }
}

/// Swaps endianness for a contiguous array of `Count` sub-elements.
///
/// This is used for swapping the endianness of data types like
/// `std::complex<T>`; in that case, `SubElementSize = sizeof(T)` and
/// `Count = 2`.
///
/// In generic code, this can also be used with `SubElementSize = 1` as an
/// equivalent for `std::memcpy(dest, source, Count)`.
///
/// \tparam SubElementSize Size in bytes of each sub-element.
/// \tparam Count Number of elements.
/// \param source Pointer to source array of `SubElementSize*Count` bytes.
/// \param dest Pointer to destination array of `SubElementSize*Count` bytes.
template <size_t SubElementSize, size_t Count>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SwapEndianUnaligned(const void* source,
                                                             void* dest) {
  if constexpr (SubElementSize == 1) {
    std::memcpy(dest, source, Count);
  } else {
    for (size_t i = 0; i < Count; ++i) {
      SwapEndianUnaligned<SubElementSize>(source, dest);
      source = reinterpret_cast<const char*>(source) + SubElementSize;
      dest = reinterpret_cast<char*>(dest) + SubElementSize;
    }
  }
}

/// Swaps endianness in-place.
///
/// There is no alignment requirement on `data`.
///
/// This is equivalent to `SwapEndianUnaligned<ElementSize>(data, data)`.
template <size_t ElementSize>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SwapEndianUnalignedInplace(
    void* data) {
  SwapEndianUnaligned<ElementSize>(data, data);
}

/// Swaps endianness in-place for a contiguous array of `Count` elements.
///
/// There is no alignment requirement on `data`.
///
/// This is equivalent to
/// `SwapEndianUnaligned<SubElementSize, Count>(data, data)`.
template <size_t SubElementSize, size_t Count>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SwapEndianUnalignedInplace(
    void* data) {
  SwapEndianUnaligned<SubElementSize, Count>(data, data);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ENDIAN_H_
