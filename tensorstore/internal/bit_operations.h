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

#ifndef TENSORSTORE_INTERNAL_BIT_OPERATIONS_H_
#define TENSORSTORE_INTERNAL_BIT_OPERATIONS_H_

/// \file Defines common bit-wise operations.

#include <cstdint>

// The implementation of `CountLeadingZeros64` was copied from
// absl/base/internal/bits.h

// Clang on Windows has __builtin_clzll; otherwise we need to use the
// windows intrinsic functions.
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif

namespace tensorstore {
namespace internal_bits {

constexpr inline int CountLeadingZeros64Slow(std::uint64_t n) {
  int zeroes = 60;
  if (n >> 32) zeroes -= 32, n >>= 32;
  if (n >> 16) zeroes -= 16, n >>= 16;
  if (n >> 8) zeroes -= 8, n >>= 8;
  if (n >> 4) zeroes -= 4, n >>= 4;
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[n] + zeroes;
}

// On MSVC 2019, the intrinsics are not constexpr.
#if !defined(_MSC_VER) || defined(__clang__)
#define TENSORSTORE_INTERNAL_CONSTEXPR_UNLESS_MSVC constexpr
#else
#define TENSORSTORE_INTERNAL_CONSTEXPR_UNLESS_MSVC
#endif

TENSORSTORE_INTERNAL_CONSTEXPR_UNLESS_MSVC inline int CountLeadingZeros64(
    std::uint64_t n) {
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
  // MSVC does not have __buitin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(_MSC_VER) && !defined(__clang__)
  // MSVC does not have __buitin_clzll. Compose two calls to _BitScanReverse
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((n >> 32) && _BitScanReverse(&result, n >> 32)) {
    return 31 - result;
  }
  if (_BitScanReverse(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(__GNUC__)
  // Use __builtin_clzll, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  static_assert(sizeof(unsigned long long) == sizeof(n),  // NOLINT(runtime/int)
                "__builtin_clzll does not take 64-bit arg");

#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return __lzcnt64(n);
#elif defined(__aarch64__) || defined(__powerpc64__)
  // Empirically verified that __builtin_clzll(0) works as expected.
  return __builtin_clzll(n);
#endif

  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  if (n == 0) {
    return 64;
  }
  return __builtin_clzll(n);
#else
  return CountLeadingZeros64Slow(n);
#endif
}
}  // namespace internal_bits

namespace internal {

/// Returns the number of bits required to represent `x`.
///
/// This is equivalent to the C++20 function of the same name.
///
/// \returns `1 + floor(log2(x))`, or `0` if `x == 0`
TENSORSTORE_INTERNAL_CONSTEXPR_UNLESS_MSVC inline int bit_width(
    std::uint64_t x) {
  return 64 - internal_bits::CountLeadingZeros64(x);
}

#undef TENSORSTORE_INTERNAL_CONSTEXPR_UNLESS_MSVC

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BIT_OPERATIONS_H_
