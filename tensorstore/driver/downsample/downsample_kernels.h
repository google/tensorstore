// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_KERNELS_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_KERNELS_H_

/// \file
/// Package-private 1-d window partitioning, byte-width window staging,
/// canonical compute-type traits, order-preserving unsigned radix mapping, and
/// contiguous row reduction kernels for `downsample_nditerable.cc`.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/numeric/int128.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/insertion_sort.h"
#include "tensorstore/internal/meta/attributes.h"
#include "tensorstore/internal/scatter_gather_util.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {
namespace internal_downsample {

using internal::TrivialBytes;

static_assert((-1 >> 1) == -1, "Arithmetic signed right shift is required.");

/// Partition of a 1-d `[0, base_len)` block into an optional leading partial
/// window (`out_idx == 0` when `offset != 0`), `full_out_count` full windows of
/// width `factor`, and an optional trailing partial window (`out_idx ==
/// out_len - 1`).
struct DimPartition {
  Index head_src_len;
  Index full_out_begin;
  Index full_out_count;
  Index full_src_begin;
  Index tail_src_len;
  Index tail_src_begin;
};

/// Computes the 1-d window partition for `[0, base_len)` given `offset` in
/// `[0, factor)` relative to the start of output cell 0.  Clamping
/// `head_src_len` to `base_len` handles single-cell blocks clipped at both
/// ends (`offset + base_len < factor`).
inline DimPartition PartitionDim(Index out_len, Index base_len, Index offset,
                                 Index factor) {
  assert(out_len > 0 && base_len > 0 && factor > 0);
  assert(offset >= 0 && offset < factor);
  DimPartition p;
  if (offset != 0) {
    p.head_src_len = std::min(base_len, factor - offset);
    p.full_out_begin = 1;
  } else {
    p.head_src_len = 0;
    p.full_out_begin = 0;
  }
  p.full_src_begin = p.head_src_len;
  const Index rem = base_len - p.head_src_len;
  p.full_out_count = rem / factor;
  p.tail_src_len = rem % factor;
  p.tail_src_begin = p.full_src_begin + p.full_out_count * factor;
  assert(p.full_out_begin + p.full_out_count + (p.tail_src_len > 0 ? 1 : 0) ==
         out_len);
  return p;
}

/// Stages `out_count` windows of width `seg_factor` (`elem_size` bytes per
/// element) into `[out_w, max_total_elements]` slots at outer offset
/// `row_call_idx * seg_factor`.  `acc_row_void` and `src_row_void` must not
/// overlap.
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void StoreSegmentTrivial(
    void* TENSORSTORE_INTERNAL_RESTRICT acc_row_void,
    const void* TENSORSTORE_INTERNAL_RESTRICT src_row_void, Index out_x_begin,
    Index out_count, Index src_x_begin, Index seg_factor,
    Index max_total_elements, Index row_call_idx, size_t elem_size) {
  const size_t seg_bytes = static_cast<size_t>(seg_factor) * elem_size;
  const size_t slot_bytes = static_cast<size_t>(max_total_elements) * elem_size;
  unsigned char* TENSORSTORE_INTERNAL_RESTRICT acc =
      static_cast<unsigned char*>(acc_row_void) +
      static_cast<size_t>(out_x_begin) * slot_bytes +
      static_cast<size_t>(row_call_idx) * seg_bytes;
  const unsigned char* TENSORSTORE_INTERNAL_RESTRICT src =
      static_cast<const unsigned char*>(src_row_void) +
      static_cast<size_t>(src_x_begin) * elem_size;
  internal::ScatterStridedTrivial(src, acc, static_cast<Index>(slot_bytes),
                                  out_count, seg_bytes);
}

/// Stages `out_count` windows of width `seg_factor` for non-trivial `dtype`
/// (`string_t`, `ustring_t`, `json_t`).  `acc_row_void` and `src_row_void` must
/// not overlap.
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void StoreSegmentNonTrivial(
    DataType dtype, void* TENSORSTORE_INTERNAL_RESTRICT acc_row_void,
    const void* TENSORSTORE_INTERNAL_RESTRICT src_row_void, Index out_x_begin,
    Index out_count, Index src_x_begin, Index seg_factor,
    Index max_total_elements, Index row_call_idx) {
  const Index elem_size = dtype.size();
  const size_t elem_bytes = static_cast<size_t>(elem_size);
  const size_t seg_bytes = static_cast<size_t>(seg_factor) * elem_bytes;
  const size_t slot_bytes =
      static_cast<size_t>(max_total_elements) * elem_bytes;
  unsigned char* acc = static_cast<unsigned char*>(acc_row_void) +
                       static_cast<size_t>(out_x_begin) * slot_bytes +
                       static_cast<size_t>(row_call_idx) * seg_bytes;
  const unsigned char* src = static_cast<const unsigned char*>(src_row_void) +
                             static_cast<size_t>(src_x_begin) * elem_bytes;
  const auto& copy_fn = dtype->copy_assign;
  if (seg_factor == max_total_elements) {
    copy_fn[internal::IterationBufferKind::kContiguous](
        nullptr, {1, out_count * seg_factor},
        internal::IterationBufferPointer(const_cast<unsigned char*>(src),
                                         Index(0), elem_size),
        internal::IterationBufferPointer(acc, Index(0), elem_size), nullptr);
  } else if (seg_factor == 1) {
    copy_fn[internal::IterationBufferKind::kStrided](
        nullptr, {1, out_count},
        internal::IterationBufferPointer(const_cast<unsigned char*>(src),
                                         Index(0), elem_size),
        internal::IterationBufferPointer(acc, Index(0),
                                         static_cast<Index>(slot_bytes)),
        nullptr);
  } else {
    copy_fn[internal::IterationBufferKind::kContiguous](
        nullptr, {out_count, seg_factor},
        internal::IterationBufferPointer(const_cast<unsigned char*>(src),
                                         static_cast<Index>(seg_bytes),
                                         elem_size),
        internal::IterationBufferPointer(acc, static_cast<Index>(slot_bytes),
                                         elem_size),
        nullptr);
  }
}

/// Canonical compute-type mapping.  Low-precision floats promote to IEEE
/// `float32_t` (providing hardware SIMD math and true `+/-infinity` identities
/// for finite-only float8 formats in `kMin`/`kMax`), and low-precision integers
/// promote to `int8_t` via `GetDataTypeConverter`.  Layout-identical byte types
/// (`char_t`, `byte_t`) share canonical `8`-bit kernels with
/// `kIsPromoted == false`.
template <typename Element>
struct ComputeTypeTraits {
  using ComputeElement = Element;
  static constexpr bool kIsPromoted = false;
};

template <>
struct ComputeTypeTraits<::tensorstore::dtypes::char_t> {
  using ComputeElement =
      std::conditional_t<std::is_signed_v<char>, int8_t, uint8_t>;
  static constexpr bool kIsPromoted = false;
};

template <>
struct ComputeTypeTraits<::tensorstore::dtypes::byte_t> {
  using ComputeElement = uint8_t;
  static constexpr bool kIsPromoted = false;
};

// TODO(summivox): b/295577703: uint4_t
#define TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_INT(T, ...) \
  template <>                                                  \
  struct ComputeTypeTraits<::tensorstore::dtypes::T> {         \
    using ComputeElement = int8_t;                             \
    static constexpr bool kIsPromoted = true;                  \
  };
TENSORSTORE_FOR_EACH_LOW_PRECISION_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_INT)
#undef TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_INT

#define TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_FLOAT(T, ...) \
  template <>                                                    \
  struct ComputeTypeTraits<::tensorstore::dtypes::T> {           \
    using ComputeElement = ::tensorstore::dtypes::float32_t;     \
    static constexpr bool kIsPromoted = true;                    \
  };
TENSORSTORE_FOR_EACH_LOW_PRECISION_FLOAT_DATA_TYPE(
    TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_FLOAT)
#undef TENSORSTORE_INTERNAL_PROMOTE_LOW_PRECISION_FLOAT

/// Metafunction mapping a canonical `ComputeElement` to the accumulator type
/// for `DownsampleMethod::kMean` (`void` if unsupported).
///
/// Floating-point and complex types accumulate in place.  `bool` and integer
/// types narrower than 64 bits use 64-bit accumulators to prevent overflow;
/// 64-bit integers use 128-bit accumulators.
template <typename ComputeElement>
struct MeanAccumulateType {
  using type = void;
};
template <>
struct MeanAccumulateType<bool> {
  using type = int64_t;
};
template <>
struct MeanAccumulateType<int8_t> {
  using type = int64_t;
};
template <>
struct MeanAccumulateType<uint8_t> {
  using type = uint64_t;
};
template <>
struct MeanAccumulateType<int16_t> {
  using type = int64_t;
};
template <>
struct MeanAccumulateType<uint16_t> {
  using type = uint64_t;
};
template <>
struct MeanAccumulateType<int32_t> {
  using type = int64_t;
};
template <>
struct MeanAccumulateType<uint32_t> {
  using type = uint64_t;
};
template <>
struct MeanAccumulateType<int64_t> {
  using type = absl::int128;
};
template <>
struct MeanAccumulateType<uint64_t> {
  using type = absl::uint128;
};
template <>
struct MeanAccumulateType<::tensorstore::dtypes::float32_t> {
  using type = ::tensorstore::dtypes::float32_t;
};
template <>
struct MeanAccumulateType<::tensorstore::dtypes::float64_t> {
  using type = ::tensorstore::dtypes::float64_t;
};
template <>
struct MeanAccumulateType<::tensorstore::dtypes::complex64_t> {
  using type = ::tensorstore::dtypes::complex64_t;
};
template <>
struct MeanAccumulateType<::tensorstore::dtypes::complex128_t> {
  using type = ::tensorstore::dtypes::complex128_t;
};

struct SumOp {
  template <typename Acc, typename Src>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static Acc Combine(Acc acc, Src x) {
    return acc + static_cast<Acc>(x);
  }
};

// `MinOp` and `MaxOp` match `std::min(acc, x)` / `std::max(acc, x)` operand
// order: starting from `+inf` / `-inf`, any non-NaN `x` replaces the initial
// infinity while `NaN` inputs evaluate false and are ignored.  If an entire
// window is `NaN`, types with IEEE `has_infinity` retain the `+inf` / `-inf`
// identity, while finite-only `float8_*` types (which promote to `float32_t`
// for accumulation) replace untouched `+/-inf` accumulators with `+qNaN`.
struct MinOp {
  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static T Combine(T acc, T x) {
    return (x < acc) ? x : acc;
  }
};

struct MaxOp {
  template <typename T>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static T Combine(T acc, T x) {
    return (acc < x) ? x : acc;
  }
};

using AccumulateRowFn = void (*)(void* acc, const void* src, Index count,
                                 Index factor);
using FinalizeMeanRowFn = void (*)(void* acc, void* dst, Index count,
                                   Index divisor);
using ReduceStoredRowFn = void (*)(void* staged, Index stride, void* dst,
                                   Index count, Index divisor);

/// Accumulates `count` contiguous windows of width `factor` from `src_void`
/// into `acc_void` for `kMean`, `kMin`, and `kMax`.  `factor == 1` is
/// SIMD-vectorized across `count`, while `factor >= 2` reduces each window
/// without scalar-to-XMM strided-gather packing (`vpinsr*`) (retaining
/// inner-loop SIMD vectorization for widening integer `SumOp`).
template <typename Op, typename Acc, typename Src>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void AccumulateRow(
    void* TENSORSTORE_INTERNAL_RESTRICT acc_void,
    const void* TENSORSTORE_INTERNAL_RESTRICT src_void, Index count,
    Index factor) {
  auto* acc = static_cast<Acc*>(acc_void);
  const auto* src = static_cast<const Src*>(src_void);
  if (factor == 1) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      acc[i] = Op::Combine(acc[i], src[i]);
    }
  } else {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const Src* row = src + i * factor;
      Acc v = acc[i];
      if constexpr (std::is_same_v<Op, SumOp> && sizeof(Src) < sizeof(Acc)) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
        for (Index k = 0; k < factor; ++k) {
          v = Op::Combine(v, row[k]);
        }
      } else {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index k = 0; k < factor; ++k) {
          v = Op::Combine(v, row[k]);
        }
      }
      acc[i] = v;
    }
  }
}

/// Divides `bool` sums by `divisor` with round-to-nearest-even (`false` == 0 is
/// even, so exact `divisor / 2` ties round to `false`).  Also implements `bool`
/// `kMedian` and `kMode`, which have identical tie-breaking semantics.
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void FinalizeMeanBoolRow(
    void* TENSORSTORE_INTERNAL_RESTRICT acc_void,
    void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count, Index divisor) {
  const auto* acc = static_cast<const int64_t*>(acc_void);
  auto* out = static_cast<bool*>(dst_void);
  for (Index i = 0; i < count; ++i) {
    out[i] = (acc[i] * 2 > divisor);
  }
}

/// Granlund-Montgomery 64-bit unsigned division multiplier for a
/// non-power-of-two `u_div` in `[3, INT64_MAX]`, exact for all `uint64_t`
/// numerators.
struct UnsignedMagicDiv {
  uint64_t mul;
  uint64_t u_div;
  int shift;
};

TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline UnsignedMagicDiv
MakeUnsignedMagicDiv(uint64_t u_div) {
  assert(u_div >= 3 && (u_div & (u_div - 1)) != 0);
  const int l = 64 - absl::countl_zero(u_div - 1);
  const uint64_t rem_pow2 = (uint64_t{1} << l) - u_div;
  const uint64_t mul =
      static_cast<uint64_t>(absl::MakeUint128(rem_pow2, 0) / u_div) + 1;
  return {mul, u_div, l - 1};
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline uint64_t RoundDivUnsignedMagic(
    uint64_t x, const UnsignedMagicDiv& m) {
  const uint64_t t1 = absl::Uint128High64(absl::uint128{x} * m.mul);
  const uint64_t q = (t1 + ((x - t1) >> 1)) >> m.shift;
  const uint64_t r = x - q * m.u_div;
  return q + (r * 2 + (q & 1) > m.u_div);
}

ABSL_ATTRIBUTE_ALWAYS_INLINE inline int64_t RoundDivSignedMagic(
    int64_t x, const UnsignedMagicDiv& m) {
  const uint64_t ux = static_cast<uint64_t>(x);
  const uint64_t u_abs = (x >= 0) ? ux : (uint64_t{0} - ux);
  const uint64_t q_mag = RoundDivUnsignedMagic(u_abs, m);
  return static_cast<int64_t>((x >= 0) ? q_mag : (uint64_t{0} - q_mag));
}

/// 32-bit unsigned magic multiplier parameters for non-power-of-two `d` in
/// `[3, 0x3fffffff]` exact for all `uint32_t` dividends `x <= 0x80000000u`.
struct UnsignedMagicDiv32 {
  uint32_t multiplier;
  int shift;
};

constexpr UnsignedMagicDiv32 MakeUnsignedMagicDiv32(uint32_t d) {
  const int l = 32 - absl::countl_zero(d - 1);
  const int shift = 31 + l;
  const uint32_t m = static_cast<uint32_t>((uint64_t{1} << shift) / d + 1);
  return {m, shift};
}

/// Divides `count` unsigned 64-bit sums (`acc[i] <= max_elem_abs * divisor`) in
/// place by `divisor` with round-half-to-even.
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
DivideMeanUint64Row(uint64_t* acc, Index count, Index divisor,
                    uint64_t max_elem_abs) {
  if (divisor == 1) return;
  const uint64_t u_div = static_cast<uint64_t>(divisor);
  if ((u_div & (u_div - 1)) == 0) {
    const int shift = absl::countr_zero(u_div);
    const uint64_t mask = u_div - 1;
    const uint64_t half = u_div >> 1;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const uint64_t x = acc[i];
      const uint64_t q = x >> shift;
      const uint64_t r = x & mask;
      acc[i] = q + ((r + (q & 1)) > half);
    }
    return;
  }
  if (u_div <= 0x3fffffffu && max_elem_abs <= (0x80000000ULL / 3) &&
      max_elem_abs * u_div <= 0x80000000ULL) {
    const uint32_t d32 = static_cast<uint32_t>(u_div);
    const UnsignedMagicDiv32 m32 = MakeUnsignedMagicDiv32(d32);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const uint32_t x = static_cast<uint32_t>(acc[i]);
      const uint32_t q =
          static_cast<uint32_t>((uint64_t{x} * m32.multiplier) >> m32.shift);
      const uint32_t r = x - q * d32;
      acc[i] = q + (r * 2u + (q & 1u) > d32);
    }
    return;
  }
  if (count == 1) {
    const uint64_t x = acc[0];
    const uint64_t q = x / u_div;
    const uint64_t r = x % u_div;
    acc[0] = q + (r * 2 + (q & 1) > u_div);
    return;
  }
  const UnsignedMagicDiv magic = MakeUnsignedMagicDiv(u_div);
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    acc[i] = RoundDivUnsignedMagic(acc[i], magic);
  }
}

/// Divides `count` signed 64-bit sums (`|acc[i]| <= max_elem_abs * divisor`) in
/// place by `divisor` with round-half-to-even.
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
DivideMeanInt64Row(int64_t* acc, Index count, Index divisor,
                   uint64_t max_elem_abs) {
  if (divisor == 1) return;
  const uint64_t u_div = static_cast<uint64_t>(divisor);
  if ((u_div & (u_div - 1)) == 0) {
    // Exact branchless round-to-nearest-even for `divisor = 2^shift` using
    // Euclidean remainder `x = q * 2^shift + r` (`0 <= r < 2^shift`).
    const int shift = absl::countr_zero(u_div);
    const uint64_t mask = u_div - 1;
    const uint64_t half = u_div >> 1;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const int64_t x = acc[i];
      const int64_t q = x >> shift;
      const uint64_t r = static_cast<uint64_t>(x) & mask;
      const uint64_t q_odd = static_cast<uint64_t>(q) & 1;
      acc[i] = q + static_cast<int64_t>(r + q_odd > half);
    }
    return;
  }
  if (u_div <= 0x3fffffffu && max_elem_abs <= (0x80000000ULL / 3) &&
      max_elem_abs * u_div <= 0x80000000ULL) {
    const uint32_t d32 = static_cast<uint32_t>(u_div);
    const UnsignedMagicDiv32 m32 = MakeUnsignedMagicDiv32(d32);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const int32_t x = static_cast<int32_t>(acc[i]);
      const uint32_t ux = static_cast<uint32_t>(x);
      const uint32_t u_abs = (x >= 0) ? ux : (0u - ux);
      uint32_t q_mag = static_cast<uint32_t>(
          (uint64_t{u_abs} * m32.multiplier) >> m32.shift);
      const uint32_t r = u_abs - q_mag * d32;
      q_mag += (r * 2u + (q_mag & 1u) > d32);
      const int64_t q_signed = static_cast<int64_t>(q_mag);
      acc[i] = (x >= 0) ? q_signed : -q_signed;
    }
    return;
  }
  if (count == 1) {
    const int64_t x = acc[0];
    const uint64_t ux = static_cast<uint64_t>(x);
    const uint64_t u_abs = (x >= 0) ? ux : (uint64_t{0} - ux);
    const uint64_t q = u_abs / u_div;
    const uint64_t r = u_abs % u_div;
    const uint64_t q_mag = q + (r * 2 + (q & 1) > u_div);
    acc[0] = static_cast<int64_t>((x >= 0) ? q_mag : (uint64_t{0} - q_mag));
    return;
  }
  const UnsignedMagicDiv magic = MakeUnsignedMagicDiv(u_div);
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    acc[i] = RoundDivSignedMagic(acc[i], magic);
  }
}

/// Divides integer sums in `acc_void` by `divisor`, rounding to nearest and
/// breaking exact half-way ties to even.  Narrow outputs (`8/16/32`-bit) share
/// `DivideMeanUint64Row` / `DivideMeanInt64Row` in place before narrowing.
template <typename Acc, typename Out>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void FinalizeMeanIntRow(
    void* TENSORSTORE_INTERNAL_RESTRICT acc_void,
    void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count, Index divisor) {
  auto* out = static_cast<Out*>(dst_void);
  if constexpr (std::is_same_v<Acc, uint64_t>) {
    static_assert(sizeof(Out) < sizeof(uint64_t));
    auto* acc = static_cast<uint64_t*>(acc_void);
    constexpr uint64_t kMaxAbs =
        static_cast<uint64_t>(std::numeric_limits<Out>::max());
    DivideMeanUint64Row(acc, count, divisor, kMaxAbs);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      out[i] = static_cast<Out>(acc[i]);
    }
  } else if constexpr (std::is_same_v<Acc, int64_t>) {
    static_assert(sizeof(Out) < sizeof(int64_t));
    auto* acc = static_cast<int64_t*>(acc_void);
    constexpr uint64_t kMaxAbs =
        static_cast<uint64_t>(std::numeric_limits<Out>::max()) + 1;
    DivideMeanInt64Row(acc, count, divisor, kMaxAbs);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      out[i] = static_cast<Out>(acc[i]);
    }
  } else {
    const auto* acc = static_cast<const Acc*>(acc_void);
    if (divisor == 1) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        out[i] = static_cast<Out>(acc[i]);
      }
      return;
    }
    const uint64_t u_div = static_cast<uint64_t>(divisor);
    if ((u_div & (u_div - 1)) == 0) {
      const int shift = absl::countr_zero(u_div);
      const uint64_t mask = u_div - 1;
      const uint64_t half = u_div >> 1;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        const Acc x = acc[i];
        const Acc q = x >> shift;
        const uint64_t r = static_cast<uint64_t>(x) & mask;
        const uint64_t q_odd = static_cast<uint64_t>(q) & 1;
        out[i] = static_cast<Out>(q + static_cast<Acc>(r + q_odd > half));
      }
      return;
    }
    if constexpr (std::is_same_v<Acc, absl::uint128>) {
      const absl::uint128 div128 = u_div;
      if (count == 1) {
        const absl::uint128 x = acc[0];
        if (ABSL_PREDICT_TRUE(absl::Uint128High64(x) == 0)) {
          const uint64_t x64 = absl::Uint128Low64(x);
          const uint64_t q = x64 / u_div;
          const uint64_t r = x64 % u_div;
          out[0] = q + (r * 2 + (q & 1) > u_div);
        } else {
          const absl::uint128 q = x / div128;
          const uint64_t r = static_cast<uint64_t>(x % div128);
          const uint64_t q64 = static_cast<uint64_t>(q);
          out[0] = q64 + (r * 2 + (q64 & 1) > u_div);
        }
        return;
      }
      // Fast-path 128-bit sums that fit in 64 bits via `magic` to avoid
      // software `__udivti3`.
      const UnsignedMagicDiv magic = MakeUnsignedMagicDiv(u_div);
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        const absl::uint128 x = acc[i];
        if (ABSL_PREDICT_TRUE(absl::Uint128High64(x) == 0)) {
          out[i] = RoundDivUnsignedMagic(absl::Uint128Low64(x), magic);
        } else {
          const absl::uint128 q = x / div128;
          const uint64_t r = static_cast<uint64_t>(x % div128);
          const uint64_t q64 = static_cast<uint64_t>(q);
          out[i] = q64 + (r * 2 + (q64 & 1) > u_div);
        }
      }
    } else {
      static_assert(std::is_same_v<Acc, absl::int128>);
      const absl::int128 div128 = divisor;
      if (count == 1) {
        const absl::int128 x = acc[0];
        const int64_t x64 = static_cast<int64_t>(x);
        if (ABSL_PREDICT_TRUE(absl::int128(x64) == x)) {
          const uint64_t ux = static_cast<uint64_t>(x64);
          const uint64_t u_abs = (x64 >= 0) ? ux : (uint64_t{0} - ux);
          const uint64_t q = u_abs / u_div;
          const uint64_t r = u_abs % u_div;
          const uint64_t q_mag = q + (r * 2 + (q & 1) > u_div);
          out[0] =
              static_cast<int64_t>((x64 >= 0) ? q_mag : (uint64_t{0} - q_mag));
        } else {
          const absl::int128 q = x / div128;
          const int64_t r = static_cast<int64_t>(x % div128);
          const uint64_t u_r = (r >= 0)
                                   ? static_cast<uint64_t>(r)
                                   : (uint64_t{0} - static_cast<uint64_t>(r));
          const uint64_t q_odd = static_cast<uint64_t>(q) & 1;
          const int adjust = (u_r * 2 + q_odd > u_div) ? 1 : 0;
          out[0] = static_cast<int64_t>((x >= 0) ? (q + adjust) : (q - adjust));
        }
        return;
      }
      // Fast-path signed 128-bit sums that fit in `int64_t` via `magic` to
      // avoid software `__divti3`.
      const UnsignedMagicDiv magic = MakeUnsignedMagicDiv(u_div);
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        const absl::int128 x = acc[i];
        const int64_t x64 = static_cast<int64_t>(x);
        if (ABSL_PREDICT_TRUE(absl::int128(x64) == x)) {
          out[i] = RoundDivSignedMagic(x64, magic);
        } else {
          const absl::int128 q = x / div128;
          const int64_t r = static_cast<int64_t>(x % div128);
          const uint64_t u_r = (r >= 0)
                                   ? static_cast<uint64_t>(r)
                                   : (uint64_t{0} - static_cast<uint64_t>(r));
          const uint64_t q_odd = static_cast<uint64_t>(q) & 1;
          const int adjust = (u_r * 2 + q_odd > u_div) ? 1 : 0;
          out[i] = static_cast<int64_t>((x >= 0) ? (q + adjust) : (q - adjust));
        }
      }
    }
  }
}

/// Divides real floating-point sums in `acc_void` by `divisor`.
template <typename Scalar>
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
FinalizeMeanFloatRow(void* TENSORSTORE_INTERNAL_RESTRICT acc_void,
                     void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count,
                     Index divisor) {
  const auto* acc = static_cast<const Scalar*>(acc_void);
  auto* out = static_cast<Scalar*>(dst_void);
  const uint64_t u_div = static_cast<uint64_t>(divisor);
  if ((u_div & (u_div - 1)) == 0) {
    // Multiplication by an exact power-of-two reciprocal is IEEE-754 bit-exact.
    const Scalar inv_div = Scalar{1} / static_cast<Scalar>(divisor);
    for (Index i = 0; i < count; ++i) {
      out[i] = acc[i] * inv_div;
    }
  } else {
    const Scalar div = static_cast<Scalar>(divisor);
    for (Index i = 0; i < count; ++i) {
      out[i] = acc[i] / div;
    }
  }
}

/// Divides interleaved `std::complex<Scalar>` sums componentwise by real
/// `divisor` (avoiding Annex G `std::complex::operator/=` scaling, which rounds
/// less accurately and turns `(inf + 0i) / n` into `(inf + NaN*i)`).
template <typename Scalar>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void FinalizeMeanComplexRow(
    void* acc_void, void* dst_void, Index count, Index divisor) {
  FinalizeMeanFloatRow<Scalar>(acc_void, dst_void, count * 2, divisor);
}

/// Maps IEEE-754 `float` (`uint32_t`) or `double` (`uint64_t`) raw bits `u` to
/// an order-preserving unsigned key:
///   - Negative values (`[-inf, -0.0)`): `~u` (`< kSignBit - 1`).
///   - `-0.0`: `kSignBit - 1`.
///   - `+0.0`: `kSignBit`.
///   - Positive values (`(+0.0, +inf]`): `u | kSignBit` up to `kInfKey`.
///   - Any `NaN` (`mag > kInfMag`): `kNaNKey` (`> kInfKey`), which sorts after
///     `+inf` and decodes to canonical
///     `std::numeric_limits<Float>::quiet_NaN()` (input NaN sign/payload bits
///     are not preserved on all-NaN windows).
///
/// Per-width constants are specialized rather than selected with a ternary,
/// since `uint32_t{0x7ff0000000000000ULL}` in the unselected arm is an
/// ill-formed narrowing conversion that MSVC rejects (C2397).
template <typename UInt>
struct FloatKeyConstants;

template <>
struct FloatKeyConstants<uint32_t> {
  static constexpr uint32_t kInfMag = 0x7f800000u;
  static constexpr uint32_t kInfKey = 0xff800000u;
  static constexpr uint32_t kNaNKey = 0xffc00000u;
};

template <>
struct FloatKeyConstants<uint64_t> {
  static constexpr uint64_t kInfMag = 0x7ff0000000000000ULL;
  static constexpr uint64_t kInfKey = 0xfff0000000000000ULL;
  static constexpr uint64_t kNaNKey = 0xfff8000000000000ULL;
};

template <typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline UInt FloatBitsToOrderedKey(UInt u) {
  static_assert(sizeof(UInt) == 4 || sizeof(UInt) == 8);
  using SInt = std::make_signed_t<UInt>;
  constexpr int kBits = sizeof(UInt) * 8;
  constexpr UInt kSignBit = UInt{1} << (kBits - 1);
  constexpr UInt kMagMask = kSignBit - 1;
  constexpr UInt kInfMag = FloatKeyConstants<UInt>::kInfMag;
  constexpr UInt kNaNKey = FloatKeyConstants<UInt>::kNaNKey;
  const UInt mag = u & kMagMask;
  const UInt mask =
      static_cast<UInt>(static_cast<SInt>(u) >> (kBits - 1)) | kSignBit;
  const UInt flipped = u ^ mask;
  return (mag > kInfMag) ? kNaNKey : flipped;
}

template <typename Float>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline auto FloatToOrderedKey(Float x) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  using UInt = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  UInt u;
  std::memcpy(&u, &x, sizeof(u));
  return FloatBitsToOrderedKey(u);
}

/// Inverse of `FloatBitsToOrderedKey`.  `kNaNKey ^ kSignBit` produces canonical
/// `quiet_NaN()` bits directly without a branch.
template <typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline UInt OrderedKeyToFloatBits(UInt k) {
  static_assert(sizeof(UInt) == 4 || sizeof(UInt) == 8);
  constexpr int kBits = sizeof(UInt) * 8;
  constexpr UInt kSignBit = UInt{1} << (kBits - 1);
  const UInt inv_mask = ((k >> (kBits - 1)) - UInt{1}) | kSignBit;
  return k ^ inv_mask;
}

template <typename Float, typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline Float OrderedKeyToFloat(UInt k) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  const UInt u = OrderedKeyToFloatBits(k);
  Float out;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

/// Branchless small-window `kMedian` and `kMode` networks (`divisor` in 3..4).
/// Ties in even-sized `kMedian` take the lower median (`(valid_n - 1) / 2`),
/// and ties in `kMode` select the smallest value.  Since any NaN key equals
/// `kNaNKey > max_valid_key`, sorting places all NaNs at the top of the
/// network (`c` and `d`), allowing partial-NaN and all-NaN windows to be
/// resolved with a single comparison against `max_valid_key` without a
/// fallback call.
template <typename T>
constexpr T DefaultMaxValidKey() {
  if constexpr (std::is_unsigned_v<T>) {
    return ~T{0};
  } else {
    return T{};
  }
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void Sort2(T& x, T& y) {
  const T lo = std::min(x, y);
  const T hi = std::max(x, y);
  x = lo;
  y = hi;
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T Median3(
    T a, T b, T c, T max_valid_key = DefaultMaxValidKey<T>()) {
  Sort2(a, b);
  Sort2(b, c);
  Sort2(a, b);
  if constexpr (std::is_unsigned_v<T> && sizeof(T) >= 4) {
    return (c <= max_valid_key) ? b : a;
  } else {
    (void)max_valid_key;
    return b;
  }
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T LowerMedian4(
    T a, T b, T c, T d, T max_valid_key = DefaultMaxValidKey<T>()) {
  Sort2(a, b);
  Sort2(c, d);
  Sort2(a, c);
  b = std::min(b, d);
  Sort2(b, c);
  if constexpr (std::is_unsigned_v<T> && sizeof(T) >= 4) {
    return (c <= max_valid_key) ? b : a;
  } else {
    (void)max_valid_key;
    return b;
  }
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T Mode3(
    T a, T b, T c, T max_valid_key = DefaultMaxValidKey<T>()) {
  Sort2(a, b);
  Sort2(b, c);
  Sort2(a, b);
  if constexpr (std::is_unsigned_v<T> && sizeof(T) >= 4) {
    return (b == c && c <= max_valid_key) ? b : a;
  } else {
    (void)max_valid_key;
    return (b == c) ? b : a;
  }
}

template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T Mode4(
    T a, T b, T c, T d, T max_valid_key = DefaultMaxValidKey<T>()) {
  Sort2(a, b);
  Sort2(c, d);
  Sort2(a, c);
  Sort2(b, d);
  Sort2(b, c);
  if constexpr (std::is_unsigned_v<T> && sizeof(T) >= 4) {
    const T r2 = (c == d && d <= max_valid_key) ? c : a;
    const T r1 = (b == c && c <= max_valid_key) ? b : r2;
    return (a == b) ? a : r1;
  } else {
    (void)max_valid_key;
    const T r2 = (c == d) ? c : a;
    const T r1 = (b == c) ? b : r2;
    return (a == b) ? a : r1;
  }
}

constexpr Index kInsertionSortThreshold = 20;
constexpr Index kHistogram8BitThreshold = 32;

/// Returns true if all `n` elements of `w` equal `w[0]`.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline bool IsUniformWindow(const T* w, Index n) {
  const T& first = w[0];
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (Index j = 1; j < n; ++j) {
    if (!(w[j] == first)) return false;
  }
  return true;
}

/// Scans a non-empty sorted window `w[0..n)` and returns the most frequent
/// value, breaking ties in favor of the earliest (smallest) value.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline T& FindMostFrequentSorted(T* w, Index n) {
  Index most_frequent_index = 0;
  size_t most_frequent_count = 1;
  size_t cur_count = 1;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (Index j = 1; j < n; ++j) {
    if (w[j] == w[j - 1]) {
      ++cur_count;
    } else {
      if (cur_count > most_frequent_count) {
        most_frequent_count = cur_count;
        most_frequent_index = j - 1;
      }
      cur_count = 1;
    }
  }
  if (cur_count > most_frequent_count) {
    most_frequent_index = n - 1;
  }
  return w[most_frequent_index];
}

ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline uint8_t
MedianHistogram8Bit(const uint8_t* w, Index divisor, Index median_idx) {
  size_t hist[256] = {};
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (Index j = 0; j < divisor; ++j) {
    ++hist[w[j]];
  }
  Index rem = median_idx;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (unsigned v = 0; v < 255; ++v) {
    rem -= static_cast<Index>(hist[v]);
    if (rem < 0) return static_cast<uint8_t>(v);
  }
  return 255;
}

ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline uint8_t
ModeHistogram8Bit(const uint8_t* w, Index divisor) {
  size_t hist[256] = {};
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (Index j = 0; j < divisor; ++j) {
    ++hist[w[j]];
  }
  uint8_t best_v = 0;
  size_t best_c = 0;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (unsigned v = 0; v < 256; ++v) {
    if (hist[v] > best_c) {
      best_c = hist[v];
      best_v = static_cast<uint8_t>(v);
    }
  }
  return best_v;
}

/// Reduces an unsigned key window `w[0..divisor)` to its lower median
/// (`(valid_n - 1) / 2`), ignoring keys `> max_valid_key` (float NaN keys
/// `kNaNKey > kInfKey`) unless the entire window exceeds `max_valid_key`.
/// Uses an outlined 256-bin histogram for `uint8_t` when
/// `divisor > kHistogram8BitThreshold`.
template <typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline UInt MedianWindowUnsigned(
    UInt* w, Index divisor, UInt max_valid_key) {
  static_assert(std::is_unsigned_v<UInt>);
  if (divisor > 4 && IsUniformWindow(w, divisor)) {
    return w[0];
  }
  if constexpr (sizeof(UInt) == 1) {
    (void)max_valid_key;
    const Index median_idx =
        static_cast<Index>(static_cast<size_t>(divisor - 1) >> 1);
    if (divisor <= kHistogram8BitThreshold) {
      internal::InsertionSort(w, w + divisor);
      return w[median_idx];
    }
    return MedianHistogram8Bit(w, divisor, median_idx);
  } else {
    Index valid_n = divisor;
    if constexpr (sizeof(UInt) >= 4) {
      for (Index j = 0; j < valid_n;) {
        if (ABSL_PREDICT_FALSE(w[j] > max_valid_key)) {
          --valid_n;
          std::swap(w[j], w[valid_n]);
        } else {
          ++j;
        }
      }
      if (ABSL_PREDICT_FALSE(valid_n == 0)) return w[0];
    }
    const Index median_idx =
        static_cast<Index>(static_cast<size_t>(valid_n - 1) >> 1);
    if (valid_n <= kInsertionSortThreshold) {
      internal::InsertionSort(w, w + valid_n);
    } else {
      std::nth_element(w, w + median_idx, w + valid_n);
    }
    return w[median_idx];
  }
}

template <typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void StoreUnalignedKey(
    internal::TrivialBytes<sizeof(UInt)>& dst, UInt v) {
  std::memcpy(&dst, &v, sizeof(UInt));
}

/// Shared unsigned lower-median row kernel (`UInt` in `uint8/16/32/64_t`).
/// Writes `dst_void` via `std::memcpy` so callers with live `Float[]` output
/// buffers never incur `UInt*` strict-aliasing UB.
template <typename UInt>
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
ComputeMedianUnsignedImpl(void* TENSORSTORE_INTERNAL_RESTRICT staged_void,
                          Index stride,
                          void* TENSORSTORE_INTERNAL_RESTRICT dst_void,
                          Index count, Index divisor, UInt max_valid_key) {
  static_assert(std::is_unsigned_v<UInt>);
  using OutBytes = internal::TrivialBytes<sizeof(UInt)>;
  auto* in = static_cast<UInt*>(staged_void);
  auto* out = static_cast<OutBytes*>(dst_void);
  if (divisor == 1) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      StoreUnalignedKey(out[i], in[i * stride]);
    }
    return;
  }
  if (divisor == stride) {
    if (divisor == 2) {
      for (Index i = 0; i < count; ++i) {
        StoreUnalignedKey(out[i], std::min(in[2 * i], in[2 * i + 1]));
      }
      return;
    }
    if (divisor == 3) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        const UInt* w = in + 3 * i;
        StoreUnalignedKey(out[i], Median3(w[0], w[1], w[2], max_valid_key));
      }
      return;
    }
    if (divisor == 4) {
      for (Index i = 0; i < count; ++i) {
        const UInt* w = in + 4 * i;
        StoreUnalignedKey(out[i],
                          LowerMedian4(w[0], w[1], w[2], w[3], max_valid_key));
      }
      return;
    }
  }
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    StoreUnalignedKey(
        out[i], MedianWindowUnsigned(in + i * stride, divisor, max_valid_key));
  }
}

template <typename UInt>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void ComputeMedianUnsignedRow(
    void* staged_void, Index stride, void* dst_void, Index count,
    Index divisor) {
  ComputeMedianUnsignedImpl<UInt>(staged_void, stride, dst_void, count, divisor,
                                  ~UInt{0});
}

/// Reduces an unsigned key window `w[0..divisor)` to its mode, ignoring keys
/// `> max_valid_key` (float NaN keys `kNaNKey > kInfKey`) unless the entire
/// window exceeds `max_valid_key`.
template <typename UInt>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline UInt ModeWindowUnsigned(
    UInt* w, Index divisor, UInt max_valid_key) {
  static_assert(std::is_unsigned_v<UInt>);
  if (divisor > 4 && IsUniformWindow(w, divisor)) {
    return w[0];
  }
  if constexpr (sizeof(UInt) == 1) {
    (void)max_valid_key;
    if (divisor <= kHistogram8BitThreshold) {
      internal::InsertionSort(w, w + divisor);
      return FindMostFrequentSorted(w, divisor);
    }
    return ModeHistogram8Bit(w, divisor);
  } else {
    internal::ShellSort(w, w + divisor);
    Index valid_n = divisor;
    if constexpr (sizeof(UInt) >= 4) {
      if (ABSL_PREDICT_FALSE(w[valid_n - 1] > max_valid_key)) {
        do {
          --valid_n;
        } while (valid_n > 0 && w[valid_n - 1] > max_valid_key);
        if (valid_n == 0) return w[0];
      }
    }
    return FindMostFrequentSorted(w, valid_n);
  }
}

/// Shared unsigned mode row kernel (`UInt` in `uint8/16/32/64_t`).
/// Writes `dst_void` via `std::memcpy` so callers with live `Float[]` output
/// buffers never incur `UInt*` strict-aliasing UB.
template <typename UInt>
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
ComputeModeUnsignedImpl(void* TENSORSTORE_INTERNAL_RESTRICT staged_void,
                        Index stride,
                        void* TENSORSTORE_INTERNAL_RESTRICT dst_void,
                        Index count, Index divisor, UInt max_valid_key) {
  static_assert(std::is_unsigned_v<UInt>);
  using OutBytes = internal::TrivialBytes<sizeof(UInt)>;
  auto* in = static_cast<UInt*>(staged_void);
  auto* out = static_cast<OutBytes*>(dst_void);
  if (divisor == 1) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      StoreUnalignedKey(out[i], in[i * stride]);
    }
    return;
  }
  if (divisor == stride) {
    if (divisor == 2) {
      for (Index i = 0; i < count; ++i) {
        StoreUnalignedKey(out[i], std::min(in[2 * i], in[2 * i + 1]));
      }
      return;
    }
    if (divisor == 3) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        const UInt* w = in + 3 * i;
        StoreUnalignedKey(out[i], Mode3(w[0], w[1], w[2], max_valid_key));
      }
      return;
    }
    if (divisor == 4) {
      for (Index i = 0; i < count; ++i) {
        const UInt* w = in + 4 * i;
        StoreUnalignedKey(out[i], Mode4(w[0], w[1], w[2], w[3], max_valid_key));
      }
      return;
    }
  }
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    StoreUnalignedKey(
        out[i], ModeWindowUnsigned(in + i * stride, divisor, max_valid_key));
  }
}

template <typename UInt>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void ComputeModeUnsignedRow(
    void* staged_void, Index stride, void* dst_void, Index count,
    Index divisor) {
  ComputeModeUnsignedImpl<UInt>(staged_void, stride, dst_void, count, divisor,
                                ~UInt{0});
}

/// Shared sign-bit bias wrapper mapping `intN_t` windows to `uintN_t` for both
/// `kMedian` and `kMode` (`x ^ kSignBit` is a strict order + equality
/// bijection).
template <typename UInt>
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
RunSignedWindowReduction(void* staged_void, Index stride, void* dst_void,
                         Index count, Index divisor,
                         ReduceStoredRowFn unsigned_kernel) {
  constexpr UInt kSignBit = UInt{1} << (sizeof(UInt) * 8 - 1);
  if (divisor > 1) {
    auto* in = static_cast<UInt*>(staged_void);
    if (stride == divisor) {
      const Index total = count * divisor;
      for (Index idx = 0; idx < total; ++idx) {
        in[idx] ^= kSignBit;
      }
    } else {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index i = 0; i < count; ++i) {
        UInt* w = in + i * stride;
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (Index k = 0; k < divisor; ++k) {
          w[k] ^= kSignBit;
        }
      }
    }
  }
  unsigned_kernel(staged_void, stride, dst_void, count, divisor);
  if (divisor > 1) {
    auto* out = static_cast<UInt*>(dst_void);
    for (Index i = 0; i < count; ++i) {
      out[i] ^= kSignBit;
    }
  }
}

/// Shared radix-bijection wrapper mapping `float` / `double` windows to
/// `uint32_t` / `uint64_t` for both `kMedian` (`is_mode == false`) and `kMode`
/// (`is_mode == true`).  For `kMode`, `-0.0` and `+0.0` within a window
/// coalesce to the first zero bit-pattern in that window so mixed signed zeros
/// count as equal while an all-`-0.0` window preserves `-0.0`.  Reads and
/// writes `Float` representations via `std::memcpy` / `TrivialBytes` and starts
/// `UInt` lifetimes via placement `new` so `Float*` and `UInt*` never alias.
template <typename Float>
ABSL_ATTRIBUTE_NOINLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
RunFloatWindowReduction(void* staged_void, Index stride, void* dst_void,
                        Index count, Index divisor, bool is_mode) {
  using UInt = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  using Bytes = internal::TrivialBytes<sizeof(Float)>;
  constexpr UInt kSignBit = UInt{1} << (sizeof(UInt) * 8 - 1);
  constexpr UInt kMagMask = kSignBit - 1;
  constexpr UInt kInfKey = FloatKeyConstants<UInt>::kInfKey;
  if (divisor == 1) {
    const auto* TENSORSTORE_INTERNAL_RESTRICT in_b =
        static_cast<const Bytes*>(staged_void);
    auto* TENSORSTORE_INTERNAL_RESTRICT out_b = static_cast<Bytes*>(dst_void);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      out_b[i] = in_b[i * stride];
    }
    return;
  }
  const auto* in_bytes = static_cast<const unsigned char*>(staged_void);
  auto* in_u = static_cast<UInt*>(staged_void);
  if (!is_mode && stride == divisor) {
    const Index total = count * divisor;
    for (Index idx = 0; idx < total; ++idx) {
      UInt u;
      std::memcpy(&u, in_bytes + idx * sizeof(UInt), sizeof(UInt));
      new (&in_u[idx]) UInt(FloatBitsToOrderedKey(u));
    }
  } else {
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const unsigned char* wb =
          in_bytes + i * stride * static_cast<Index>(sizeof(UInt));
      UInt* wu = in_u + i * stride;
      UInt zero_bits = UInt{1};
#if defined(__clang__)
#pragma clang loop vectorize(disable) unroll(disable)
#endif
      for (Index k = 0; k < divisor; ++k) {
        UInt u;
        std::memcpy(&u, wb + k * sizeof(UInt), sizeof(UInt));
        if (is_mode && (u & kMagMask) == 0) {
          if (zero_bits == 1) zero_bits = u;
          u = zero_bits;
        }
        new (&wu[k]) UInt(FloatBitsToOrderedKey(u));
      }
    }
  }
  if (is_mode) {
    ComputeModeUnsignedImpl<UInt>(staged_void, stride, dst_void, count, divisor,
                                  kInfKey);
  } else {
    ComputeMedianUnsignedImpl<UInt>(staged_void, stride, dst_void, count,
                                    divisor, kInfKey);
  }
  auto* dst_bytes = static_cast<unsigned char*>(dst_void);
  for (Index i = 0; i < count; ++i) {
    UInt k;
    std::memcpy(&k, dst_bytes + i * sizeof(UInt), sizeof(UInt));
    const UInt u = OrderedKeyToFloatBits(k);
    std::memcpy(dst_bytes + i * sizeof(UInt), &u, sizeof(UInt));
  }
}

/// Lexicographical order over non-NaN `std::complex<T>` values.
template <typename T>
struct ComplexLess {
  bool operator()(const std::complex<T>& a, const std::complex<T>& b) const {
    return (a.real() < b.real()) ||
           (a.real() == b.real() && a.imag() < b.imag());
  }
};

/// Computes the mode across `count` windows for non-scalar types (`complex`,
/// `string_t`, `ustring_t`, `json_t`) using `ShellSort` directly on `T`.
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void
ComputeModeGenericRow(void* TENSORSTORE_INTERNAL_RESTRICT staged_void,
                      Index stride,
                      void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count,
                      Index divisor) {
  using std::swap;
  auto* in = static_cast<T*>(staged_void);
  auto* out = static_cast<T*>(dst_void);
  if (divisor == 1) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      swap(out[i], in[i * stride]);
    }
    return;
  }
  for (Index i = 0; i < count; ++i) {
    T* w = in + i * stride;
    if (divisor > 2 && IsUniformWindow(w, divisor)) {
      swap(out[i], w[0]);
      continue;
    }
    Index valid_n = divisor;
    if constexpr (IsComplex<T>) {
      for (Index j = 0; j < valid_n;) {
        if (ABSL_PREDICT_FALSE(std::isnan(w[j].real()) ||
                               std::isnan(w[j].imag()))) {
          --valid_n;
          swap(w[j], w[valid_n]);
        } else {
          ++j;
        }
      }
      if (ABSL_PREDICT_FALSE(valid_n == 0)) {
        swap(out[i], w[0]);
        continue;
      }
      internal::ShellSort(w, w + valid_n,
                          ComplexLess<typename T::value_type>{});
    } else {
      internal::ShellSort(w, w + valid_n);
    }
    swap(out[i], FindMostFrequentSorted(w, valid_n));
  }
}

/// Computes the lower median (`(divisor - 1) / 2`) across `count` staged
/// windows separated by `stride` elements.
template <typename T>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void ComputeMedianRow(
    void* staged_void, Index stride, void* dst_void, Index count,
    Index divisor) {
  if constexpr (std::is_unsigned_v<T>) {
    ComputeMedianUnsignedRow<T>(staged_void, stride, dst_void, count, divisor);
  } else if constexpr (std::is_integral_v<T>) {
    using UInt = std::make_unsigned_t<T>;
    RunSignedWindowReduction<UInt>(staged_void, stride, dst_void, count,
                                   divisor, &ComputeMedianUnsignedRow<UInt>);
  } else {
    static_assert(std::is_floating_point_v<T>);
    RunFloatWindowReduction<T>(staged_void, stride, dst_void, count, divisor,
                               /*is_mode=*/false);
  }
}

/// Computes the mode across `count` staged windows separated by `stride`
/// elements.
template <typename T>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN inline void ComputeModeRow(
    void* staged_void, Index stride, void* dst_void, Index count,
    Index divisor) {
  if constexpr (std::is_unsigned_v<T>) {
    ComputeModeUnsignedRow<T>(staged_void, stride, dst_void, count, divisor);
  } else if constexpr (std::is_integral_v<T>) {
    using UInt = std::make_unsigned_t<T>;
    RunSignedWindowReduction<UInt>(staged_void, stride, dst_void, count,
                                   divisor, &ComputeModeUnsignedRow<UInt>);
  } else if constexpr (std::is_floating_point_v<T>) {
    RunFloatWindowReduction<T>(staged_void, stride, dst_void, count, divisor,
                               /*is_mode=*/true);
  } else if constexpr (std::is_same_v<T, Utf8String>) {
    // `Utf8String` is a single-member wrapper around `std::string` with
    // identical size, alignment, and comparison semantics.
    static_assert(sizeof(Utf8String) == sizeof(std::string) &&
                  alignof(Utf8String) == alignof(std::string) &&
                  std::is_standard_layout_v<Utf8String> ==
                      std::is_standard_layout_v<std::string>);
    ComputeModeRow<std::string>(staged_void, stride, dst_void, count, divisor);
  } else {
    ComputeModeGenericRow<T>(staged_void, stride, dst_void, count, divisor);
  }
}

}  // namespace internal_downsample
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_KERNELS_H_
