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

#include "tensorstore/data_type_conversion.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/meta/attributes.h"
#include "tensorstore/internal/scatter_gather_util.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/util/bfloat16.h"  // IWYU pragma: keep
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/float8.h"   // IWYU pragma: keep
#include "tensorstore/util/int2.h"     // IWYU pragma: keep
#include "tensorstore/util/int4.h"     // IWYU pragma: keep
#include "tensorstore/util/mxfloat.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {

/// Define conversions between canonical data types.

namespace {

// Returns `true` if `T` is any of `Ts`.
template <typename T, typename... Ts>
constexpr bool IsAnyOf = (std::is_same_v<T, Ts> || ...);

// Returns `true` if `T` is a 1-byte float data type.
template <typename T>
constexpr bool IsOneByteFloat = false;

#define TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT(T, ...) \
  template <>                                          \
  constexpr bool IsOneByteFloat<::tensorstore::dtypes::T> = true;
TENSORSTORE_FOR_EACH_FLOAT8_DATA_TYPE(TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT)
TENSORSTORE_FOR_EACH_MXFLOAT_DATA_TYPE(TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT)
#undef TENSORSTORE_INTERNAL_IS_ONE_BYTE_FLOAT

template <typename T>
constexpr bool IsLowPrecisionInt = false;

#define TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT(T, ...) \
  template <>                                             \
  constexpr bool IsLowPrecisionInt<::tensorstore::dtypes::T> = true;
TENSORSTORE_FOR_EACH_LOW_PRECISION_INT_DATA_TYPE(
    TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT)
#undef TENSORSTORE_INTERNAL_IS_LOW_PRECISION_INT

// Returns `true` if `T` is a 1-byte int data type.
template <typename T>
constexpr bool IsOneByteInt =
    IsLowPrecisionInt<T> ||
    IsAnyOf<T, ::tensorstore::dtypes::int8_t, ::tensorstore::dtypes::uint8_t>;

template <typename T>
constexpr bool IsOneByteType = IsOneByteFloat<T> || IsOneByteInt<T> ||
                               std::is_same_v<T, ::tensorstore::dtypes::bool_t>;

template <typename T>
constexpr bool IsIntDataType =
    IsOneByteInt<T> ||
    (std::is_integral_v<T> &&
     !IsAnyOf<T, ::tensorstore::dtypes::bool_t, ::tensorstore::dtypes::char_t,
              ::tensorstore::dtypes::byte_t>);

template <typename T>
constexpr bool IsUnsignedIntDataType =
    IsIntDataType<T> && !std::numeric_limits<T>::is_signed;

template <typename T>
constexpr bool IsFloatDataType = IsOneByteFloat<T> ||
                                 IsAnyOf<T, ::tensorstore::dtypes::float16_t,
                                         ::tensorstore::dtypes::bfloat16_t> ||
                                 std::is_floating_point_v<T>;

template <typename T>
constexpr bool IsScalarNumericDataType =
    IsIntDataType<T> || IsFloatDataType<T> ||
    std::is_same_v<T, ::tensorstore::dtypes::bool_t>;

template <typename T>
constexpr bool IsComplexDataType =
    IsAnyOf<T, ::tensorstore::dtypes::complex64_t,
            ::tensorstore::dtypes::complex128_t>;

template <typename T>
ABSL_ATTRIBUTE_NOINLINE void NumberToString(T x, std::string* out) {
  out->clear();
  absl::StrAppend(out, x);
}

ABSL_ATTRIBUTE_NOINLINE void ComplexToString(double real, double imag,
                                             std::string* out) {
  out->clear();
  absl::StrAppend(out, "(", real, ",", imag, ")");
}

// Bespoke conversion functors for non-scalar edge pairs (Layer 3).
template <typename T>
ABSL_ATTRIBUTE_NOINLINE void AssignJson(::tensorstore::dtypes::json_t* to,
                                        T val) {
  *to = val;
}

ABSL_ATTRIBUTE_NOINLINE void AssignComplexJson(
    ::tensorstore::dtypes::json_t* to, double real, double imag) {
  ::tensorstore::dtypes::json_t::array_t arr(2);
  arr[0] = real;
  arr[1] = imag;
  *to = std::move(arr);
}

template <typename From, typename To>
struct ConvertDataType {
  static constexpr bool kUseStridedForContiguous = true;
  void operator()(const From* from, To* to, void*) const {
    *to = static_cast<To>(*from);
  }
};

ABSL_ATTRIBUTE_NOINLINE bool ValidateUtf8(
    const ::tensorstore::dtypes::string_t* from, void* status) {
  if (ABSL_PREDICT_TRUE(internal::IsValidUtf8(*from))) return true;
  *static_cast<absl::Status*>(status) =
      absl::InvalidArgumentError("Invalid UTF-8 sequence encountered");
  return false;
}

template <typename From, typename To>
struct ConvertEdgeElement {
  static bool Run(const void* from_void, void* to_void,
                  [[maybe_unused]] void* status) {
    const auto* from = static_cast<const From*>(from_void);
    auto* to = static_cast<To*>(to_void);
    if constexpr (IsComplexDataType<From> &&
                  std::is_same_v<To, dtypes::json_t>) {
      AssignComplexJson(to, from->real(), from->imag());
      return true;
    } else if constexpr (IsComplexDataType<From> &&
                         std::is_same_v<To, dtypes::string_t>) {
      ComplexToString(from->real(), from->imag(), to);
      return true;
    } else if constexpr (std::is_same_v<From, dtypes::ustring_t> &&
                         std::is_same_v<To, dtypes::json_t>) {
      AssignJson<const std::string&>(to, from->utf8);
      return true;
    } else if constexpr (std::is_same_v<From, dtypes::json_t> &&
                         std::is_same_v<To, dtypes::string_t>) {
      auto s = internal_json::JsonRequireValueAs(*from, to, /*strict=*/false);
      if (ABSL_PREDICT_TRUE(s.ok())) return true;
      *static_cast<absl::Status*>(status) = std::move(s);
      return false;
    } else if constexpr (std::is_same_v<From, dtypes::string_t> &&
                         std::is_same_v<To, dtypes::ustring_t>) {
      if (ABSL_PREDICT_FALSE(!ValidateUtf8(from, status))) return false;
      to->utf8 = *from;
      return true;
    } else {
      static_assert(std::is_same_v<From, dtypes::string_t> &&
                    std::is_same_v<To, dtypes::json_t>);
      if (ABSL_PREDICT_FALSE(!ValidateUtf8(from, status))) return false;
      AssignJson<const std::string&>(to, *from);
      return true;
    }
  }
};

// ---------------------------------------------------------------------------
// Layer 1: Direct 256-entry lookup tables (`1B -> 1B` and `1B float -> bf16`)
// ---------------------------------------------------------------------------

template <typename T>
constexpr T ByteFromUint8(uint8_t byte) {
  if constexpr (std::is_same_v<T, ::tensorstore::dtypes::bool_t>) {
    return byte != 0;
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(byte);
  } else {
    return absl::bit_cast<T>(byte);
  }
}

template <typename T>
constexpr uint8_t ByteToUint8(T val) {
  if constexpr (std::is_same_v<T, ::tensorstore::dtypes::bool_t>) {
    return static_cast<uint8_t>(val ? 1 : 0);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<uint8_t>(val);
  } else {
    return absl::bit_cast<uint8_t>(val);
  }
}

// Compile-time 256-entry lookup table for all `1B -> 1B` conversions
// (`bool`, `int2`, `int4`, `int8`, `uint8`, `Float8*`, `Float4*`).
template <typename From, typename To>
constexpr std::array<uint8_t, 256> kUint8ToUint8Table = [] {
  std::array<uint8_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    const From from_val = ByteFromUint8<From>(static_cast<uint8_t>(i));
    To to_val{};
    if constexpr (IsOneByteFloat<From> && IsIntDataType<To>) {
      to_val = internal::SaturatedFloatToInt<To>(from_val);
    } else {
      to_val = static_cast<To>(from_val);
    }
    table[i] = ByteToUint8<To>(to_val);
  }
  return table;
}();

// Deduplicates byte-for-byte identical 256-byte lookup tables in Layer 1.
template <typename From, typename To>
constexpr const std::array<uint8_t, 256>& GetUint8ToUint8Table() {
  using dtypes::bool_t;
  using dtypes::float4_e2m1fn_t;
  using dtypes::float8_e3m4_t;
  using dtypes::float8_e4m3b11fnuz_t;
  using dtypes::float8_e4m3fn_t;
  using dtypes::float8_e4m3fnuz_t;
  using dtypes::float8_e5m2_t;
  using dtypes::float8_e5m2fnuz_t;
  using dtypes::int2_t;
  using dtypes::int4_t;
  using dtypes::int8_t;
  using dtypes::uint8_t;
  if constexpr ((std::is_same_v<From, bool_t> && IsOneByteInt<To>) ||
                (std::is_same_v<To, bool_t> &&
                 IsAnyOf<From, int8_t, uint8_t, float8_e4m3fnuz_t,
                         float8_e4m3b11fnuz_t, float8_e5m2fnuz_t>)) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<uint8_t, bool_t>);
    return kUint8ToUint8Table<uint8_t, bool_t>;
  } else if constexpr (std::is_same_v<To, bool_t> &&
                       IsAnyOf<From, float8_e3m4_t, float8_e4m3fn_t,
                               float8_e5m2_t>) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<float8_e3m4_t, bool_t>);
    return kUint8ToUint8Table<float8_e3m4_t, bool_t>;
  } else if constexpr (std::is_same_v<From, bool_t> &&
                       IsAnyOf<To, float8_e4m3fnuz_t, float8_e5m2fnuz_t>) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<bool_t, float8_e4m3fnuz_t>);
    return kUint8ToUint8Table<bool_t, float8_e4m3fnuz_t>;
  } else if constexpr (IsOneByteInt<From> && IsOneByteInt<To> &&
                       (std::is_same_v<From, int2_t> ||
                        std::is_same_v<To, int2_t>)) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<int8_t, int2_t>);
    return kUint8ToUint8Table<int8_t, int2_t>;
  } else if constexpr (IsOneByteInt<From> && IsOneByteInt<To> &&
                       (std::is_same_v<From, int4_t> ||
                        std::is_same_v<To, int4_t>)) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<int8_t, int4_t>);
    return kUint8ToUint8Table<int8_t, int4_t>;
  } else if constexpr (std::is_same_v<From, float4_e2m1fn_t> &&
                       IsAnyOf<To, int4_t, int8_t>) {
    static_assert(kUint8ToUint8Table<From, To> ==
                  kUint8ToUint8Table<float4_e2m1fn_t, int8_t>);
    return kUint8ToUint8Table<float4_e2m1fn_t, int8_t>;
  } else {
    return kUint8ToUint8Table<From, To>;
  }
}

struct Uint8TableLookupConvert {
  static constexpr bool kUseStridedForContiguous = true;
  void operator()(const std::array<uint8_t, 256>& table, const uint8_t* from,
                  uint8_t* to, void*) const {
    *to = table[*from];
  }
};

using Uint8TableLookupFn = internal::SimpleElementwiseFunction<
    internal_elementwise_function::Stateless<const std::array<uint8_t, 256>,
                                             Uint8TableLookupConvert>(
        const uint8_t, uint8_t),
    void*>;

// Every 1-byte float format has <= 4 mantissa bits and min exponent >= -127,
// so the lower 16 bits of its IEEE-754 float32 representation are identically
// zero and its upper 16 bits equal its BFloat16 bit representation.
template <typename From>
constexpr std::array<uint16_t, 256> kByteToBFloat16BitsTable = [] {
  std::array<uint16_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    const uint32_t f32_bits = absl::bit_cast<uint32_t>(
        static_cast<float>(ByteFromUint8<From>(static_cast<uint8_t>(i))));
    if ((f32_bits & 0xffffu) != 0) ABSL_UNREACHABLE();
    table[i] = static_cast<uint16_t>(f32_bits >> 16);
  }
  return table;
}();

struct Uint16TableLookupConvert {
  static constexpr bool kUseStridedForContiguous = true;
  void operator()(const std::array<uint16_t, 256>& table, const uint8_t* from,
                  uint16_t* to, void*) const {
    *to = table[*from];
  }
};

using Uint16TableLookupFn = internal::SimpleElementwiseFunction<
    internal_elementwise_function::Stateless<const std::array<uint16_t, 256>,
                                             Uint16TableLookupConvert>(
        const uint8_t, uint16_t),
    void*>;

// ---------------------------------------------------------------------------
// Layer 2: Unified Symmetric Descriptor-Driven Hub (`float`, `double`, `int64`)
//
// Converts `From -> Hub -> To` in 64-element blocks using `O(M + N)` batch
// functions instead of `O(M * N)` pairwise loop instantiations. Each `(From,
// To)` pair is represented by a 4-byte `HubDescriptor` in `.rodata` that
// indexes into `kBatchSources` and `kBatchTargets`.
// ---------------------------------------------------------------------------

// Batch conversion function signature shared by both `kBatchSources` (`src` is
// strided by `byte_stride`, `dst` is contiguous `Hub[count]`) and
// `kBatchTargets` (`src` is contiguous `const Hub[count]`, `dst` is strided by
// `byte_stride`). Sharing one signature allows `HubLoopTemplate` to merge
// source-identity and target-identity bypass loops and unifies all three hubs
// (`float`, `double`, `int64_t`) into a single loop implementation.
using BatchFn = void (*)(const void* src, Index byte_stride, void* dst,
                         Index count);

// Converts a source element `x` to `Hub`. For sub-byte signed integers
// (`int2_t`, `int4_t`), sign-extension is performed in 32-bit `int32_t` rather
// than 8-bit `int8_t` so x86 SSE2/AVX can lower it to hardware `vpslld` +
// `vpsrad` (x86 lacks an 8-bit arithmetic right shift instruction).
template <typename From, typename Hub>
inline Hub DecodeSourceElement(From x) {
  if constexpr (IsLowPrecisionInt<From>) {
    constexpr int kShift = 32 - (std::numeric_limits<From>::digits + 1);
    const uint32_t raw = absl::bit_cast<uint8_t>(x);
    const int32_t sext = static_cast<int32_t>(raw << kShift) >> kShift;
    return static_cast<Hub>(sext);
  } else {
    return static_cast<Hub>(x);
  }
}

#if defined(__F16C__) || defined(__aarch64__) || defined(_M_ARM64)
constexpr bool kHasHardwareFloat16 = true;
#else
constexpr bool kHasHardwareFloat16 = false;
#endif

template <typename From, typename Hub>
constexpr bool kVectorizesContiguousSource =
    !std::is_same_v<From, ::tensorstore::dtypes::float16_t> ||
    kHasHardwareFloat16;

// Vectorizable `From -> Hub` batch decoder with a unit-stride fast path.
template <typename From, typename Hub>
struct SourceToHubBatch {
  ABSL_ATTRIBUTE_NOINLINE static void Run(
      const void* TENSORSTORE_INTERNAL_RESTRICT src, Index byte_stride,
      void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count) {
    Hub* TENSORSTORE_INTERNAL_RESTRICT dst = static_cast<Hub*>(dst_void);
    ByteStridedPointer<const From> p(src);
    if constexpr (kVectorizesContiguousSource<From, Hub>) {
      if (byte_stride == static_cast<Index>(sizeof(From))) {
        const From* TENSORSTORE_INTERNAL_RESTRICT from = p.get();
        for (Index i = 0; i < count; ++i) {
          dst[i] = DecodeSourceElement<From, Hub>(from[i]);
        }
        return;
      }
    }
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      dst[i] = DecodeSourceElement<From, Hub>(
          *(p + internal::wrap_on_overflow::Multiply(byte_stride, i)));
    }
  }
};

template <typename T>
ABSL_ATTRIBUTE_NOINLINE float RoundToOddFloatOutOfLine(T x) {
  return float8_internal::RoundToOddFloat(x);
}

// Narrowing `From -> float` batch decoder using round-to-odd (`double` and
// 32/64-bit integers -> `float`) so a subsequent round-to-nearest-even cast to
// a 1-byte float produces the exact single-rounded result.
template <typename From>
struct SourceToHubRTOBatch {
  ABSL_ATTRIBUTE_NOINLINE static void Run(
      const void* TENSORSTORE_INTERNAL_RESTRICT src, Index byte_stride,
      void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count) {
    float* TENSORSTORE_INTERNAL_RESTRICT dst = static_cast<float*>(dst_void);
    using Wide = std::conditional_t<
        std::is_floating_point_v<From>, double,
        std::conditional_t<std::is_signed_v<From>, int64_t, uint64_t>>;
    ByteStridedPointer<const From> p(src);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      const From val =
          *(p + internal::wrap_on_overflow::Multiply(byte_stride, i));
      dst[i] = RoundToOddFloatOutOfLine(static_cast<Wide>(val));
    }
  }
};

// Shared `1-byte float -> float` batch decoder parameterized by `table` in the
// 5th argument so each per-type wrapper compiles to a 2-instruction (`lea` +
// `jmp`) tail call.
ABSL_ATTRIBUTE_NOINLINE void ByteTableToFloatBatchImpl(
    const void* TENSORSTORE_INTERNAL_RESTRICT src, Index byte_stride,
    void* TENSORSTORE_INTERNAL_RESTRICT dst_void, Index count,
    const uint16_t* TENSORSTORE_INTERNAL_RESTRICT table) {
  float* TENSORSTORE_INTERNAL_RESTRICT dst = static_cast<float*>(dst_void);
  if (ABSL_PREDICT_TRUE(byte_stride == 1)) {
    const uint8_t* TENSORSTORE_INTERNAL_RESTRICT s =
        static_cast<const uint8_t*>(src);
    for (Index i = 0; i < count; ++i) {
      dst[i] = absl::bit_cast<float>(static_cast<uint32_t>(table[s[i]]) << 16);
    }
    return;
  }
  ByteStridedPointer<const uint8_t> p(src);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    const uint8_t b =
        *(p + internal::wrap_on_overflow::Multiply(byte_stride, i));
    dst[i] = absl::bit_cast<float>(static_cast<uint32_t>(table[b]) << 16);
  }
}

template <typename From>
struct ByteFloatSourceToHubBatch {
  static void Run(const void* TENSORSTORE_INTERNAL_RESTRICT src,
                  Index byte_stride, void* TENSORSTORE_INTERNAL_RESTRICT dst,
                  Index count) {
    ByteTableToFloatBatchImpl(src, byte_stride, dst, count,
                              kByteToBFloat16BitsTable<From>.data());
  }
};

// Converts a `Hub` value (`float`, `double`, or `int64_t`) to a trivial target
// type `To` (excluding `float/double -> {int2_t, int4_t, int8_t}`, which are
// handled directly by `HubToClampedInt8BatchImpl`). Floating-point -> integer
// conversions saturate (`NaN -> 0`, out-of-range -> `[lowest(), max()]`).
template <typename To, typename Hub>
inline To ConvertFromHub(Hub v) {
  if constexpr (std::is_floating_point_v<Hub>) {
    if constexpr (std::is_integral_v<To> && !std::is_same_v<To, bool>) {
      return internal::SaturatedFloatToInt<To>(v);
    } else {
      return static_cast<To>(v);
    }
  } else {
    static_assert(std::is_same_v<Hub, int64_t>);
    if constexpr (IsLowPrecisionInt<To>) {
      constexpr int kShift = 32 - (std::numeric_limits<To>::digits + 1);
      const int32_t sext =
          static_cast<int32_t>(static_cast<uint32_t>(v) << kShift) >> kShift;
      return To(typename To::bitcast_construct_t{}, static_cast<int8_t>(sext));
    } else if constexpr (std::is_same_v<To, bool>) {
      return v != 0;
    } else {
      return static_cast<To>(v);
    }
  }
}

// Shared `float/double -> {int2_t, int4_t, int8_t}` batch encoder. Because
// `int2_t` and `int4_t` store a sign-extended `int8_t` in memory, all three
// targets share identical SIMD clamping and packing code parameterized by
// `[lowest, max_val]` in `xmm0`/`xmm1`.
template <typename Hub>
ABSL_ATTRIBUTE_NOINLINE void HubToClampedInt8BatchImpl(
    const void* TENSORSTORE_INTERNAL_RESTRICT src_void, Index to_byte_stride,
    void* TENSORSTORE_INTERNAL_RESTRICT to, Index count, Hub lowest,
    Hub max_val) {
  const Hub* TENSORSTORE_INTERNAL_RESTRICT from =
      static_cast<const Hub*>(src_void);
  auto convert_one = [=](Hub f) -> int8_t {
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
    using UInt =
        std::conditional_t<std::is_same_v<Hub, double>, uint64_t, uint32_t>;
    constexpr UInt kAbsMask =
        std::is_same_v<Hub, double> ? 0x7fffffffffffffffULL : 0x7fffffffU;
    constexpr UInt kInfBits =
        std::is_same_v<Hub, double> ? 0x7ff0000000000000ULL : 0x7f800000U;
    if ((__builtin_bit_cast(UInt, f) & kAbsMask) > kInfBits) return 0;
#endif
    const Hub lo = (f > lowest) ? f : lowest;
    const Hub clamped = (lo < max_val) ? lo : max_val;
    const int32_t conv = static_cast<int32_t>(clamped);
    return static_cast<int8_t>((f == f) ? conv : 0);
  };
  ByteStridedPointer<int8_t> dst(to);
  if (to_byte_stride == 1) {
    int8_t* TENSORSTORE_INTERNAL_RESTRICT p = dst.get();
    for (Index i = 0; i < count; ++i) {
      p[i] = convert_one(from[i]);
    }
    return;
  }
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    *(dst + internal::wrap_on_overflow::Multiply(to_byte_stride, i)) =
        convert_one(from[i]);
  }
}

#if defined(__AVX512DQ__) || defined(__aarch64__) || defined(_M_ARM64)
constexpr bool kHasVectorFloatToInt64 = true;
#else
constexpr bool kHasVectorFloatToInt64 = false;
#endif

template <typename Hub, typename To>
constexpr bool kVectorizesContiguousHubTarget =
    std::is_same_v<Hub, int64_t> ||
    (std::is_arithmetic_v<To> &&
     !(std::is_same_v<To, ::tensorstore::dtypes::float16_t> &&
       !kHasHardwareFloat16) &&
     !(std::is_integral_v<To> && sizeof(To) == 8 && !kHasVectorFloatToInt64)) ||
    std::is_same_v<To, ::tensorstore::dtypes::bfloat16_t> ||
    IsLowPrecisionInt<To> || IsComplexDataType<To>;

// Vectorizable `Hub -> To` batch encoder with a unit-stride fast path for
// targets whose conversion lowers to inline arithmetic/SIMD instructions.
template <typename Hub, typename To>
struct HubToTargetBatch {
  ABSL_ATTRIBUTE_NOINLINE static void Run(
      const void* TENSORSTORE_INTERNAL_RESTRICT src_void, Index to_byte_stride,
      void* TENSORSTORE_INTERNAL_RESTRICT to, Index count) {
    if constexpr (std::is_floating_point_v<Hub> &&
                  (IsLowPrecisionInt<To> || std::is_same_v<To, int8_t>)) {
      using TLimits = std::numeric_limits<To>;
      constexpr Hub kLowest =
          static_cast<Hub>(static_cast<int8_t>(TLimits::lowest()));
      constexpr Hub kMax =
          static_cast<Hub>(static_cast<int8_t>(TLimits::max()));
      HubToClampedInt8BatchImpl<Hub>(src_void, to_byte_stride, to, count,
                                     kLowest, kMax);
      return;
    }
    const Hub* TENSORSTORE_INTERNAL_RESTRICT from =
        static_cast<const Hub*>(src_void);
    ByteStridedPointer<To> dst(to);
    if constexpr (kVectorizesContiguousHubTarget<Hub, To>) {
      if (to_byte_stride == static_cast<Index>(sizeof(To))) {
        To* TENSORSTORE_INTERNAL_RESTRICT p = dst.get();
        for (Index i = 0; i < count; ++i) {
          p[i] = ConvertFromHub<To>(from[i]);
        }
        return;
      }
    }
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      *(dst + internal::wrap_on_overflow::Multiply(to_byte_stride, i)) =
          ConvertFromHub<To>(from[i]);
    }
  }
};

template <typename Hub, typename CastType>
struct HubToStringBatch {
  ABSL_ATTRIBUTE_NOINLINE static void Run(
      const void* TENSORSTORE_INTERNAL_RESTRICT src_void, Index to_byte_stride,
      void* TENSORSTORE_INTERNAL_RESTRICT to, Index count) {
    const Hub* TENSORSTORE_INTERNAL_RESTRICT from =
        static_cast<const Hub*>(src_void);
    ByteStridedPointer<::tensorstore::dtypes::string_t> dst(to);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      NumberToString<CastType>(
          static_cast<CastType>(from[i]),
          (dst + internal::wrap_on_overflow::Multiply(to_byte_stride, i))
              .get());
    }
  }
};

template <typename Hub, typename JsonValType = double>
struct HubToJsonBatch {
  ABSL_ATTRIBUTE_NOINLINE static void Run(
      const void* TENSORSTORE_INTERNAL_RESTRICT src_void, Index to_byte_stride,
      void* TENSORSTORE_INTERNAL_RESTRICT to, Index count) {
    const Hub* TENSORSTORE_INTERNAL_RESTRICT from =
        static_cast<const Hub*>(src_void);
    ByteStridedPointer<::tensorstore::dtypes::json_t> dst(to);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
    for (Index i = 0; i < count; ++i) {
      JsonValType val;
      if constexpr (std::is_same_v<JsonValType, bool>) {
        val = (from[i] != 0);
      } else {
        val = static_cast<JsonValType>(from[i]);
      }
      AssignJson<JsonValType>(
          (dst + internal::wrap_on_overflow::Multiply(to_byte_stride, i)).get(),
          val);
    }
  }
};

// Compile-time operation table mapping each `Ops` type to its `uint8_t` slot
// in `kFunctions` via `std::is_same_v` (avoiding `constexpr` function-pointer
// comparisons and template-address overload deduction).
template <typename FnType, typename... Ops>
struct OpTable {
  static constexpr FnType kFunctions[] = {&Ops::Run...};

  template <typename TargetOp>
  static constexpr uint8_t kIndex = []() constexpr {
    static_assert(sizeof...(Ops) <= 256);
    constexpr bool kMatches[] = {std::is_same_v<TargetOp, Ops>...};
    for (size_t i = 0; i < sizeof...(Ops); ++i) {
      if (kMatches[i]) return static_cast<uint8_t>(i);
    }
    ABSL_UNREACHABLE();
    return uint8_t{0};
  }();
};

// Unified source and target batch tables across all three hubs (`float`,
// `double`, `int64_t`). Indices `0..1` are reserved for the 4-byte (`float`)
// and 8-byte (`double` / `int64_t`) identity hub-type copies, enabling
// `HubLoopTemplate` to test `op < 2` to bypass `hub_buf` when either endpoint
// is already a unit-strided hub array.

// clang-format off
using BatchSourceTable = OpTable<
    BatchFn,
    // Indices 0..1: Identity hub decoders (`ScalarFrom == Hub`).
    SourceToHubBatch<float, float>,
    SourceToHubBatch<int64_t, int64_t>,
    // `float` hub decoders:
    SourceToHubBatch<::tensorstore::dtypes::float16_t, float>,
    SourceToHubBatch<::tensorstore::dtypes::bfloat16_t, float>,
    SourceToHubBatch<double, float>,
    SourceToHubRTOBatch<double>,
    SourceToHubBatch<bool, float>,
    SourceToHubBatch<::tensorstore::dtypes::int2_t, float>,
    SourceToHubBatch<::tensorstore::dtypes::int4_t, float>,
    SourceToHubBatch<int8_t, float>,
    SourceToHubBatch<uint8_t, float>,
    SourceToHubBatch<int16_t, float>,
    SourceToHubBatch<uint16_t, float>,
    SourceToHubBatch<int32_t, float>,
    SourceToHubBatch<uint32_t, float>,
    SourceToHubBatch<int64_t, float>,
    SourceToHubBatch<uint64_t, float>,
    SourceToHubRTOBatch<int32_t>,
    SourceToHubRTOBatch<uint32_t>,
    SourceToHubRTOBatch<int64_t>,
    SourceToHubRTOBatch<uint64_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e3m4_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e4m3fn_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e4m3fnuz_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e4m3b11fnuz_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e5m2_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e5m2fnuz_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float8_e8m0fnu_t>,
    ByteFloatSourceToHubBatch<::tensorstore::dtypes::float4_e2m1fn_t>,
    // `double` hub decoders:
    SourceToHubBatch<int32_t, double>,
    SourceToHubBatch<uint32_t, double>,
    SourceToHubBatch<int64_t, double>,
    SourceToHubBatch<uint64_t, double>,
    // `int64_t` hub decoders:
    SourceToHubBatch<bool, int64_t>,
    SourceToHubBatch<::tensorstore::dtypes::int2_t, int64_t>,
    SourceToHubBatch<::tensorstore::dtypes::int4_t, int64_t>,
    SourceToHubBatch<int8_t, int64_t>,
    SourceToHubBatch<uint8_t, int64_t>,
    SourceToHubBatch<int16_t, int64_t>,
    SourceToHubBatch<uint16_t, int64_t>,
    SourceToHubBatch<int32_t, int64_t>,
    SourceToHubBatch<uint32_t, int64_t>>;

constexpr auto& kBatchSources = BatchSourceTable::kFunctions;

using BatchTargetTable = OpTable<
    BatchFn,
    // Indices 0..1: Identity hub encoders (`To == Hub`).
    HubToTargetBatch<float, float>,
    HubToTargetBatch<int64_t, int64_t>,
    // `float` hub encoders:
    HubToTargetBatch<float, ::tensorstore::dtypes::float16_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::bfloat16_t>,
    HubToTargetBatch<float, double>,
    HubToTargetBatch<float, ::tensorstore::dtypes::complex64_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::complex128_t>,
    HubToTargetBatch<float, bool>,
    HubToTargetBatch<float, ::tensorstore::dtypes::int2_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::int4_t>,
    HubToTargetBatch<float, int8_t>,
    HubToTargetBatch<float, uint8_t>,
    HubToTargetBatch<float, int16_t>,
    HubToTargetBatch<float, uint16_t>,
    HubToTargetBatch<float, int32_t>,
    HubToTargetBatch<float, uint32_t>,
    HubToTargetBatch<float, int64_t>,
    HubToTargetBatch<float, uint64_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e3m4_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e4m3fn_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e4m3fnuz_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e4m3b11fnuz_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e5m2_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e5m2fnuz_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float8_e8m0fnu_t>,
    HubToTargetBatch<float, ::tensorstore::dtypes::float4_e2m1fn_t>,
    HubToStringBatch<float, double>,
    HubToJsonBatch<float, double>,
    // `double` hub encoders:
    HubToTargetBatch<double, ::tensorstore::dtypes::complex128_t>,
    HubToTargetBatch<double, bool>,
    HubToTargetBatch<double, ::tensorstore::dtypes::int2_t>,
    HubToTargetBatch<double, ::tensorstore::dtypes::int4_t>,
    HubToTargetBatch<double, int8_t>,
    HubToTargetBatch<double, uint8_t>,
    HubToTargetBatch<double, int16_t>,
    HubToTargetBatch<double, uint16_t>,
    HubToTargetBatch<double, int32_t>,
    HubToTargetBatch<double, uint32_t>,
    HubToTargetBatch<double, int64_t>,
    HubToTargetBatch<double, uint64_t>,
    HubToStringBatch<double, double>,
    HubToJsonBatch<double, double>,
    // `int64_t` hub encoders:
    HubToTargetBatch<int64_t, int32_t>,
    HubToTargetBatch<int64_t, int16_t>,
    HubToTargetBatch<int64_t, int8_t>,
    HubToTargetBatch<int64_t, ::tensorstore::dtypes::int4_t>,
    HubToTargetBatch<int64_t, ::tensorstore::dtypes::int2_t>,
    HubToTargetBatch<int64_t, bool>,
    HubToStringBatch<int64_t, int64_t>,
    HubToStringBatch<int64_t, uint64_t>,
    HubToJsonBatch<int64_t, bool>,
    HubToJsonBatch<int64_t, int64_t>,
    HubToJsonBatch<int64_t, uint64_t>>;

constexpr auto& kBatchTargets = BatchTargetTable::kFunctions;
// clang-format on

// Relocation-free `.rodata` context passed to `HubLoopTemplate::Loop`.
// `src_op` and `dst_op` index `kBatchSources` and `kBatchTargets` (`0..1` =
// identity copy where `ScalarFrom == Hub` or `To == Hub`); `src_size` and
// `dst_size` give the element byte sizes (`dst_size == 0` for non-trivial
// `string_t`/`json_t` targets that must be written in place).
struct HubDescriptor {
  uint8_t src_op;
  uint8_t dst_op;
  uint8_t src_size;
  uint8_t dst_size;
};

template <uint8_t SrcOp, uint8_t DstOp, uint8_t SrcSize, uint8_t DstSize>
inline constexpr HubDescriptor kHubDescriptor = {SrcOp, DstOp, SrcSize,
                                                 DstSize};

// Loads `table[index]` through an opaque pointer barrier so the compiler emits
// a single indexed load and indirect call rather than devirtualizing internal
// `constexpr` tables into multi-way switch/jump-table dispatchers and bounds
// traps at each call site.
template <typename Fn, size_t N>
inline Fn LoadBatchFn(const Fn (&table)[N], uint8_t index) {
  assert(index < N);
  const Fn* ptr = table;
#if defined(__clang__) || defined(__GNUC__)
  asm("" : "+r"(ptr));
#endif
  return ptr[index];
}

// Checks whether the 2D strided byte spans `[from_lo, from_hi)` and
// `[to_lo, to_hi)` are strictly disjoint, accounting for negative or zero
// strides, offset slices, and transposed views on the same backing buffer.
ABSL_ATTRIBUTE_NOINLINE bool BuffersAreDisjoint(
    internal::IterationBufferShape shape,
    const internal::IterationBufferPointer& from_ptr, uint8_t from_elem_size,
    const internal::IterationBufferPointer& to_ptr, uint8_t to_elem_size) {
  if (to_elem_size == 0) return false;
  if (shape[0] <= 0 || shape[1] <= 0) return true;
  auto compute_span =
      [&](const internal::IterationBufferPointer& ptr,
          uint8_t elem_size) -> std::pair<uintptr_t, uintptr_t> {
    const Index outer_extent = internal::wrap_on_overflow::Multiply(
        shape[0] - 1, ptr.outer_byte_stride);
    const Index inner_extent = internal::wrap_on_overflow::Multiply(
        shape[1] - 1, ptr.inner_byte_stride);
    const Index min_off = internal::wrap_on_overflow::Add(
        std::min<Index>(0, outer_extent), std::min<Index>(0, inner_extent));
    const Index max_off = internal::wrap_on_overflow::Add(
        std::max<Index>(0, outer_extent), std::max<Index>(0, inner_extent));
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr.pointer.get());
    const uintptr_t lo = base + static_cast<uintptr_t>(min_off);
    const uintptr_t hi = base + static_cast<uintptr_t>(max_off) +
                         static_cast<uintptr_t>(elem_size);
    return {lo, hi};
  };
  const auto [from_lo, from_hi] = compute_span(from_ptr, from_elem_size);
  const auto [to_lo, to_hi] = compute_span(to_ptr, to_elem_size);
  return from_hi <= to_lo || to_hi <= from_lo;
}

// Encodes `count` contiguous `hub_buf` elements (`1 <= count <= 64`) and
// scatters them to offset-indexed destination positions via
// `ScatterIndexedTrivial` (or writes non-trivial `string_t`/`json_t` targets in
// place when `desc.dst_size == 0`). Keeping the `dst_size` branch out-of-line
// prevents the compiler from loop-unswitching `Loop<kIndexed>`.
ABSL_ATTRIBUTE_NOINLINE void FlushIndexedTarget(
    BatchFn dst_fn, const void* hub_buf, void* to_base, const Index* to_offsets,
    Index count, HubDescriptor desc) {
  if (count <= 0 || count > 64) ABSL_UNREACHABLE();
  if (ABSL_PREDICT_TRUE(desc.dst_size != 0)) {
    alignas(16) unsigned char raw_out[64 * 16];
    dst_fn(hub_buf, desc.dst_size, raw_out, count);
    internal::ScatterIndexedTrivial(raw_out, to_base, to_offsets, count,
                                    desc.dst_size);
    return;
  }
  constexpr uint8_t kFloatToStringOp =
      BatchTargetTable::kIndex<HubToStringBatch<float, double>>;
  constexpr uint8_t kFloatToJsonOp =
      BatchTargetTable::kIndex<HubToJsonBatch<float, double>>;
  const size_t hub_elem_size =
      (desc.dst_op == kFloatToStringOp || desc.dst_op == kFloatToJsonOp)
          ? sizeof(float)
          : sizeof(double);
  const auto* hub_bytes = static_cast<const unsigned char*>(hub_buf);
  char* base_ptr = static_cast<char*>(to_base);
#ifdef __clang__
#pragma clang loop vectorize(disable) unroll(disable)
#endif
  for (Index i = 0; i < count; ++i) {
    dst_fn(hub_bytes + i * hub_elem_size, 0, base_ptr + to_offsets[i], 1);
  }
}

// Shared 2D conversion loop across all three hubs (`float`, `double`, `int64`):
// - Strided/Contiguous (`Kind != kIndexed`):
//   1. If `src_op < 2` (`ScalarFrom == Hub` with unit stride) or `dst_op < 2`
//      (`To == Hub` with unit stride) and the buffers are disjoint, streams
//      each row directly through `dst_fn` or `src_fn` without an intermediate
//      buffer.
//   2. Otherwise, converts in 64-element blocks via `hub_buf` on the stack.
// - Indexed (`Kind == kIndexed`): gathers 64 indexed source elements into
//   `raw_in`, runs contiguous `src_fn`, and flushes to the indexed target via
//   `FlushIndexedTarget`.
struct TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN HubLoopTemplate {
  template <internal::IterationBufferKind Kind>
  TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN static bool Loop(
      void* context, internal::IterationBufferShape shape,
      internal::IterationBufferPointer from_ptr,
      internal::IterationBufferPointer to_ptr, [[maybe_unused]] void* status) {
    using Accessor = internal::IterationBufferAccessor<Kind>;
    const HubDescriptor desc = *static_cast<const HubDescriptor*>(context);
    const BatchFn src_fn = LoadBatchFn(kBatchSources, desc.src_op);
    const BatchFn dst_fn = LoadBatchFn(kBatchTargets, desc.dst_op);
    constexpr Index kBlockSize = 64;
    alignas(8) unsigned char hub_buf[kBlockSize * sizeof(double)];
    if constexpr (Kind != internal::IterationBufferKind::kIndexed) {
      const bool can_bypass_src =
          desc.src_op < 2 &&
          from_ptr.inner_byte_stride == static_cast<Index>(desc.src_size);
      const bool can_bypass_dst =
          desc.dst_op < 2 &&
          to_ptr.inner_byte_stride == static_cast<Index>(desc.dst_size);
      if ((can_bypass_src || can_bypass_dst) &&
          BuffersAreDisjoint(shape, from_ptr, desc.src_size, to_ptr,
                             desc.dst_size)) {
        const BatchFn direct_fn = can_bypass_src ? dst_fn : src_fn;
        const Index direct_stride = can_bypass_src ? to_ptr.inner_byte_stride
                                                   : from_ptr.inner_byte_stride;
        for (Index outer = 0; outer < shape[0]; ++outer) {
          direct_fn(
              Accessor::template GetPointerAtPosition<const void>(from_ptr,
                                                                  outer, 0),
              direct_stride,
              Accessor::template GetPointerAtPosition<char>(to_ptr, outer, 0),
              shape[1]);
        }
        return true;
      }
      for (Index outer = 0; outer < shape[0]; ++outer) {
        for (Index inner = 0; inner < shape[1]; inner += kBlockSize) {
          const Index block = std::min(kBlockSize, shape[1] - inner);
          src_fn(Accessor::template GetPointerAtPosition<const void>(
                     from_ptr, outer, inner),
                 from_ptr.inner_byte_stride, hub_buf, block);
          dst_fn(hub_buf, to_ptr.inner_byte_stride,
                 Accessor::template GetPointerAtPosition<char>(to_ptr, outer,
                                                               inner),
                 block);
        }
      }
    } else {
      alignas(8) unsigned char raw_in[kBlockSize * 8];
      for (Index outer = 0; outer < shape[0]; ++outer) {
        const Index* from_row_offsets =
            from_ptr.byte_offsets +
            internal::wrap_on_overflow::Multiply(
                from_ptr.byte_offsets_outer_stride, outer);
        const Index* to_row_offsets =
            to_ptr.byte_offsets + internal::wrap_on_overflow::Multiply(
                                      to_ptr.byte_offsets_outer_stride, outer);
        for (Index inner = 0; inner < shape[1]; inner += kBlockSize) {
          const Index block = std::min(kBlockSize, shape[1] - inner);
          internal::GatherIndexedTrivial(from_ptr.pointer.get(),
                                         from_row_offsets + inner, block,
                                         desc.src_size, raw_in);
          src_fn(raw_in, desc.src_size, hub_buf, block);
          FlushIndexedTarget(dst_fn, hub_buf, to_ptr.pointer.get(),
                             to_row_offsets + inner, block, desc);
        }
      }
    }
    return true;
  }

  template <typename ArrayAccessor>
  static constexpr internal::SpecializedElementwiseFunctionPointer<2, void*>
  GetLoopFn() {
    if constexpr (ArrayAccessor::buffer_kind ==
                  internal::IterationBufferKind::kContiguous) {
      return &Loop<internal::IterationBufferKind::kStrided>;
    } else {
      return &Loop<ArrayAccessor::buffer_kind>;
    }
  }
};

constexpr internal::ElementwiseFunction<2, void*> kHubLoopFn(HubLoopTemplate{});

template <typename Hub, typename ScalarFrom, typename To>
constexpr uint8_t GetHubSourceOp() {
  if constexpr (std::is_same_v<Hub, float>) {
    if constexpr (IsOneByteFloat<ScalarFrom>) {
      return BatchSourceTable::kIndex<ByteFloatSourceToHubBatch<ScalarFrom>>;
    } else if constexpr (std::is_same_v<ScalarFrom, dtypes::float64_t>) {
      using Op =
          std::conditional_t<IsAnyOf<To, dtypes::float16_t, dtypes::bfloat16_t,
                                     dtypes::float32_t, dtypes::complex64_t>,
                             SourceToHubBatch<double, float>,
                             SourceToHubRTOBatch<double>>;
      return BatchSourceTable::kIndex<Op>;
    } else if constexpr (IsAnyOf<ScalarFrom, int32_t, uint32_t, int64_t,
                                 uint64_t> &&
                         IsOneByteFloat<To>) {
      return BatchSourceTable::kIndex<SourceToHubRTOBatch<ScalarFrom>>;
    } else {
      return BatchSourceTable::kIndex<SourceToHubBatch<ScalarFrom, float>>;
    }
  } else if constexpr (std::is_same_v<Hub, double>) {
    if constexpr (std::is_same_v<ScalarFrom, double>) {
      return BatchSourceTable::kIndex<SourceToHubBatch<int64_t, int64_t>>;
    } else {
      return BatchSourceTable::kIndex<SourceToHubBatch<ScalarFrom, double>>;
    }
  } else {
    static_assert(std::is_same_v<Hub, int64_t>);
    using Src = std::conditional_t<std::is_same_v<ScalarFrom, uint64_t>,
                                   int64_t, ScalarFrom>;
    return BatchSourceTable::kIndex<SourceToHubBatch<Src, int64_t>>;
  }
}

template <typename Hub, typename ScalarFrom, typename To>
constexpr uint8_t GetHubTargetOp() {
  if constexpr (std::is_same_v<Hub, float> || std::is_same_v<Hub, double>) {
    if constexpr (std::is_same_v<To, dtypes::string_t>) {
      return BatchTargetTable::kIndex<HubToStringBatch<Hub, double>>;
    } else if constexpr (std::is_same_v<To, dtypes::json_t>) {
      return BatchTargetTable::kIndex<HubToJsonBatch<Hub, double>>;
    } else if constexpr (std::is_same_v<Hub, double> &&
                         std::is_same_v<To, double>) {
      return BatchTargetTable::kIndex<HubToTargetBatch<int64_t, int64_t>>;
    } else {
      return BatchTargetTable::kIndex<HubToTargetBatch<Hub, To>>;
    }
  } else {
    static_assert(std::is_same_v<Hub, int64_t>);
    if constexpr (std::is_same_v<To, dtypes::string_t>) {
      using CastType = std::conditional_t<IsUnsignedIntDataType<ScalarFrom>,
                                          uint64_t, int64_t>;
      return BatchTargetTable::kIndex<HubToStringBatch<int64_t, CastType>>;
    } else if constexpr (std::is_same_v<To, dtypes::json_t>) {
      using JsonValType = std::conditional_t<
          std::is_same_v<ScalarFrom, dtypes::bool_t>, bool,
          std::conditional_t<IsUnsignedIntDataType<ScalarFrom>, uint64_t,
                             int64_t>>;
      return BatchTargetTable::kIndex<HubToJsonBatch<int64_t, JsonValType>>;
    } else if constexpr (std::is_integral_v<To> && !std::is_same_v<To, bool>) {
      return BatchTargetTable::kIndex<
          HubToTargetBatch<int64_t, std::make_signed_t<To>>>;
    } else {
      return BatchTargetTable::kIndex<HubToTargetBatch<int64_t, To>>;
    }
  }
}

template <typename T>
struct ScalarSourceType {
  using type = T;
};
template <typename R>
struct ScalarSourceType<std::complex<R>> {
  using type = R;
};

template <typename From, typename To>
constexpr bool UsesHubLoop =
    (IsScalarNumericDataType<From> &&
     (IsScalarNumericDataType<To> || IsComplexDataType<To> ||
      IsAnyOf<To, ::tensorstore::dtypes::string_t,
              ::tensorstore::dtypes::json_t>)) ||
    (IsComplexDataType<From> && IsScalarNumericDataType<To>);

// Selects the intermediate hub type (`int64_t`, `double`, or `float`) and
// returns the closure `{&kHubLoopFn, &kHubDescriptor<...>}`:
// - `int64_t` hub: all `int/bool -> int/bool/string/json` conversions.
// - `double` hub: `float64` -> `{float64, complex128, bool, int*, string,
//   json}` and `int32/uint32/int64/uint64` -> `{float64, complex128}` (which
//   require >24 bits of mantissa or vectorize directly from `double`).
// - `float` hub: all remaining numeric conversions (with `SourceToHubRTOBatch`
//   when narrowing `double` or 32/64-bit integers to 1-byte floats).
// For `complex<R> -> Scalar`, `ScalarSourceType<From>` strips `std::complex`
// so the strided `R -> Scalar` decoder reads the real part at `sizeof(From)`
// byte stride.
template <typename From, typename To>
constexpr internal::ElementwiseClosure<2, void*> GetHubClosure() {
  using ScalarFrom = typename ScalarSourceType<From>::type;
  constexpr uint8_t kSrcSize = static_cast<uint8_t>(sizeof(ScalarFrom));
  constexpr uint8_t kDstSize = IsAnyOf<To, ::tensorstore::dtypes::string_t,
                                       ::tensorstore::dtypes::json_t>
                                   ? 0
                                   : static_cast<uint8_t>(sizeof(To));

  using Hub = std::conditional_t<
      (!IsFloatDataType<ScalarFrom> && !IsFloatDataType<To> &&
       !IsComplexDataType<To>),
      int64_t,
      std::conditional_t<
          (std::is_same_v<ScalarFrom, ::tensorstore::dtypes::float64_t> &&
           (IsIntDataType<To> || IsAnyOf<To, ::tensorstore::dtypes::bool_t,
                                         ::tensorstore::dtypes::float64_t,
                                         ::tensorstore::dtypes::complex128_t,
                                         ::tensorstore::dtypes::string_t,
                                         ::tensorstore::dtypes::json_t>)) ||
              (IsAnyOf<To, ::tensorstore::dtypes::float64_t,
                       ::tensorstore::dtypes::complex128_t> &&
               IsAnyOf<ScalarFrom, int32_t, uint32_t, int64_t, uint64_t,
                       ::tensorstore::dtypes::float64_t>),
          double, float>>;
  constexpr uint8_t kSrcOp = GetHubSourceOp<Hub, ScalarFrom, To>();
  constexpr uint8_t kDstOp = GetHubTargetOp<Hub, ScalarFrom, To>();
  return {&kHubLoopFn,
          const_cast<HubDescriptor*>(
              &kHubDescriptor<kSrcOp, kDstOp, kSrcSize, kDstSize>)};
}

// Fallible JSON -> numeric/bool hub loop (`json_t -> {bool, int*, float*}`).

enum class JsonParseKind : uint8_t {
  kFloat32,
  kFloat32RTO,
  kFloat64,
  kSignedInt,
  kUnsignedInt,
  kBool,
};

// `.rodata` descriptor for `JsonToTargetLoopTemplate`, specifying how each
// `json_t` element is parsed/validated (`kind`, `min_val`, `max_val`) and
// which `kBatchTargets[dst_op]` encoder writes the validated 64-element block
// to the target buffer.
struct JsonTargetDescriptor {
  JsonParseKind kind;
  uint8_t dst_op;
  uint8_t dst_size;
  int64_t min_val;
  uint64_t max_val;
};

ABSL_ATTRIBUTE_NOINLINE bool ParseJsonElement(
    const ::tensorstore::dtypes::json_t* from, const JsonTargetDescriptor& desc,
    void* hub_buf, Index i, void* status) {
  absl::Status s;
  switch (desc.kind) {
    case JsonParseKind::kSignedInt:
      s = internal_json::JsonRequireIntegerImpl<int64_t>::Execute(
          *from, static_cast<int64_t*>(hub_buf) + i, /*strict=*/false,
          desc.min_val, static_cast<int64_t>(desc.max_val));
      break;
    case JsonParseKind::kUnsignedInt: {
      uint64_t u = 0;
      s = internal_json::JsonRequireIntegerImpl<uint64_t>::Execute(
          *from, &u, /*strict=*/false, 0, desc.max_val);
      static_cast<int64_t*>(hub_buf)[i] = static_cast<int64_t>(u);
      break;
    }
    case JsonParseKind::kBool: {
      bool b = false;
      s = internal_json::JsonRequireValueAs(*from, &b, /*strict=*/false);
      static_cast<int64_t*>(hub_buf)[i] = b ? 1 : 0;
      break;
    }
    case JsonParseKind::kFloat64:
    case JsonParseKind::kFloat32RTO:
    case JsonParseKind::kFloat32: {
      double val = 0;
      s = internal_json::JsonRequireValueAs(*from, &val, /*strict=*/false);
      if (desc.kind == JsonParseKind::kFloat64) {
        static_cast<double*>(hub_buf)[i] = val;
      } else if (desc.kind == JsonParseKind::kFloat32RTO) {
        static_cast<float*>(hub_buf)[i] = RoundToOddFloatOutOfLine(val);
      } else {
        static_cast<float*>(hub_buf)[i] = static_cast<float>(val);
      }
      break;
    }
  }
  if (ABSL_PREDICT_TRUE(s.ok())) return true;
  *static_cast<absl::Status*>(status) = std::move(s);
  return false;
}

// Parses and validates `json_t` elements in 64-element blocks before flushing
// each validated block via `kBatchTargets[desc.dst_op]`. If an element fails
// validation, earlier blocks (not earlier elements within the failing block)
// have been written to `to_ptr`; callers discard the destination buffer on
// conversion failure.
struct TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN JsonToTargetLoopTemplate {
  template <internal::IterationBufferKind Kind>
  TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN static bool Loop(
      void* context, internal::IterationBufferShape shape,
      internal::IterationBufferPointer from_ptr,
      internal::IterationBufferPointer to_ptr, void* status) {
    using Accessor = internal::IterationBufferAccessor<Kind>;
    const auto& desc = *static_cast<const JsonTargetDescriptor*>(context);
    const BatchFn dst_fn = LoadBatchFn(kBatchTargets, desc.dst_op);
    constexpr Index kBlockSize = 64;
    alignas(8) unsigned char hub_buf[kBlockSize * sizeof(double)];

    for (Index outer = 0; outer < shape[0]; ++outer) {
      for (Index inner = 0; inner < shape[1]; inner += kBlockSize) {
        const Index block = std::min(kBlockSize, shape[1] - inner);
        for (Index i = 0; i < block; ++i) {
          const auto* src = Accessor::template GetPointerAtPosition<
              const ::tensorstore::dtypes::json_t>(from_ptr, outer, inner + i);
          if (ABSL_PREDICT_FALSE(
                  !ParseJsonElement(src, desc, hub_buf, i, status))) {
            return false;
          }
        }
        if constexpr (Kind != internal::IterationBufferKind::kIndexed) {
          dst_fn(hub_buf, to_ptr.inner_byte_stride,
                 Accessor::template GetPointerAtPosition<char>(to_ptr, outer,
                                                               inner),
                 block);
        } else {
          const Index* to_row_offsets =
              to_ptr.byte_offsets +
              internal::wrap_on_overflow::Multiply(
                  to_ptr.byte_offsets_outer_stride, outer);
          FlushIndexedTarget(dst_fn, hub_buf, to_ptr.pointer.get(),
                             to_row_offsets + inner, block,
                             {0, desc.dst_op, 0, desc.dst_size});
        }
      }
    }
    return true;
  }

  template <typename ArrayAccessor>
  static constexpr internal::SpecializedElementwiseFunctionPointer<2, void*>
  GetLoopFn() {
    if constexpr (ArrayAccessor::buffer_kind ==
                  internal::IterationBufferKind::kContiguous) {
      return &Loop<internal::IterationBufferKind::kStrided>;
    } else {
      return &Loop<ArrayAccessor::buffer_kind>;
    }
  }
};

constexpr internal::ElementwiseFunction<2, void*> kJsonToTargetLoopFn(
    JsonToTargetLoopTemplate{});

template <typename To>
constexpr JsonTargetDescriptor MakeJsonTargetDescriptor() {
  if constexpr (std::is_same_v<To, ::tensorstore::dtypes::bool_t>) {
    return {JsonParseKind::kBool, GetHubTargetOp<int64_t, bool, bool>(),
            static_cast<uint8_t>(sizeof(bool)), 0, 0};
  } else if constexpr (IsIntDataType<To>) {
    if constexpr (std::numeric_limits<To>::is_signed) {
      return {JsonParseKind::kSignedInt, GetHubTargetOp<int64_t, To, To>(),
              static_cast<uint8_t>(sizeof(To)),
              static_cast<int64_t>(std::numeric_limits<To>::min()),
              static_cast<uint64_t>(
                  static_cast<int64_t>(std::numeric_limits<To>::max()))};
    } else {
      return {JsonParseKind::kUnsignedInt, GetHubTargetOp<int64_t, To, To>(),
              static_cast<uint8_t>(sizeof(To)), 0,
              static_cast<uint64_t>(std::numeric_limits<To>::max())};
    }
  } else if constexpr (std::is_same_v<To, ::tensorstore::dtypes::float64_t>) {
    return {JsonParseKind::kFloat64, GetHubTargetOp<double, double, double>(),
            static_cast<uint8_t>(sizeof(double)), 0, 0};
  } else if constexpr (IsOneByteFloat<To>) {
    return {JsonParseKind::kFloat32RTO, GetHubTargetOp<float, float, To>(),
            static_cast<uint8_t>(sizeof(To)), 0, 0};
  } else {
    return {JsonParseKind::kFloat32, GetHubTargetOp<float, float, To>(),
            static_cast<uint8_t>(sizeof(To)), 0, 0};
  }
}

template <typename To>
inline constexpr JsonTargetDescriptor kJsonTargetDescriptor =
    MakeJsonTargetDescriptor<To>();

// Shared 2D per-element loop for the 8 non-vectorizable non-scalar edge pairs
// (`complex -> string/json`, `string -> ustring/json`, `ustring -> json`, and
// `json -> string`). Because `IterationBufferAccessor<{kStrided, kIndexed}>`
// computes element addresses from runtime byte strides or byte offsets without
// depending on `sizeof(From)` or `sizeof(To)`, all 8 pairs share a single pair
// of `Loop` functions parameterized by a 1-byte index into `kEdgeConvertFns`.
using EdgeConvertFn = bool (*)(const void* from, void* to, void* status);

// clang-format off
using EdgeConvertTable = OpTable<
    EdgeConvertFn,
    ConvertEdgeElement<::tensorstore::dtypes::complex64_t,
                       ::tensorstore::dtypes::json_t>,
    ConvertEdgeElement<::tensorstore::dtypes::complex128_t,
                       ::tensorstore::dtypes::json_t>,
    ConvertEdgeElement<::tensorstore::dtypes::complex64_t,
                       ::tensorstore::dtypes::string_t>,
    ConvertEdgeElement<::tensorstore::dtypes::complex128_t,
                       ::tensorstore::dtypes::string_t>,
    ConvertEdgeElement<::tensorstore::dtypes::ustring_t,
                       ::tensorstore::dtypes::json_t>,
    ConvertEdgeElement<::tensorstore::dtypes::json_t,
                       ::tensorstore::dtypes::string_t>,
    ConvertEdgeElement<::tensorstore::dtypes::string_t,
                       ::tensorstore::dtypes::ustring_t>,
    ConvertEdgeElement<::tensorstore::dtypes::string_t,
                       ::tensorstore::dtypes::json_t>>;

constexpr auto& kEdgeConvertFns = EdgeConvertTable::kFunctions;
// clang-format on

template <uint8_t Op>
inline constexpr uint8_t kEdgeOp = Op;

struct TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN EdgeLoopTemplate {
  template <internal::IterationBufferKind Kind>
  TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN static bool Loop(
      void* context, internal::IterationBufferShape shape,
      internal::IterationBufferPointer from_ptr,
      internal::IterationBufferPointer to_ptr, void* status) {
    using Accessor = internal::IterationBufferAccessor<Kind>;
    const EdgeConvertFn fn =
        LoadBatchFn(kEdgeConvertFns, *static_cast<const uint8_t*>(context));
    for (Index outer = 0; outer < shape[0]; ++outer) {
      for (Index inner = 0; inner < shape[1]; ++inner) {
        if (ABSL_PREDICT_FALSE(
                !fn(Accessor::template GetPointerAtPosition<const char>(
                        from_ptr, outer, inner),
                    Accessor::template GetPointerAtPosition<char>(to_ptr, outer,
                                                                  inner),
                    status))) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename ArrayAccessor>
  static constexpr internal::SpecializedElementwiseFunctionPointer<2, void*>
  GetLoopFn() {
    if constexpr (ArrayAccessor::buffer_kind ==
                  internal::IterationBufferKind::kContiguous) {
      return &Loop<internal::IterationBufferKind::kStrided>;
    } else {
      return &Loop<ArrayAccessor::buffer_kind>;
    }
  }
};

constexpr internal::ElementwiseFunction<2, void*> kEdgeLoopFn(
    EdgeLoopTemplate{});

// ---------------------------------------------------------------------------
// Top-level conversion closure table
// ---------------------------------------------------------------------------

// Returns the `ElementwiseClosure` for converting `From -> To` across the
// 30x30 canonical type matrix, routing each pair to one of four layers:
// - Layer 0 (Identity / Reinterpret / Alias):
//   - Unsupported, identity (`From == To`), or `kCanReinterpretCast` pairs
//     store a null closure (`GetDataTypeConverter` substitutes `copy_assign`).
//   - `byte_t -> char_t` aliases `dtype_v<byte_t>->copy_assign` directly
//     (supported explicit conversion without `kCanReinterpretCast`).
//   - `From -> ustring_t` (`From != string_t`) aliases `From -> string_t`
//     because `ustring_t` is layout-compatible with `string_t` and all
//     non-`string_t` sources produce guaranteed valid UTF-8 ASCII.
// - Layer 1 (Direct 256-entry LUTs):
//   - `1B -> 1B`: `Uint8TableLookupFn` with deduplicated `256 x uint8_t`
//     tables (`GetUint8ToUint8Table`).
//   - `1B float -> bfloat16_t`: `Uint16TableLookupFn` over
//     `kByteToBFloat16BitsTable<From>`.
// - Layer 2 (Descriptor-Driven Hub Loops):
//   - `UsesHubLoop<From, To>`: `GetHubClosure<From, To>()` via
//     `kHubLoopFn` and `kHubDescriptor<...>`.
//   - `json_t -> {bool, int*, float*}`: `kJsonToTargetLoopFn` with
//     `kJsonTargetDescriptor<To>`.
// - Layer 3 (Non-Scalar Edge Pairs):
//   - `complex64_t <-> complex128_t`: vectorized in-place conversion via
//     `SimpleElementwiseFunction<ConvertDataType<From, To>>`.
//   - Remaining 8 non-vectorizable pairs (`complex -> string/json`, `string ->
//     ustring/json`, `ustring -> json`, `json -> string`) share `kEdgeLoopFn`
//     indexed by `kEdgeOp`.
template <typename From, typename To>
constexpr internal::ElementwiseClosure<2, void*> GetConvertClosure() {
  if constexpr ((DataTypeConversionTraits<From, To>::flags &
                 (DataTypeConversionFlags::kSupported |
                  DataTypeConversionFlags::kCanReinterpretCast)) !=
                    DataTypeConversionFlags::kSupported ||
                std::is_same_v<From, To>) {
    // Layer 0a: Unsupported, identity, or bit-reinterpret conversions.
    return {};
  } else if constexpr (std::is_same_v<From, ::tensorstore::dtypes::byte_t> &&
                       std::is_same_v<To, ::tensorstore::dtypes::char_t>) {
    // Layer 0b: Raw byte copy without setting `kCanReinterpretCast`.
    return {&dtype_v<::tensorstore::dtypes::byte_t>->copy_assign, nullptr};
  } else if constexpr (std::is_same_v<To, ::tensorstore::dtypes::ustring_t> &&
                       !std::is_same_v<From, ::tensorstore::dtypes::string_t>) {
    // Layer 0c: Redirect `From -> ustring_t` to `From -> string_t`.
    static_assert(
        IsDataTypeConversionSupported<From, ::tensorstore::dtypes::string_t>);
    return GetConvertClosure<From, ::tensorstore::dtypes::string_t>();
  } else if constexpr (IsOneByteType<From> && IsOneByteType<To>) {
    // Layer 1a: Direct 256-byte lookup table (`1B -> 1B`).
    return {&Uint8TableLookupFn::function,
            const_cast<std::array<uint8_t, 256>*>(
                &GetUint8ToUint8Table<From, To>())};
  } else if constexpr (IsOneByteFloat<From> &&
                       std::is_same_v<To, ::tensorstore::dtypes::bfloat16_t>) {
    // Layer 1b: Direct 256 x uint16_t lookup table (`1B float -> bfloat16_t`).
    return {&Uint16TableLookupFn::function,
            const_cast<std::array<uint16_t, 256>*>(
                &kByteToBFloat16BitsTable<From>)};
  } else if constexpr (UsesHubLoop<From, To>) {
    // Layer 2a: Unified symmetric hub (`float`, `double`, or `int64_t`).
    return GetHubClosure<From, To>();
  } else if constexpr (std::is_same_v<From, ::tensorstore::dtypes::json_t> &&
                       IsScalarNumericDataType<To>) {
    // Layer 2b: Fallible `json_t -> numeric/bool` block loop.
    return {&kJsonToTargetLoopFn,
            const_cast<JsonTargetDescriptor*>(&kJsonTargetDescriptor<To>)};
  } else if constexpr (IsComplexDataType<From> && IsComplexDataType<To>) {
    // Layer 3a: Vectorized `complex64_t <-> complex128_t`.
    return {&internal::SimpleElementwiseFunction<
                ConvertDataType<From, To>(const From, To), void*>::function,
            nullptr};
  } else {
    // Layer 3b: Shared 2D loop for non-vectorizable string/json/complex edges.
    constexpr uint8_t kOp =
        EdgeConvertTable::kIndex<ConvertEdgeElement<From, To>>;
    return {&kEdgeLoopFn, const_cast<uint8_t*>(&kEdgeOp<kOp>)};
  }
}

struct CanonicalConversionOperations {
  // Closure for converting to/from canonical data type.
  //
  // The `void*` pointer is a pointer to an `absl::Status` that may be set to
  // an error status in the case of failure.
  std::array<internal::ElementwiseClosure<2, void*>, kNumDataTypeIds> convert;
  std::array<DataTypeConversionFlags, kNumDataTypeIds> flags;
};

template <typename From>
constexpr CanonicalConversionOperations GetConvertToCanonicalOperations() {
  return {
      /*.convert=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return GetConvertClosure<From, X>();
      }),
      /*.flags=*/MapCanonicalDataTypes([](auto dtype) {
        using X = typename decltype(dtype)::Element;
        return DataTypeConversionTraits<From, X>::flags;
      }),
  };
}

/// Outer array is indexed by source type.  Inner array is indexed by target
/// type.
constexpr std::array<CanonicalConversionOperations, kNumDataTypeIds>
    canonical_data_type_conversions = MapCanonicalDataTypes([](auto d) {
      using X = typename decltype(d)::Element;
      return GetConvertToCanonicalOperations<X>();
    });

}  // namespace

namespace internal {

DataTypeConversionLookupResult GetDataTypeConverter(DataType from,
                                                    DataType to) {
  assert(from.valid());
  assert(to.valid());
  DataTypeConversionLookupResult lookup_result = {};
  if (from == to) {
    lookup_result.closure.function = &from->copy_assign;
    lookup_result.flags = DataTypeConversionFlags::kSupported |
                          DataTypeConversionFlags::kCanReinterpretCast |
                          DataTypeConversionFlags::kIdentity |
                          DataTypeConversionFlags::kSafeAndImplicit;
    return lookup_result;
  }
  const DataTypeId from_id = from->id;
  const DataTypeId to_id = to->id;
  if (from_id == DataTypeId::custom || to_id == DataTypeId::custom) {
    return lookup_result;
  }
  const auto& ops =
      canonical_data_type_conversions[static_cast<size_t>(from_id)];
  lookup_result.flags = ops.flags[static_cast<size_t>(to_id)];
  if ((lookup_result.flags & DataTypeConversionFlags::kCanReinterpretCast) ==
      DataTypeConversionFlags::kCanReinterpretCast) {
    lookup_result.closure.function = &from->copy_assign;
  } else {
    lookup_result.closure = ops.convert[static_cast<size_t>(to_id)];
  }
  return lookup_result;
}

Result<DataTypeConversionLookupResult> GetDataTypeConverterOrError(
    DataType from, DataType to, DataTypeConversionFlags required_flags) {
  auto lookup_result = GetDataTypeConverter(from, to);
  required_flags = (required_flags | DataTypeConversionFlags::kSupported);
  if ((lookup_result.flags & required_flags) != required_flags) {
    if (!!(lookup_result.flags & DataTypeConversionFlags::kSupported)) {
      if (!!(required_flags & DataTypeConversionFlags::kSafeAndImplicit) &&
          !(lookup_result.flags & DataTypeConversionFlags::kSafeAndImplicit)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Explicit data type conversion required to convert %v -> %v", from,
            to));
      }
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot convert %v -> %v", from, to));
  }
  return lookup_result;
}

}  // namespace internal
}  // namespace tensorstore
