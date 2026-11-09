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

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/testing/half_gtest.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DataTypeConversionFlags;
using ::tensorstore::DataTypeConversionTraits;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::IsDataTypeConversionSupported;
using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::GetDataTypeConverter;
using ::tensorstore::internal::GetDataTypeConverterOrError;
using ::tensorstore::internal::GetElementCopyErrorStatus;
using ::tensorstore::internal::IterationBufferKind;
using ::tensorstore::internal::IterationBufferPointer;

#define X(T, ...)                 \
  using ::tensorstore::dtypes::T; \
  /**/
TENSORSTORE_FOR_EACH_DATA_TYPE(X)
#undef X

constexpr DataTypeConversionFlags kSupported =
    DataTypeConversionFlags::kSupported;
constexpr DataTypeConversionFlags kIdentity =
    DataTypeConversionFlags::kIdentity;
constexpr DataTypeConversionFlags kSafeAndImplicit =
    DataTypeConversionFlags::kSafeAndImplicit;
constexpr DataTypeConversionFlags kCanReinterpretCast =
    DataTypeConversionFlags::kCanReinterpretCast;

template <typename From, typename To>
void TestUnsupported() {
  static_assert(DataTypeConversionTraits<From, To>::flags ==
                DataTypeConversionFlags{});
  static_assert(!IsDataTypeConversionSupported<From, To>);
  auto r = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
  EXPECT_EQ(DataTypeConversionFlags{}, r.flags);
}

template <typename To, typename From>
Result<To> TestConversion(
    From from, DataTypeConversionFlags flags = DataTypeConversionFlags{}) {
  SCOPED_TRACE(absl::StrFormat("TestConversion<To=%v, From=%v>", dtype_v<To>,
                               dtype_v<From>)
                   .c_str());
  flags = flags | kSupported;
  if constexpr (!std::is_same_v<To, From>) {
    EXPECT_EQ(flags, (DataTypeConversionTraits<From, To>::flags));
  }
  EXPECT_EQ(!!(flags & kSafeAndImplicit),
            (IsDataTypeConversionSupported<From, To, kSafeAndImplicit>));
  EXPECT_TRUE((IsDataTypeConversionSupported<From, To>));
  auto r = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
  EXPECT_EQ(flags, r.flags);
  To value;
  absl::Status status;
  if ((*r.closure.function)[IterationBufferKind::kContiguous](
          r.closure.context, {1, 1},
          IterationBufferPointer(&from, Index(0), Index(0)),
          IterationBufferPointer(&value, Index(0), Index(0)), &status) != 1) {
    return GetElementCopyErrorStatus(std::move(status));
  }
  return value;
}

TEST(DataTypeConversionTest, Bool) {
  EXPECT_EQ(false, TestConversion<bool_t>(false, kSafeAndImplicit | kIdentity |
                                                     kCanReinterpretCast));
  EXPECT_EQ(true, TestConversion<bool_t>(true, kSafeAndImplicit | kIdentity |
                                                   kCanReinterpretCast));
  EXPECT_EQ(0, TestConversion<int4_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int4_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int8_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int8_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int16_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int16_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int32_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int32_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int64_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int64_t>(true, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(0u, TestConversion<uint8_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1u, TestConversion<uint8_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0u, TestConversion<uint16_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1u, TestConversion<uint16_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0u, TestConversion<uint32_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1u, TestConversion<uint32_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0u, TestConversion<uint64_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1u, TestConversion<uint64_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<float16_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<float16_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<bfloat16_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<bfloat16_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<float32_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<float32_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<float64_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<float64_t>(true, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(0),
            TestConversion<complex64_t>(false, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(1),
            TestConversion<complex64_t>(true, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(0),
            TestConversion<complex128_t>(false, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(1),
            TestConversion<complex128_t>(true, kSafeAndImplicit));
  EXPECT_EQ(json_t(false), TestConversion<json_t>(false, kSafeAndImplicit));
  EXPECT_EQ(json_t(true), TestConversion<json_t>(true, kSafeAndImplicit));
  TestUnsupported<bool, string_t>();
  TestUnsupported<bool, ustring_t>();
}

TEST(DataTypeConversionTest, Int4) {
  using T = int4_t;
  constexpr T pos{7};
  constexpr T neg{-8};
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(static_cast<int8_t>(neg),
            TestConversion<int8_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(static_cast<int8_t>(pos),
            TestConversion<int8_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(static_cast<int16_t>(neg),
            TestConversion<int16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(static_cast<int32_t>(neg),
            TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(static_cast<int64_t>(neg),
            TestConversion<int64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(static_cast<uint8_t>(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(static_cast<uint8_t>(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(static_cast<uint16_t>(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(static_cast<uint16_t>(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(static_cast<uint32_t>(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(static_cast<uint32_t>(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(static_cast<uint64_t>(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(static_cast<uint64_t>(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(bfloat16_t(neg), TestConversion<bfloat16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)),
            TestConversion<complex64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("-8", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"-8"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int8) {
  using T = int8_t;
  constexpr T pos = 42;
  constexpr T neg = -42;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg),
            TestConversion<int8_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int8_t(pos),
            TestConversion<int8_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(bfloat16_t(neg), TestConversion<bfloat16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)),
            TestConversion<complex64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("-42", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"-42"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint8) {
  using T = uint8_t;
  constexpr T pos = 42;
  constexpr T neg = -42;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg),
            TestConversion<uint8_t>(
                neg, kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_EQ(uint8_t(pos),
            TestConversion<uint8_t>(
                pos, kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(bfloat16_t(neg), TestConversion<bfloat16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)),
            TestConversion<complex64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("214", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"214"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int16) {
  using T = int16_t;
  constexpr T pos = 12345;
  constexpr T neg = -12345;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg),
            TestConversion<int16_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg));
  EXPECT_EQ(bfloat16_t(neg), TestConversion<bfloat16_t>(neg));
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)),
            TestConversion<complex64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("-12345", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"-12345"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint16) {
  using T = uint16_t;
  constexpr T pos = 12345;
  constexpr T neg = -12345;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg),
            TestConversion<uint16_t>(
                neg, kCanReinterpretCast | kIdentity | kSafeAndImplicit));
  EXPECT_EQ(uint16_t(pos),
            TestConversion<uint16_t>(
                pos, kCanReinterpretCast | kIdentity | kSafeAndImplicit));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg));
  EXPECT_EQ(bfloat16_t(neg), TestConversion<bfloat16_t>(neg));
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)),
            TestConversion<complex64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("53191", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"53191"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int32) {
  using T = int32_t;
  constexpr T pos = 123456789;
  constexpr T neg = -123456789;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg),
            TestConversion<int32_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(static_cast<float>(neg)),
            TestConversion<float16_t>(neg));  // inexact
  EXPECT_EQ(bfloat16_t(static_cast<float>(neg)),
            TestConversion<bfloat16_t>(neg));  // inexact
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)), TestConversion<complex64_t>(neg));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("-123456789", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"-123456789"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint32) {
  using T = uint32_t;
  constexpr T pos = 123456789;
  constexpr T neg = -123456789;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg),
            TestConversion<uint32_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(uint32_t(pos),
            TestConversion<uint32_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float16_t(static_cast<float>(neg)),
            TestConversion<float16_t>(neg));  // inexact
  EXPECT_EQ(bfloat16_t(static_cast<float>(neg)),
            TestConversion<bfloat16_t>(neg));  // inexact
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(neg)), TestConversion<complex64_t>(neg));
  EXPECT_EQ(complex128_t(float64_t(neg)),
            TestConversion<complex128_t>(neg, kSafeAndImplicit));
  EXPECT_EQ("4171510507", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"4171510507"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int64) {
  using T = int64_t;
  constexpr T pos = 123456789012345;
  constexpr T neg = -123456789012345;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg));
  EXPECT_EQ(int64_t(neg),
            TestConversion<int64_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(float16_t(static_cast<float>(neg)),
            TestConversion<float16_t>(neg));  // inexact
  EXPECT_EQ(bfloat16_t(static_cast<float>(neg)),
            TestConversion<bfloat16_t>(neg));  // inexact
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg));
  EXPECT_EQ(complex64_t(float32_t(neg)), TestConversion<complex64_t>(neg));
  EXPECT_EQ(complex128_t(float64_t(neg)), TestConversion<complex128_t>(neg));
  EXPECT_EQ("-123456789012345", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"-123456789012345"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint64) {
  using T = uint64_t;
  constexpr T pos = 123456789012345;
  constexpr T neg = -123456789012345;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int4_t(neg), TestConversion<int4_t>(neg));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kCanReinterpretCast));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg),
            TestConversion<uint64_t>(
                neg, kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_EQ(uint64_t(pos),
            TestConversion<uint64_t>(
                pos, kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_EQ(float16_t(static_cast<float>(neg)),
            TestConversion<float16_t>(neg));  // inexact
  EXPECT_EQ(bfloat16_t(static_cast<float>(neg)),
            TestConversion<bfloat16_t>(neg));  // inexact
  EXPECT_EQ(float32_t(neg), TestConversion<float32_t>(neg));
  EXPECT_EQ(float64_t(neg), TestConversion<float64_t>(neg));
  EXPECT_EQ(complex64_t(float32_t(neg)), TestConversion<complex64_t>(neg));
  EXPECT_EQ(complex128_t(float64_t(neg)), TestConversion<complex128_t>(neg));
  EXPECT_EQ("18446620616920539271", TestConversion<string_t>(neg));
  EXPECT_EQ(ustring_t{"18446620616920539271"}, TestConversion<ustring_t>(neg));
  EXPECT_EQ(json_t(neg), TestConversion<json_t>(neg, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float16) {
  using T = float16_t;
  const T pos(42.5);
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(static_cast<int4_t>(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(static_cast<int8_t>(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(static_cast<int16_t>(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(static_cast<int32_t>(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(static_cast<int64_t>(pos), TestConversion<int64_t>(pos));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(static_cast<uint8_t>(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(static_cast<uint16_t>(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(static_cast<uint32_t>(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(static_cast<uint64_t>(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(static_cast<float16_t>(pos),
            TestConversion<float16_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(static_cast<bfloat16_t>(pos), TestConversion<bfloat16_t>(pos));
  EXPECT_EQ(static_cast<float32_t>(pos),
            TestConversion<float32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(static_cast<float64_t>(pos),
            TestConversion<float64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(pos)),
            TestConversion<complex64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  EXPECT_EQ("42.5", TestConversion<string_t>(pos));
  EXPECT_EQ(ustring_t{"42.5"}, TestConversion<ustring_t>(pos));
  EXPECT_EQ(json_t(42.5), TestConversion<json_t>(pos, kSafeAndImplicit));
}

template <typename InternalFloat>
class InternalFloat8Test : public ::testing::Test {};

using InternalFloat8Types =
    ::testing::Types<float8_e3m4_t, float8_e4m3fn_t, float8_e4m3fnuz_t,
                     float8_e4m3b11fnuz_t, float8_e5m2_t, float8_e5m2fnuz_t,
                     float8_e8m0fnu_t>;

TYPED_TEST_SUITE(InternalFloat8Test, InternalFloat8Types);

TYPED_TEST(InternalFloat8Test, DataTypeConversionTest_InternalFloat8Types) {
  using T = TypeParam;
  const T pos(3.5);
  if constexpr (!std::is_same_v<T, float8_e8m0fnu_t>) {
    EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  }
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(T(pos), TestConversion<T>(pos, kSafeAndImplicit | kIdentity |
                                               kCanReinterpretCast));
  if (!std::is_same_v<T, float8_e3m4_t>) {
    EXPECT_EQ(float8_e3m4_t(pos), TestConversion<float8_e3m4_t>(pos));
  }
  if (!std::is_same_v<T, float8_e4m3fn_t>) {
    EXPECT_EQ(float8_e4m3fn_t(pos), TestConversion<float8_e4m3fn_t>(pos));
  }
  if (!std::is_same_v<T, float8_e4m3fnuz_t>) {
    EXPECT_EQ(float8_e4m3fnuz_t(pos), TestConversion<float8_e4m3fnuz_t>(pos));
  }
  if (!std::is_same_v<T, float8_e4m3b11fnuz_t>) {
    EXPECT_EQ(float8_e4m3b11fnuz_t(pos),
              TestConversion<float8_e4m3b11fnuz_t>(pos));
  }
  if (!std::is_same_v<T, float8_e5m2fnuz_t>) {
    if (std::is_same_v<T, float8_e5m2_t>) {
      EXPECT_EQ(float8_e5m2fnuz_t(pos),
                TestConversion<float8_e5m2fnuz_t>(pos, kSafeAndImplicit));
    } else {
      EXPECT_EQ(float8_e5m2fnuz_t(pos), TestConversion<float8_e5m2fnuz_t>(pos));
    }
  }
  if (!std::is_same_v<T, float8_e5m2_t>) {
    EXPECT_EQ(float8_e5m2_t(pos), TestConversion<float8_e5m2_t>(pos));
  }
  if (!std::is_same_v<T, float8_e8m0fnu_t>) {
    EXPECT_EQ(float8_e8m0fnu_t(pos), TestConversion<float8_e8m0fnu_t>(pos));
  }
  if (std::is_same_v<T, float8_e5m2fnuz_t> ||
      std::is_same_v<T, float8_e8m0fnu_t>) {
    EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
  } else {
    EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos, kSafeAndImplicit));
  }
  EXPECT_EQ(bfloat16_t(pos), TestConversion<bfloat16_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float32_t(pos), TestConversion<float32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float64_t(pos), TestConversion<float64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(pos)),
            TestConversion<complex64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  if constexpr (std::numeric_limits<T>::digits == 1) {
    EXPECT_EQ("4", TestConversion<string_t>(pos));
    EXPECT_EQ(ustring_t{"4"}, TestConversion<ustring_t>(pos));
    EXPECT_EQ(json_t(4.0), TestConversion<json_t>(pos, kSafeAndImplicit));
  } else {
    EXPECT_EQ("3.5", TestConversion<string_t>(pos));
    EXPECT_EQ(ustring_t{"3.5"}, TestConversion<ustring_t>(pos));
    EXPECT_EQ(json_t(3.5), TestConversion<json_t>(pos, kSafeAndImplicit));
  }
}

template <typename InternalFloat>
class InternalMXFloatTest : public ::testing::Test {};

using InternalMXFloatTypes = ::testing::Types<float4_e2m1fn_t>;

TYPED_TEST_SUITE(InternalMXFloatTest, InternalMXFloatTypes);

TYPED_TEST(InternalMXFloatTest, DataTypeConversionTest_InternalFloat8Types) {
  using T = TypeParam;
  const T pos(1.5);
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(T(pos), TestConversion<T>(pos, kSafeAndImplicit | kIdentity |
                                               kCanReinterpretCast));
  if (!std::is_same_v<T, float4_e2m1fn_t>) {
    EXPECT_EQ(float4_e2m1fn_t(pos), TestConversion<float4_e2m1fn_t>(pos));
  }
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(bfloat16_t(pos), TestConversion<bfloat16_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float32_t(pos), TestConversion<float32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float64_t(pos), TestConversion<float64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(pos)),
            TestConversion<complex64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  EXPECT_EQ("1.5", TestConversion<string_t>(pos));
  EXPECT_EQ(ustring_t{"1.5"}, TestConversion<ustring_t>(pos));
  EXPECT_EQ(json_t(1.5), TestConversion<json_t>(pos, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Bfloat16) {
  using T = bfloat16_t;
  const T pos(42.5);
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
  EXPECT_EQ(bfloat16_t(pos),
            TestConversion<bfloat16_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(float32_t(pos), TestConversion<float32_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(float64_t(pos), TestConversion<float64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(pos)),
            TestConversion<complex64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  EXPECT_EQ("42.5", TestConversion<string_t>(pos));
  EXPECT_EQ(ustring_t{"42.5"}, TestConversion<ustring_t>(pos));
  EXPECT_EQ(json_t(42.5), TestConversion<json_t>(pos, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float32) {
  using T = float32_t;
  constexpr T pos = 42.5;
  // Don't test overflow of float -> int conversion, because that is undefined
  // behavior according to the C++ standard, and the behavior differs in
  // practice between x86 and PPC64le.
  //
  // See: https://github.com/numpy/numpy/issues/9040
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
  EXPECT_EQ(bfloat16_t(pos), TestConversion<bfloat16_t>(pos));
  EXPECT_EQ(float32_t(pos),
            TestConversion<float32_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(float64_t(pos), TestConversion<float64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex64_t(float32_t(pos)),
            TestConversion<complex64_t>(pos, kSafeAndImplicit));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  EXPECT_EQ("42.5", TestConversion<string_t>(pos));
  EXPECT_EQ(ustring_t{"42.5"}, TestConversion<ustring_t>(pos));
  EXPECT_EQ(json_t(pos), TestConversion<json_t>(pos, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float64) {
  using T = float64_t;
  constexpr T pos = 42.5;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(int4_t(pos), TestConversion<int4_t>(pos));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
  EXPECT_EQ(bfloat16_t(pos), TestConversion<bfloat16_t>(pos));
  EXPECT_EQ(float32_t(pos), TestConversion<float32_t>(pos));
  EXPECT_EQ(float64_t(pos),
            TestConversion<float64_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(complex64_t(float32_t(pos)), TestConversion<complex64_t>(pos));
  EXPECT_EQ(complex128_t(float64_t(pos)),
            TestConversion<complex128_t>(pos, kSafeAndImplicit));
  EXPECT_EQ("42.5", TestConversion<string_t>(pos));
  EXPECT_EQ(ustring_t{"42.5"}, TestConversion<ustring_t>(pos));
  EXPECT_EQ(json_t(pos), TestConversion<json_t>(pos, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Complex64) {
  using T = complex64_t;
  constexpr T value(42.5, 43.5);
  EXPECT_EQ(int4_t(value.real()), TestConversion<int4_t>(value));
  EXPECT_EQ(int8_t(value.real()), TestConversion<int8_t>(value));
  EXPECT_EQ(int16_t(value.real()), TestConversion<int16_t>(value));
  EXPECT_EQ(int32_t(value.real()), TestConversion<int32_t>(value));
  EXPECT_EQ(int64_t(value.real()), TestConversion<int64_t>(value));
  // TODO(summivox): b/295577703 uint8_t
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(float16_t(value.real()), TestConversion<float16_t>(value));
  EXPECT_EQ(bfloat16_t(value.real()), TestConversion<bfloat16_t>(value));
  EXPECT_EQ(float32_t(value.real()), TestConversion<float32_t>(value));
  EXPECT_EQ(float64_t(value.real()), TestConversion<float64_t>(value));
  EXPECT_EQ(complex64_t(value),
            TestConversion<complex64_t>(
                value, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(complex128_t(value),
            TestConversion<complex128_t>(value, kSafeAndImplicit));
  EXPECT_EQ("(42.5,43.5)", TestConversion<string_t>(value));
  EXPECT_EQ(ustring_t{"(42.5,43.5)"}, TestConversion<ustring_t>(value));
  EXPECT_EQ(json_t(json_t::array_t{value.real(), value.imag()}),
            TestConversion<json_t>(value, kSafeAndImplicit));
  TestUnsupported<T, bool>();
}

TEST(DataTypeConversionTest, Complex128) {
  using T = complex128_t;
  constexpr T value(42.5, 43.5);
  EXPECT_EQ(int4_t(value.real()), TestConversion<int4_t>(value));
  EXPECT_EQ(int8_t(value.real()), TestConversion<int8_t>(value));
  EXPECT_EQ(int16_t(value.real()), TestConversion<int16_t>(value));
  EXPECT_EQ(int32_t(value.real()), TestConversion<int32_t>(value));
  EXPECT_EQ(int64_t(value.real()), TestConversion<int64_t>(value));
  // TODO(summivox): b/295577703 uint8_t
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(float16_t(value.real()), TestConversion<float16_t>(value));
  EXPECT_EQ(bfloat16_t(value.real()), TestConversion<bfloat16_t>(value));
  EXPECT_EQ(float32_t(value.real()), TestConversion<float32_t>(value));
  EXPECT_EQ(float64_t(value.real()), TestConversion<float64_t>(value));
  EXPECT_EQ(complex64_t(value), TestConversion<complex64_t>(value));
  EXPECT_EQ(complex128_t(value),
            TestConversion<complex128_t>(
                value, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ("(42.5,43.5)", TestConversion<string_t>(value));
  EXPECT_EQ(ustring_t{"(42.5,43.5)"}, TestConversion<ustring_t>(value));
  EXPECT_EQ(json_t(json_t::array_t{value.real(), value.imag()}),
            TestConversion<json_t>(value, kSafeAndImplicit));
  TestUnsupported<T, bool>();
}

TEST(DataTypeConversionTest, String) {
  using T = string_t;
  T value = "test";
  T invalid_utf8 = "test\xa0";
  TestUnsupported<T, bool>();
  TestUnsupported<T, int4_t>();
  TestUnsupported<T, int8_t>();
  // TODO(summivox): b/295577703 uint4
  TestUnsupported<T, uint8_t>();
  TestUnsupported<T, int16_t>();
  TestUnsupported<T, uint16_t>();
  TestUnsupported<T, int32_t>();
  TestUnsupported<T, uint32_t>();
  TestUnsupported<T, int64_t>();
  TestUnsupported<T, uint64_t>();
  TestUnsupported<T, float16_t>();
  TestUnsupported<T, bfloat16_t>();
  TestUnsupported<T, float32_t>();
  TestUnsupported<T, float64_t>();
  TestUnsupported<T, complex64_t>();
  TestUnsupported<T, complex128_t>();
  EXPECT_EQ(value,
            TestConversion<string_t>(
                value, kSafeAndImplicit | kCanReinterpretCast | kIdentity));
  EXPECT_EQ(ustring_t{value}, TestConversion<ustring_t>(value));
  EXPECT_THAT(TestConversion<ustring_t>(invalid_utf8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Invalid UTF-8 sequence encountered"));
  EXPECT_EQ(json_t("test"), TestConversion<json_t>(value));
  EXPECT_THAT(TestConversion<json_t>(invalid_utf8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Invalid UTF-8 sequence encountered"));
}

TEST(DataTypeConversionTest, Ustring) {
  using T = ustring_t;
  T value{"test"};
  TestUnsupported<T, bool>();
  TestUnsupported<T, int4_t>();
  TestUnsupported<T, int8_t>();
  // TODO(summivox): b/295577703 uint4
  TestUnsupported<T, uint8_t>();
  TestUnsupported<T, int16_t>();
  TestUnsupported<T, uint16_t>();
  TestUnsupported<T, int32_t>();
  TestUnsupported<T, uint32_t>();
  TestUnsupported<T, int64_t>();
  TestUnsupported<T, uint64_t>();
  TestUnsupported<T, float16_t>();
  TestUnsupported<T, bfloat16_t>();
  TestUnsupported<T, float32_t>();
  TestUnsupported<T, float64_t>();
  TestUnsupported<T, complex64_t>();
  TestUnsupported<T, complex128_t>();
  EXPECT_EQ(value.utf8, TestConversion<string_t>(
                            value, kSafeAndImplicit | kCanReinterpretCast));
  EXPECT_EQ(value,
            TestConversion<ustring_t>(
                value, kSafeAndImplicit | kCanReinterpretCast | kIdentity));
  EXPECT_EQ(json_t("test"), TestConversion<json_t>(value, kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Json) {
  EXPECT_THAT(TestConversion<bool_t>(json_t("hello")),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<bool_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int4_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int8_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int16_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int32_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int64_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint8_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint16_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint32_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint64_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float16_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<bfloat16_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float32_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float64_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<ustring_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(false, TestConversion<bool_t>(json_t(false)));
  EXPECT_EQ(false, TestConversion<bool_t>(json_t("false")));
  EXPECT_EQ(true, TestConversion<bool_t>(json_t(true)));
  EXPECT_EQ(true, TestConversion<bool_t>(json_t("true")));
  EXPECT_EQ(int4_t(-8), TestConversion<int4_t>(json_t(-8)));
  EXPECT_EQ(int8_t(58), TestConversion<int8_t>(json_t(58)));
  EXPECT_EQ(int16_t(1234), TestConversion<int16_t>(json_t(1234)));
  EXPECT_EQ(int16_t(1234), TestConversion<int16_t>(json_t("1234")));
  EXPECT_EQ(int32_t(123456789), TestConversion<int32_t>(json_t(123456789)));
  EXPECT_EQ(int64_t(1234567890123),
            TestConversion<int64_t>(json_t(1234567890123)));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_EQ(uint8_t(254), TestConversion<uint8_t>(json_t(254u)));
  EXPECT_EQ(uint16_t(45123), TestConversion<uint16_t>(json_t(45123u)));
  EXPECT_EQ(uint32_t(4012356789),
            TestConversion<uint32_t>(json_t(4012356789u)));
  EXPECT_EQ(uint64_t(40123567891234),
            TestConversion<uint64_t>(json_t(40123567891234)));
  EXPECT_EQ(float16_t(42.5), TestConversion<float16_t>(json_t(42.5)));
  EXPECT_EQ(float16_t(42.5), TestConversion<float16_t>(json_t("42.5")));
  EXPECT_EQ(bfloat16_t(42.5), TestConversion<bfloat16_t>(json_t(42.5)));
  EXPECT_EQ(bfloat16_t(42.5), TestConversion<bfloat16_t>(json_t("42.5")));
  EXPECT_EQ(float32_t(42.5), TestConversion<float32_t>(json_t(42.5)));
  EXPECT_EQ(float64_t(42.5), TestConversion<float64_t>(json_t(42.5)));
  EXPECT_EQ(float64_t(42.5), TestConversion<float64_t>(json_t("42.5")));
  TestUnsupported<json_t, complex64_t>();
  TestUnsupported<json_t, complex128_t>();
  EXPECT_EQ("hello", TestConversion<string_t>(json_t("hello")));
  EXPECT_THAT(TestConversion<string_t>(json_t(7)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(true)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(1.5)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t::array({2, 3})),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(ustring_t{"hello"}, TestConversion<ustring_t>(json_t("hello")));
  EXPECT_EQ(json_t("hello"), TestConversion<json_t>(
                                 json_t("hello"), kSafeAndImplicit | kIdentity |
                                                      kCanReinterpretCast));
}

TEST(GetDataTypeConverterOrErrorTest, Basic) {
  TENSORSTORE_EXPECT_OK(
      GetDataTypeConverterOrError(dtype_v<int32_t>, dtype_v<int32_t>));
  TENSORSTORE_EXPECT_OK(GetDataTypeConverterOrError(
      dtype_v<int32_t>, dtype_v<int32_t>, kIdentity));
  TENSORSTORE_EXPECT_OK(GetDataTypeConverterOrError(
      dtype_v<int32_t>, dtype_v<int64_t>, kSafeAndImplicit));
  TENSORSTORE_EXPECT_OK(GetDataTypeConverterOrError(
      dtype_v<int32_t>, dtype_v<uint32_t>, kCanReinterpretCast));
  EXPECT_THAT(
      GetDataTypeConverterOrError(dtype_v<json_t>, dtype_v<complex64_t>),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Cannot convert json -> complex64"));
  EXPECT_THAT(
      GetDataTypeConverterOrError(dtype_v<uint32_t>, dtype_v<int32_t>,
                                  kSafeAndImplicit),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Explicit data type conversion required to convert uint32 -> int32"));
}

using LowPrecisionByteTestTypes =
    ::testing::Types<float8_e3m4_t, float8_e4m3fn_t, float8_e4m3fnuz_t,
                     float8_e4m3b11fnuz_t, float8_e5m2_t, float8_e5m2fnuz_t,
                     float8_e8m0fnu_t, float4_e2m1fn_t, int2_t, int4_t>;

template <typename T>
class LowPrecisionByteConversionTest : public ::testing::Test {};
TYPED_TEST_SUITE(LowPrecisionByteConversionTest, LowPrecisionByteTestTypes);

template <typename From, typename To>
void RunAllBufferKindsAndVerifyBitIdentical(
    const From* in, size_t n_elems,
    std::unique_ptr<To[]>* out_contiguous_out = nullptr) {
  const Index n = static_cast<Index>(n_elems);
  const auto conv = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
  ASSERT_EQ(conv.flags & kSupported, kSupported);

  auto out_contiguous = std::make_unique<To[]>(n);
  auto in_strided = std::make_unique<From[]>(n * 2);
  auto out_strided = std::make_unique<To[]>(n * 2);
  std::vector<Index> in_offsets(n);
  std::vector<Index> out_offsets(n);
  auto out_indexed = std::make_unique<To[]>(n);
  for (Index i = 0; i < n; ++i) {
    in_strided[i * 2] = in[i];
    in_offsets[i] = i * static_cast<Index>(sizeof(From));
    out_offsets[i] = (n - 1 - i) * static_cast<Index>(sizeof(To));
  }

  absl::Status status;
  ASSERT_TRUE((*conv.closure.function)[IterationBufferKind::kContiguous](
      conv.closure.context, {1, n},
      IterationBufferPointer{const_cast<From*>(in),
                             static_cast<Index>(sizeof(From)) * n,
                             static_cast<Index>(sizeof(From))},
      IterationBufferPointer{out_contiguous.get(),
                             static_cast<Index>(sizeof(To)) * n,
                             static_cast<Index>(sizeof(To))},
      &status))
      << status;

  ASSERT_TRUE((*conv.closure.function)[IterationBufferKind::kStrided](
      conv.closure.context, {1, n},
      IterationBufferPointer{in_strided.get(),
                             static_cast<Index>(sizeof(From)) * n * 2,
                             static_cast<Index>(sizeof(From)) * 2},
      IterationBufferPointer{out_strided.get(),
                             static_cast<Index>(sizeof(To)) * n * 2,
                             static_cast<Index>(sizeof(To)) * 2},
      &status))
      << status;

  ASSERT_TRUE((*conv.closure.function)[IterationBufferKind::kIndexed](
      conv.closure.context, {1, n},
      IterationBufferPointer{const_cast<From*>(in), n, in_offsets.data()},
      IterationBufferPointer{out_indexed.get(), n, out_offsets.data()},
      &status))
      << status;

  for (Index i = 0; i < n; ++i) {
    const To& c = out_contiguous[i];
    const To& s = out_strided[i * 2];
    const To& x = out_indexed[n - 1 - i];
    if constexpr (tensorstore::IsTrivial<To>) {
      EXPECT_EQ(std::memcmp(&c, &s, sizeof(To)), 0) << "index=" << i;
      EXPECT_EQ(std::memcmp(&c, &x, sizeof(To)), 0) << "index=" << i;
    } else if constexpr (std::is_same_v<To, json_t>) {
      if (c.is_number_float() && std::isnan(c.template get<double>())) {
        EXPECT_TRUE(s.is_number_float() && std::isnan(s.template get<double>()))
            << "index=" << i;
        EXPECT_TRUE(x.is_number_float() && std::isnan(x.template get<double>()))
            << "index=" << i;
      } else {
        EXPECT_TRUE(tensorstore::internal_data_type::CompareIdentical(c, s))
            << "index=" << i;
        EXPECT_TRUE(tensorstore::internal_data_type::CompareIdentical(c, x))
            << "index=" << i;
      }
    } else {
      EXPECT_TRUE(tensorstore::internal_data_type::CompareIdentical(c, s))
          << "index=" << i;
      EXPECT_TRUE(tensorstore::internal_data_type::CompareIdentical(c, x))
          << "index=" << i;
    }
  }
  if (out_contiguous_out != nullptr) {
    *out_contiguous_out = std::move(out_contiguous);
  }
}

TYPED_TEST(LowPrecisionByteConversionTest, Exhaustive256ByteRepresentations) {
  static_assert(tensorstore::internal::SaturatedFloatToInt<int8_t>(
                    std::numeric_limits<float8_e5m2_t>::infinity()) == 127);
  static_assert(tensorstore::internal::SaturatedFloatToInt<int8_t>(
                    -std::numeric_limits<float8_e5m2_t>::infinity()) == -128);
  static_assert(tensorstore::internal::SaturatedFloatToInt<int8_t>(
                    std::numeric_limits<float8_e5m2_t>::quiet_NaN()) == 0);
  static_assert(static_cast<float8_e4m3fn_t>(int4_t(3)) ==
                float8_e4m3fn_t(3.0f));
  static_assert(static_cast<float8_e4m3fn_t>(int2_t(-2)) ==
                float8_e4m3fn_t(-2.0f));
  using LowPrec = TypeParam;
  std::array<LowPrec, 256> all_bytes;
  for (int i = 0; i < 256; ++i) {
    all_bytes[i] = std::bit_cast<LowPrec>(static_cast<uint8_t>(i));
  }

  tensorstore::MapCanonicalDataTypes([&](auto d) {
    using Other = typename decltype(d)::Element;
    const auto to_other =
        GetDataTypeConverter(dtype_v<LowPrec>, dtype_v<Other>);
    std::unique_ptr<Other[]> converted;
    if ((to_other.flags & kSupported) == kSupported &&
        to_other.closure.function != nullptr) {
      RunAllBufferKindsAndVerifyBitIdentical<LowPrec, Other>(
          all_bytes.data(), all_bytes.size(), &converted);
      for (int i = 0; i < 256; ++i) {
        if constexpr (std::is_same_v<Other, float16_t> ||
                      std::is_same_v<Other, bfloat16_t> ||
                      (std::is_same_v<Other, float64_t> &&
                       std::numeric_limits<LowPrec>::is_integer)) {
          using Hub = std::conditional_t<std::is_same_v<LowPrec, int2_t> ||
                                             std::is_same_v<LowPrec, int4_t>,
                                         int8_t, float>;
          const Hub hub_val = static_cast<Hub>(all_bytes[i]);
          if constexpr (std::is_same_v<Hub, float>) {
            if (std::isnan(hub_val)) {
              EXPECT_TRUE(std::isnan(static_cast<float>(converted[i])))
                  << "i=" << i << " to=" << dtype_v<Other>;
              continue;
            }
          }
          const Other expected = static_cast<Other>(hub_val);
          EXPECT_EQ(std::memcmp(&converted[i], &expected, sizeof(Other)), 0)
              << "i=" << i << " to=" << dtype_v<Other>;
        } else if constexpr (std::numeric_limits<Other>::is_integer &&
                             !std::is_same_v<Other, bool> &&
                             !std::numeric_limits<LowPrec>::is_integer) {
          const Other expected =
              tensorstore::internal::SaturatedFloatToInt<Other>(all_bytes[i]);
          EXPECT_EQ(std::memcmp(&converted[i], &expected, sizeof(Other)), 0)
              << "i=" << i << " to=" << dtype_v<Other>;
        } else if constexpr (std::is_arithmetic_v<Other> ||
                             std::is_same_v<Other, int2_t> ||
                             std::is_same_v<Other, int4_t>) {
          const Other expected = static_cast<Other>(all_bytes[i]);
          EXPECT_EQ(std::memcmp(&converted[i], &expected, sizeof(Other)), 0)
              << "i=" << i << " to=" << dtype_v<Other>;
        }
      }
    }

    const auto from_other =
        GetDataTypeConverter(dtype_v<Other>, dtype_v<LowPrec>);
    if ((from_other.flags & kSupported) == kSupported &&
        from_other.closure.function != nullptr && converted != nullptr) {
      auto valid_inputs = std::make_unique<Other[]>(256);
      size_t num_valid = 0;
      for (size_t i = 0; i < 256; ++i) {
        if constexpr (std::is_same_v<Other, json_t>) {
          if (converted[i].is_null()) continue;
        }
        valid_inputs[num_valid++] = converted[i];
      }
      if (num_valid > 0) {
        std::unique_ptr<LowPrec[]> roundtrip;
        RunAllBufferKindsAndVerifyBitIdentical<Other, LowPrec>(
            valid_inputs.get(), num_valid, &roundtrip);
        if constexpr (std::is_arithmetic_v<Other> ||
                      std::is_same_v<Other, int2_t> ||
                      std::is_same_v<Other, int4_t> ||
                      std::is_same_v<Other, bfloat16_t>) {
          for (size_t i = 0; i < num_valid; ++i) {
            LowPrec expected;
            if constexpr (std::is_same_v<Other, float16_t> &&
                          !std::is_same_v<LowPrec, int2_t> &&
                          !std::is_same_v<LowPrec, int4_t>) {
              expected =
                  static_cast<LowPrec>(static_cast<float>(valid_inputs[i]));
            } else {
              expected = static_cast<LowPrec>(valid_inputs[i]);
            }
            EXPECT_EQ(std::memcmp(&roundtrip[i], &expected, sizeof(LowPrec)), 0)
                << "i=" << i << " from=" << dtype_v<Other>;
          }
        }
      }
    }
    return 0;
  });
}

template <typename From, typename To>
void BM_ConvertContiguous(benchmark::State& state) {
  constexpr Index kCount = 4096;
  std::vector<From> src(kCount);
  for (Index i = 0; i < kCount; ++i) {
    src[i] = static_cast<From>((i & 63) * 0.25f - 8.0f);
  }
  std::vector<To> dst(kCount);
  auto r = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
  absl::Status status;
  const auto fn = (*r.closure.function)[IterationBufferKind::kContiguous];
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    fn(r.closure.context, {1, kCount},
       IterationBufferPointer(src.data(), Index(0), sizeof(From)),
       IterationBufferPointer(dst.data(), Index(0), sizeof(To)), &status);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kCount);
}

template <typename From, typename To>
void BM_ScalarStaticCast(benchmark::State& state) {
  constexpr Index kCount = 4096;
  std::vector<From> src(kCount);
  for (Index i = 0; i < kCount; ++i) {
    src[i] = static_cast<From>((i & 63) * 0.25f - 8.0f);
  }
  std::vector<To> dst(kCount);
  for (auto _ : state) {
    const From* s = src.data();
    To* d = dst.data();
    benchmark::DoNotOptimize(s);
    for (Index i = 0; i < kCount; ++i) {
      d[i] = static_cast<To>(s[i]);
    }
    benchmark::DoNotOptimize(d);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kCount);
}

// Regime B: 1-byte low-precision -> multi-byte upcasts
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, bfloat16_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float32_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float64_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, int32_t>);
BENCHMARK(BM_ConvertContiguous<float4_e2m1fn_t, bfloat16_t>);

// Regime A: 1-byte -> 1-byte conversions
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float8_e5m2_t>);
BENCHMARK(BM_ConvertContiguous<int4_t, float8_e4m3fn_t>);

// Regime C: multi-byte -> 1-byte downcasts
BENCHMARK(BM_ConvertContiguous<float32_t, float8_e4m3fn_t>);
BENCHMARK(BM_ConvertContiguous<bfloat16_t, float8_e4m3fn_t>);

// Standard multi-byte contiguous conversions
BENCHMARK(BM_ConvertContiguous<float32_t, bfloat16_t>);
BENCHMARK(BM_ConvertContiguous<bfloat16_t, float32_t>);
BENCHMARK(BM_ConvertContiguous<float32_t, float64_t>);
BENCHMARK(BM_ConvertContiguous<float32_t, int32_t>);

// Direct scalar static_cast<To>(From) loop outside data_type_conversion.cc
BENCHMARK(BM_ScalarStaticCast<float8_e4m3fn_t, float32_t>);
BENCHMARK(BM_ScalarStaticCast<float32_t, float8_e4m3fn_t>);

}  // namespace
