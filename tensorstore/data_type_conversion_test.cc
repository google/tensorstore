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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/half_gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using namespace tensorstore::data_types;  // NOLINT
using tensorstore::DataTypeConversionFlags;
using tensorstore::DataTypeConversionTraits;
using tensorstore::DataTypeOf;
using tensorstore::Index;
using tensorstore::IsDataTypeConversionSupported;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::StrCat;
using tensorstore::internal::GetDataTypeConverter;
using tensorstore::internal::GetDataTypeConverterOrError;
using tensorstore::internal::GetElementCopyErrorStatus;
using tensorstore::internal::IterationBufferKind;
using tensorstore::internal::IterationBufferPointer;

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
  static_assert(!IsDataTypeConversionSupported<From, To>::value);
  auto r = GetDataTypeConverter(DataTypeOf<From>(), DataTypeOf<To>());
  EXPECT_EQ(DataTypeConversionFlags{}, r.flags);
}

template <typename To, typename From>
Result<To> TestConversion(
    From from, DataTypeConversionFlags flags = DataTypeConversionFlags{}) {
  SCOPED_TRACE(StrCat("TestConversion<To=", DataTypeOf<To>(),
                      ", From=", DataTypeOf<From>(), ">")
                   .c_str());
  flags = flags | kSupported;
  if (!std::is_same<To, From>::value) {
    EXPECT_EQ(flags, (DataTypeConversionTraits<From, To>::flags));
  }
  EXPECT_EQ(!!(flags & kSafeAndImplicit),
            (IsDataTypeConversionSupported<From, To, kSafeAndImplicit>::value));
  EXPECT_TRUE((IsDataTypeConversionSupported<From, To>::value));
  auto r = GetDataTypeConverter(DataTypeOf<From>(), DataTypeOf<To>());
  EXPECT_EQ(flags, r.flags);
  To value;
  Status status;
  if ((*r.closure.function)[IterationBufferKind::kContiguous](
          r.closure.context, 1, IterationBufferPointer(&from, Index(0)),
          IterationBufferPointer(&value, Index(0)), &status) != 1) {
    return GetElementCopyErrorStatus(std::move(status));
  }
  return value;
}

TEST(DataTypeConversionTest, Bool) {
  EXPECT_EQ(false, TestConversion<bool_t>(false, kSafeAndImplicit | kIdentity |
                                                     kCanReinterpretCast));
  EXPECT_EQ(true, TestConversion<bool_t>(true, kSafeAndImplicit | kIdentity |
                                                   kCanReinterpretCast));
  EXPECT_EQ(0, TestConversion<int8_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int8_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int16_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int16_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int32_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int32_t>(true, kSafeAndImplicit));
  EXPECT_EQ(0, TestConversion<int64_t>(false, kSafeAndImplicit));
  EXPECT_EQ(1, TestConversion<int64_t>(true, kSafeAndImplicit));
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

TEST(DataTypeConversionTest, Int8) {
  using T = int8_t;
  constexpr T pos = 42;
  constexpr T neg = -42;
  EXPECT_EQ(false, TestConversion<bool_t>(T(0)));
  EXPECT_EQ(true, TestConversion<bool_t>(pos));
  EXPECT_EQ(true, TestConversion<bool_t>(neg));
  EXPECT_EQ(int8_t(neg),
            TestConversion<int8_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int8_t(pos),
            TestConversion<int8_t>(
                pos, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg, kSafeAndImplicit));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg),
            TestConversion<int16_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(uint8_t(neg), TestConversion<uint8_t>(neg));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(neg), TestConversion<uint16_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos, kCanReinterpretCast));
  EXPECT_EQ(uint32_t(neg), TestConversion<uint32_t>(neg));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(neg), TestConversion<uint64_t>(neg));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(neg), TestConversion<float16_t>(neg));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kSafeAndImplicit));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg),
            TestConversion<int32_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg, kCanReinterpretCast));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kSafeAndImplicit));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg));
  EXPECT_EQ(int64_t(neg),
            TestConversion<int64_t>(
                neg, kSafeAndImplicit | kIdentity | kCanReinterpretCast));
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
  EXPECT_EQ(int8_t(neg), TestConversion<int8_t>(neg));
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(neg), TestConversion<int16_t>(neg));
  EXPECT_EQ(int32_t(neg), TestConversion<int32_t>(neg));
  EXPECT_EQ(int64_t(neg), TestConversion<int64_t>(neg, kCanReinterpretCast));
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
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos),
            TestConversion<float16_t>(
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
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
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
  EXPECT_EQ(int8_t(pos), TestConversion<int8_t>(pos));
  EXPECT_EQ(int16_t(pos), TestConversion<int16_t>(pos));
  EXPECT_EQ(int32_t(pos), TestConversion<int32_t>(pos));
  EXPECT_EQ(int64_t(pos), TestConversion<int64_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint8_t(pos), TestConversion<uint8_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint16_t(pos), TestConversion<uint16_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint32_t(pos), TestConversion<uint32_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(uint64_t(pos), TestConversion<uint64_t>(pos));
  EXPECT_EQ(float16_t(pos), TestConversion<float16_t>(pos));
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
  EXPECT_EQ(int8_t(value.real()), TestConversion<int8_t>(value));
  EXPECT_EQ(int16_t(value.real()), TestConversion<int16_t>(value));
  EXPECT_EQ(int32_t(value.real()), TestConversion<int32_t>(value));
  EXPECT_EQ(int64_t(value.real()), TestConversion<int64_t>(value));
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(float16_t(value.real()), TestConversion<float16_t>(value));
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
  EXPECT_EQ(int8_t(value.real()), TestConversion<int8_t>(value));
  EXPECT_EQ(int16_t(value.real()), TestConversion<int16_t>(value));
  EXPECT_EQ(int32_t(value.real()), TestConversion<int32_t>(value));
  EXPECT_EQ(int64_t(value.real()), TestConversion<int64_t>(value));
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint8_t(value.real()), TestConversion<uint8_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint16_t(value.real()), TestConversion<uint16_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint32_t(value.real()), TestConversion<uint32_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(uint64_t(value.real()), TestConversion<uint64_t>(value));
  EXPECT_EQ(float16_t(value.real()), TestConversion<float16_t>(value));
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
  TestUnsupported<T, int8_t>();
  TestUnsupported<T, uint8_t>();
  TestUnsupported<T, int16_t>();
  TestUnsupported<T, uint16_t>();
  TestUnsupported<T, int32_t>();
  TestUnsupported<T, uint32_t>();
  TestUnsupported<T, int64_t>();
  TestUnsupported<T, uint64_t>();
  TestUnsupported<T, float16_t>();
  TestUnsupported<T, float32_t>();
  TestUnsupported<T, float64_t>();
  TestUnsupported<T, complex64_t>();
  TestUnsupported<T, complex128_t>();
  EXPECT_EQ(value,
            TestConversion<string_t>(
                value, kSafeAndImplicit | kCanReinterpretCast | kIdentity));
  EXPECT_EQ(ustring_t{value}, TestConversion<ustring_t>(value));
  EXPECT_THAT(TestConversion<ustring_t>(invalid_utf8),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid UTF-8 sequence encountered"));
  EXPECT_EQ(json_t("test"), TestConversion<json_t>(value));
  EXPECT_THAT(TestConversion<json_t>(invalid_utf8),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid UTF-8 sequence encountered"));
}

TEST(DataTypeConversionTest, Ustring) {
  using T = ustring_t;
  T value{"test"};
  TestUnsupported<T, bool>();
  TestUnsupported<T, int8_t>();
  TestUnsupported<T, uint8_t>();
  TestUnsupported<T, int16_t>();
  TestUnsupported<T, uint16_t>();
  TestUnsupported<T, int32_t>();
  TestUnsupported<T, uint32_t>();
  TestUnsupported<T, int64_t>();
  TestUnsupported<T, uint64_t>();
  TestUnsupported<T, float16_t>();
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
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<bool_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int8_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int16_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int32_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<int64_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint8_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint16_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint32_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<uint64_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float16_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float32_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<float64_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<ustring_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(nullptr)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(false, TestConversion<bool_t>(json_t(false)));
  EXPECT_EQ(false, TestConversion<bool_t>(json_t("false")));
  EXPECT_EQ(true, TestConversion<bool_t>(json_t(true)));
  EXPECT_EQ(true, TestConversion<bool_t>(json_t("true")));
  EXPECT_EQ(int8_t(58), TestConversion<int8_t>(json_t(58)));
  EXPECT_EQ(int16_t(1234), TestConversion<int16_t>(json_t(1234)));
  EXPECT_EQ(int16_t(1234), TestConversion<int16_t>(json_t("1234")));
  EXPECT_EQ(int32_t(123456789), TestConversion<int32_t>(json_t(123456789)));
  EXPECT_EQ(int64_t(1234567890123),
            TestConversion<int64_t>(json_t(1234567890123)));
  EXPECT_EQ(uint8_t(254), TestConversion<uint8_t>(json_t(254u)));
  EXPECT_EQ(uint16_t(45123), TestConversion<uint16_t>(json_t(45123u)));
  EXPECT_EQ(uint32_t(4012356789),
            TestConversion<uint32_t>(json_t(4012356789u)));
  EXPECT_EQ(uint64_t(40123567891234),
            TestConversion<uint64_t>(json_t(40123567891234)));
  EXPECT_EQ(float16_t(42.5), TestConversion<float16_t>(json_t(42.5)));
  EXPECT_EQ(float16_t(42.5), TestConversion<float16_t>(json_t("42.5")));
  EXPECT_EQ(float32_t(42.5), TestConversion<float32_t>(json_t(42.5)));
  EXPECT_EQ(float64_t(42.5), TestConversion<float64_t>(json_t(42.5)));
  EXPECT_EQ(float64_t(42.5), TestConversion<float64_t>(json_t("42.5")));
  TestUnsupported<json_t, complex64_t>();
  TestUnsupported<json_t, complex128_t>();
  EXPECT_EQ("hello", TestConversion<string_t>(json_t("hello")));
  EXPECT_THAT(TestConversion<string_t>(json_t(7)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(true)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t(1.5)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TestConversion<string_t>(json_t::array({2, 3})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(ustring_t{"hello"}, TestConversion<ustring_t>(json_t("hello")));
  EXPECT_EQ(json_t("hello"), TestConversion<json_t>(
                                 json_t("hello"), kSafeAndImplicit | kIdentity |
                                                      kCanReinterpretCast));
}

TEST(GetDataTypeConverterOrErrorTest, Basic) {
  EXPECT_EQ(Status(),
            GetStatus(GetDataTypeConverterOrError(DataTypeOf<std::int32_t>(),
                                                  DataTypeOf<std::int32_t>())));
  EXPECT_EQ(Status(),
            GetStatus(GetDataTypeConverterOrError(
                DataTypeOf<int32_t>(), DataTypeOf<int32_t>(), kIdentity)));
  EXPECT_EQ(Status(), GetStatus(GetDataTypeConverterOrError(
                          DataTypeOf<int32_t>(), DataTypeOf<int64_t>(),
                          kSafeAndImplicit)));
  EXPECT_EQ(Status(), GetStatus(GetDataTypeConverterOrError(
                          DataTypeOf<int32_t>(), DataTypeOf<uint32_t>(),
                          kCanReinterpretCast)));
  EXPECT_THAT(GetDataTypeConverterOrError(DataTypeOf<json_t>(),
                                          DataTypeOf<complex64_t>()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot convert json -> complex64"));
  EXPECT_THAT(
      GetDataTypeConverterOrError(DataTypeOf<uint32_t>(), DataTypeOf<int32_t>(),
                                  kSafeAndImplicit),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Explicit data type conversion required to convert uint32 -> int32"));
}

}  // namespace
