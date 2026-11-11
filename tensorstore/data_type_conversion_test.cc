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
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/testing/half_gtest.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AllocateArray;
using ::tensorstore::Array;
using ::tensorstore::ArrayView;
using ::tensorstore::DataType;
using ::tensorstore::DataTypeConversionFlags;
using ::tensorstore::DataTypeConversionTraits;
using ::tensorstore::dtype_v;
using ::tensorstore::ElementPointer;
using ::tensorstore::Index;
using ::tensorstore::IsDataTypeConversionSupported;
using ::tensorstore::MatchesArrayIdentically;
using ::tensorstore::Result;
using ::tensorstore::SharedArray;
using ::tensorstore::StatusIs;
using ::tensorstore::StridedLayout;
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

template <typename To>
class ConvertsToMatcher {
 public:
  using is_gtest_matcher = void;

  ConvertsToMatcher(::testing::Matcher<Result<To>> value_matcher,
                    DataTypeConversionFlags flags)
      : value_matcher_(std::move(value_matcher)), flags_(flags | kSupported) {}

  template <typename From>
  bool MatchAndExplain(const From& from,
                       ::testing::MatchResultListener* listener) const {
    if constexpr (!std::is_same_v<To, From>) {
      constexpr auto kTraitFlags = DataTypeConversionTraits<From, To>::flags;
      if (kTraitFlags != flags_) {
        *listener << "where DataTypeConversionTraits<" << dtype_v<From> << ", "
                  << dtype_v<To> << ">::flags is "
                  << static_cast<int>(kTraitFlags) << " (expected "
                  << static_cast<int>(flags_) << ")";
        return false;
      }
    }
    if ((!!(flags_ & kSafeAndImplicit)) !=
        IsDataTypeConversionSupported<From, To, kSafeAndImplicit>) {
      *listener << "where IsDataTypeConversionSupported<"
                << dtype_v<From> << ", "
                << dtype_v<To> << ", kSafeAndImplicit> mismatches";
      return false;
    }
    if (!IsDataTypeConversionSupported<From, To>) {
      *listener << "where IsDataTypeConversionSupported<"
                << dtype_v<From> << ", " << dtype_v<To> << "> is false";
      return false;
    }
    const auto r = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
    if (r.flags != flags_) {
      *listener << "where GetDataTypeConverter(" << dtype_v<From> << ", "
                << dtype_v<To> << ").flags is " << static_cast<int>(r.flags)
                << " (expected " << static_cast<int>(flags_) << ")";
      return false;
    }
    To value{};
    absl::Status status;
    Result<To> result = [&]() -> Result<To> {
      if ((*r.closure.function)[IterationBufferKind::kContiguous](
              r.closure.context, {1, 1},
              IterationBufferPointer(const_cast<From*>(&from), Index(0),
                                     Index(0)),
              IterationBufferPointer(&value, Index(0), Index(0)),
              &status) == 1) {
        return value;
      }
      return GetElementCopyErrorStatus(std::move(status));
    }();
    *listener << "which converts (" << dtype_v<From> << " -> "
              << dtype_v<To> << ") to " << ::testing::PrintToString(result);
    return ::testing::ExplainMatchResult(value_matcher_, result, listener);
  }

  void DescribeTo(std::ostream* os) const {
    *os << "converts to " << dtype_v<To> << " with flags "
        << static_cast<int>(flags_) << " and result that ";
    value_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "does not convert to " << dtype_v<To> << " with flags "
        << static_cast<int>(flags_) << " and result that ";
    value_matcher_.DescribeNegationTo(os);
  }

 private:
  ::testing::Matcher<Result<To>> value_matcher_;
  DataTypeConversionFlags flags_;
};

template <typename To>
ConvertsToMatcher<To> ConvertsTo(
    To expected, DataTypeConversionFlags flags = DataTypeConversionFlags{}) {
  return ConvertsToMatcher<To>(::testing::Optional(std::move(expected)), flags);
}

template <typename To, typename InnerMatcher>
ConvertsToMatcher<To> ConvertsToMatching(
    InnerMatcher inner_matcher,
    DataTypeConversionFlags flags = DataTypeConversionFlags{}) {
  return ConvertsToMatcher<To>(
      ::testing::SafeMatcherCast<Result<To>>(std::move(inner_matcher)), flags);
}

TEST(DataTypeConversionTest, Bool) {
  EXPECT_THAT(false, ConvertsTo<bool_t>(false, kSafeAndImplicit | kIdentity |
                                                   kCanReinterpretCast));
  EXPECT_THAT(true, ConvertsTo<bool_t>(true, kSafeAndImplicit | kIdentity |
                                                 kCanReinterpretCast));
  EXPECT_THAT(false, ConvertsTo(int4_t{0}, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(int4_t{1}, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<int8_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<int8_t>(1, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<int16_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<int16_t>(1, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<int32_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<int32_t>(1, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<int64_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<int64_t>(1, kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(false, ConvertsTo<uint8_t>(0u, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<uint8_t>(1u, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<uint16_t>(0u, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<uint16_t>(1u, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<uint32_t>(0u, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<uint32_t>(1u, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<uint64_t>(0u, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<uint64_t>(1u, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo(float16_t{0}, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(float16_t{1}, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo(bfloat16_t{0}, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(bfloat16_t{1}, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<float32_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<float32_t>(1, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo<float64_t>(0, kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo<float64_t>(1, kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo(complex64_t(0), kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(complex64_t(1), kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo(complex128_t(0), kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(complex128_t(1), kSafeAndImplicit));
  EXPECT_THAT(false, ConvertsTo(json_t(false), kSafeAndImplicit));
  EXPECT_THAT(true, ConvertsTo(json_t(true), kSafeAndImplicit));
  TestUnsupported<bool, string_t>();
  TestUnsupported<bool, ustring_t>();
}

TEST(DataTypeConversionTest, Int4) {
  using T = int4_t;
  constexpr T pos{7};
  constexpr T neg{-8};
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(float16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(bfloat16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("-8"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"-8"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int8) {
  using T = int8_t;
  constexpr T pos = 42;
  constexpr T neg = -42;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<int8_t>(neg),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<int8_t>(pos),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(float16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(bfloat16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("-42"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"-42"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint8) {
  using T = uint8_t;
  constexpr T pos = 42;
  constexpr T neg = -42;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<uint8_t>(neg),
                         kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<uint8_t>(pos),
                         kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(bfloat16_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("214"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"214"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int16) {
  using T = int16_t;
  constexpr T pos = 12345;
  constexpr T neg = -12345;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<int16_t>(neg),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(float16_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(bfloat16_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("-12345"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"-12345"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint16) {
  using T = uint16_t;
  constexpr T pos = 12345;
  constexpr T neg = -12345;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<uint16_t>(neg),
                         kCanReinterpretCast | kIdentity | kSafeAndImplicit));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<uint16_t>(pos),
                         kCanReinterpretCast | kIdentity | kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float16_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(bfloat16_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("53191"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"53191"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int32) {
  using T = int32_t;
  constexpr T pos = 123456789;
  constexpr T neg = -123456789;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<int32_t>(neg),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(float16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg,
              ConvertsTo(bfloat16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg))));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("-123456789"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"-123456789"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint32) {
  using T = uint32_t;
  constexpr T pos = 123456789;
  constexpr T neg = -123456789;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kSafeAndImplicit));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<uint32_t>(neg),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<uint32_t>(pos),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(neg,
              ConvertsTo(float16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg,
              ConvertsTo(bfloat16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg))));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg)), kSafeAndImplicit));
  EXPECT_THAT(neg, ConvertsTo<string_t>("4171510507"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"4171510507"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Int64) {
  using T = int64_t;
  constexpr T pos = 123456789012345;
  constexpr T neg = -123456789012345;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<int64_t>(neg),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint64_t>(neg), kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos), kCanReinterpretCast));
  EXPECT_THAT(neg,
              ConvertsTo(float16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg,
              ConvertsTo(bfloat16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg))));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg))));
  EXPECT_THAT(neg, ConvertsTo<string_t>("-123456789012345"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"-123456789012345"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Uint64) {
  using T = uint64_t;
  constexpr T pos = 123456789012345;
  constexpr T neg = -123456789012345;
  EXPECT_THAT(T{0}, ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo<bool_t>(true));
  EXPECT_THAT(neg, ConvertsTo(int4_t(neg)));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int16_t>(neg)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int32_t>(neg)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<int64_t>(neg), kCanReinterpretCast));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint8_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint16_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(neg, ConvertsTo(static_cast<uint32_t>(neg)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(neg,
              ConvertsTo(static_cast<uint64_t>(neg),
                         kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<uint64_t>(pos),
                         kCanReinterpretCast | kSafeAndImplicit | kIdentity));
  EXPECT_THAT(neg,
              ConvertsTo(float16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg,
              ConvertsTo(bfloat16_t(static_cast<float>(neg))));  // inexact
  EXPECT_THAT(neg, ConvertsTo(float32_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(float64_t(neg)));
  EXPECT_THAT(neg, ConvertsTo(complex64_t(float32_t(neg))));
  EXPECT_THAT(neg, ConvertsTo(complex128_t(float64_t(neg))));
  EXPECT_THAT(neg, ConvertsTo<string_t>("18446620616920539271"));
  EXPECT_THAT(neg, ConvertsTo(ustring_t{"18446620616920539271"}));
  EXPECT_THAT(neg, ConvertsTo(json_t(neg), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float16) {
  using T = float16_t;
  const T pos(42.5);
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int4_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos,
              ConvertsTo(static_cast<float16_t>(pos),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(static_cast<bfloat16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<float32_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(static_cast<float64_t>(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo<string_t>("42.5"));
  EXPECT_THAT(pos, ConvertsTo(ustring_t{"42.5"}));
  EXPECT_THAT(pos, ConvertsTo(json_t(42.5), kSafeAndImplicit));
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
    EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  }
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(T(pos), kSafeAndImplicit | kIdentity |
                                          kCanReinterpretCast));
  if (!std::is_same_v<T, float8_e3m4_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e3m4_t(pos)));
  }
  if (!std::is_same_v<T, float8_e4m3fn_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e4m3fn_t(pos)));
  }
  if (!std::is_same_v<T, float8_e4m3fnuz_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e4m3fnuz_t(pos)));
  }
  if (!std::is_same_v<T, float8_e4m3b11fnuz_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e4m3b11fnuz_t(pos)));
  }
  if (!std::is_same_v<T, float8_e5m2fnuz_t>) {
    if (std::is_same_v<T, float8_e5m2_t>) {
      EXPECT_THAT(pos, ConvertsTo(float8_e5m2fnuz_t(pos), kSafeAndImplicit));
    } else {
      EXPECT_THAT(pos, ConvertsTo(float8_e5m2fnuz_t(pos)));
    }
  }
  if (!std::is_same_v<T, float8_e5m2_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e5m2_t(pos)));
  }
  if (!std::is_same_v<T, float8_e8m0fnu_t>) {
    EXPECT_THAT(pos, ConvertsTo(float8_e8m0fnu_t(pos)));
  }
  if (std::is_same_v<T, float8_e5m2fnuz_t> ||
      std::is_same_v<T, float8_e8m0fnu_t>) {
    EXPECT_THAT(pos, ConvertsTo(float16_t(pos)));
  } else {
    EXPECT_THAT(pos, ConvertsTo(float16_t(pos), kSafeAndImplicit));
  }
  EXPECT_THAT(pos, ConvertsTo(bfloat16_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(float32_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(float64_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  if constexpr (std::numeric_limits<T>::digits == 1) {
    EXPECT_THAT(pos, ConvertsTo<string_t>("4"));
    EXPECT_THAT(pos, ConvertsTo(ustring_t{"4"}));
    EXPECT_THAT(pos, ConvertsTo(json_t(4.0), kSafeAndImplicit));
  } else {
    EXPECT_THAT(pos, ConvertsTo<string_t>("3.5"));
    EXPECT_THAT(pos, ConvertsTo(ustring_t{"3.5"}));
    EXPECT_THAT(pos, ConvertsTo(json_t(3.5), kSafeAndImplicit));
  }
}

template <typename InternalFloat>
class InternalMXFloatTest : public ::testing::Test {};

using InternalMXFloatTypes = ::testing::Types<float4_e2m1fn_t>;

TYPED_TEST_SUITE(InternalMXFloatTest, InternalMXFloatTypes);

TYPED_TEST(InternalMXFloatTest, DataTypeConversionTest_InternalFloat8Types) {
  using T = TypeParam;
  const T pos(1.5);
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(T(pos), kSafeAndImplicit | kIdentity |
                                          kCanReinterpretCast));
  if (!std::is_same_v<T, float4_e2m1fn_t>) {
    EXPECT_THAT(pos, ConvertsTo(float4_e2m1fn_t(pos)));
  }
  EXPECT_THAT(pos, ConvertsTo(float16_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(bfloat16_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(float32_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(float64_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo<string_t>("1.5"));
  EXPECT_THAT(pos, ConvertsTo(ustring_t{"1.5"}));
  EXPECT_THAT(pos, ConvertsTo(json_t(1.5), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Bfloat16) {
  using T = bfloat16_t;
  const T pos(42.5);
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(float16_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(bfloat16_t(pos), kSafeAndImplicit | kIdentity |
                                                   kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(float32_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(float64_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo<string_t>("42.5"));
  EXPECT_THAT(pos, ConvertsTo(ustring_t{"42.5"}));
  EXPECT_THAT(pos, ConvertsTo(json_t(42.5), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float32) {
  using T = float32_t;
  constexpr T pos = 42.5;
  // Don't test overflow of float -> int conversion, because that is undefined
  // behavior according to the C++ standard, and the behavior differs in
  // practice between x86 and PPC64le.
  //
  // See: https://github.com/numpy/numpy/issues/9040
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(float16_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(bfloat16_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(float32_t(pos), kSafeAndImplicit | kIdentity |
                                                  kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(float64_t(pos), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo<string_t>("42.5"));
  EXPECT_THAT(pos, ConvertsTo(ustring_t{"42.5"}));
  EXPECT_THAT(pos, ConvertsTo(json_t(pos), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Float64) {
  using T = float64_t;
  constexpr T pos = 42.5;
  EXPECT_THAT(T(0), ConvertsTo<bool_t>(false));
  EXPECT_THAT(pos, ConvertsTo<bool_t>(true));
  EXPECT_THAT(pos, ConvertsTo(int4_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<int64_t>(pos)));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint8_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint16_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint32_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(static_cast<uint64_t>(pos)));
  EXPECT_THAT(pos, ConvertsTo(float16_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(bfloat16_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(float32_t(pos)));
  EXPECT_THAT(pos, ConvertsTo(float64_t(pos), kSafeAndImplicit | kIdentity |
                                                  kCanReinterpretCast));
  EXPECT_THAT(pos, ConvertsTo(complex64_t(float32_t(pos))));
  EXPECT_THAT(pos, ConvertsTo(complex128_t(float64_t(pos)), kSafeAndImplicit));
  EXPECT_THAT(pos, ConvertsTo<string_t>("42.5"));
  EXPECT_THAT(pos, ConvertsTo(ustring_t{"42.5"}));
  EXPECT_THAT(pos, ConvertsTo(json_t(pos), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Complex64) {
  using T = complex64_t;
  constexpr T value(42.5, 43.5);
  EXPECT_THAT(value, ConvertsTo(int4_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int8_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int16_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int32_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int64_t>(value.real())));
  // TODO(summivox): b/295577703 uint8_t
  EXPECT_THAT(value, ConvertsTo(static_cast<uint8_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint8_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint16_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint16_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint32_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint32_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint64_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint64_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(float16_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(bfloat16_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(float32_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(float64_t(value.real())));
  EXPECT_THAT(value,
              ConvertsTo(complex64_t(value),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(value, ConvertsTo(complex128_t(value), kSafeAndImplicit));
  EXPECT_THAT(value, ConvertsTo<string_t>("(42.5,43.5)"));
  EXPECT_THAT(value, ConvertsTo(ustring_t{"(42.5,43.5)"}));
  EXPECT_THAT(value,
              ConvertsTo(json_t(json_t::array_t{value.real(), value.imag()}),
                         kSafeAndImplicit));
  TestUnsupported<T, bool>();
}

TEST(DataTypeConversionTest, Complex128) {
  using T = complex128_t;
  constexpr T value(42.5, 43.5);
  EXPECT_THAT(value, ConvertsTo(int4_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int8_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int16_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int32_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<int64_t>(value.real())));
  // TODO(summivox): b/295577703 uint8_t
  EXPECT_THAT(value, ConvertsTo(static_cast<uint8_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint16_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint32_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(static_cast<uint64_t>(value.real())));
  EXPECT_THAT(value, ConvertsTo(float16_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(bfloat16_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(float32_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(float64_t(value.real())));
  EXPECT_THAT(value, ConvertsTo(complex64_t(value)));
  EXPECT_THAT(value,
              ConvertsTo(complex128_t(value),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
  EXPECT_THAT(value, ConvertsTo<string_t>("(42.5,43.5)"));
  EXPECT_THAT(value, ConvertsTo(ustring_t{"(42.5,43.5)"}));
  EXPECT_THAT(value,
              ConvertsTo(json_t(json_t::array_t{value.real(), value.imag()}),
                         kSafeAndImplicit));
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
  EXPECT_THAT(value, ConvertsTo(value, kSafeAndImplicit | kCanReinterpretCast |
                                           kIdentity));
  EXPECT_THAT(value, ConvertsTo(ustring_t{value}));
  EXPECT_THAT(invalid_utf8, ConvertsToMatching<ustring_t>(StatusIs(
                                absl::StatusCode::kInvalidArgument,
                                "Invalid UTF-8 sequence encountered")));
  EXPECT_THAT(value, ConvertsTo(json_t("test")));
  EXPECT_THAT(invalid_utf8, ConvertsToMatching<json_t>(StatusIs(
                                absl::StatusCode::kInvalidArgument,
                                "Invalid UTF-8 sequence encountered")));
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
  EXPECT_THAT(value,
              ConvertsTo(value.utf8, kSafeAndImplicit | kCanReinterpretCast));
  EXPECT_THAT(value, ConvertsTo(value, kSafeAndImplicit | kCanReinterpretCast |
                                           kIdentity));
  EXPECT_THAT(value, ConvertsTo(json_t("test"), kSafeAndImplicit));
}

TEST(DataTypeConversionTest, Json) {
  EXPECT_THAT(
      json_t("hello"),
      ConvertsToMatching<bool_t>(StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(
      json_t(nullptr),
      ConvertsToMatching<bool_t>(StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(
      json_t(nullptr),
      ConvertsToMatching<int4_t>(StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(
      json_t(nullptr),
      ConvertsToMatching<int8_t>(StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<int16_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<int32_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<int64_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<uint8_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<uint16_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<uint32_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<uint64_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<float16_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<bfloat16_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<float32_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<float64_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<string_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<ustring_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(nullptr), ConvertsToMatching<string_t>(StatusIs(
                                   absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(false), ConvertsTo<bool_t>(false));
  EXPECT_THAT(json_t("false"), ConvertsTo<bool_t>(false));
  EXPECT_THAT(json_t(true), ConvertsTo<bool_t>(true));
  EXPECT_THAT(json_t("true"), ConvertsTo<bool_t>(true));
  EXPECT_THAT(json_t(-8), ConvertsTo(int4_t(-8)));
  EXPECT_THAT(json_t(58), ConvertsTo(int8_t{58}));
  EXPECT_THAT(json_t(1234), ConvertsTo(int16_t{1234}));
  EXPECT_THAT(json_t("1234"), ConvertsTo(int16_t{1234}));
  EXPECT_THAT(json_t(123456789), ConvertsTo(int32_t{123456789}));
  EXPECT_THAT(json_t(1234567890123), ConvertsTo(int64_t{1234567890123}));
  // TODO(summivox): b/295577703 uint4_t
  EXPECT_THAT(json_t(254u), ConvertsTo(uint8_t{254}));
  EXPECT_THAT(json_t(45123u), ConvertsTo(uint16_t{45123}));
  EXPECT_THAT(json_t(4012356789u), ConvertsTo(uint32_t{4012356789}));
  EXPECT_THAT(json_t(40123567891234), ConvertsTo(uint64_t{40123567891234}));
  EXPECT_THAT(json_t(42.5), ConvertsTo(float16_t(42.5)));
  EXPECT_THAT(json_t("42.5"), ConvertsTo(float16_t(42.5)));
  EXPECT_THAT(json_t(42.5), ConvertsTo(bfloat16_t(42.5)));
  EXPECT_THAT(json_t("42.5"), ConvertsTo(bfloat16_t(42.5)));
  EXPECT_THAT(json_t(42.5), ConvertsTo(float32_t(42.5)));
  EXPECT_THAT(json_t(42.5), ConvertsTo(float64_t(42.5)));
  EXPECT_THAT(json_t("42.5"), ConvertsTo(float64_t(42.5)));
  TestUnsupported<json_t, complex64_t>();
  TestUnsupported<json_t, complex128_t>();
  EXPECT_THAT(json_t("hello"), ConvertsTo<string_t>("hello"));
  EXPECT_THAT(json_t(7), ConvertsToMatching<string_t>(
                             StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(true), ConvertsToMatching<string_t>(
                                StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t(1.5), ConvertsToMatching<string_t>(
                               StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t::array({2, 3}), ConvertsToMatching<string_t>(StatusIs(
                                         absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(json_t("hello"), ConvertsTo(ustring_t{"hello"}));
  EXPECT_THAT(json_t("hello"),
              ConvertsTo(json_t("hello"),
                         kSafeAndImplicit | kIdentity | kCanReinterpretCast));
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

bool ElementsAreIdentical(ArrayView<const void, 2> a,
                          ArrayView<const void, 2> b) {
  if (a.dtype() != b.dtype() || a.domain() != b.domain()) return false;
  if (a.dtype() == dtype_v<json_t>) {
    // Low-precision float -> `json_t` converts via `float32 -> json_t`, storing
    // `NaN` as a JSON `number_float` (where `JsonSame` returns false for
    // `NaN == NaN`) rather than JSON `null`.
    auto aj =
        tensorstore::StaticDataTypeCast<const json_t, tensorstore::unchecked>(
            a);
    auto bj =
        tensorstore::StaticDataTypeCast<const json_t, tensorstore::unchecked>(
            b);
    return tensorstore::IterateOverIndexRange(
        a.domain(), [&](tensorstore::span<const Index> idx) {
          const json_t& av = aj(idx);
          const json_t& bv = bj(idx);
          if (av.is_number_float() && std::isnan(av.get<double>())) {
            return bv.is_number_float() && std::isnan(bv.get<double>());
          }
          return tensorstore::internal_data_type::CompareIdentical(av, bv);
        });
  }
  return tensorstore::AreArraysEqual(
      a, b, tensorstore::EqualityComparisonKind::identical);
}

MATCHER_P(MatchesConvertedArray, expected, "") {
  if (ElementsAreIdentical(arg, expected)) return true;
  return ::testing::ExplainMatchResult(MatchesArrayIdentically(expected), arg,
                                       result_listener);
}

struct AllBufferKindsRunResult {
  DataType from_dtype;
  DataType to_dtype;
  DataTypeConversionFlags flags = DataTypeConversionFlags{};
  Index n = 0;
  Index rows = 0;
  Index cols = 0;
  Index strided_row_elems = 0;
  bool copy_strided_ok = false;
  bool contiguous_ok = false;
  bool strided_ok = false;
  bool indexed_ok = false;
  absl::Status status;
  SharedArray<void, 2> out_contiguous;
  SharedArray<void, 2> out_strided_backing;
  SharedArray<void, 1> out_indexed_backing;

  SharedArray<const void, 2> contiguous_view() const { return out_contiguous; }
  Array<const void, 2> strided_view() const {
    return Array<const void, 2>(
        ElementPointer<const void>(out_strided_backing.data(), to_dtype),
        StridedLayout<2>({rows, cols}, {strided_row_elems * to_dtype->size,
                                        2 * to_dtype->size}));
  }
  Array<const void, 2> indexed_view() const {
    const void* last = static_cast<const char*>(out_indexed_backing.data()) +
                       (n - 1) * to_dtype->size;
    return Array<const void, 2>(
        ElementPointer<const void>(last, to_dtype),
        StridedLayout<2>({rows, cols},
                         {-cols * to_dtype->size, -to_dtype->size}));
  }
};

AllBufferKindsRunResult RunAllBufferKinds(DataType from_dtype,
                                          DataType to_dtype, const void* in,
                                          size_t n_elems) {
  AllBufferKindsRunResult res;
  res.from_dtype = from_dtype;
  res.to_dtype = to_dtype;
  res.n = static_cast<Index>(n_elems);
  const auto conv = GetDataTypeConverter(from_dtype, to_dtype);
  res.flags = conv.flags;
  if ((conv.flags & kSupported) != kSupported ||
      conv.closure.function == nullptr) {
    return res;
  }

  res.rows = (res.n >= 2 && (res.n % 2 == 0)) ? 2 : 1;
  res.cols = res.n / res.rows;
  constexpr Index kStridedRowPad = 6;
  constexpr Index kIndexedRowPad = 5;
  res.strided_row_elems = res.cols * 2 + kStridedRowPad;

  res.out_contiguous = AllocateArray({res.rows, res.cols}, tensorstore::c_order,
                                     tensorstore::default_init, to_dtype);
  auto in_strided =
      AllocateArray({res.rows, res.strided_row_elems}, tensorstore::c_order,
                    tensorstore::default_init, from_dtype);
  res.out_strided_backing =
      AllocateArray({res.rows, res.strided_row_elems}, tensorstore::c_order,
                    tensorstore::default_init, to_dtype);
  res.out_indexed_backing = AllocateArray({res.n}, tensorstore::c_order,
                                          tensorstore::default_init, to_dtype);

  std::vector<Index> in_offsets(res.rows * (res.cols + kIndexedRowPad));
  std::vector<Index> out_offsets(res.rows * (res.cols + kIndexedRowPad));
  for (Index r = 0; r < res.rows; ++r) {
    for (Index c = 0; c < res.cols; ++c) {
      const Index i = r * res.cols + c;
      in_offsets[r * (res.cols + kIndexedRowPad) + c] = i * from_dtype->size;
      out_offsets[r * (res.cols + kIndexedRowPad) + c] =
          (res.n - 1 - i) * to_dtype->size;
    }
  }
  res.copy_strided_ok = from_dtype->copy_assign[IterationBufferKind::kStrided](
      nullptr, {res.rows, res.cols},
      IterationBufferPointer{const_cast<void*>(in), from_dtype->size * res.cols,
                             from_dtype->size},
      IterationBufferPointer{in_strided.data(),
                             from_dtype->size * res.strided_row_elems,
                             from_dtype->size * 2},
      &res.status);

  res.contiguous_ok =
      (*conv.closure.function)[IterationBufferKind::kContiguous](
          conv.closure.context, {res.rows, res.cols},
          IterationBufferPointer{const_cast<void*>(in),
                                 from_dtype->size * res.cols, from_dtype->size},
          IterationBufferPointer{res.out_contiguous.data(),
                                 to_dtype->size * res.cols, to_dtype->size},
          &res.status);

  res.strided_ok = (*conv.closure.function)[IterationBufferKind::kStrided](
      conv.closure.context, {res.rows, res.cols},
      IterationBufferPointer{in_strided.data(),
                             from_dtype->size * res.strided_row_elems,
                             from_dtype->size * 2},
      IterationBufferPointer{res.out_strided_backing.data(),
                             to_dtype->size * res.strided_row_elems,
                             to_dtype->size * 2},
      &res.status);

  res.indexed_ok = (*conv.closure.function)[IterationBufferKind::kIndexed](
      conv.closure.context, {res.rows, res.cols},
      IterationBufferPointer{const_cast<void*>(in), res.cols + kIndexedRowPad,
                             in_offsets.data()},
      IterationBufferPointer{res.out_indexed_backing.data(),
                             res.cols + kIndexedRowPad, out_offsets.data()},
      &res.status);
  return res;
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
    AllBufferKindsRunResult to_res;
    if ((to_other.flags & kSupported) == kSupported &&
        to_other.closure.function != nullptr) {
      to_res = RunAllBufferKinds(dtype_v<LowPrec>, dtype_v<Other>,
                                 all_bytes.data(), all_bytes.size());
      EXPECT_EQ(to_res.flags & kSupported, kSupported);
      EXPECT_TRUE(to_res.copy_strided_ok) << to_res.status;
      EXPECT_TRUE(to_res.contiguous_ok) << to_res.status;
      EXPECT_TRUE(to_res.strided_ok) << to_res.status;
      EXPECT_TRUE(to_res.indexed_ok) << to_res.status;
      EXPECT_THAT(to_res.strided_view(),
                  MatchesConvertedArray(to_res.contiguous_view()));
      EXPECT_THAT(to_res.indexed_view(),
                  MatchesConvertedArray(to_res.contiguous_view()));
      const Other* converted =
          static_cast<const Other*>(to_res.out_contiguous.data());
      for (int i = 0; i < 256; ++i) {
        if constexpr (std::is_same_v<Other, float16_t> ||
                      std::is_same_v<Other, bfloat16_t> ||
                      (std::is_same_v<Other, float64_t> &&
                       std::numeric_limits<LowPrec>::is_integer)) {
          // Sub-byte integers (`int2_t`, `int4_t`) convert via `int8_t` and
          // 1-byte floats convert via `float32_t` when targeting `float16_t` or
          // `bfloat16_t`.
          using Intermediate =
              std::conditional_t<std::is_same_v<LowPrec, int2_t> ||
                                     std::is_same_v<LowPrec, int4_t>,
                                 int8_t, float>;
          const Intermediate intermediate_val =
              static_cast<Intermediate>(all_bytes[i]);
          if constexpr (std::is_same_v<Intermediate, float>) {
            if (std::isnan(intermediate_val)) {
              EXPECT_TRUE(std::isnan(static_cast<float>(converted[i])))
                  << "i=" << i << " to=" << dtype_v<Other>;
              continue;
            }
          }
          const Other expected = static_cast<Other>(intermediate_val);
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
        from_other.closure.function != nullptr &&
        to_res.out_contiguous.data() != nullptr) {
      const Other* converted =
          static_cast<const Other*>(to_res.out_contiguous.data());
      auto valid_inputs = std::make_unique<Other[]>(256);
      size_t num_valid = 0;
      for (size_t i = 0; i < 256; ++i) {
        if constexpr (std::is_same_v<Other, json_t>) {
          if (converted[i].is_null()) continue;
        }
        valid_inputs[num_valid++] = converted[i];
      }
      if (num_valid > 0) {
        auto roundtrip_res = RunAllBufferKinds(dtype_v<Other>, dtype_v<LowPrec>,
                                               valid_inputs.get(), num_valid);
        EXPECT_EQ(roundtrip_res.flags & kSupported, kSupported);
        EXPECT_TRUE(roundtrip_res.copy_strided_ok) << roundtrip_res.status;
        EXPECT_TRUE(roundtrip_res.contiguous_ok) << roundtrip_res.status;
        EXPECT_TRUE(roundtrip_res.strided_ok) << roundtrip_res.status;
        EXPECT_TRUE(roundtrip_res.indexed_ok) << roundtrip_res.status;
        EXPECT_THAT(roundtrip_res.strided_view(),
                    MatchesConvertedArray(roundtrip_res.contiguous_view()));
        EXPECT_THAT(roundtrip_res.indexed_view(),
                    MatchesConvertedArray(roundtrip_res.contiguous_view()));
        if constexpr (std::is_arithmetic_v<Other> ||
                      std::is_same_v<Other, int2_t> ||
                      std::is_same_v<Other, int4_t> ||
                      std::is_same_v<Other, bfloat16_t>) {
          const LowPrec* roundtrip =
              static_cast<const LowPrec*>(roundtrip_res.out_contiguous.data());
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

#if 0
// Fails at HEAD on `50331647 = 1.5 * 2^25 - 1`, `201326591 = 1.5 * 2^27 - 1`,
// and `(1LL << 40) + (1LL << 39) - 1`: HEAD converts `int32/uint32/int64/uint64
// -> float8_e8m0fnu_t` via `static_cast<float>`, which double-rounds values
// just below `1.5 * 2^E` up to `1.5 * 2^E` in `float32` and then up again to
// `2^(E+1)` in `float8_e8m0fnu_t` instead of rounding down to `2^E` (as direct
// `static_cast<float8_e8m0fnu_t>(wide_int)` via `RoundToOddFloat` does).
TEST(DataTypeConversionTest, WideIntToFloat8E8m0fnuNoDoubleRounding) {
  EXPECT_THAT(int32_t{50331647}, ConvertsTo(float8_e8m0fnu_t(33554432.0f)));
  EXPECT_THAT(int32_t{50331649}, ConvertsTo(float8_e8m0fnu_t(67108864.0f)));
  EXPECT_THAT(int32_t{201326591}, ConvertsTo(float8_e8m0fnu_t(134217728.0f)));
  EXPECT_THAT(int32_t{201326593}, ConvertsTo(float8_e8m0fnu_t(268435456.0f)));
  EXPECT_THAT(uint32_t{50331647}, ConvertsTo(float8_e8m0fnu_t(33554432.0f)));
  EXPECT_THAT(int64_t{50331647}, ConvertsTo(float8_e8m0fnu_t(33554432.0f)));
  EXPECT_THAT(uint64_t{50331647}, ConvertsTo(float8_e8m0fnu_t(33554432.0f)));
  EXPECT_THAT(int64_t{(1LL << 40) + (1LL << 39) - 1},
              ConvertsTo(float8_e8m0fnu_t(static_cast<float>(1ULL << 40))));
  EXPECT_THAT(uint64_t{(1ULL << 40) + (1ULL << 39) - 1},
              ConvertsTo(float8_e8m0fnu_t(static_cast<float>(1ULL << 40))));
}
#endif

TEST(DataTypeConversionTest, ByteAndCharConversions) {
  // `char_t -> byte_t` allows `kCanReinterpretCast` (any char is a raw byte),
  // whereas `byte_t -> char_t` does not.
  EXPECT_THAT(byte_t{'x'}, ConvertsTo(char_t{'x'}, kSafeAndImplicit));
  EXPECT_THAT(char_t{'x'},
              ConvertsTo(byte_t{'x'}, kSafeAndImplicit | kCanReinterpretCast));
}

// Excludes `complex64_t` / `complex128_t` (not convertible to `json_t`),
// `byte_t` (not convertible to `json_t`), and `char_t` / `string_t` /
// `ustring_t` (which convert to `json_t::value_t::string`).
using JsonTypeTagTestTypes = ::testing::Types<
    bool_t, uint8_t, uint16_t, uint32_t, uint64_t, int2_t, int4_t, int8_t,
    int16_t, int32_t, int64_t, float4_e2m1fn_t, float8_e3m4_t, float8_e4m3fn_t,
    float8_e4m3fnuz_t, float8_e4m3b11fnuz_t, float8_e5m2_t, float8_e5m2fnuz_t,
    float8_e8m0fnu_t, bfloat16_t, float16_t, float32_t, float64_t>;

template <typename T>
class ToJsonTypeTagTest : public ::testing::Test {};
TYPED_TEST_SUITE(ToJsonTypeTagTest, JsonTypeTagTestTypes);

TYPED_TEST(ToJsonTypeTagTest, PreservesJsonTypeTag) {
  using T = TypeParam;
  auto has_json_type = [](json_t::value_t expected_type) {
    return ConvertsToMatching<json_t>(
        ::testing::Optional(::testing::ResultOf(
            [](const json_t& j) { return j.type(); }, expected_type)),
        kSafeAndImplicit);
  };
  if constexpr (std::is_same_v<T, bool_t>) {
    EXPECT_THAT(bool_t{true}, has_json_type(json_t::value_t::boolean));
  } else if constexpr (std::is_unsigned_v<T>) {
    EXPECT_THAT(T{42}, has_json_type(json_t::value_t::number_unsigned));
  } else if constexpr (std::numeric_limits<T>::is_integer) {
    EXPECT_THAT(T{1}, has_json_type(json_t::value_t::number_integer));
  } else {
    EXPECT_THAT(T{1.0f}, has_json_type(json_t::value_t::number_float));
  }
}

template <typename T, typename... Ts>
constexpr bool IsAnyOf = (std::is_same_v<T, Ts> || ...);

template <typename T>
constexpr bool IsComplex = IsAnyOf<T, complex64_t, complex128_t>;

template <typename T>
constexpr bool IsStringOrJson = IsAnyOf<T, string_t, ustring_t, json_t>;

template <typename From, typename To>
bool PopulateExpectedElements(const void* in_ptr, void* expected_ptr,
                              size_t n_elems) {
  if constexpr (IsStringOrJson<To>) {
    return false;
  } else {
    const From* in = static_cast<const From*>(in_ptr);
    To* expected = static_cast<To*>(expected_ptr);
    for (size_t i = 0; i < n_elems; ++i) {
      if constexpr (std::is_same_v<From, json_t>) {
        expected[i] = static_cast<To>(in[i].template get<double>());
      } else if constexpr (IsComplex<From>) {
        if constexpr (IsComplex<To>) {
          using ToReal = typename To::value_type;
          expected[i] = To{static_cast<ToReal>(in[i].real()),
                           static_cast<ToReal>(in[i].imag())};
        } else {
          expected[i] = static_cast<To>(in[i].real());
        }
      } else {
        using ScalarIn =
            std::conditional_t<IsAnyOf<From, float16_t, bfloat16_t> ||
                                   (std::is_integral_v<From> &&
                                    std::is_same_v<To, float16_t>),
                               float, From>;
        const ScalarIn val = static_cast<ScalarIn>(in[i]);
        if constexpr (IsComplex<To>) {
          using ToReal = typename To::value_type;
          expected[i] = To{static_cast<ToReal>(val), ToReal{0}};
        } else {
          expected[i] = static_cast<To>(val);
        }
      }
    }
    return true;
  }
}

template <typename From, typename To>
void PopulateMultiBlockTestInputs(From* in, size_t n_elems) {
  for (size_t i = 0; i < n_elems; ++i) {
    if constexpr (std::is_same_v<From, int16_t>) {
      constexpr int16_t kSpecial[] = {-32768, -300, -129, -128, -2,  -1,   0,
                                      1,      2,    127,  128,  300, 32767};
      in[i] = (i < std::size(kSpecial)) ? kSpecial[i]
                                        : static_cast<int16_t>(i % 64);
    } else if constexpr (std::is_same_v<From, uint16_t>) {
      constexpr uint16_t kSpecial[] = {
          0u, 1u, 2u, 127u, 128u, 255u, 256u, 32767u, 32768u, 65534u, 65535u};
      in[i] = (i < std::size(kSpecial)) ? kSpecial[i]
                                        : static_cast<uint16_t>(i % 64);
    } else if constexpr (std::is_same_v<From, int32_t>) {
      constexpr int32_t kSpecial[] = {
          0, 1, -1, 127, -128, 32767, -32768, 50331649, 201326593, 0x7fffffff};
      in[i] = (i < std::size(kSpecial)) ? kSpecial[i]
                                        : static_cast<int32_t>(i % 64);
    } else if constexpr (std::is_same_v<From, uint32_t>) {
      constexpr uint32_t kSpecial[] = {
          0u,          1u,          127u,        128u,        255u,
          32767u,      32768u,      65535u,      50331649u,   201326593u,
          0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
      in[i] = (i < std::size(kSpecial))
                  ? kSpecial[i]
                  : static_cast<uint32_t>(i * 2654435761u);
    } else if constexpr (std::is_same_v<From, int64_t>) {
      constexpr int64_t kSpecial[] = {0,
                                      1,
                                      -1,
                                      50331649,
                                      201326593,
                                      (1LL << 40) + (1LL << 39) + 1,
                                      (1LL << 40) + (1LL << 39)};
      in[i] = (i < std::size(kSpecial)) ? kSpecial[i]
                                        : static_cast<int64_t>(i % 64);
    } else if constexpr (std::is_same_v<From, uint64_t>) {
      constexpr uint64_t kSpecial[] = {0ULL,
                                       1ULL,
                                       50331649ULL,
                                       201326593ULL,
                                       (1ULL << 40) + (1ULL << 39) + 1ULL,
                                       (1ULL << 40) + (1ULL << 39)};
      in[i] = (i < std::size(kSpecial)) ? kSpecial[i]
                                        : static_cast<uint64_t>(i % 64);
    } else {
      // Non-negative for unsigned integer targets (avoiding C++ float-to-uint
      // negative UB); signed & negative values for signed/float targets.
      const float v =
          (std::is_unsigned_v<To> && !std::is_same_v<To, bool_t>)
              ? static_cast<float>(i % 64) * 0.5f
              : (static_cast<float>(static_cast<int>(i % 64) - 32) * 0.5f);
      if constexpr (std::is_same_v<From, complex64_t>) {
        in[i] = complex64_t{v, -v};
      } else if constexpr (std::is_same_v<From, complex128_t>) {
        in[i] = complex128_t{static_cast<double>(v), -static_cast<double>(v)};
      } else if constexpr (std::is_same_v<From, json_t>) {
        in[i] = static_cast<double>(v);
      } else {
        in[i] = static_cast<From>(v);
      }
    }
  }
}

struct MultiBlockConversionTestCase {
  DataType from_dtype;
  DataType to_dtype;
  void (*populate_inputs)(void* in, size_t n_elems);
  bool (*populate_expected)(const void* in, void* expected, size_t n_elems);
};

template <typename From, typename To>
MultiBlockConversionTestCase MakeMultiBlockConversionTestCase() {
  return {
      dtype_v<From>,
      dtype_v<To>,
      [](void* in, size_t n) {
        PopulateMultiBlockTestInputs<From, To>(static_cast<From*>(in), n);
      },
      &PopulateExpectedElements<From, To>,
  };
}

std::ostream& operator<<(std::ostream& os,
                         const MultiBlockConversionTestCase& tc) {
  return os << tc.from_dtype << "_to_" << tc.to_dtype;
}

template <typename From>
void AddMultiBlockConversionTestCasesForSource(
    std::vector<MultiBlockConversionTestCase>* cases) {
  tensorstore::MapCanonicalDataTypes([cases](auto d) {
    using To = typename decltype(d)::Element;
    if constexpr ((DataTypeConversionTraits<From, To>::flags &
                   (kSupported | kCanReinterpretCast)) == kSupported &&
                  !std::is_same_v<From, To>) {
      cases->push_back(MakeMultiBlockConversionTestCase<From, To>());
    }
    return 0;
  });
}

std::vector<MultiBlockConversionTestCase> GetMultiBlockConversionTestCases() {
  std::vector<MultiBlockConversionTestCase> cases;
  AddMultiBlockConversionTestCasesForSource<int16_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<uint16_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<float16_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<bfloat16_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<float32_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<float64_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<complex64_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<complex128_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<int32_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<uint32_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<int64_t>(&cases);
  AddMultiBlockConversionTestCasesForSource<uint64_t>(&cases);

  cases.push_back(MakeMultiBlockConversionTestCase<json_t, float16_t>());
  cases.push_back(MakeMultiBlockConversionTestCase<json_t, bfloat16_t>());
  cases.push_back(MakeMultiBlockConversionTestCase<json_t, float32_t>());
  cases.push_back(MakeMultiBlockConversionTestCase<json_t, float64_t>());
  cases.push_back(MakeMultiBlockConversionTestCase<json_t, float8_e4m3fn_t>());
  cases.push_back(MakeMultiBlockConversionTestCase<json_t, float4_e2m1fn_t>());
  return cases;
}

class MultiBlockBufferKindConversionsTest
    : public ::testing::TestWithParam<MultiBlockConversionTestCase> {};

TEST_P(MultiBlockBufferKindConversionsTest, AllBufferKindsAndReference) {
  const auto& param = GetParam();
  constexpr size_t kNum = 136;  // Multiple 64-element blocks + tail, 2 rows.
  auto in = AllocateArray({static_cast<Index>(kNum)}, tensorstore::c_order,
                          tensorstore::default_init, param.from_dtype);
  param.populate_inputs(in.data(), kNum);

  auto res =
      RunAllBufferKinds(param.from_dtype, param.to_dtype, in.data(), kNum);
  ASSERT_EQ(res.flags & kSupported, kSupported);
  ASSERT_TRUE(res.copy_strided_ok) << res.status;
  ASSERT_TRUE(res.contiguous_ok) << res.status;
  ASSERT_TRUE(res.strided_ok) << res.status;
  ASSERT_TRUE(res.indexed_ok) << res.status;

  EXPECT_THAT(res.strided_view(), MatchesConvertedArray(res.contiguous_view()));
  EXPECT_THAT(res.indexed_view(), MatchesConvertedArray(res.contiguous_view()));

  auto expected = AllocateArray({res.rows, res.cols}, tensorstore::c_order,
                                tensorstore::default_init, param.to_dtype);
  if (param.populate_expected(in.data(), expected.data(), kNum)) {
    EXPECT_THAT(res.contiguous_view(), MatchesConvertedArray(expected));
  }
}

INSTANTIATE_TEST_SUITE_P(
    DataTypeConversion, MultiBlockBufferKindConversionsTest,
    ::testing::ValuesIn(GetMultiBlockConversionTestCases()),
    [](const ::testing::TestParamInfo<MultiBlockConversionTestCase>& info) {
      return absl::StrFormat("%v_to_%v", info.param.from_dtype,
                             info.param.to_dtype);
    });

struct JsonToFloatErrorParam {
  DataType to_dtype;
  IterationBufferKind kind;
};

class JsonToFloatSecondBlockErrorTest
    : public ::testing::TestWithParam<JsonToFloatErrorParam> {};

TEST_P(JsonToFloatSecondBlockErrorTest, FailsInSecondBlock) {
  const auto& [to_dtype, kind] = GetParam();
  constexpr Index kRows = 2;
  constexpr Index kCols = 70;
  std::vector<json_t> json_in(kRows * kCols, json_t(1.5));
  json_in[kCols + 5] = "not_a_number";
  std::vector<Index> in_offsets(kRows * kCols);
  std::vector<Index> out_offsets(kRows * kCols);
  for (Index i = 0; i < kRows * kCols; ++i) {
    in_offsets[i] = i * static_cast<Index>(sizeof(json_t));
    out_offsets[i] = i * to_dtype->size;
  }
  auto out_buf = AllocateArray({kRows, kCols}, tensorstore::c_order,
                               tensorstore::default_init, to_dtype);
  const auto conv = GetDataTypeConverter(dtype_v<json_t>, to_dtype);
  absl::Status status;
  IterationBufferPointer from_ptr =
      (kind == IterationBufferKind::kIndexed)
          ? IterationBufferPointer{json_in.data(), kCols, in_offsets.data()}
          : IterationBufferPointer{json_in.data(),
                                   static_cast<Index>(sizeof(json_t)) * kCols,
                                   static_cast<Index>(sizeof(json_t))};
  IterationBufferPointer to_ptr =
      (kind == IterationBufferKind::kIndexed)
          ? IterationBufferPointer{out_buf.data(), kCols, out_offsets.data()}
          : IterationBufferPointer{out_buf.data(), to_dtype->size * kCols,
                                   to_dtype->size};
  EXPECT_FALSE((*conv.closure.function)[kind](
      conv.closure.context, {kRows, kCols}, from_ptr, to_ptr, &status));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    DataTypeConversion, JsonToFloatSecondBlockErrorTest,
    ::testing::Values(JsonToFloatErrorParam{dtype_v<float32_t>,
                                            IterationBufferKind::kContiguous},
                      JsonToFloatErrorParam{dtype_v<float32_t>,
                                            IterationBufferKind::kStrided},
                      JsonToFloatErrorParam{dtype_v<float32_t>,
                                            IterationBufferKind::kIndexed},
                      JsonToFloatErrorParam{dtype_v<bfloat16_t>,
                                            IterationBufferKind::kContiguous},
                      JsonToFloatErrorParam{dtype_v<bfloat16_t>,
                                            IterationBufferKind::kStrided},
                      JsonToFloatErrorParam{dtype_v<bfloat16_t>,
                                            IterationBufferKind::kIndexed},
                      JsonToFloatErrorParam{dtype_v<float64_t>,
                                            IterationBufferKind::kContiguous},
                      JsonToFloatErrorParam{dtype_v<float64_t>,
                                            IterationBufferKind::kStrided},
                      JsonToFloatErrorParam{dtype_v<float64_t>,
                                            IterationBufferKind::kIndexed},
                      JsonToFloatErrorParam{dtype_v<float8_e4m3fn_t>,
                                            IterationBufferKind::kContiguous},
                      JsonToFloatErrorParam{dtype_v<float8_e4m3fn_t>,
                                            IterationBufferKind::kStrided},
                      JsonToFloatErrorParam{dtype_v<float8_e4m3fn_t>,
                                            IterationBufferKind::kIndexed},
                      JsonToFloatErrorParam{dtype_v<float4_e2m1fn_t>,
                                            IterationBufferKind::kContiguous},
                      JsonToFloatErrorParam{dtype_v<float4_e2m1fn_t>,
                                            IterationBufferKind::kStrided},
                      JsonToFloatErrorParam{dtype_v<float4_e2m1fn_t>,
                                            IterationBufferKind::kIndexed}),
    [](const ::testing::TestParamInfo<JsonToFloatErrorParam>& info) {
      absl::string_view kind_str = "Contiguous";
      if (info.param.kind == IterationBufferKind::kStrided) {
        kind_str = "Strided";
      } else if (info.param.kind == IterationBufferKind::kIndexed) {
        kind_str = "Indexed";
      }
      return absl::StrFormat("%v_%s", info.param.to_dtype, kind_str);
    });

template <typename From, typename To, int kOffset = 0>
struct InPlaceConversionCase {
  using FromType = From;
  using ToType = To;
  static constexpr int offset = kOffset;
};

using InPlaceConversionTestTypes =
    ::testing::Types<InPlaceConversionCase<float32_t, int32_t, -20>,
                     InPlaceConversionCase<int32_t, float32_t, -20>,
                     InPlaceConversionCase<float32_t, uint32_t, 0>,
                     InPlaceConversionCase<float64_t, int64_t, -50>,
                     InPlaceConversionCase<float64_t, uint64_t, 0>,
                     InPlaceConversionCase<bfloat16_t, int16_t, -30>,
                     InPlaceConversionCase<int16_t, bfloat16_t, -30>,
                     InPlaceConversionCase<float16_t, int16_t, -30>,
                     InPlaceConversionCase<int16_t, float16_t, -30>>;

template <typename T>
class InPlaceStridedConversionTest : public ::testing::Test {};
TYPED_TEST_SUITE(InPlaceStridedConversionTest, InPlaceConversionTestTypes);

TYPED_TEST(InPlaceStridedConversionTest, MultiBlockStrided) {
  using From = typename TypeParam::FromType;
  using To = typename TypeParam::ToType;
  constexpr int kOffset = TypeParam::offset;
  constexpr Index kCount = 150;  // Multiple 64-element blocks + tail.
  static_assert(sizeof(From) == sizeof(To));
  alignas(std::max(alignof(From), alignof(To)))
      std::array<char, kCount * sizeof(From)>
          buf{};
  auto* from_ptr = reinterpret_cast<From*>(buf.data());
  auto* to_ptr = reinterpret_cast<To*>(buf.data());
  for (Index i = 0; i < kCount; ++i) {
    from_ptr[i] = static_cast<From>(static_cast<float>(i + kOffset));
  }
  const auto conv = GetDataTypeConverter(dtype_v<From>, dtype_v<To>);
  absl::Status status;
  ASSERT_TRUE((*conv.closure.function)[IterationBufferKind::kStrided](
      conv.closure.context, {1, kCount},
      IterationBufferPointer{buf.data(), Index(0),
                             static_cast<Index>(sizeof(From))},
      IterationBufferPointer{buf.data(), Index(0),
                             static_cast<Index>(sizeof(To))},
      &status))
      << status;
  for (Index i = 0; i < kCount; ++i) {
    EXPECT_EQ(static_cast<To>(static_cast<float>(i + kOffset)), to_ptr[i])
        << "i=" << i;
  }
}

TEST(DataTypeConversionTest, EmptyShapesConversion) {
  const auto f32_to_i32 =
      GetDataTypeConverter(dtype_v<float32_t>, dtype_v<int32_t>);
  absl::Status status;
  std::array<float32_t, 10> dummy_buf{};
  EXPECT_TRUE((*f32_to_i32.closure.function)[IterationBufferKind::kContiguous](
      f32_to_i32.closure.context, {0, 10},
      IterationBufferPointer{tensorstore::ByteStridedPointer<void>{}, Index(0),
                             static_cast<Index>(sizeof(float32_t))},
      IterationBufferPointer{tensorstore::ByteStridedPointer<void>{}, Index(0),
                             static_cast<Index>(sizeof(int32_t))},
      &status));
  EXPECT_TRUE((*f32_to_i32.closure.function)[IterationBufferKind::kContiguous](
      f32_to_i32.closure.context, {10, 0},
      IterationBufferPointer{dummy_buf.data(), Index(0),
                             static_cast<Index>(sizeof(float32_t))},
      IterationBufferPointer{dummy_buf.data(), Index(0),
                             static_cast<Index>(sizeof(int32_t))},
      &status));
}

TEST(DataTypeConversionTest, ZeroInnerStrideBroadcastConversion) {
  const auto f32_to_i32 =
      GetDataTypeConverter(dtype_v<float32_t>, dtype_v<int32_t>);
  absl::Status status;
  const float32_t broadcast_src = 7.0f;
  std::vector<int32_t> broadcast_dst(70, 0);
  ASSERT_TRUE((*f32_to_i32.closure.function)[IterationBufferKind::kStrided](
      f32_to_i32.closure.context, {1, 70},
      IterationBufferPointer{const_cast<float32_t*>(&broadcast_src), Index(0),
                             Index(0)},
      IterationBufferPointer{broadcast_dst.data(), Index(0),
                             static_cast<Index>(sizeof(int32_t))},
      &status));
  EXPECT_THAT(broadcast_dst, ::testing::Each(7));
}

TEST(DataTypeConversionTest, NegativeInnerStrideConversion) {
  const auto f32_to_i32 =
      GetDataTypeConverter(dtype_v<float32_t>, dtype_v<int32_t>);
  absl::Status status;
  std::vector<float32_t> rev_src(70);
  for (int i = 0; i < 70; ++i) rev_src[i] = static_cast<float32_t>(i);
  std::vector<int32_t> rev_dst(70, -1);
  ASSERT_TRUE((*f32_to_i32.closure.function)[IterationBufferKind::kStrided](
      f32_to_i32.closure.context, {1, 70},
      IterationBufferPointer{rev_src.data() + 69, Index(0),
                             -static_cast<Index>(sizeof(float32_t))},
      IterationBufferPointer{rev_dst.data(), Index(0),
                             static_cast<Index>(sizeof(int32_t))},
      &status));
  for (int i = 0; i < 70; ++i) {
    EXPECT_EQ(69 - i, rev_dst[i]);
  }

  std::fill(rev_dst.begin(), rev_dst.end(), -1);
  ASSERT_TRUE((*f32_to_i32.closure.function)[IterationBufferKind::kStrided](
      f32_to_i32.closure.context, {1, 70},
      IterationBufferPointer{rev_src.data(), Index(0),
                             static_cast<Index>(sizeof(float32_t))},
      IterationBufferPointer{rev_dst.data() + 69, Index(0),
                             -static_cast<Index>(sizeof(int32_t))},
      &status));
  for (int i = 0; i < 70; ++i) {
    EXPECT_EQ(69 - i, rev_dst[i]);
  }
}

// Excludes `float8_e8m0fnu_t` (unsigned exponent-only format with no `-0.0` or
// sign bit) and `*fnuz` types (whose `0x80` encoding is `NaN`, not `-0.0`).
using FloatSpecialConversionTestTypes =
    ::testing::Types<float4_e2m1fn_t, float8_e3m4_t, float8_e4m3fn_t,
                     float8_e5m2_t, bfloat16_t, float16_t, float32_t,
                     float64_t>;

template <typename T>
class FloatSpecialConversionTest : public ::testing::Test {};
TYPED_TEST_SUITE(FloatSpecialConversionTest, FloatSpecialConversionTestTypes);

TYPED_TEST(FloatSpecialConversionTest, NegativeZeroSignbitPreservation) {
  using T = TypeParam;
  constexpr DataTypeConversionFlags kExpectedFlags =
      (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>)
          ? (kSupported | kSafeAndImplicit |
             (std::is_same_v<T, float32_t> ? kIdentity | kCanReinterpretCast
                                           : DataTypeConversionFlags{}))
          : kSupported;
  EXPECT_THAT(
      float32_t{-0.0f},
      ConvertsToMatching<T>(
          ::testing::Optional(::testing::ResultOf(
              [](T v) { return std::signbit(static_cast<float>(v)); }, true)),
          kExpectedFlags));
  // IEEE-754 `-0.0 == 0.0`, so converting `-0.0` to `bool` yields `false`.
  EXPECT_THAT(T{-0.0f}, ConvertsTo(bool_t{false}));
}

TYPED_TEST(FloatSpecialConversionTest, NaNFromFloat32AndToBool) {
  using T = TypeParam;
  if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
    constexpr DataTypeConversionFlags kExpectedFlags =
        (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>)
            ? (kSupported | kSafeAndImplicit |
               (std::is_same_v<T, float32_t> ? kIdentity | kCanReinterpretCast
                                             : DataTypeConversionFlags{}))
            : kSupported;
    for (float32_t f32_nan : {std::numeric_limits<float32_t>::quiet_NaN(),
                              -std::numeric_limits<float32_t>::quiet_NaN()}) {
      EXPECT_THAT(
          f32_nan,
          ConvertsToMatching<T>(
              ::testing::Optional(::testing::ResultOf(
                  [](T v) { return std::isnan(static_cast<float>(v)); }, true)),
              kExpectedFlags));
    }
    // In C++, `NaN != 0`, so converting `NaN` to `bool` yields `true`.
    EXPECT_THAT(std::numeric_limits<T>::quiet_NaN(), ConvertsTo(bool_t{true}));
  }
}

TYPED_TEST(FloatSpecialConversionTest, InfinityToBool) {
  using T = TypeParam;
  if constexpr (std::numeric_limits<T>::has_infinity) {
    EXPECT_THAT(std::numeric_limits<T>::infinity(), ConvertsTo(bool_t{true}));
    EXPECT_THAT(-std::numeric_limits<T>::infinity(), ConvertsTo(bool_t{true}));
  }
}

using AllIntegerTargetTestTypes =
    ::testing::Types<int2_t, int4_t, int8_t, uint8_t, int16_t, uint16_t,
                     int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
class Float8ToIntegerNaNConversionTest : public ::testing::Test {};
TYPED_TEST_SUITE(Float8ToIntegerNaNConversionTest, AllIntegerTargetTestTypes);

TYPED_TEST(Float8ToIntegerNaNConversionTest, NaNConvertsToZero) {
  // Unlike C++ `static_cast<Int>(NaN)` (which is UB), `SaturatedFloatToInt`
  // defines `NaN -> 0` for all integer targets.
  using Int = TypeParam;
  EXPECT_THAT(std::numeric_limits<float8_e4m3fn_t>::quiet_NaN(),
              ConvertsTo(Int{0}));
}

// Excludes `int2_t` and `int4_t` (covered in `#if 0` below).
using StandardIntegerSaturationTestTypes =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                     int64_t, uint64_t>;

template <typename T>
class Float8ToIntegerSaturationTest : public ::testing::Test {};
TYPED_TEST_SUITE(Float8ToIntegerSaturationTest,
                 StandardIntegerSaturationTestTypes);

TYPED_TEST(Float8ToIntegerSaturationTest, InfinitySaturatesToBounds) {
  // Out-of-range / infinite 1-byte floats saturate to `[min(), max()]` (so
  // `-Inf` clamps to `0` on unsigned integer targets).
  using Int = TypeParam;
  EXPECT_THAT(std::numeric_limits<float8_e5m2_t>::infinity(),
              ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(-std::numeric_limits<float8_e5m2_t>::infinity(),
              ConvertsTo(std::numeric_limits<Int>::min()));
}

#if 0
// Fails at HEAD (`-1` instead of `max()` and `0` instead of `min()`):
// `SaturatedFloatToInt` clamps to `int8_t` bounds (`[-128, 127]`) before
// `static_cast<IntPadded<N>>`, which masks to the low `N` bits (`0x7f -> -1`,
// `0x80 -> 0`) rather than saturating to `[-2^(N-1), 2^(N-1)-1]`.
using SubByteIntegerSaturationTestTypes = ::testing::Types<int2_t, int4_t>;

template <typename T>
class Float8ToSubByteIntegerSaturationTest : public ::testing::Test {};
TYPED_TEST_SUITE(Float8ToSubByteIntegerSaturationTest,
                 SubByteIntegerSaturationTestTypes);

TYPED_TEST(Float8ToSubByteIntegerSaturationTest, InfinitySaturatesToBounds) {
  using Int = TypeParam;
  EXPECT_THAT(std::numeric_limits<float8_e5m2_t>::infinity(),
              ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(-std::numeric_limits<float8_e5m2_t>::infinity(),
              ConvertsTo(std::numeric_limits<Int>::min()));
}

// Fails / triggers UBSan `float-cast-overflow` at HEAD: multi-byte floats
// (`float16_t`, `bfloat16_t`, `float32_t`, `float64_t`) convert to integers via
// raw `static_cast<Int>(*from)` rather than `SaturatedFloatToInt<Int>(*from)`.
template <typename T>
class MultiByteFloatToIntegerSaturationTest : public ::testing::Test {};
TYPED_TEST_SUITE(MultiByteFloatToIntegerSaturationTest,
                 AllIntegerTargetTestTypes);

TYPED_TEST(MultiByteFloatToIntegerSaturationTest,
           NaNAndInfinitySaturateToZeroAndBounds) {
  using Int = TypeParam;
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THAT(float16_t{kNaN}, ConvertsTo(Int{0}));
  EXPECT_THAT(bfloat16_t{kNaN}, ConvertsTo(Int{0}));
  EXPECT_THAT(float32_t{kNaN}, ConvertsTo(Int{0}));
  EXPECT_THAT(float64_t{kNaN}, ConvertsTo(Int{0}));

  constexpr float kInf = std::numeric_limits<float>::infinity();
  EXPECT_THAT(float16_t{kInf}, ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(float16_t{-kInf}, ConvertsTo(std::numeric_limits<Int>::min()));
  EXPECT_THAT(bfloat16_t{kInf}, ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(bfloat16_t{-kInf}, ConvertsTo(std::numeric_limits<Int>::min()));
  EXPECT_THAT(float32_t{kInf}, ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(float32_t{-kInf}, ConvertsTo(std::numeric_limits<Int>::min()));
  EXPECT_THAT(float64_t{kInf}, ConvertsTo(std::numeric_limits<Int>::max()));
  EXPECT_THAT(float64_t{-kInf}, ConvertsTo(std::numeric_limits<Int>::min()));
}
#endif

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

// 1-byte low-precision -> multi-byte upcasts
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, bfloat16_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float32_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float64_t>);
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, int32_t>);
BENCHMARK(BM_ConvertContiguous<float4_e2m1fn_t, bfloat16_t>);

// 1-byte -> 1-byte conversions
BENCHMARK(BM_ConvertContiguous<float8_e4m3fn_t, float8_e5m2_t>);
BENCHMARK(BM_ConvertContiguous<int4_t, float8_e4m3fn_t>);

// Multi-byte -> 1-byte downcasts
BENCHMARK(BM_ConvertContiguous<float32_t, float8_e4m3fn_t>);
BENCHMARK(BM_ConvertContiguous<bfloat16_t, float8_e4m3fn_t>);

// Standard multi-byte contiguous conversions
BENCHMARK(BM_ConvertContiguous<float32_t, bfloat16_t>);
BENCHMARK(BM_ConvertContiguous<bfloat16_t, float32_t>);
BENCHMARK(BM_ConvertContiguous<float32_t, float64_t>);
BENCHMARK(BM_ConvertContiguous<float32_t, int32_t>);
BENCHMARK(BM_ConvertContiguous<int32_t, float32_t>);

// Direct scalar static_cast<To>(From) loop outside data_type_conversion.cc
BENCHMARK(BM_ScalarStaticCast<float8_e4m3fn_t, float32_t>);
BENCHMARK(BM_ScalarStaticCast<float32_t, float8_e4m3fn_t>);

}  // namespace
