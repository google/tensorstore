// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/driver/zarr3/dtype.h"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::kInfIndex;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_zarr3::ChooseBaseDType;
using ::tensorstore::internal_zarr3::ParseBaseDType;
using ::tensorstore::internal_zarr3::ParseDType;
using ::tensorstore::internal_zarr3::ZarrDType;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

void CheckBaseDType(std::string dtype, DataType r,
                    std::vector<Index> flexible_shape) {
  EXPECT_THAT(ParseBaseDType(dtype), ::testing::Optional(ZarrDType::BaseDType{
                                         dtype, r, flexible_shape}))
      << dtype;
}

TEST(ParseBaseDType, Success) {
  CheckBaseDType("bool", dtype_v<bool>, {});
  CheckBaseDType("int8", dtype_v<int8_t>, {});
  CheckBaseDType("uint8", dtype_v<uint8_t>, {});
  CheckBaseDType("int16", dtype_v<int16_t>, {});
  CheckBaseDType("uint16", dtype_v<uint16_t>, {});
  CheckBaseDType("int32", dtype_v<int32_t>, {});
  CheckBaseDType("uint32", dtype_v<uint32_t>, {});
  CheckBaseDType("int64", dtype_v<int64_t>, {});
  CheckBaseDType("uint64", dtype_v<uint64_t>, {});
  CheckBaseDType("float16", dtype_v<tensorstore::dtypes::float16_t>, {});
  CheckBaseDType("bfloat16", dtype_v<tensorstore::dtypes::bfloat16_t>, {});
  CheckBaseDType("float32", dtype_v<tensorstore::dtypes::float32_t>, {});
  CheckBaseDType("float64", dtype_v<tensorstore::dtypes::float64_t>, {});
  CheckBaseDType("complex64", dtype_v<tensorstore::dtypes::complex64_t>, {});
  CheckBaseDType("complex128", dtype_v<tensorstore::dtypes::complex128_t>, {});
  CheckBaseDType("r8", dtype_v<tensorstore::dtypes::byte_t>, {1});
  CheckBaseDType("r16", dtype_v<tensorstore::dtypes::byte_t>, {2});
  CheckBaseDType("r64", dtype_v<tensorstore::dtypes::byte_t>, {8});
}

TEST(ParseBaseDType, Failure) {
  EXPECT_THAT(
      ParseBaseDType(""),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("data type is not one of the supported data types")));
  EXPECT_THAT(ParseBaseDType("float"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("string"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<i4"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("r"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("data type is invalid; expected r<N>")));
  EXPECT_THAT(ParseBaseDType("r7"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("data type is invalid; expected r<N>")));
  EXPECT_THAT(ParseBaseDType("r0"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("data type is invalid; expected r<N>")));
}

void CheckDType(const ::nlohmann::json& json, const ZarrDType& expected) {
  SCOPED_TRACE(json.dump());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(json));
  EXPECT_EQ(expected, dtype);
  // Check round trip.
  EXPECT_EQ(json, ::nlohmann::json(dtype));
}

TEST(ParseDType, SimpleStringBool) {
  CheckDType("bool", ZarrDType{
                         /*.has_fields=*/false,
                         /*.fields=*/
                         {
                             {{
                                  /*.encoded_dtype=*/"bool",
                                  /*.dtype=*/dtype_v<bool>,
                                  /*.flexible_shape=*/{},
                              },
                              /*.outer_shape=*/{},
                              /*.name=*/"",
                              /*.field_shape=*/{},
                              /*.num_inner_elements=*/1,
                              /*.byte_offset=*/0,
                              /*.num_bytes=*/1},
                         },
                         /*.bytes_per_outer_element=*/1,
                     });
}

TEST(ParseDType, SingleNamedFieldChar) {
  // Zarr 3 doesn't support fixed size strings natively in core, so we use uint8 for testing bytes
  CheckDType(::nlohmann::json::array_t{{"x", "uint8"}},
             ZarrDType{
                 /*.has_fields=*/true,
                 /*.fields=*/
                 {
                     {{
                          /*.encoded_dtype=*/"uint8",
                          /*.dtype=*/dtype_v<uint8_t>,
                          /*.flexible_shape=*/{},
                      },
                      /*.outer_shape=*/{},
                      /*.name=*/"x",
                      /*.field_shape=*/{},
                      /*.num_inner_elements=*/1,
                      /*.byte_offset=*/0,
                      /*.num_bytes=*/1},
                 },
                 /*.bytes_per_outer_element=*/1,
             });
}

TEST(ParseDType, TwoNamedFields) {
  CheckDType(
      ::nlohmann::json::array_t{{"x", "int8", {2, 3}}, {"y", "int16", {5}}},
      ZarrDType{
          /*.has_fields=*/true,
          /*.fields=*/
          {
              {{
                   /*.encoded_dtype=*/"int8",
                   /*.dtype=*/dtype_v<int8_t>,
                   /*.flexible_shape=*/{},
               },
               /*.outer_shape=*/{2, 3},
               /*.name=*/"x",
               /*.field_shape=*/{2, 3},
               /*.num_inner_elements=*/2 * 3,
               /*.byte_offset=*/0,
               /*.num_bytes=*/1 * 2 * 3},
              {{
                   /*.encoded_dtype=*/"int16",
                   /*.dtype=*/dtype_v<int16_t>,
                   /*.flexible_shape=*/{},
               },
               /*.outer_shape=*/{5},
               /*.name=*/"y",
               /*.field_shape=*/{5},
               /*.num_inner_elements=*/5,
               /*.byte_offset=*/1 * 2 * 3,
               /*.num_bytes=*/2 * 5},
          },
          /*.bytes_per_outer_element=*/1 * 2 * 3 + 2 * 5,
      });
}

TEST(ParseDType, FieldSpecTooShort) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x"}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Error parsing value at position 0: "
                    "Expected array of size 2 or 3, but received: [\"x\"]")));
}

TEST(ParseDType, FieldSpecTooLong) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "int16", {2, 3}, 5}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Error parsing value at position 0: "
                    "Expected array of size 2 or 3, but received: "
                    "[\"x\",\"int16\",[2,3],5]")));
}

TEST(ParseDType, InvalidFieldName) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{3, "int16"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Error parsing value at position 0: "
                         "Error parsing value at position 0: "
                         "Expected non-empty string, but received: 3")));
}

TEST(ParseDType, EmptyFieldName) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"", "int16"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Error parsing value at position 0: "
                         "Error parsing value at position 0: "
                         "Expected non-empty string, but received: \"\"")));
}

TEST(ParseDType, DuplicateFieldName) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "int16"}, {"x", "uint16"}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Field name \"x\" occurs more than once")));
}

TEST(ParseDType, NonStringFieldBaseDType) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", 3}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Error parsing value at position 0: "
                                 "Error parsing value at position 1: "
                                 "Expected string, but received: 3")));
}

TEST(ParseDType, InvalidFieldBaseDType) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "unknown"}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Error parsing value at position 0: "
                                 "Error parsing value at position 1: "
                                 "unknown data type is not one of the "
                                 "supported data types")));
}

TEST(ParseDType, ProductOfDimensionsOverflow) {
  EXPECT_THAT(
      ParseDType(
          ::nlohmann::json::array_t{{"x", "int8", {kInfIndex, kInfIndex}}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               MatchesRegex(".*Product of dimensions .* is too large.*")));
}

TEST(ParseDType, FieldSizeInBytesOverflow) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "float64", {kInfIndex}}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Field size in bytes is too large")));
}

TEST(ParseDType, BytesPerOuterElementOverflow) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "int16", {kInfIndex}},
                                           {"y", "int16", {kInfIndex}}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Total number of bytes per outer array element is too large")));
}

TEST(ChooseBaseDTypeTest, RoundTrip) {
  constexpr tensorstore::DataType kSupportedDataTypes[] = {
      dtype_v<bool>, dtype_v<uint8_t>, dtype_v<uint16_t>, dtype_v<uint32_t>,
      dtype_v<uint64_t>, dtype_v<int8_t>, dtype_v<int16_t>,
      dtype_v<int32_t>,  dtype_v<int64_t>,
      dtype_v<tensorstore::dtypes::bfloat16_t>,
      dtype_v<tensorstore::dtypes::float16_t>,
      dtype_v<tensorstore::dtypes::float32_t>,
      dtype_v<tensorstore::dtypes::float64_t>,
      dtype_v<tensorstore::dtypes::complex64_t>,
      dtype_v<tensorstore::dtypes::complex128_t>,
      dtype_v<tensorstore::dtypes::byte_t>,
      dtype_v<tensorstore::dtypes::char_t>,
  };
  for (auto dtype : kSupportedDataTypes) {
    SCOPED_TRACE(tensorstore::StrCat("dtype=", dtype));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_zarr_dtype,
                                     ChooseBaseDType(dtype));
    // byte_t and char_t both encode as r8, which parses back to byte_t
    DataType expected_dtype = dtype;
    if (dtype == dtype_v<tensorstore::dtypes::char_t>) {
      expected_dtype = dtype_v<tensorstore::dtypes::byte_t>;
    }
    EXPECT_EQ(expected_dtype, base_zarr_dtype.dtype);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto parsed, ParseBaseDType(base_zarr_dtype.encoded_dtype));
    EXPECT_EQ(expected_dtype, parsed.dtype);
    EXPECT_EQ(base_zarr_dtype.flexible_shape, parsed.flexible_shape);
    EXPECT_EQ(base_zarr_dtype.encoded_dtype, parsed.encoded_dtype);
  }
}

TEST(ChooseBaseDTypeTest, Invalid) {
  struct X {};
  EXPECT_THAT(ChooseBaseDType(dtype_v<X>),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Data type not supported")));
  EXPECT_THAT(ChooseBaseDType(dtype_v<::tensorstore::dtypes::string_t>),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Data type not supported: string")));
}

}  // namespace
