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

#include "tensorstore/driver/zarr/dtype.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/driver/zarr/metadata_testutil.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::bfloat16_t;
using ::tensorstore::complex128_t;
using ::tensorstore::complex64_t;
using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::endian;
using ::tensorstore::float16_t;
using ::tensorstore::float32_t;
using ::tensorstore::float64_t;
using ::tensorstore::Index;
using ::tensorstore::kInfIndex;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_zarr::ChooseBaseDType;
using ::tensorstore::internal_zarr::ParseBaseDType;
using ::tensorstore::internal_zarr::ParseDType;
using ::tensorstore::internal_zarr::ZarrDType;

void CheckBaseDType(std::string dtype, DataType r, endian e,
                    std::vector<Index> flexible_shape) {
  EXPECT_THAT(ParseBaseDType(dtype), ::testing::Optional(ZarrDType::BaseDType{
                                         dtype, r, e, flexible_shape}))
      << dtype;
}

TEST(ParseBaseDType, Success) {
  CheckBaseDType("|b1", dtype_v<bool>, endian::native, {});
  CheckBaseDType("<b1", dtype_v<bool>, endian::native, {});
  CheckBaseDType(">b1", dtype_v<bool>, endian::native, {});
  CheckBaseDType("|S150", dtype_v<char>, endian::native, {150});
  CheckBaseDType(">S150", dtype_v<char>, endian::native, {150});
  CheckBaseDType("<S150", dtype_v<char>, endian::native, {150});
  CheckBaseDType("|S9223372036854775807", dtype_v<char>, endian::native,
                 {9223372036854775807});
  CheckBaseDType("|V150", dtype_v<std::byte>, endian::native, {150});
  CheckBaseDType("<V150", dtype_v<std::byte>, endian::native, {150});
  CheckBaseDType(">V150", dtype_v<std::byte>, endian::native, {150});
  CheckBaseDType("|i1", dtype_v<std::int8_t>, endian::native, {});
  CheckBaseDType("<i1", dtype_v<std::int8_t>, endian::native, {});
  CheckBaseDType(">i1", dtype_v<std::int8_t>, endian::native, {});
  CheckBaseDType("|u1", dtype_v<std::uint8_t>, endian::native, {});
  CheckBaseDType("<u1", dtype_v<std::uint8_t>, endian::native, {});
  CheckBaseDType(">u1", dtype_v<std::uint8_t>, endian::native, {});
  CheckBaseDType("<i2", dtype_v<std::int16_t>, endian::little, {});
  CheckBaseDType("<i4", dtype_v<std::int32_t>, endian::little, {});
  CheckBaseDType("<i8", dtype_v<std::int64_t>, endian::little, {});
  CheckBaseDType("<u2", dtype_v<std::uint16_t>, endian::little, {});
  CheckBaseDType("<u4", dtype_v<std::uint32_t>, endian::little, {});
  CheckBaseDType("<u8", dtype_v<std::uint64_t>, endian::little, {});

  CheckBaseDType(">i2", dtype_v<std::int16_t>, endian::big, {});
  CheckBaseDType(">i4", dtype_v<std::int32_t>, endian::big, {});
  CheckBaseDType(">i8", dtype_v<std::int64_t>, endian::big, {});
  CheckBaseDType(">u2", dtype_v<std::uint16_t>, endian::big, {});
  CheckBaseDType(">u4", dtype_v<std::uint32_t>, endian::big, {});
  CheckBaseDType(">u8", dtype_v<std::uint64_t>, endian::big, {});

  CheckBaseDType("<f2", dtype_v<float16_t>, endian::little, {});
  CheckBaseDType("bfloat16", dtype_v<bfloat16_t>, endian::little, {});
  CheckBaseDType("<f4", dtype_v<float32_t>, endian::little, {});
  CheckBaseDType("<f8", dtype_v<float64_t>, endian::little, {});
  CheckBaseDType(">f2", dtype_v<float16_t>, endian::big, {});
  CheckBaseDType(">f4", dtype_v<float32_t>, endian::big, {});
  CheckBaseDType(">f8", dtype_v<float64_t>, endian::big, {});

  CheckBaseDType("<c8", dtype_v<complex64_t>, endian::little, {});
  CheckBaseDType("<c16", dtype_v<complex128_t>, endian::little, {});
  CheckBaseDType(">c8", dtype_v<complex64_t>, endian::big, {});
  CheckBaseDType(">c16", dtype_v<complex128_t>, endian::big, {});
}

TEST(ParseBaseDType, Failure) {
  EXPECT_THAT(ParseBaseDType(""),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Unsupported zarr dtype: \"\""));
  EXPECT_THAT(ParseBaseDType("|f4"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|f8"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|c8"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|c16"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|b2"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|i2"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<i9"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<u9"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<S"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|S999999999999999999999999999"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|S9223372036854775808"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|Sa"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("|S "),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<f5"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<c5"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<m8"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<M8"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseBaseDType("<X5"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

void CheckDType(const ::nlohmann::json& json, const ZarrDType& expected) {
  SCOPED_TRACE(json.dump());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(json));
  EXPECT_EQ(expected, dtype);
  // Check round trip.
  EXPECT_EQ(json, ::nlohmann::json(dtype));
}

TEST(ParseDType, SimpleStringBool) {
  CheckDType("|b1", ZarrDType{
                        /*.has_fields=*/false,
                        /*.fields=*/
                        {
                            {{
                                 /*.encoded_dtype=*/"|b1",
                                 /*.dtype=*/dtype_v<bool>,
                                 /*.endian=*/endian::native,
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
  CheckDType(::nlohmann::json::array_t{{"x", "|S10"}},
             ZarrDType{
                 /*.has_fields=*/true,
                 /*.fields=*/
                 {
                     {{
                          /*.encoded_dtype=*/"|S10",
                          /*.dtype=*/dtype_v<char>,
                          /*.endian=*/endian::native,
                          /*.flexible_shape=*/{10},
                      },
                      /*.outer_shape=*/{},
                      /*.name=*/"x",
                      /*.field_shape=*/{10},
                      /*.num_inner_elements=*/10,
                      /*.byte_offset=*/0,
                      /*.num_bytes=*/10},
                 },
                 /*.bytes_per_outer_element=*/10,
             });
}

TEST(ParseDType, TwoNamedFieldsCharAndInt) {
  CheckDType(
      ::nlohmann::json::array_t{{"x", "|S10", {2, 3}}, {"y", "<i2", {5}}},
      ZarrDType{
          /*.has_fields=*/true,
          /*.fields=*/
          {
              {{
                   /*.encoded_dtype=*/"|S10",
                   /*.dtype=*/dtype_v<char>,
                   /*.endian=*/endian::native,
                   /*.flexible_shape=*/{10},
               },
               /*.outer_shape=*/{2, 3},
               /*.name=*/"x",
               /*.field_shape=*/{2, 3, 10},
               /*.num_inner_elements=*/10 * 2 * 3,
               /*.byte_offset=*/0,
               /*.num_bytes=*/10 * 2 * 3},
              {{
                   /*.encoded_dtype=*/"<i2",
                   /*.dtype=*/
                   dtype_v<std::int16_t>,
                   /*.endian=*/endian::little,
                   /*.flexible_shape=*/{},
               },
               /*.outer_shape=*/{5},
               /*.name=*/"y",
               /*.field_shape=*/{5},
               /*.num_inner_elements=*/5,
               /*.byte_offset=*/10 * 2 * 3,
               /*.num_bytes=*/2 * 5},
          },
          /*.bytes_per_outer_element=*/10 * 2 * 3 + 2 * 5,
      });
}

TEST(ParseDType, FieldSpecTooShort) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x"}}),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Error parsing value at position 0: "
                  "Expected array of size 2 or 3, but received: \\[\"x\"\\]"));
}

TEST(ParseDType, FieldSpecTooLong) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "<i2", {2, 3}, 5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Expected array of size 2 or 3, but received: "
                            "\\[\"x\",\"<i2\",\\[2,3\\],5\\]"));
}

TEST(ParseDType, InvalidFieldName) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{3, "<i2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Error parsing value at position 0: "
                            "Expected non-empty string, but received: 3"));
}

TEST(ParseDType, EmptyFieldName) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"", "<i2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Error parsing value at position 0: "
                            "Expected non-empty string, but received: \"\""));
}

TEST(ParseDType, DuplicateFieldName) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "<i2"}, {"x", "<u2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Field name \"x\" occurs more than once"));
}

TEST(ParseDType, NonStringFieldBaseDType) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", 3}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Error parsing value at position 1: "
                            "Expected string, but received: 3"));
}

TEST(ParseDType, InvalidFieldBaseDType) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "<X2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Error parsing value at position 1: "
                            "Unsupported zarr dtype: \"<X2\""));
}

TEST(ParseDType, ProductOfDimensionsOverflow) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{
                  {"x", "|i1", {kInfIndex, kInfIndex}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Product of dimensions .* is too large"));
}

TEST(ParseDType, FieldSizeInBytesOverflow) {
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "<f8", {kInfIndex}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Field size in bytes is too large"));
}

TEST(ParseDType, BytesPerOuterElementOverflow) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "<i2", {kInfIndex}},
                                           {"y", "<i2", {kInfIndex}}}),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Total number of bytes per outer array element is too large"));
}

TEST(ChooseBaseDTypeTest, RoundTrip) {
  constexpr tensorstore::DataType kSupportedDataTypes[] = {
      dtype_v<bool>,
      dtype_v<uint8_t>,
      dtype_v<uint16_t>,
      dtype_v<uint32_t>,
      dtype_v<uint64_t>,
      dtype_v<int8_t>,
      dtype_v<int16_t>,
      dtype_v<int32_t>,
      dtype_v<int64_t>,
      dtype_v<tensorstore::float16_t>,
      dtype_v<tensorstore::bfloat16_t>,
      dtype_v<tensorstore::float32_t>,
      dtype_v<tensorstore::float64_t>,
      dtype_v<tensorstore::complex64_t>,
      dtype_v<tensorstore::complex128_t>,
  };
  for (auto dtype : kSupportedDataTypes) {
    SCOPED_TRACE(tensorstore::StrCat("dtype=", dtype));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_zarr_dtype,
                                     ChooseBaseDType(dtype));
    EXPECT_EQ(dtype, base_zarr_dtype.dtype);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto parsed, ParseBaseDType(base_zarr_dtype.encoded_dtype));
    EXPECT_EQ(dtype, parsed.dtype);
    EXPECT_EQ(base_zarr_dtype.endian, parsed.endian);
    EXPECT_EQ(base_zarr_dtype.flexible_shape, parsed.flexible_shape);
    EXPECT_EQ(base_zarr_dtype.encoded_dtype, parsed.encoded_dtype);
  }
}

TEST(ChooseBaseDTypeTest, Invalid) {
  struct X {};
  EXPECT_THAT(ChooseBaseDType(dtype_v<X>),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Data type not supported: .*"));
  EXPECT_THAT(ChooseBaseDType(dtype_v<tensorstore::string_t>),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Data type not supported: string"));
}

}  // namespace
