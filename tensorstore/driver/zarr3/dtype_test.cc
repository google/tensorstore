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
#include "absl/strings/str_cat.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_zarr3::ChooseBaseDType;
using ::tensorstore::internal_zarr3::ParseBaseDType;
using ::tensorstore::internal_zarr3::ParseDType;
using ::tensorstore::internal_zarr3::ZarrDType;
using ::testing::HasSubstr;
using ::tensorstore::IsOkAndHolds;

// Matcher to check if a string parses successfully to a specific ZarrDType::BaseDType.
MATCHER_P2(ParsesAsBaseDType, expected_data_type, expected_flexible_shape, "") {
  auto parsed = ParseBaseDType(arg);
  return ExplainMatchResult(
      ::testing::Optional(::testing::AllOf(
          ::testing::Field("encoded_dtype", &ZarrDType::BaseDType::encoded_dtype,
                           arg),
          ::testing::Field("dtype", &ZarrDType::BaseDType::dtype,
                           expected_data_type),
          ::testing::Field("flexible_shape",
                           &ZarrDType::BaseDType::flexible_shape,
                           expected_flexible_shape))),
      parsed, result_listener);
}

// Helper to add a field to ZarrDType, computing byte_offset and updating
// bytes_per_outer_element. The byte_offset in the passed field is ignored and
// computed based on the current bytes_per_outer_element. If the field has a
// non-empty name, has_fields is set to true.
void AddFieldToZarrDType(ZarrDType& dtype, ZarrDType::Field field) {
  field.byte_offset = dtype.bytes_per_outer_element;
  dtype.bytes_per_outer_element += field.num_bytes;
  if (!field.name.empty()) {
    dtype.has_fields = true;
  }
  dtype.fields.push_back(std::move(field));
}

TEST(ParseBaseDType, Success) {
  EXPECT_THAT("bool", ParsesAsBaseDType(dtype_v<bool>, std::vector<Index>{}));
  EXPECT_THAT("int8", ParsesAsBaseDType(dtype_v<int8_t>, std::vector<Index>{}));
  EXPECT_THAT("uint8", ParsesAsBaseDType(dtype_v<uint8_t>, std::vector<Index>{}));
  EXPECT_THAT("int16", ParsesAsBaseDType(dtype_v<int16_t>, std::vector<Index>{}));
  EXPECT_THAT("uint16", ParsesAsBaseDType(dtype_v<uint16_t>, std::vector<Index>{}));
  EXPECT_THAT("int32", ParsesAsBaseDType(dtype_v<int32_t>, std::vector<Index>{}));
  EXPECT_THAT("uint32", ParsesAsBaseDType(dtype_v<uint32_t>, std::vector<Index>{}));
  EXPECT_THAT("int64", ParsesAsBaseDType(dtype_v<int64_t>, std::vector<Index>{}));
  EXPECT_THAT("uint64", ParsesAsBaseDType(dtype_v<uint64_t>, std::vector<Index>{}));
  EXPECT_THAT("float16", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::float16_t>, std::vector<Index>{}));
  EXPECT_THAT("bfloat16", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::bfloat16_t>, std::vector<Index>{}));
  EXPECT_THAT("float32", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::float32_t>, std::vector<Index>{}));
  EXPECT_THAT("float64", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::float64_t>, std::vector<Index>{}));
  EXPECT_THAT("complex64", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::complex64_t>, std::vector<Index>{}));
  EXPECT_THAT("complex128", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::complex128_t>, std::vector<Index>{}));
  EXPECT_THAT("r8", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>, std::vector<Index>{1}));
  EXPECT_THAT("r16", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>, std::vector<Index>{2}));
  EXPECT_THAT("r64", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>, std::vector<Index>{8}));
  // Large N must parse (suffix may exceed 31-bit signed int; N fits in uint64).
  EXPECT_THAT("r1024", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>,
                                        std::vector<Index>{128}));
  EXPECT_THAT("r8388608", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>,
                                             std::vector<Index>{1048576}));
  EXPECT_THAT("r8589934592", ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>,
                                                std::vector<Index>{1073741824}));
  // Max r<N>: N = largest multiple of 8 in uint64_t (18446744073709551608 bits -> 2305843009213693951
  // bytes). Values above uint64_t fail parse; UINT64_MAX is not a multiple of 8.
  EXPECT_THAT(
      "r18446744073709551608",
      ParsesAsBaseDType(dtype_v<tensorstore::dtypes::byte_t>,
                        std::vector<Index>{2305843009213693951LL}));
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

TEST(ParseDType, SimpleStringBool) {
  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"bool",
                        /*.dtype=*/dtype_v<bool>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/1});
  EXPECT_THAT(ParseDType("bool"), IsOkAndHolds(expected));
}

TEST(ParseDType, SingleNamedFieldChar) {
  // Zarr 3 doesn't support fixed size strings natively in core, so we use uint8 for testing bytes
  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"uint8",
                        /*.dtype=*/dtype_v<uint8_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"x",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/1});
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "uint8"}}),
              IsOkAndHolds(expected));
}

TEST(ParseDType, TwoNamedFields) {
  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"int8",
                        /*.dtype=*/dtype_v<int8_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"x",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/1});
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"int16",
                        /*.dtype=*/dtype_v<int16_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"y",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/2});
  EXPECT_THAT(ParseDType(::nlohmann::json::array_t{{"x", "int8"}, {"y", "int16"}}),
              IsOkAndHolds(expected));
}

TEST(ParseDType, FieldSpecTooShort) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x"}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Error parsing value at position 0: "
                    "Expected array of size 2, but received: [\"x\"]")));
}

TEST(ParseDType, FieldSpecTooLong) {
  EXPECT_THAT(
      ParseDType(::nlohmann::json::array_t{{"x", "int16", {2, 3}}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Error parsing value at position 0: "
                    "Expected array of size 2, but received: "
                    "[\"x\",\"int16\",[2,3]]")));
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
    SCOPED_TRACE(absl::StrCat("dtype=", dtype));
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

TEST(ParseDType, StructNameNewFormat) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                  {{"name", "y"}, {"data_type", "int16"}}})}}}};

  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"uint8",
                        /*.dtype=*/dtype_v<uint8_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"x",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/1});
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"int16",
                        /*.dtype=*/dtype_v<int16_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"y",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/2});

  EXPECT_THAT(ParseDType(input), IsOkAndHolds(expected));

  // Verify output uses Zarr v3 spec format
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(input));
  ::nlohmann::json output = dtype;
  EXPECT_EQ(output["name"], "struct");
  EXPECT_TRUE(output.contains("configuration"));
  EXPECT_TRUE(output["configuration"]["fields"].is_array());
  EXPECT_EQ(output["configuration"]["fields"].size(), 2);
  EXPECT_EQ(output["configuration"]["fields"][0]["name"], "x");
  EXPECT_EQ(output["configuration"]["fields"][0]["data_type"], "uint8");
}

TEST(ParseDType, StructuredNameLegacy) {
  // "structured" (legacy) with tuple format fields
  ::nlohmann::json input = {
      {"name", "structured"},
      {"configuration",
       {{"fields", ::nlohmann::json::array({{"a", "float32"}})}}}};

  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"float32",
                        /*.dtype=*/dtype_v<tensorstore::dtypes::float32_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"a",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/4});

  EXPECT_THAT(ParseDType(input), IsOkAndHolds(expected));
}

TEST(ParseDType, ObjectFieldFormat) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", "field1"}, {"data_type", "uint32"}}})}}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(input));
  ASSERT_EQ(dtype.fields.size(), 1);
  EXPECT_EQ(dtype.fields[0].name, "field1");
  EXPECT_EQ(dtype.fields[0].encoded_dtype, "uint32");
  EXPECT_EQ(dtype.fields[0].dtype, dtype_v<uint32_t>);
}

TEST(ParseDType, StructuredWithTupleFields) {
  // "structured" (legacy) requires tuple format fields
  ::nlohmann::json input = {{"name", "structured"},
                            {"configuration",
                             {{"fields",
                               ::nlohmann::json::array({{"field1", "uint32"}})}}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(input));
  ASSERT_EQ(dtype.fields.size(), 1);
  EXPECT_EQ(dtype.fields[0].name, "field1");
  EXPECT_EQ(dtype.fields[0].dtype, dtype_v<uint32_t>);
}

TEST(ParseDType, StructWithTupleFieldsRejected) {
  // "struct" (new) must NOT accept tuple format fields
  ::nlohmann::json input = {{"name", "struct"},
                            {"configuration",
                             {{"fields",
                               ::nlohmann::json::array({{"field1", "uint32"}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("struct dtype requires fields as objects")));
}

TEST(ParseDType, StructuredWithObjectFieldsRejected) {
  // "structured" (legacy) must NOT accept object format fields
  ::nlohmann::json input = {
      {"name", "structured"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", "field1"}, {"data_type", "uint32"}}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("structured dtype requires fields as arrays")));
}

TEST(ParseDType, StructWithMixedFieldsRejected) {
  // "struct" with mixed formats should fail on the tuple field
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", "obj_field"}, {"data_type", "int8"}},
                                  {"tuple_field", "int16"}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("struct dtype requires fields as objects")));
}

TEST(ParseDType, ObjectFieldMissingName) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields", ::nlohmann::json::array({{{"data_type", "uint8"}}})}}}};

  EXPECT_THAT(
      ParseDType(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Field object must contain 'name' and 'data_type'")));
}

TEST(ParseDType, ObjectFieldMissingDataType) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields", ::nlohmann::json::array({{{"name", "x"}}})}}}};

  EXPECT_THAT(
      ParseDType(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Field object must contain 'name' and 'data_type'")));
}

TEST(ParseDType, ObjectFieldEmptyName) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", ""}, {"data_type", "uint8"}}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Field 'name' must be non-empty")));
}

TEST(ParseDType, StructEmptyFieldsRejected) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration", {{"fields", ::nlohmann::json::array()}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("struct data type requires at least one field")));
}

TEST(ParseDType, StructuredEmptyFieldsRejected) {
  ::nlohmann::json input = {
      {"name", "structured"},
      {"configuration", {{"fields", ::nlohmann::json::array()}}}};

  EXPECT_THAT(
      ParseDType(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("structured data type requires at least one field")));
}

TEST(ParseDType, BareArrayEmptyFieldsRejected) {
  ::nlohmann::json input = ::nlohmann::json::array();

  EXPECT_THAT(
      ParseDType(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("structured data type requires at least one field")));
}

TEST(ParseDType, NestedStructNotSupported) {
  // Nested struct types are valid per Zarr v3 spec but not supported by
  // TensorStore
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array(
             {{{"name", "point"},
               {"data_type",
                {{"name", "struct"},
                 {"configuration",
                  {{"fields",
                    ::nlohmann::json::array(
                        {{{"name", "x"}, {"data_type", "float32"}},
                         {{"name", "y"}, {"data_type", "float32"}}})}}}}}}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Nested struct types and extension data types "
                                 "with configuration are not supported")));
}

TEST(ParseDType, ExtensionDataTypeWithConfigNotSupported) {
  // Extension data types with configuration (e.g., numpy.datetime64) are valid
  // per Zarr v3 spec but not supported by TensorStore
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array(
             {{{"name", "timestamp"},
               {"data_type",
                {{"name", "numpy.datetime64"},
                 {"configuration", {{"unit", "s"}, {"scale_factor", 1}}}}}}})}}}};

  EXPECT_THAT(ParseDType(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Nested struct types and extension data types "
                                 "with configuration are not supported")));
}

TEST(ParseDType, SerializationUsesNewFormat) {
  ::nlohmann::json legacy_input =
      ::nlohmann::json::array_t{{"x", "uint8"}, {"y", "int16"}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(legacy_input));

  ::nlohmann::json output = dtype;
  EXPECT_EQ(output["name"], "struct");
  EXPECT_TRUE(output.contains("configuration"));
  EXPECT_TRUE(output["configuration"]["fields"].is_array());
  EXPECT_EQ(output["configuration"]["fields"].size(), 2);
  EXPECT_TRUE(output["configuration"]["fields"][0].is_object());
  EXPECT_EQ(output["configuration"]["fields"][0]["name"], "x");
  EXPECT_EQ(output["configuration"]["fields"][0]["data_type"], "uint8");
}

TEST(ParseDType, Raw64BitFieldShape) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields",
         ::nlohmann::json::array({{{"name", "x"}, {"data_type", "r64"}}})}}}};
  // r64 results in field_shape [8]
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(input));
  ASSERT_EQ(dtype.fields.size(), 1);
  EXPECT_THAT(dtype.fields[0].field_shape, ::testing::ElementsAre(8));
  EXPECT_EQ(dtype.fields[0].num_bytes, 8);
}

TEST(ParseDType, StructuredWithFieldShape) {
  ::nlohmann::json input = {
      {"name", "struct"},
      {"configuration",
       {{"fields", ::nlohmann::json::array(
                       {{{"name", "scalar"}, {"data_type", "int32"}},
                        {{"name", "array"}, {"data_type", "r16"}}})}}}};

  ZarrDType expected{};
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"int32",
                        /*.dtype=*/dtype_v<int32_t>,
                        /*.flexible_shape=*/{}},
                       /*.name=*/"scalar",
                       /*.field_shape=*/{},
                       /*.num_inner_elements=*/1,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/4});
  AddFieldToZarrDType(expected,
                      {{/*.encoded_dtype=*/"r16",
                        /*.dtype=*/dtype_v<tensorstore::dtypes::byte_t>,
                        /*.flexible_shape=*/{2}},
                       /*.name=*/"array",
                       /*.field_shape=*/{2},
                       /*.num_inner_elements=*/2,
                       /*.byte_offset=*/0,
                       /*.num_bytes=*/2});

  EXPECT_THAT(ParseDType(input), IsOkAndHolds(expected));
}

TEST(ParseDType, ManyFieldsOffsets) {
  // Verify that many fields are supported and byte offsets are computed correctly.
  // There is no explicit limit on the number of fields; the limit is bounded only
  // by memory and integer overflow in byte offset calculations.
  ::nlohmann::json::array_t fields;
  for (int i = 0; i < 1000; ++i) {
    fields.push_back({{"name", absl::StrCat("f", i)}, {"data_type", "int64"}});
  }
  ::nlohmann::json input = {{"name", "struct"},
                            {"configuration", {{"fields", fields}}}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto dtype, ParseDType(input));
  EXPECT_EQ(dtype.fields.size(), 1000);
  EXPECT_EQ(dtype.bytes_per_outer_element, 8000);
  EXPECT_EQ(dtype.fields[99].byte_offset, 99 * 8);
  EXPECT_EQ(dtype.fields[999].byte_offset, 999 * 8);
}

}  // namespace
