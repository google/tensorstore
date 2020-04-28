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

#include "tensorstore/driver/zarr/metadata.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array_testutil.h"
#include "tensorstore/driver/zarr/metadata_testutil.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using tensorstore::ContiguousLayoutOrder;
using tensorstore::DataTypeOf;
using tensorstore::MakeArray;
using tensorstore::MakeScalarArray;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::EncodeFillValue;
using tensorstore::internal_zarr::ParseDType;
using tensorstore::internal_zarr::ParseFillValue;
using tensorstore::internal_zarr::ParseMetadata;
using tensorstore::internal_zarr::ParseOrder;
using tensorstore::internal_zarr::ZarrMetadata;
using testing::ElementsAre;

TEST(ParseOrderTest, Success) {
  EXPECT_EQ(ContiguousLayoutOrder::c, ParseOrder("C"));
  EXPECT_EQ(ContiguousLayoutOrder::fortran, ParseOrder("F"));
}

TEST(ParseOrderTest, Failure) {
  EXPECT_THAT(ParseOrder("x"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected \"C\" or \"F\", but received: \"x\""));
  EXPECT_THAT(ParseOrder(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected \"C\" or \"F\", but received: 3"));
}

void TestFillValueRoundTrip(
    const ::nlohmann::json& dtype, const ::nlohmann::json& encoded_fill_value,
    std::vector<tensorstore::SharedArray<const void>> fill_values,
    std::vector<tensorstore::ArrayMatcher> fill_values_matcher) {
  auto parsed_dtype = ParseDType(dtype);
  ASSERT_EQ(Status(), GetStatus(parsed_dtype)) << "dtype=" << dtype.dump();
  EXPECT_THAT(
      ParseFillValue(encoded_fill_value, *parsed_dtype),
      ::testing::Optional(::testing::ElementsAreArray(fill_values_matcher)))
      << "dtype=" << dtype.dump()
      << ", encoded_fill_value=" << encoded_fill_value.dump()
      << ", fill_values=" << ::testing::PrintToString(fill_values);
  EXPECT_EQ(encoded_fill_value, EncodeFillValue(*parsed_dtype, fill_values))
      << "dtype=" << dtype.dump()
      << ", encoded_fill_value=" << encoded_fill_value.dump()
      << ", fill_values=" << ::testing::PrintToString(fill_values);
}

void TestFillValueRoundTrip(
    const ::nlohmann::json& dtype, const ::nlohmann::json& encoded_fill_value,
    std::vector<tensorstore::SharedArray<const void>> fill_values) {
  std::vector<tensorstore::ArrayMatcher> fill_values_matcher(
      fill_values.begin(), fill_values.end());
  return TestFillValueRoundTrip(dtype, encoded_fill_value, fill_values,
                                fill_values_matcher);
}

template <typename FloatType>
void TestFillValueRoundTripFloat(const ::nlohmann::json& dtype) {
  TestFillValueRoundTrip(dtype, 3.5, {MakeScalarArray<FloatType>(3.5)});
  TestFillValueRoundTrip(dtype, "Infinity",
                         {MakeScalarArray<FloatType>(INFINITY)});
  TestFillValueRoundTrip(dtype, "-Infinity",
                         {MakeScalarArray<FloatType>(-INFINITY)});
  TestFillValueRoundTrip(dtype, "NaN", {MakeScalarArray<FloatType>(NAN)},
                         {tensorstore::MatchesScalarArray<FloatType>(
                             ::testing::internal::FloatingEqMatcher<FloatType>(
                                 NAN, /*nan_eq_nan=*/true))});
}

template <typename FloatType>
void TestFillValueRoundTripComplex(const ::nlohmann::json& dtype) {
  using Complex = std::complex<FloatType>;
  TestFillValueRoundTrip(dtype, {3.5, 4.5},
                         {MakeScalarArray<Complex>({3.5, 4.5})});
  TestFillValueRoundTrip(dtype, {"Infinity", 4.5},
                         {MakeScalarArray<Complex>(Complex{INFINITY, 4.5})});
}

TEST(ParseFillValueTest, FloatingPointSuccess) {
  TestFillValueRoundTripFloat<float>("<f4");
  TestFillValueRoundTripFloat<float>(">f4");
  TestFillValueRoundTripFloat<double>("<f8");
  TestFillValueRoundTripFloat<double>(">f8");
}

TEST(ParseFillValueTest, ComplexSuccess) {
  TestFillValueRoundTripComplex<float>("<c8");
  TestFillValueRoundTripComplex<float>(">c8");
  TestFillValueRoundTripComplex<double>("<c16");
  TestFillValueRoundTripComplex<double>(">c16");
}

TEST(ParseFillValueTest, FloatingPointFailure) {
  EXPECT_THAT(ParseFillValue("x", ParseDType("<f4").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid floating-point value: \"x\""));
}

TEST(ParseFillValueTest, ComplexFailure) {
  EXPECT_THAT(
      ParseFillValue(3, ParseDType("<c8").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected 8 base64-encoded bytes, but received: 3"));
  EXPECT_THAT(
      ParseFillValue(::nlohmann::json::array_t{3}, ParseDType("<c8").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Array has length 1 but should have length 2"));
  EXPECT_THAT(ParseFillValue(::nlohmann::json::array_t{3, 4, 5},
                             ParseDType("<c8").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseFillValue({"x", "y"}, ParseDType("<c16").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing value at position 0: "
                            "Invalid floating-point value: \"x\""));
}

TEST(ParseFillValueTest, BoolSuccess) {
  TestFillValueRoundTrip("|b1", true, {MakeScalarArray<bool>(true)});
  TestFillValueRoundTrip("|b1", false, {MakeScalarArray<bool>(false)});
}

TEST(ParseFillValueTest, BoolFailure) {
  EXPECT_THAT(ParseFillValue("x", ParseDType("|b1").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected boolean, but received: \"x\""));
}

TEST(ParseFillValueTest, IntegerSuccess) {
  TestFillValueRoundTrip("|i1", -124, {MakeScalarArray<std::int8_t>(-124)});
  TestFillValueRoundTrip("|i1", 124, {MakeScalarArray<std::int8_t>(124)});
  TestFillValueRoundTrip("<i2", -31000,
                         {MakeScalarArray<std::int16_t>(-31000)});
  TestFillValueRoundTrip("<i2", 31000, {MakeScalarArray<std::int16_t>(31000)});
  TestFillValueRoundTrip("<i4", -310000000,
                         {MakeScalarArray<std::int32_t>(-310000000)});
  TestFillValueRoundTrip("<i4", 310000000,
                         {MakeScalarArray<std::int32_t>(310000000)});
  TestFillValueRoundTrip("<i8", -31000000000,
                         {MakeScalarArray<std::int64_t>(-31000000000)});
  TestFillValueRoundTrip("<i8", 31000000000,
                         {MakeScalarArray<std::int64_t>(31000000000)});

  TestFillValueRoundTrip("|u1", 124, {MakeScalarArray<std::uint8_t>(124)});
  TestFillValueRoundTrip("<u2", 31000, {MakeScalarArray<std::uint16_t>(31000)});
  TestFillValueRoundTrip("<u4", 310000000,
                         {MakeScalarArray<std::uint32_t>(310000000)});
  TestFillValueRoundTrip("<u8", 31000000000,
                         {MakeScalarArray<std::uint64_t>(31000000000)});
}

TEST(ParseFillValueTest, IntegerFailure) {
  EXPECT_THAT(ParseFillValue(500, ParseDType("|i1").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected integer in the range \\[-128, 127\\], "
                            "but received: 500"));
  EXPECT_THAT(ParseFillValue(500, ParseDType("|u1").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected integer in the range \\[0, 255\\], "
                            "but received: 500"));
  EXPECT_THAT(
      ParseFillValue(45000, ParseDType("<i2").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected integer in the range \\[-32768, 32767\\], "
                    "but received: 45000"));
  EXPECT_THAT(ParseFillValue(90000, ParseDType("<u2").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected integer in the range \\[0, 65535\\], "
                            "but received: 90000"));
  EXPECT_THAT(
      ParseFillValue("x", ParseDType("<i4").value()),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected integer in the range \\[-2147483648, 2147483647\\], "
          "but received: \"x\""));
  EXPECT_THAT(
      ParseFillValue("x", ParseDType("<u4").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected integer in the range \\[0, 4294967295\\], "
                    "but received: \"x\""));
  EXPECT_THAT(ParseFillValue("x", ParseDType("<i8").value()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected integer in the range "
                            "\\[-9223372036854775808, 9223372036854775807\\], "
                            "but received: \"x\""));
  EXPECT_THAT(
      ParseFillValue("x", ParseDType("<u8").value()),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected integer in the range \\[0, 18446744073709551615\\], "
          "but received: \"x\""));
}

TEST(ParseFillValueTest, Base64Success) {
  TestFillValueRoundTrip(
      "|S10",
      "YWJjZGVmZ2hpag==",  // Base64 encoding of "abcdefghij"
      {MakeArray<char>({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'})});
  TestFillValueRoundTrip(
      "|V10",
      "YWJjZGVmZ2hpag==",  // Base64 encoding of "abcdefghij"
      {MakeArray<std::byte>({std::byte('a'), std::byte('b'), std::byte('c'),
                             std::byte('d'), std::byte('e'), std::byte('f'),
                             std::byte('g'), std::byte('h'), std::byte('i'),
                             std::byte('j')})});
  TestFillValueRoundTrip(
      ::nlohmann::json::array_t{{"x", "<i2", {2}}, {"y", ">u4", {3}}},
      "x8/3X0mWAtIbOgwUzgpqFA==",
      {MakeArray<std::int16_t>({-12345, 24567}),
       MakeArray<std::uint32_t>({1234567890, 456789012, 3456789012})});
}

TEST(ParseFillValueTest, Base64Failure) {
  EXPECT_THAT(
      ParseFillValue("YWJjZGVmZ2hp",  // Base64 encoding of "abcdefghi"
                     ParseDType("|S10").value()),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected 10 base64-encoded bytes, but received: \"YWJjZGVmZ2hp\""));
  EXPECT_THAT(
      ParseFillValue("x", ParseDType("|S10").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected 10 base64-encoded bytes, but received: \"x\""));
  EXPECT_THAT(
      ParseFillValue(10, ParseDType("|S10").value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected 10 base64-encoded bytes, but received: 10"));
}

// Many of the following test cases are derived from:
// https://github.com/zarr-developers/zarr/blob/master/zarr/tests/test_meta.py

// Corresponds to the zarr test_encode_decode_array_1 test case.
TEST(EncodeDecodeMetadataTest, Array1) {
  absl::string_view metadata_text = R"(
{
        "chunks": [10],
        "compressor": {"id": "zlib", "level": 1},
        "dtype": "<f8",
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100],
        "zarr_format": 2
}
)";
  nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                           /*allow_exceptions=*/false);
  ASSERT_FALSE(j.is_discarded());
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(100));
  EXPECT_THAT(metadata.chunks, ElementsAre(10));

  EXPECT_FALSE(metadata.dtype.has_fields);
  EXPECT_EQ(1, metadata.dtype.fields.size());
  EXPECT_EQ(DataTypeOf<double>(), metadata.dtype.fields[0].data_type);
  EXPECT_FALSE(metadata.fill_values[0].valid());
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(8, metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("", metadata.dtype.fields[0].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {8}),
            metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {8}),
            metadata.chunk_layout.fields[0].decoded_chunk_layout);

  EXPECT_EQ(8, metadata.dtype.bytes_per_outer_element);
  EXPECT_EQ(80, metadata.chunk_layout.bytes_per_chunk);
  EXPECT_EQ(10, metadata.chunk_layout.num_outer_elements);
  EXPECT_EQ(ContiguousLayoutOrder::c, metadata.order);

  EXPECT_EQ(j, ::nlohmann::json(metadata));
}

// Corresponds to the zarr test_encode_decode_array_2 test case, except that
// "filters" are `null`.
TEST(EncodeDecodeMetadataTest, Array2) {
  absl::string_view metadata_text = R"(
{
        "chunks": [10, 10],
        "compressor": {
            "id": "blosc",
            "clevel": 3,
            "cname": "lz4",
            "shuffle": 2,
            "blocksize": 0
        },
        "dtype": [["a", "<i4"], ["b", "|S10"]],
        "fill_value": "AAAAAAAAAAAAAAAAAAA=",
        "filters": null,
        "order": "F",
        "shape": [100, 100],
        "zarr_format": 2
}
)";
  nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                           /*allow_exceptions=*/false);
  ASSERT_FALSE(j.is_discarded());
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(100, 100));
  EXPECT_THAT(metadata.chunks, ElementsAre(10, 10));
  EXPECT_TRUE(metadata.dtype.has_fields);
  EXPECT_EQ(2, metadata.dtype.fields.size());
  EXPECT_EQ(DataTypeOf<std::int32_t>(), metadata.dtype.fields[0].data_type);
  EXPECT_TRUE(metadata.fill_values[0].valid());
  EXPECT_EQ(metadata.fill_values[0], MakeScalarArray<std::int32_t>(0));
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(4, metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("a", metadata.dtype.fields[0].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10}, {14, 140}),
            metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10}, {4, 40}),
            metadata.chunk_layout.fields[0].decoded_chunk_layout);

  EXPECT_EQ(DataTypeOf<char>(), metadata.dtype.fields[1].data_type);
  EXPECT_TRUE(metadata.fill_values[1].valid());
  EXPECT_EQ(metadata.fill_values[1],
            tensorstore::MakeArray<char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(tensorstore::endian::native, metadata.dtype.fields[1].endian);
  EXPECT_THAT(metadata.dtype.fields[1].field_shape, ElementsAre(10));
  EXPECT_EQ(4, metadata.dtype.fields[1].byte_offset);
  EXPECT_EQ(10, metadata.dtype.fields[1].num_bytes);
  EXPECT_EQ("b", metadata.dtype.fields[1].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {14, 140, 1}),
            metadata.chunk_layout.fields[1].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {10, 100, 1}),
            metadata.chunk_layout.fields[1].decoded_chunk_layout);

  EXPECT_EQ(14, metadata.dtype.bytes_per_outer_element);
  EXPECT_EQ(14 * 10 * 10, metadata.chunk_layout.bytes_per_chunk);
  EXPECT_EQ(10 * 10, metadata.chunk_layout.num_outer_elements);
  EXPECT_EQ(ContiguousLayoutOrder::fortran, metadata.order);

  EXPECT_EQ(j, ::nlohmann::json(metadata));
}

// Corresponds to the zarr test_encode_decode_array_2 test case, except that
// "filters" are `null` and the "fill_value" is changed to not be all zero.
TEST(EncodeDecodeMetadataTest, Array2Modified) {
  absl::string_view metadata_text = R"(
{
        "chunks": [10, 10],
        "compressor": {
            "id": "blosc",
            "clevel": 3,
            "cname": "lz4",
            "shuffle": 2,
            "blocksize": 0
        },
        "dtype": [["a", "<i4"], ["b", "|S10"]],
        "fill_value": "Fc1bB2FiY2RlZmdoaWo=",
        "filters": null,
        "order": "F",
        "shape": [100, 100],
        "zarr_format": 2
}
)";
  nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                           /*allow_exceptions=*/false);
  ASSERT_FALSE(j.is_discarded());
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(100, 100));
  EXPECT_THAT(metadata.chunks, ElementsAre(10, 10));
  EXPECT_TRUE(metadata.dtype.has_fields);
  EXPECT_EQ(2, metadata.dtype.fields.size());
  EXPECT_EQ(DataTypeOf<std::int32_t>(), metadata.dtype.fields[0].data_type);
  EXPECT_TRUE(metadata.fill_values[0].valid());
  EXPECT_EQ(metadata.fill_values[0], MakeScalarArray<std::int32_t>(123456789));
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(4, metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("a", metadata.dtype.fields[0].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10}, {14, 140}),
            metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10}, {4, 40}),
            metadata.chunk_layout.fields[0].decoded_chunk_layout);

  EXPECT_EQ(DataTypeOf<char>(), metadata.dtype.fields[1].data_type);
  EXPECT_TRUE(metadata.fill_values[1].valid());
  EXPECT_EQ(metadata.fill_values[1],
            tensorstore::MakeArray<char>(
                {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}));
  EXPECT_EQ(tensorstore::endian::native, metadata.dtype.fields[1].endian);
  EXPECT_THAT(metadata.dtype.fields[1].field_shape, ElementsAre(10));
  EXPECT_EQ(4, metadata.dtype.fields[1].byte_offset);
  EXPECT_EQ(10, metadata.dtype.fields[1].num_bytes);
  EXPECT_EQ("b", metadata.dtype.fields[1].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {14, 140, 1}),
            metadata.chunk_layout.fields[1].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {10, 100, 1}),
            metadata.chunk_layout.fields[1].decoded_chunk_layout);

  EXPECT_EQ(14, metadata.dtype.bytes_per_outer_element);
  EXPECT_EQ(14 * 10 * 10, metadata.chunk_layout.bytes_per_chunk);
  EXPECT_EQ(10 * 10, metadata.chunk_layout.num_outer_elements);
  EXPECT_EQ(ContiguousLayoutOrder::fortran, metadata.order);

  EXPECT_EQ(j, ::nlohmann::json(metadata));
}

// Corresponds to the zarr test_encode_decode_array_structured test case.
TEST(EncodeDecodeMetadataTest, ArrayStructured) {
  absl::string_view metadata_text = R"(
{
        "chunks": [10],
        "compressor": {"id": "zlib", "level": 1},
        "dtype": [["f0", "<i8"], ["f1", "<f8", [10, 10]], ["f2", "|u1", [5, 10, 15]]],
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100],
        "zarr_format": 2
}
)";
  nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                           /*allow_exceptions=*/false);
  ASSERT_FALSE(j.is_discarded());
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(100));
  EXPECT_THAT(metadata.chunks, ElementsAre(10));
  EXPECT_TRUE(metadata.dtype.has_fields);
  EXPECT_EQ(3, metadata.dtype.fields.size());
  EXPECT_EQ(DataTypeOf<std::int64_t>(), metadata.dtype.fields[0].data_type);
  EXPECT_FALSE(metadata.fill_values[0].valid());
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(8, metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("f0", metadata.dtype.fields[0].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {1558}),
            metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {8}),
            metadata.chunk_layout.fields[0].decoded_chunk_layout);

  EXPECT_EQ(DataTypeOf<double>(), metadata.dtype.fields[1].data_type);
  EXPECT_FALSE(metadata.fill_values[1].valid());
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[1].endian);
  EXPECT_THAT(metadata.dtype.fields[1].field_shape, ElementsAre(10, 10));
  EXPECT_EQ(8, metadata.dtype.fields[1].byte_offset);
  EXPECT_EQ(800, metadata.dtype.fields[1].num_bytes);
  EXPECT_EQ("f1", metadata.dtype.fields[1].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {1558, 80, 8}),
            metadata.chunk_layout.fields[1].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({10, 10, 10}, {800, 80, 8}),
            metadata.chunk_layout.fields[1].decoded_chunk_layout);

  EXPECT_EQ(DataTypeOf<std::uint8_t>(), metadata.dtype.fields[2].data_type);
  EXPECT_FALSE(metadata.fill_values[2].valid());
  EXPECT_EQ(tensorstore::endian::native, metadata.dtype.fields[2].endian);
  EXPECT_THAT(metadata.dtype.fields[2].field_shape, ElementsAre(5, 10, 15));
  EXPECT_EQ(808, metadata.dtype.fields[2].byte_offset);
  EXPECT_EQ(750, metadata.dtype.fields[2].num_bytes);
  EXPECT_EQ("f2", metadata.dtype.fields[2].name);
  EXPECT_EQ(
      tensorstore::StridedLayoutView<>({10, 5, 10, 15}, {1558, 150, 15, 1}),
      metadata.chunk_layout.fields[2].encoded_chunk_layout);
  EXPECT_EQ(
      tensorstore::StridedLayoutView<>({10, 5, 10, 15}, {750, 150, 15, 1}),
      metadata.chunk_layout.fields[2].decoded_chunk_layout);

  EXPECT_EQ(1558, metadata.dtype.bytes_per_outer_element);
  EXPECT_EQ(10 * 1558, metadata.chunk_layout.bytes_per_chunk);
  EXPECT_EQ(10, metadata.chunk_layout.num_outer_elements);
  EXPECT_EQ(ContiguousLayoutOrder::c, metadata.order);

  EXPECT_EQ(j, ::nlohmann::json(metadata));
}

// Corresponds to the zarr test_encode_decode_fill_values_nan test case.
TEST(EncodeDecodeMetadataTest, FillValuesNan) {
  for (const auto& pair : std::vector<std::pair<double, std::string>>{
           {std::numeric_limits<double>::quiet_NaN(), "NaN"},
           {std::numeric_limits<double>::infinity(), "Infinity"},
           {-std::numeric_limits<double>::infinity(), "-Infinity"}}) {
    std::string metadata_text = R"(
{
            "chunks": [10],
            "compressor": {"id": "zlib", "level": 1},
            "dtype": "<f8",
            "fill_value": ")";
    metadata_text += pair.second;
    metadata_text += R"(",
            "filters": null,
            "order": "C",
            "shape": [100],
            "zarr_format": 2
}
)";
    nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                             /*allow_exceptions=*/false);
    ASSERT_FALSE(j.is_discarded());
    ZarrMetadata metadata;
    EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
    EXPECT_EQ(2, metadata.zarr_format);
    EXPECT_THAT(metadata.shape, ElementsAre(100));
    EXPECT_THAT(metadata.chunks, ElementsAre(10));

    EXPECT_FALSE(metadata.dtype.has_fields);
    EXPECT_EQ(1, metadata.dtype.fields.size());
    EXPECT_EQ(DataTypeOf<double>(), metadata.dtype.fields[0].data_type);
    EXPECT_TRUE(metadata.fill_values[0].valid());
    EXPECT_THAT(metadata.fill_values[0].shape(), ElementsAre());
    if (std::isnan(pair.first)) {
      EXPECT_TRUE(std::isnan(
          *static_cast<const double*>(metadata.fill_values[0].data())));
    } else {
      EXPECT_EQ(MakeScalarArray<double>(pair.first), metadata.fill_values[0]);
    }
    EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
    EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
    EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
    EXPECT_EQ(8, metadata.dtype.fields[0].num_bytes);
    EXPECT_EQ("", metadata.dtype.fields[0].name);
    EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {8}),
              metadata.chunk_layout.fields[0].encoded_chunk_layout);
    EXPECT_EQ(tensorstore::StridedLayoutView<>({10}, {8}),
              metadata.chunk_layout.fields[0].decoded_chunk_layout);

    EXPECT_EQ(8, metadata.dtype.bytes_per_outer_element);
    EXPECT_EQ(80, metadata.chunk_layout.bytes_per_chunk);
    EXPECT_EQ(10, metadata.chunk_layout.num_outer_elements);
    EXPECT_EQ(ContiguousLayoutOrder::c, metadata.order);

    EXPECT_EQ(j, ::nlohmann::json(metadata));
  }
}

template <typename T>
void EncodeDecodeMetadataTestArrayComplex(std::string zarr_dtype) {
  nlohmann::json j{{"chunks", {10, 10}},
                   {"compressor",
                    {{"id", "blosc"},
                     {"clevel", 3},
                     {"shuffle", 2},
                     {"blocksize", 0},
                     {"cname", "lz4"}}},
                   {"dtype", zarr_dtype},
                   {"fill_value", {"NaN", -1.0}},
                   {"filters", nullptr},
                   {"order", "F"},
                   {"shape", {100, 100}},
                   {"zarr_format", 2}};
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(100, 100));
  EXPECT_THAT(metadata.chunks, ElementsAre(10, 10));

  EXPECT_FALSE(metadata.dtype.has_fields);
  EXPECT_EQ(1, metadata.dtype.fields.size());
  ASSERT_EQ(DataTypeOf<T>(), metadata.dtype.fields[0].data_type);
  EXPECT_TRUE(metadata.fill_values[0].valid());
  EXPECT_THAT(metadata.fill_values[0].shape(), ElementsAre());
  EXPECT_TRUE(std::isnan(
      static_cast<const T*>(metadata.fill_values[0].data())->real()));
  EXPECT_EQ(-1.0f,
            static_cast<const T*>(metadata.fill_values[0].data())->imag());
  EXPECT_EQ(tensorstore::endian::little, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(sizeof(T), metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("", metadata.dtype.fields[0].name);
  EXPECT_EQ(
      tensorstore::StridedLayoutView<>({10, 10}, {sizeof(T), sizeof(T) * 10}),
      metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(
      tensorstore::StridedLayoutView<>({10, 10}, {sizeof(T), sizeof(T) * 10}),
      metadata.chunk_layout.fields[0].decoded_chunk_layout);
  EXPECT_EQ(sizeof(T), metadata.dtype.bytes_per_outer_element);
  EXPECT_EQ(sizeof(T) * 10 * 10, metadata.chunk_layout.bytes_per_chunk);
  EXPECT_EQ(10 * 10, metadata.chunk_layout.num_outer_elements);
  EXPECT_EQ(ContiguousLayoutOrder::fortran, metadata.order);
  EXPECT_EQ(j, ::nlohmann::json(metadata));
}

// Corresponds to the zarr test_encode_decode_array_complex test case.
TEST(EncodeDecodeMetadataTest, ArrayComplex8) {
  EncodeDecodeMetadataTestArrayComplex<tensorstore::complex64_t>("<c8");
}

// Corresponds to the zarr test_encode_decode_array_complex test case.
TEST(EncodeDecodeMetadataTest, ArrayComplex16) {
  EncodeDecodeMetadataTestArrayComplex<tensorstore::complex128_t>("<c16");
}

TEST(ParseMetadataTest, Simple) {
  absl::string_view metadata_text = R"(
{
    "chunks": [
        100
    ],
    "compressor": null,
    "dtype": "|i1",
    "fill_value": 0,
    "filters": null,
    "order": "F",
    "shape": [
        1111
    ],
    "zarr_format": 2
}
)";
  nlohmann::json j = nlohmann::json::parse(metadata_text, nullptr,
                                           /*allow_exceptions=*/false);
  ASSERT_FALSE(j.is_discarded());
  ZarrMetadata metadata;
  EXPECT_EQ(Status(), ParseMetadata(j, &metadata));
  EXPECT_EQ(2, metadata.zarr_format);
  EXPECT_THAT(metadata.shape, ElementsAre(1111));
  EXPECT_THAT(metadata.chunks, ElementsAre(100));
  EXPECT_EQ(1, metadata.dtype.fields.size());
  EXPECT_EQ(DataTypeOf<std::int8_t>(), metadata.dtype.fields[0].data_type);
  EXPECT_EQ(tensorstore::endian::native, metadata.dtype.fields[0].endian);
  EXPECT_THAT(metadata.dtype.fields[0].field_shape, ElementsAre());
  EXPECT_EQ(0, metadata.dtype.fields[0].byte_offset);
  EXPECT_EQ(1, metadata.dtype.fields[0].num_bytes);
  EXPECT_EQ("", metadata.dtype.fields[0].name);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({100}, {1}),
            metadata.chunk_layout.fields[0].encoded_chunk_layout);
  EXPECT_EQ(tensorstore::StridedLayoutView<>({100}, {1}),
            metadata.chunk_layout.fields[0].decoded_chunk_layout);
}

TEST(ParseMetadataTest, InvalidChunks) {
  ZarrMetadata metadata;
  // Chunk dimensions must be > 0.
  EXPECT_THAT(
      ParseMetadata(::nlohmann::json{{"chunks", {0}},
                                     {"compressor", nullptr},
                                     {"dtype", "|i1"},
                                     {"fill_value", 0},
                                     {"filters", nullptr},
                                     {"order", "F"},
                                     {"shape", {10}},
                                     {"zarr_format", 2}},
                    &metadata),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"chunks\".*"));
}

}  // namespace
