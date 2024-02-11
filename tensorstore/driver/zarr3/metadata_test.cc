// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/driver/zarr3/metadata.h"

#include <stdint.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_types.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/unit.h"

namespace {

namespace jb = ::tensorstore::internal_json_binding;

using ::tensorstore::ChunkLayout;
using ::tensorstore::CodecSpec;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::Schema;
using ::tensorstore::SharedArray;
using ::tensorstore::StatusIs;
using ::tensorstore::Unit;
using ::tensorstore::dtypes::bfloat16_t;
using ::tensorstore::dtypes::complex128_t;
using ::tensorstore::dtypes::complex64_t;
using ::tensorstore::dtypes::float16_t;
using ::tensorstore::dtypes::float32_t;
using ::tensorstore::dtypes::float64_t;
using ::tensorstore::internal::uint_t;
using ::tensorstore::internal_zarr3::FillValueJsonBinder;
using ::tensorstore::internal_zarr3::ZarrMetadata;
using ::tensorstore::internal_zarr3::ZarrMetadataConstraints;

::nlohmann::json GetBasicMetadata() {
  return {
      {"zarr_format", 3},
      {"node_type", "array"},
      {"shape", {10, 11, 12}},
      {"data_type", "uint16"},
      {"chunk_grid",
       {{"name", "regular"}, {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
      {"chunk_key_encoding", {{"name", "default"}}},
      {"fill_value", 0},
      {"codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
      {"attributes", {{"a", "b"}, {"c", "d"}}},
      {"dimension_names", {"a", nullptr, ""}},
  };
}

TEST(MetadataTest, ParseValid) {
  auto json = GetBasicMetadata();
  tensorstore::TestJsonBinderRoundTripJsonOnly<ZarrMetadata>({json});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson(json));
  EXPECT_THAT(metadata.shape, ::testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(metadata.chunk_shape, ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(metadata.data_type, tensorstore::dtype_v<uint16_t>);
  EXPECT_THAT(metadata.dimension_names,
              ::testing::ElementsAre("a", std::nullopt, ""));
  EXPECT_THAT(metadata.user_attributes, MatchesJson({{"a", "b"}, {"c", "d"}}));
}

TEST(MetadataTest, DuplicateDimensionNames) {
  auto json = GetBasicMetadata();
  json["dimension_names"] = {"a", "a", "b"};
  tensorstore::TestJsonBinderRoundTripJsonOnly<ZarrMetadata>({json});
}

TEST(MetadataTest, ParseValidNoDimensionNames) {
  auto json = GetBasicMetadata();
  json.erase("dimension_names");
  tensorstore::TestJsonBinderRoundTripJsonOnly<ZarrMetadata>({json});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson(json));
  EXPECT_THAT(metadata.shape, ::testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(metadata.chunk_shape, ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(metadata.data_type, tensorstore::dtype_v<uint16_t>);
  EXPECT_THAT(metadata.dimension_names,
              ::testing::ElementsAre(std::nullopt, std::nullopt, std::nullopt));
  EXPECT_THAT(metadata.user_attributes, MatchesJson({{"a", "b"}, {"c", "d"}}));
}

TEST(MetadataTest, UnknownExtensionAttribute) {
  auto json = GetBasicMetadata();
  json["foo"] = 42;
  EXPECT_THAT(ZarrMetadata::FromJson(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::StartsWith(
                           "Unsupported metadata field \"foo\" is not marked "
                           "{\"must_understand\": false}")));
}

// An empty "storage_transformers" list is allowed.
TEST(MetadataTest, EmptyStorageTransformers) {
  auto json = GetBasicMetadata();
  json["storage_transformers"] = ::nlohmann::json::array_t();
  EXPECT_THAT(ZarrMetadata::FromJson(json), StatusIs(absl::StatusCode::kOk));
}

// An non-empty "storage_transformers" list is not allowed.
TEST(MetadataTest, NonEmptyStorageTransformers) {
  auto json = GetBasicMetadata();
  json["storage_transformers"] = {1};
  EXPECT_THAT(ZarrMetadata::FromJson(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::MatchesRegex(
                           ".*: No storage transformers supported")));
}

TEST(MetadataTest, DimensionUnitsRankMismatch) {
  auto json = GetBasicMetadata();
  json["attributes"]["dimension_units"] = {"m", "s"};
  EXPECT_THAT(ZarrMetadata::FromJson(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::MatchesRegex(
                           ".*: Array has length 2 but should have length 3")));
}

TEST(MetadataTest, DimensionUnitsTypeMismatch) {
  auto json = GetBasicMetadata();
  json["attributes"]["dimension_units"] = {"m", false, "s"};
  EXPECT_THAT(ZarrMetadata::FromJson(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::MatchesRegex(
                           ".*: Expected string, but received: false")));
}

TEST(MetadataTest, ParseMissingMember) {
  for (auto member : {
           "shape",
           "chunk_grid",
           "chunk_key_encoding",
           "node_type",
           "zarr_format",
           "fill_value",
           "data_type",
           "codecs",
       }) {
    SCOPED_TRACE(member);
    auto json = GetBasicMetadata();
    json.erase(member);
    EXPECT_THAT(ZarrMetadata::FromJson(json),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

TEST(MetadataConstraintsTest, FillValueWithoutDataType) {
  EXPECT_THAT(
      ZarrMetadataConstraints::FromJson({{"fill_value", 0}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: must be specified in conjunction with \"data_type\""));
}

TEST(MetadataConstraintsTest, DimensionUnitsInConstraints) {
  ::nlohmann::json constraints_json = {
      {"zarr_format", 3},
      {"node_type", "array"},
      {"shape", {10, 11, 12}},
      {"dimension_names", {"a", nullptr, ""}},
      {"chunk_key_encoding", {{"name", "default"}}},
      {"fill_value", 0},
      {"chunk_grid",
       {{"name", "regular"}, {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
      {"data_type", "uint16"},
      {"attributes", {{"a", "b"}, {"c", "d"}}},
      {"codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
      {"bar", {{"must_understand", false}, {"a", 4}}},
  };
  constraints_json["attributes"]["dimension_units"] = {"m", "m", "m"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto constraints, ZarrMetadataConstraints::FromJson(constraints_json));
  EXPECT_THAT(constraints.dimension_units,
              ::testing::Optional(
                  ::testing::ElementsAre(Unit("m"), Unit("m"), Unit("m"))));
}

absl::Status TestMetadataConstraints(::nlohmann::json metadata_json,
                                     ::nlohmann::json constraints_json) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               ZarrMetadata::FromJson(metadata_json));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto constraints, ZarrMetadataConstraints::FromJson(constraints_json));
  return ValidateMetadata(metadata, constraints);
}

TEST(MetadataConstraintsTest, ValidateMetadataErrors) {
  ::nlohmann::json metadata_json = {
      {"zarr_format", 3},
      {"node_type", "array"},
      {"shape", {10, 11, 12}},
      {"dimension_names", {"a", nullptr, ""}},
      {"chunk_key_encoding", {{"name", "default"}}},
      {"fill_value", 0},
      {"chunk_grid",
       {{"name", "regular"}, {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
      {"data_type", "uint16"},
      {"attributes", {{"a", "b"}, {"c", "d"}}},
      {"codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
      {"bar", {{"must_understand", false}, {"a", 4}}},
  };
  {
    auto constraints_json = metadata_json;
    constraints_json["shape"] = {11, 12, 13};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"shape\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["data_type"] = "uint32";
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"data_type\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["chunk_grid"]["configuration"]["chunk_shape"] = {1, 2, 4};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"chunk_shape\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["chunk_key_encoding"]["configuration"] = {
        {"separator", "."}};
    EXPECT_THAT(
        TestMetadataConstraints(metadata_json, constraints_json),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 ::testing::StartsWith("Expected \"chunk_key_encoding\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["fill_value"] = 3;
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"fill_value\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["codecs"] = {
        {{"name", "bytes"}, {"configuration", {{"endian", "big"}}}}};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Mismatch in \"codecs\":")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["dimension_names"] = {"a", "b", "c"};
    EXPECT_THAT(
        TestMetadataConstraints(metadata_json, constraints_json),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 ::testing::StartsWith("Expected \"dimension_names\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["attributes"] = {{"d", "e"}};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Mismatch in \"attributes\":")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["foo"] = {{"must_understand", false}};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"foo\" ")));
  }
  {
    auto constraints_json = metadata_json;
    constraints_json["bar"] = {{"must_understand", false}, {"a", 5}};
    EXPECT_THAT(TestMetadataConstraints(metadata_json, constraints_json),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         ::testing::StartsWith("Expected \"bar\" ")));
  }
  {
    auto metadata_json_copy = metadata_json;
    metadata_json_copy["attributes"]["dimension_units"] = {"a", "m", "s"};
    auto constraints_json = metadata_json;
    constraints_json["attributes"]["dimension_units"] = {"b", "m", "m"};
    EXPECT_THAT(
        TestMetadataConstraints(metadata_json_copy, constraints_json),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 ::testing::StartsWith("Expected \"dimension_units\" of")));
  }
}

template <typename... Option>
absl::Status TestMetadataSchema(::nlohmann::json metadata_json,
                                Option&&... option) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               ZarrMetadata::FromJson(metadata_json));
  Schema schema;
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(std::forward<Option>(option)...));
  return ValidateMetadataSchema(metadata, schema);
}

TEST(MetadataSchemaTest, ValidateMetadataSchema) {
  ::nlohmann::json metadata_json = {
      {"zarr_format", 3},
      {"node_type", "array"},
      {"shape", {10, 11, 12}},
      {"dimension_names", {"a", nullptr, ""}},
      {"chunk_key_encoding", {{"name", "default"}}},
      {"fill_value", 0},
      {"chunk_grid",
       {{"name", "regular"}, {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
      {"data_type", "uint16"},
      {"attributes",
       {
           {"a", "b"},
           {"c", "d"},
           {"dimension_units", {"m", "m", "s"}},
       }},
      {"codecs",
       {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
  };
  EXPECT_THAT(TestMetadataSchema(metadata_json, tensorstore::RankConstraint{2}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Rank specified by schema (2) does not match rank "
                       "specified by metadata (3)"));
  EXPECT_THAT(
      TestMetadataSchema(metadata_json, Schema::Shape({10, 11, 13})),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ::testing::MatchesRegex(".*: Cannot merge index domain .*")));
  EXPECT_THAT(TestMetadataSchema(metadata_json, dtype_v<uint8_t>),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "data_type from metadata (uint16) does not match dtype "
                       "in schema (uint8)"));
  EXPECT_THAT(
      TestMetadataSchema(
          metadata_json,
          Schema::FillValue{tensorstore::MakeScalarArray<uint16_t>(4)}),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               "Invalid fill_value: schema requires fill value of 4, but "
               "metadata specifies fill value of 0"));
  EXPECT_THAT(
      TestMetadataSchema(
          metadata_json,
          Schema::FillValue{tensorstore::MakeScalarArray<uint16_t>(4)}),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               "Invalid fill_value: schema requires fill value of 4, but "
               "metadata specifies fill value of 0"));
  EXPECT_THAT(TestMetadataSchema(metadata_json,
                                 ChunkLayout::CodecChunkShape(
                                     {1, 1, 1}, /*hard_constraint=*/true)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "codec_chunk_shape not supported"));
  EXPECT_THAT(
      TestMetadataSchema(
          metadata_json,
          CodecSpec::FromJson(
              {{"driver", "zarr3"},
               {"codecs",
                {{{"name", "bytes"}, {"configuration", {{"endian", "big"}}}}}}})
              .value()),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ::testing::StartsWith(
                   "codec from metadata does not match codec in schema")));
  EXPECT_THAT(
      TestMetadataSchema(metadata_json,
                         Schema::DimensionUnits({"m", "m", "ns"})),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ::testing::StartsWith("dimension_units from metadata does not "
                                     "match dimension_units in schema")));
}

template <typename... Option>
Result<std::shared_ptr<const ZarrMetadata>> TestGetNewMetadata(
    ::nlohmann::json::object_t constraints_json, Option&&... option) {
  Schema schema;
  absl::Status status;
  [[maybe_unused]] bool ok =
      ((status = schema.Set(std::forward<Option>(option))).ok() && ...);
  TENSORSTORE_RETURN_IF_ERROR(status);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto constraints, ZarrMetadataConstraints::FromJson(constraints_json));
  return GetNewMetadata(constraints, schema);
}

TEST(GetNewMetadataTest, DuplicateDimensionNames) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata,
      TestGetNewMetadata({{"dimension_names", {"a", "a", "b"}}},
                         dtype_v<uint16_t>, Schema::Shape({2, 3, 4})));
}

TEST(GetNewMetadataTest, Missing) {
  EXPECT_THAT(
      TestGetNewMetadata({}, dtype_v<uint16_t>),
      StatusIs(absl::StatusCode::kInvalidArgument, "domain must be specified"));
  EXPECT_THAT(
      TestGetNewMetadata({}, Schema::Shape({1, 2, 3})),
      StatusIs(absl::StatusCode::kInvalidArgument, "dtype must be specified"));
}

TEST(GetNewMetadataTest, DimensionUnitsInSchema) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata,
      TestGetNewMetadata({}, dtype_v<uint16_t>, Schema::Shape({2, 3, 4}),
                         Schema::DimensionUnits({"m", "m", "s"})));
  EXPECT_THAT(metadata->dimension_units,
              ::testing::Optional(
                  ::testing::ElementsAre(Unit("m"), Unit("m"), Unit("s"))));
}

TEST(GetNewMetadataTest, DimensionUnitsInMetadata) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata,
      TestGetNewMetadata(
          {{"attributes", {{"dimension_units", {"m", "m", "s"}}}}},
          dtype_v<uint16_t>, Schema::Shape({2, 3, 4})));
  EXPECT_THAT(metadata->dimension_units,
              ::testing::Optional(
                  ::testing::ElementsAre(Unit("m"), Unit("m"), Unit("s"))));
}

TEST(MetadataTest, DataTypes) {
  for (std::string_view data_type_name : {
           "bool",
           "uint8",
           "uint16",
           "uint32",
           "uint64",
           "int8",
           "int16",
           "int32",
           "int64",
           "bfloat16",
           "float16",
           "float32",
           "float64",
           "complex64",
           "complex128",
       }) {
    auto json = GetBasicMetadata();
    json["data_type"] = data_type_name;
    if (data_type_name == "bool") {
      json["fill_value"] = false;
    } else if (data_type_name == "complex64" ||
               data_type_name == "complex128") {
      json["fill_value"] = {1, 2};
    }
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                     ZarrMetadata::FromJson(json));
    EXPECT_EQ(tensorstore::GetDataType(data_type_name), metadata.data_type);
  }
}

TEST(MetadataTest, InvalidDataType) {
  auto json = GetBasicMetadata();
  json["data_type"] = "char";
  EXPECT_THAT(
      ZarrMetadata::FromJson(json),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*: char data type is not one of the supported data types: .*"));
}

template <typename T>
void TestFillValue(std::vector<std::pair<T, ::nlohmann::json>> cases,
                   bool skip_to_json = false) {
  auto binder = FillValueJsonBinder{dtype_v<T>};
  for (const auto& [value, json] : cases) {
    SharedArray<const void> expected_fill_value =
        tensorstore::MakeScalarArray(value);
    if (!skip_to_json) {
      EXPECT_THAT(jb::ToJson(expected_fill_value, binder),
                  ::testing::Optional(MatchesJson(json)))
          << "value=" << value << ", json=" << json;
    }
    EXPECT_THAT(jb::FromJson<SharedArray<const void>>(json, binder),
                ::testing::Optional(
                    tensorstore::MatchesArrayIdentically(expected_fill_value)))
        << "json=" << json;
  }
}

template <typename T>
void TestFillValueInvalid(
    std::vector<std::pair<::nlohmann::json, std::string>> cases) {
  auto binder = FillValueJsonBinder{dtype_v<T>};
  for (const auto& [json, matcher] : cases) {
    EXPECT_THAT(jb::FromJson<SharedArray<const void>>(json, binder).status(),
                MatchesStatus(absl::StatusCode::kInvalidArgument, matcher))
        << "json=" << json;
  }
}

TEST(FillValueTest, Bool) {
  TestFillValue<bool>({
      {true, true},
      {false, false},
  });
  TestFillValueInvalid<bool>({
      {0, "Expected boolean, but received: 0"},
      {"true", "Expected boolean, but received: \"true\""},
  });
}

TEST(FillValueTest, Int8) {
  TestFillValue<int8_t>({
      {127, 127},
      {-128, -128},
  });
  TestFillValueInvalid<int8_t>({
      {128, "Expected integer in the range .*, but received: 128"},
      {"0", "Expected integer in the range .*, but received: \"0\""},
      {false, "Expected integer in the range .*, but received: false"},
  });
}

template <typename T, typename Complex = void>
void TestFloatFillValue(uint_t<sizeof(T) * 8> default_nan_bits,
                        uint_t<sizeof(T) * 8> other_nan_bits) {
  const auto default_nan = absl::bit_cast<T>(default_nan_bits);
  const auto other_nan = absl::bit_cast<T>(other_nan_bits);

  TestFillValue<T>({
      {T(0.0), 0.0},
      {T(-0.0), -0.0},
      {T(-1), -1},
      {T(1), 1},
      {T(INFINITY), "Infinity"},
      {T(-INFINITY), "-Infinity"},
      {default_nan, "NaN"},
      {other_nan, absl::StrFormat("0x%x", other_nan_bits)},
  });
  TestFillValue<T>(
      {
          {T(0.0), "0x0"},
      },
      /*skip_to_json=*/true);
  TestFillValueInvalid<T>({
      {"0",
       "Expected \"Infinity\", \"-Infinity\", \"NaN\", or hex string, but "
       "received: \"0\""},
      {false, "Expected .*, but received: false"},
      {"0x", "Expected .*, but received: \"0x\""},
      {"0xg", "Expected .*, but received: \"0xg\""},
      {"0x" + std::string(sizeof(T) * 2 + 1, '0'),
       "Expected .*, but received: \"0x.*\""},
  });

  if constexpr (!std::is_void_v<Complex>) {
    TestFillValue<Complex>({
        {Complex{0.0, 0.0}, {0, 0}},
        {Complex{T(INFINITY), -T(INFINITY)}, {"Infinity", "-Infinity"}},
        {Complex{default_nan, default_nan}, {"NaN", "NaN"}},
        {Complex{1.0, -1.0}, {1, -1}},
    });
    TestFillValueInvalid<Complex>({
        {0, "Expected array, but received: 0"},
        {{0, 1, 2}, "Array has length 3 but should have length 2"},
    });
  }
}

TEST(FillValueTest, Float16) {
  TestFloatFillValue<float16_t>(
      /*default_nan_bits=*/0x7e00,
      /*other_nan_bits=*/0x7e01);
}

TEST(FillValueTest, BFloat16) {
  TestFloatFillValue<bfloat16_t>(
      /*default_nan_bits=*/0x7fc0,
      /*other_nan_bits=*/0x7fc1);
}

TEST(FillValueTest, Float32) {
  TestFloatFillValue<float32_t, complex64_t>(
      /*default_nan_bits=*/0x7fc00000,
      /*other_nan_bits=*/0x7fc00001);
}

TEST(FillValueTest, Float64) {
  TestFloatFillValue<float64_t, complex128_t>(
      /*default_nan_bits=*/0x7ff8000000000000,
      /*other_nan_bits=*/0x7ff8000000000001);
}

}  // namespace
