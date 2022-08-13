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

#include "tensorstore/driver/n5/metadata.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ArrayView;
using ::tensorstore::CodecSpec;
using ::tensorstore::fortran_order;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::StridedLayout;
using ::tensorstore::internal_n5::DecodeChunk;
using ::tensorstore::internal_n5::N5Metadata;

TEST(MetadataTest, ParseValid) {
  ::nlohmann::json attributes{
      {"dimensions", {10, 11, 12}},       {"axes", {"a", "", ""}},
      {"blockSize", {1, 2, 3}},           {"dataType", "uint16"},
      {"compression", {{"type", "raw"}}}, {"extra", "value"},
  };
  tensorstore::TestJsonBinderRoundTripJsonOnly<N5Metadata>({attributes});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   N5Metadata::FromJson(attributes));
  EXPECT_THAT(metadata.shape, ::testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(metadata.dtype, tensorstore::dtype_v<std::uint16_t>);
  EXPECT_THAT(metadata.axes, ::testing::ElementsAre("a", "", ""));
  EXPECT_THAT(metadata.extra_attributes, MatchesJson({{"extra", "value"}}));
  EXPECT_THAT(metadata.chunk_layout,
              StridedLayout(fortran_order, 2, {1, 2, 3}));
}

TEST(MetadataTest, ParseValidNoAxes) {
  ::nlohmann::json attributes{
      {"dimensions", {10, 11, 12}},
      {"blockSize", {1, 2, 3}},
      {"dataType", "uint16"},
      {"compression", {{"type", "raw"}}},
  };
  tensorstore::TestJsonBinderRoundTripJsonOnly<N5Metadata>({attributes});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   N5Metadata::FromJson(attributes));
  EXPECT_THAT(metadata.shape, ::testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(metadata.dtype, tensorstore::dtype_v<std::uint16_t>);
  EXPECT_THAT(metadata.axes, ::testing::ElementsAre("", "", ""));
  EXPECT_THAT(metadata.extra_attributes,
              MatchesJson(::nlohmann::json::object_t()));
  EXPECT_THAT(metadata.chunk_layout,
              StridedLayout(fortran_order, 2, {1, 2, 3}));
}

TEST(MetadataTest, ParseMissingDimensions) {
  EXPECT_THAT(N5Metadata::FromJson({{"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseMissingBlockSize) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseMissingDataType) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 3}},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseInvalidDataType) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", 10},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "string"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseMissingCompression) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseInvalidCompression) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", 3}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseInvalidDimensions) {
  // Exceeds rank.
  EXPECT_THAT(
      N5Metadata::FromJson({{"dimensions", ::nlohmann::json::array_t(33, 10)},
                            {"blockSize", ::nlohmann::json::array_t(33, 1)},
                            {"dataType", "uint16"},
                            {"compression", {{"type", "raw"}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", "x"},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {"x"}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {-1, 10, 11}},
                                    {"blockSize", {1, 2, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseInvalidBlockSize) {
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", "x"},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {"x"}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 0, 3}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                    {"blockSize", {1, 2, 0xFFFFFFFF}},
                                    {"dataType", "uint16"},
                                    {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MetadataTest, ParseInvalidAxes) {
  EXPECT_THAT(N5Metadata::FromJson({
                  {"dimensions", {10, 11, 12}},
                  {"axes", {"a", "b", "c", "d"}},
                  {"blockSize", {1, 2, 3}},
                  {"dataType", "uint16"},
                  {"compression", {{"type", "raw"}}},
              }),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".* \"axes\": Array has length 4 but should have length 3"));

  EXPECT_THAT(
      N5Metadata::FromJson({
          {"dimensions", {10, 11, 12}},
          {"axes", {"a", "a", ""}},
          {"blockSize", {1, 2, 3}},
          {"dataType", "uint16"},
          {"compression", {{"type", "raw"}}},
      }),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".* \"a\" not unique"));
}

TEST(MetadataTest, DataTypes) {
  // Per the specification:
  // https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
  //
  //    * dataType (one of {uint8, uint16, uint32, uint64, int8, int16, int32,
  //      int64, float32, float64})
  for (auto data_type_name :
       {"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32",
        "int64", "float32", "float64"}) {
    ::nlohmann::json attributes{{"dimensions", {10, 11, 12}},
                                {"blockSize", {1, 2, 3}},
                                {"dataType", data_type_name},
                                {"compression", {{"type", "raw"}}}};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                     N5Metadata::FromJson(attributes));
    EXPECT_EQ(tensorstore::GetDataType(data_type_name), metadata.dtype);
  }
}

// Raw chunk example from the specification:
// https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
TEST(RawCompressionTest, Golden) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x03,  //
      0x00, 0x01,              //
      0x00, 0x02,              //
      0x00, 0x03,              //
      0x00, 0x04,              //
      0x00, 0x05,              //
      0x00, 0x06,              //
  };
  std::string encoded_data(std::begin(kData), std::end(kData));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata,
      N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                            {"blockSize", {1, 2, 3}},
                            {"dataType", "uint16"},
                            {"compression", {{"type", "raw"}}}}));
  auto array = MakeArray<std::uint16_t>({{{1, 3, 5}, {2, 4, 6}}});
  EXPECT_EQ(array, DecodeChunk(metadata, absl::Cord(encoded_data)));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        EncodeChunk(span<const Index>({0, 0, 0}), metadata, array));
    EXPECT_EQ(encoded_data, buffer);
  }

  // Test with truncated array data.
  EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data.substr(
                                        0, encoded_data.size() - 1))),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with truncated header data
  EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data.substr(0, 6))),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid mode (varlength)
  {
    std::string encoded_data_invalid_mode = encoded_data;
    encoded_data_invalid_mode[1] = 0x01;
    EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_mode)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test with invalid mode (unknown)
  {
    std::string encoded_data_invalid_mode = encoded_data;
    encoded_data_invalid_mode[1] = 0x02;
    EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_mode)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test with invalid number of dimensions
  {
    std::string encoded_data_invalid_rank = encoded_data;
    encoded_data_invalid_rank[3] = 0x02;
    EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_rank)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test with too large block shape
  {
    std::string encoded_data_invalid_shape = encoded_data;
    encoded_data_invalid_shape[7] = 0x02;
    EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_shape)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

TEST(RawCompressionTest, PartialChunk) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x01,              //
      0x00, 0x02,              //
      0x00, 0x03,              //
      0x00, 0x04,              //
  };
  std::string encoded_data(std::begin(kData), std::end(kData));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata,
      N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                            {"blockSize", {1, 2, 3}},
                            {"dataType", "uint16"},
                            {"compression", {{"type", "raw"}}}}));
  auto array = MakeArray<std::uint16_t>({{{1, 3, 0}, {2, 4, 0}}});
  EXPECT_EQ(array, DecodeChunk(metadata, absl::Cord(encoded_data)));
}

TEST(N5CodecSpecTest, Merge) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec1,
                                   CodecSpec::FromJson({{"driver", "n5"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec2, CodecSpec::FromJson(
                       {{"driver", "n5"}, {"compression", {{"type", "raw"}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec3,
                                   CodecSpec::FromJson({{"driver", "n5"},
                                                        {"compression",
                                                         {{"type", "blosc"},
                                                          {"cname", "lz4"},
                                                          {"clevel", 5},
                                                          {"shuffle", 1}}}}));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec1), ::testing::Optional(codec1));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec2), ::testing::Optional(codec2));
  EXPECT_THAT(CodecSpec::Merge(codec3, codec3), ::testing::Optional(codec3));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec2), ::testing::Optional(codec2));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec3), ::testing::Optional(codec3));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge codec spec .* with .*: "
                            "\"compression\" does not match"));
}

TEST(N5CodecSpecTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<tensorstore::CodecSpec>({
      {
          {"driver", "n5"},
          {"compression", {{"type", "raw"}}},
      },
  });
}

}  // namespace
