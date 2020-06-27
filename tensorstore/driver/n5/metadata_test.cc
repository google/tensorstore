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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorstore/array.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::ArrayView;
using tensorstore::fortran_order;
using tensorstore::Index;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StridedLayout;
using tensorstore::internal_n5::DecodeChunk;
using tensorstore::internal_n5::N5Metadata;

TEST(MetadataTest, Parse) {
  ::nlohmann::json attributes{{"dimensions", {10, 11, 12}},
                              {"blockSize", {1, 2, 3}},
                              {"dataType", "uint16"},
                              {"compression", {{"type", "raw"}}},
                              {"extra", "value"}};
  auto metadata = N5Metadata::Parse(attributes);
  ASSERT_EQ(Status(), GetStatus(metadata));
  EXPECT_THAT(metadata->shape, ::testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(metadata->data_type, tensorstore::DataTypeOf<std::uint16_t>());
  EXPECT_EQ(attributes, metadata->attributes);
  EXPECT_EQ(attributes["compression"], ::nlohmann::json(metadata->compressor));
  EXPECT_EQ(StridedLayout(fortran_order, 2, {1, 2, 3}), metadata->chunk_layout);

  // Missing dimensions
  EXPECT_THAT(N5Metadata::Parse({{"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Missing blockSize
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Missing data type.
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 3}},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid data type
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", 10},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "string"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Missing compression.
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid compression
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", 3}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid dimensions
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", "x"},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {"x"}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {-1, 10, 11}},
                                 {"blockSize", {1, 2, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid blockSize
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", "x"},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {"x"}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 0, 3}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                 {"blockSize", {1, 2, 0xFFFFFFFF}},
                                 {"dataType", "uint16"},
                                 {"compression", {{"type", "raw"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
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
    auto metadata = N5Metadata::Parse(attributes);
    ASSERT_EQ(Status(), GetStatus(metadata));
    EXPECT_EQ(tensorstore::GetDataType(data_type_name), metadata->data_type);
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
  auto metadata = N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                     {"blockSize", {1, 2, 3}},
                                     {"dataType", "uint16"},
                                     {"compression", {{"type", "raw"}}}})
                      .value();
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

  // Test with invalid mode
  std::string encoded_data_invalid_mode = encoded_data;
  encoded_data_invalid_mode[1] = 0x01;
  EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_mode)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid number of dimensions
  std::string encoded_data_invalid_rank = encoded_data;
  encoded_data_invalid_rank[3] = 0x02;
  EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_rank)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with too large block shape
  std::string encoded_data_invalid_shape = encoded_data;
  encoded_data_invalid_shape[7] = 0x02;
  EXPECT_THAT(DecodeChunk(metadata, absl::Cord(encoded_data_invalid_shape)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
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
  auto metadata = N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                     {"blockSize", {1, 2, 3}},
                                     {"dataType", "uint16"},
                                     {"compression", {{"type", "raw"}}}})
                      .value();
  auto array = MakeArray<std::uint16_t>({{{1, 3, 0}, {2, 4, 0}}});
  EXPECT_EQ(array, DecodeChunk(metadata, absl::Cord(encoded_data)));
}

}  // namespace
