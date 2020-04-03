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

/// Chunk encoding/decoding tests.

#include "tensorstore/driver/neuroglancer_precomputed/chunk_encoding.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Index;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_neuroglancer_precomputed::DecodeChunk;
using tensorstore::internal_neuroglancer_precomputed::EncodeChunk;
using tensorstore::internal_neuroglancer_precomputed::MultiscaleMetadata;

template <typename T>
void TestRoundtrip(::nlohmann::json metadata_json, bool compare) {
  const auto data_type = tensorstore::DataTypeOf<T>();
  metadata_json["data_type"] = data_type.name();
  auto metadata = MultiscaleMetadata::Parse(metadata_json).value();
  auto array = tensorstore::AllocateArray<T>({metadata.num_channels, 5, 4, 3});
  for (Index i = 0, n = array.num_elements(); i < n; ++i) {
    array.data()[i] = static_cast<T>(i);
  }
  std::string out;
  std::vector<Index> chunk_indices{0, 0, 0};
  const size_t scale_index = 0;
  EXPECT_EQ(tensorstore::Status(),
            EncodeChunk(chunk_indices, metadata, scale_index, array, &out));
  tensorstore::StridedLayout chunk_layout(
      tensorstore::c_order, data_type.size(), {metadata.num_channels, 5, 4, 3});
  auto decode_result =
      DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout, out);
  EXPECT_EQ(Status(), GetStatus(decode_result));

  if (!out.empty()) {
    // Test that truncating the chunk leads to a decoding error.
    auto corrupt = out;
    corrupt.resize(corrupt.size() - 1);
    auto corrupt_decode_result = DecodeChunk(
        chunk_indices, metadata, scale_index, chunk_layout, corrupt);
    EXPECT_THAT(tensorstore::GetStatus(corrupt_decode_result),
                tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  if (!compare) return;
  EXPECT_THAT(decode_result, ::testing::Optional(array));
}

TEST(ChunkEncodingTest, Roundtrip) {
  for (const int num_channels : {1, 2, 3}) {
    const ::nlohmann::json metadata_json_raw{
        {"@type", "neuroglancer_multiscale_volume"},
        {"num_channels", num_channels},
        {"scales",
         {{{"chunk_sizes", {{3, 4, 5}}},
           {"encoding", "raw"},
           {"key", "k"},
           {"resolution", {5, 6, 7}},
           {"size", {10, 11, 12}}}}},
        {"type", "image"}};

    // Test raw
    TestRoundtrip<std::uint16_t>(metadata_json_raw, /*compare=*/true);

    // Test jpeg
    {
      if (num_channels != 1 && num_channels != 3) continue;
      auto metadata_json_jpeg = metadata_json_raw;
      metadata_json_jpeg["scales"][0]["encoding"] = "jpeg";
      // JPEG is lossy, so we can't compare values.
      TestRoundtrip<std::uint8_t>(metadata_json_jpeg, /*compare=*/false);
    }

    // Test compressed segmentation
    {
      auto metadata_json_cseg = metadata_json_raw;
      metadata_json_cseg["scales"][0]["encoding"] = "compressed_segmentation";
      metadata_json_cseg["scales"][0]["compressed_segmentation_block_size"] = {
          2, 3, 4};
      TestRoundtrip<std::uint32_t>(metadata_json_cseg, /*compare=*/true);
      TestRoundtrip<std::uint64_t>(metadata_json_cseg, /*compare=*/true);
    }
  }
}

}  // namespace
