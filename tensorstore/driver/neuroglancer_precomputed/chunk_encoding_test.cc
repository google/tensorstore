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

using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_neuroglancer_precomputed::DecodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::EncodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::MultiscaleMetadata;

template <typename T>
void TestRoundtrip(::nlohmann::json metadata_json, bool compare) {
  const auto dtype = tensorstore::dtype_v<T>;
  metadata_json["data_type"] = dtype.name();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   MultiscaleMetadata::FromJson(metadata_json));
  auto array = tensorstore::AllocateArray<T>({metadata.num_channels, 5, 4, 3});
  for (Index i = 0, n = array.num_elements(); i < n; ++i) {
    array.data()[i] = static_cast<T>(i);
  }
  std::vector<Index> chunk_indices{0, 0, 0};
  const size_t scale_index = 0;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      absl::Cord out, EncodeChunk(chunk_indices, metadata, scale_index, array));
  tensorstore::StridedLayout chunk_layout(tensorstore::c_order, dtype.size(),
                                          {metadata.num_channels, 5, 4, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decode_result,
      DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout, out));

  if (!out.empty()) {
    // Test that truncating the chunk leads to a decoding error.
    auto corrupt = out.Subcord(0, out.size() - 1);
    EXPECT_THAT(
        DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout,
                    corrupt),
        testing::AnyOf(MatchesStatus(absl::StatusCode::kDataLoss),
                       MatchesStatus(absl::StatusCode::kInvalidArgument)));
  }

  if (!compare) return;
  EXPECT_THAT(decode_result, array);
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
    TestRoundtrip<uint16_t>(metadata_json_raw, /*compare=*/true);

    // Test jpeg
    {
      if (num_channels != 1 && num_channels != 3) continue;
      auto metadata_json_jpeg = metadata_json_raw;
      metadata_json_jpeg["scales"][0]["encoding"] = "jpeg";
      // JPEG is lossy, so we can't compare values.
      TestRoundtrip<uint8_t>(metadata_json_jpeg, /*compare=*/false);
    }

    // Test compressed segmentation
    {
      auto metadata_json_cseg = metadata_json_raw;
      metadata_json_cseg["scales"][0]["encoding"] = "compressed_segmentation";
      metadata_json_cseg["scales"][0]["compressed_segmentation_block_size"] = {
          2, 3, 4};
      TestRoundtrip<uint32_t>(metadata_json_cseg, /*compare=*/true);
      TestRoundtrip<uint64_t>(metadata_json_cseg, /*compare=*/true);
    }
  }
}

}  // namespace
