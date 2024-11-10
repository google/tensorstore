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

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_neuroglancer_precomputed::DecodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::EncodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::MultiscaleMetadata;

/// Parameters for the test fixture.
struct P {
  ::nlohmann::json metadata_json;
  tensorstore::DataType dtype;
  bool compare = true;
  bool truncate = true;
};

class ChunkEncodingTest : public testing::TestWithParam<P> {
 public:
  // Allocates a SharedArray<T> and initializes the values for testing.
  template <typename T>
  tensorstore::SharedArray<void> AllocateArrayImpl(Index num_channels) {
    auto array = tensorstore::AllocateArray<T>({num_channels, 5, 4, 3});
    for (Index i = 0, n = array.num_elements(); i < n; ++i) {
      array.data()[i] = static_cast<T>(i);
    }
    return array;
  }

  tensorstore::SharedArray<void> GetArrayForDType(tensorstore::DataTypeId id,
                                                  Index num_channels) {
    switch (id) {
      case tensorstore::DataTypeId::uint8_t:
        return AllocateArrayImpl<uint8_t>(num_channels);
      case tensorstore::DataTypeId::uint16_t:
        return AllocateArrayImpl<uint16_t>(num_channels);
      case tensorstore::DataTypeId::uint32_t:
        return AllocateArrayImpl<uint32_t>(num_channels);
      case tensorstore::DataTypeId::uint64_t:
        return AllocateArrayImpl<uint64_t>(num_channels);
      default:
        ABSL_UNREACHABLE();
    }
  }
};

TEST_P(ChunkEncodingTest, Roundtrip) {
  auto metadata_json = GetParam().metadata_json;
  auto dtype = GetParam().dtype;
  metadata_json["data_type"] = dtype.name();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   MultiscaleMetadata::FromJson(metadata_json));
  auto array = GetArrayForDType(dtype.id(), metadata.num_channels);
  std::vector<Index> chunk_indices{0, 0, 0};
  const size_t scale_index = 0;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      absl::Cord out, EncodeChunk(chunk_indices, metadata, scale_index, array));
  tensorstore::StridedLayout chunk_layout(tensorstore::c_order, dtype.size(),
                                          {metadata.num_channels, 5, 4, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decode_result,
      DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout, out));

  if (!out.empty() && GetParam().truncate) {
    // Test that truncating the chunk leads to a decoding error.
    auto corrupt = out.Subcord(0, out.size() - 1);
    EXPECT_THAT(
        DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout,
                    corrupt),
        testing::AnyOf(MatchesStatus(absl::StatusCode::kDataLoss),
                       MatchesStatus(absl::StatusCode::kInvalidArgument)));
  }

  if (GetParam().compare) {
    EXPECT_THAT(decode_result, array);
  }
}

std::vector<P> GenerateParams() {
  std::vector<P> result;
  for (const int num_channels : {1, 2, 3, 4}) {
    P param;
    param.metadata_json =
        ::nlohmann::json{{"@type", "neuroglancer_multiscale_volume"},
                         {"num_channels", num_channels},
                         {"scales",
                          {{{"chunk_sizes", {{3, 4, 5}}},
                            {"encoding", "raw"},
                            {"key", "k"},
                            {"resolution", {5, 6, 7}},
                            {"size", {10, 11, 12}}}}},
                         {"type", "image"}};
    // Test raw
    param.dtype = tensorstore::dtype_v<uint16_t>;
    result.push_back(param);

    // Test png
    // Simple truncation isn't sufficient to corrupt png files.
    param.truncate = false;
    if (num_channels >= 1 && num_channels <= 4) {
      param.metadata_json["scales"][0]["encoding"] = "png";
      // uint8
      param.dtype = tensorstore::dtype_v<uint8_t>;
      result.push_back(param);
      // uint16
      if (num_channels == 1) {
        param.dtype = tensorstore::dtype_v<uint16_t>;
        result.push_back(param);
      }
    }
    param.truncate = true;

    // Test jpeg
    // JPEG is lossy, so we can't compare values.
    param.compare = false;
    if (num_channels == 1 || num_channels == 3) {
      param.metadata_json["scales"][0]["encoding"] = "jpeg";
      param.dtype = tensorstore::dtype_v<uint8_t>;
      result.push_back(param);
    }
    param.compare = true;

    // Test compressed segmentation
    param.metadata_json["scales"][0]["encoding"] = "compressed_segmentation";
    param.metadata_json["scales"][0]["compressed_segmentation_block_size"] = {
        2, 3, 4};
    param.dtype = tensorstore::dtype_v<uint32_t>;
    result.push_back(param);
    param.dtype = tensorstore::dtype_v<uint64_t>;
    result.push_back(param);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    All, ChunkEncodingTest, testing::ValuesIn(GenerateParams()),
    [](const testing::TestParamInfo<P>& info) {
      const auto& p = info.param;
      auto encoding =
          p.metadata_json["scales"][0]["encoding"].get<std::string>();
      return tensorstore::StrCat(encoding, "_", p.metadata_json["num_channels"],
                                 "_", p.dtype.name());
    });

}  // namespace
