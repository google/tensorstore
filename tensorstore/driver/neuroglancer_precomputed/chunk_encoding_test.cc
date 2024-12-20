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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json_fwd.hpp>
#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_reader.h"
#include "tensorstore/internal/image/jpeg_reader.h"
#include "tensorstore/internal/image/png_reader.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_image::ImageReader;
using ::tensorstore::internal_neuroglancer_precomputed::DecodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::EncodeChunk;
using ::tensorstore::internal_neuroglancer_precomputed::MultiscaleMetadata;

/// Parameters for the test fixture.
struct P {
  ::nlohmann::json metadata_json;
  DataType dtype;
  double max_root_mean_squared_error = 0;
  bool truncate = true;
  std::function<std::unique_ptr<ImageReader>()> get_image_reader;
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

template <typename T>
double GetRootMeanSquaredErrorImpl(tensorstore::ArrayView<const void> array_a,
                                   tensorstore::ArrayView<const void> array_b) {
  double mean_squared_error = 0;
  tensorstore::IterateOverArrays(
      [&](const T* a, const T* b) {
        double diff = static_cast<double>(*a) - static_cast<double>(*b);
        mean_squared_error += diff * diff;
      },
      /*constraints=*/{},
      tensorstore::StaticDataTypeCast<const T, tensorstore::unchecked>(array_a),
      tensorstore::StaticDataTypeCast<const T, tensorstore::unchecked>(
          array_b));
  return std::sqrt(mean_squared_error / array_a.num_elements());
}

double GetRootMeanSquaredError(tensorstore::ArrayView<const void> array_a,
                               tensorstore::ArrayView<const void> array_b) {
  assert(array_a.dtype() == array_b.dtype());
  assert(array_a.dtype() == dtype_v<uint8_t>);
  return GetRootMeanSquaredErrorImpl<uint8_t>(array_a, array_b);
}

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
    auto corrupt = out.Subcord(
        0, out.size() - std::min(static_cast<size_t>(5), out.size()));
    EXPECT_THAT(
        DecodeChunk(chunk_indices, metadata, scale_index, chunk_layout,
                    corrupt),
        testing::AnyOf(MatchesStatus(absl::StatusCode::kDataLoss),
                       MatchesStatus(absl::StatusCode::kInvalidArgument)));
  }

  if (double max_rms_error = GetParam().max_root_mean_squared_error) {
    EXPECT_LT(GetRootMeanSquaredError(decode_result, array), max_rms_error)
        << "original=" << array << ", decoded=" << decode_result;
  } else {
    EXPECT_THAT(decode_result, array);
  }

  if (GetParam().get_image_reader) {
    auto image_reader = GetParam().get_image_reader();
    riegeli::CordReader reader{&out};
    TENSORSTORE_ASSERT_OK(image_reader->Initialize(&reader));
    auto image_info = image_reader->GetImageInfo();
    EXPECT_EQ(image_info.width, 3);
    EXPECT_EQ(image_info.height, 5 * 4);
    EXPECT_EQ(image_info.num_components, metadata.num_channels);
    EXPECT_EQ(image_info.dtype, dtype);
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
    {
      P param_raw = param;
      param_raw.dtype = dtype_v<uint16_t>;
      result.push_back(param_raw);
    }

    // Test png
    // Simple truncation isn't sufficient to corrupt png files.
    if (num_channels >= 1 && num_channels <= 4) {
      P param_png = param;
      param_png.truncate = false;
      param_png.metadata_json["scales"][0]["encoding"] = "png";
      param_png.get_image_reader = [] {
        return std::make_unique<tensorstore::internal_image::PngReader>();
      };
      for (auto dtype : {
               static_cast<DataType>(dtype_v<uint8_t>),
               static_cast<DataType>(dtype_v<uint16_t>),
           }) {
        param_png.dtype = dtype;
        result.push_back(param_png);
      }
    }

    // Test jpeg
    if (num_channels == 1 || num_channels == 3) {
      // JPEG is lossy, so we can't compare values exactly.
      P param_jpeg = param;
      param_jpeg.max_root_mean_squared_error = 3;
      param_jpeg.metadata_json["scales"][0]["encoding"] = "jpeg";
      param_jpeg.dtype = dtype_v<uint8_t>;
      param_jpeg.get_image_reader = [] {
        return std::make_unique<tensorstore::internal_image::JpegReader>();
      };
      result.push_back(param_jpeg);
    }

    // Test compressed segmentation
    {
      P param_cseg = param;
      param_cseg.metadata_json["scales"][0]["encoding"] =
          "compressed_segmentation";
      param_cseg
          .metadata_json["scales"][0]["compressed_segmentation_block_size"] = {
          2, 3, 4};
      for (auto dtype : {
               static_cast<DataType>(dtype_v<uint32_t>),
               static_cast<DataType>(dtype_v<uint64_t>),
           }) {
        param_cseg.dtype = dtype;
        result.push_back(param_cseg);
      }
    }
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
