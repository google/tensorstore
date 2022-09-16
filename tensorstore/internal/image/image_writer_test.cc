// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/image/image_writer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/base/any_dependency.h"
#include "riegeli/base/base.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/internal/image/riegeli_block_writer.h"
#include "tensorstore/internal/image/tiff_reader.h"
#include "tensorstore/internal/image/tiff_writer.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal::RiegeliBlockWriter;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::ImageReader;
using ::tensorstore::internal_image::ImageView;
using ::tensorstore::internal_image::ImageWriter;
using ::tensorstore::internal_image::TiffReader;
using ::tensorstore::internal_image::TiffWriter;
using ::tensorstore::internal_image::TiffWriterOptions;

/// Returns a pointer to T if T or reference_wrapped T is contained in the
/// std::any container.
template <typename T>
const T* GetPointerFromAny(std::any* any_ptr) {
  if (!any_ptr->has_value()) {
    return nullptr;
  }
  if (auto opt = std::any_cast<T>(any_ptr); opt != nullptr) {
    return opt;
  }
  if (auto opt = std::any_cast<std::reference_wrapper<T>>(any_ptr);
      opt != nullptr) {
    return &(opt->get());
  }
  if (auto opt = std::any_cast<std::reference_wrapper<const T>>(any_ptr);
      opt != nullptr) {
    return &(opt->get());
  }
  return nullptr;
}

double ComputeRMSE(const unsigned char* a, const unsigned char* b, size_t c) {
  double squared_error = 0;
  for (size_t i = 0; i < c; ++i) {
    const int diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    squared_error += diff * diff;
  }
  return std::sqrt(squared_error / static_cast<double>(c));
}

void MakeTestImage(const ImageInfo& info,
                   tensorstore::span<unsigned char> data) {
  ImageView image(info, data);
  uint64_t lcg = info.width * info.height * info.num_components;

  for (size_t y = 0; y < info.height; ++y) {
    auto* row = image.data_row(y).data();
    for (size_t x = 0; x < info.width; ++x) {
      double gradient = static_cast<double>(x + y) /
                        static_cast<double>(info.width + info.height);
      *row++ = static_cast<unsigned char>(gradient * 255);
      if (info.num_components > 1) {
        // Channel 1 is essentially random.
        lcg = (lcg * 6364136223846793005) + 1;
        *row++ = static_cast<unsigned char>(lcg);
      }
      if (info.num_components > 2) {
        // alternately gradient / constant.
        *row++ = (y & 1) ? static_cast<unsigned char>((1.0 - gradient) * 255)
                         : static_cast<unsigned char>(x);
      }
      if (info.num_components > 3) {
        // alternately gradient / constant.
        *row++ =
            (y & 1)
                ? static_cast<unsigned char>(x)
                : static_cast<unsigned char>(std::abs(128 - gradient * 255));
      }
    }
  }
}

struct TestParam {
  std::any options;
  ImageInfo image_params;
  double rmse = 0;
};

[[maybe_unused]] std::string PrintToString(const TestParam& p) {
  return absl::StrCat(p.image_params.num_components,
                      p.rmse != 0 ? "_rmse" : "");
}

class WriterTest : public ::testing::TestWithParam<TestParam> {
 public:
  WriterTest() {
    std::any* options = const_cast<std::any*>(&GetParam().options);
    if (GetPointerFromAny<TiffWriterOptions>(options)) {
      writer.Emplace<TiffWriter>();
      reader.Emplace<TiffReader>();
    }
  }

  absl::Status InitializeWriter(riegeli::Writer* riegeli_writer) {
    std::any* options = const_cast<std::any*>(&GetParam().options);
    if (auto* ptr = GetPointerFromAny<TiffWriterOptions>(options)) {
      return reinterpret_cast<TiffWriter*>(writer.get())
          ->Initialize(riegeli_writer, *ptr);
    }
    return writer->Initialize(riegeli_writer);
  }

  riegeli::AnyDependency<ImageWriter*, TiffWriter> writer;
  riegeli::AnyDependency<ImageReader*, TiffReader> reader;
};

TEST_P(WriterTest, RoundTrip) {
  ASSERT_FALSE(writer == nullptr);
  ASSERT_FALSE(reader.get() == nullptr);

  const ImageInfo source_info = GetParam().image_params;
  std::vector<unsigned char> source(ImageRequiredBytes(source_info));
  MakeTestImage(source_info, source);

  absl::Cord encoded;
  {
    RiegeliBlockWriter buffer;

    // Despite options being passed in, initialize with defaults.
    ASSERT_THAT(InitializeWriter(&buffer), ::tensorstore::IsOk());
    ASSERT_THAT(writer->Encode(source_info, source), ::tensorstore::IsOk());
    ASSERT_THAT(writer->Done(), ::tensorstore::IsOk());
    encoded = buffer.ConvertToCord();
  }

  ImageInfo decoded_info;
  std::vector<unsigned char> decoded(source.size());
  {
    riegeli::CordReader cord_reader(&encoded);
    ASSERT_THAT(reader->Initialize(&cord_reader), ::tensorstore::IsOk());

    decoded_info = reader->GetImageInfo();
    EXPECT_EQ(decoded_info.width, source_info.width);
    EXPECT_EQ(decoded_info.height, source_info.height);
    EXPECT_EQ(decoded_info.num_components, source_info.num_components);

    EXPECT_THAT(reader->Decode(decoded), ::tensorstore::IsOk());
  }

  double rmse = ComputeRMSE(decoded.data(), source.data(), source.size());

  /// When RMSE is not 0, verify that the actual value is witin 5%.
  if (GetParam().rmse == 0) {
    EXPECT_EQ(0, rmse) << "\nA: " << source_info << " "
                       << "\nB: " << decoded_info;
  } else {
    EXPECT_NEAR(GetParam().rmse, rmse, rmse * 0.05)
        << decoded_info;  // within 5%
  }
}

INSTANTIATE_TEST_SUITE_P(
    TiffFiles, WriterTest,
    ::testing::Values(  //
        TestParam{TiffWriterOptions{}, ImageInfo{33, 100, 1}, 0},
        TestParam{TiffWriterOptions{}, ImageInfo{33, 100, 2}, 0},
        TestParam{TiffWriterOptions{}, ImageInfo{33, 100, 3}, 0},
        TestParam{TiffWriterOptions{}, ImageInfo{33, 100, 4}, 0}));

}  // namespace
