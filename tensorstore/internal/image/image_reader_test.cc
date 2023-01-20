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

#include "tensorstore/internal/image/image_reader.h"

#include <stddef.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/avif_reader.h"
#include "tensorstore/internal/image/bmp_reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/jpeg_reader.h"
#include "tensorstore/internal/image/png_reader.h"
#include "tensorstore/internal/image/tiff_reader.h"
#include "tensorstore/internal/image/webp_reader.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

ABSL_FLAG(std::string, tensorstore_test_data_dir, ".",
          "Path to directory containing test data.");

namespace {

using ::tensorstore::internal_image::AvifReader;
using ::tensorstore::internal_image::BmpReader;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::ImageReader;
using ::tensorstore::internal_image::JpegReader;
using ::tensorstore::internal_image::PngReader;
using ::tensorstore::internal_image::TiffReader;
using ::tensorstore::internal_image::WebPReader;

struct V {
  std::array<size_t, 2> yx;
  std::array<unsigned char, 3> rgb;
};

struct TestParam {
  std::string filename;
  ImageInfo info;
  std::vector<V> values;
};

[[maybe_unused]] std::string PrintToString(const TestParam& p) {
  return p.filename;
}

class ReaderTest : public ::testing::TestWithParam<TestParam> {
 public:
  ReaderTest() {
    if (IsTiff()) {
      reader = std::make_unique<TiffReader>();
    } else if (IsJpeg()) {
      reader = std::make_unique<JpegReader>();
    } else if (IsPng()) {
      reader = std::make_unique<PngReader>();
    } else if (IsBmp()) {
      reader = std::make_unique<BmpReader>();
    } else if (IsAvif()) {
      reader = std::make_unique<AvifReader>();
    } else if (IsWebP()) {
      reader = std::make_unique<WebPReader>();
    }
  }

  bool IsTiff() {
    return (absl::EndsWith(GetParam().filename, ".tiff") ||
            absl::EndsWith(GetParam().filename, ".tif"));
  }
  bool IsPng() { return absl::EndsWith(GetParam().filename, ".png"); }
  bool IsJpeg() {
    return (absl::EndsWith(GetParam().filename, ".jpg") ||
            absl::EndsWith(GetParam().filename, ".jpeg"));
  }
  bool IsAvif() { return absl::EndsWith(GetParam().filename, ".avif"); }
  bool IsBmp() { return absl::EndsWith(GetParam().filename, ".bmp"); }
  bool IsWebP() { return absl::EndsWith(GetParam().filename, ".webp"); }

  bool ReadsEntireFile() { return IsAvif() || IsJpeg(); }

  std::string GetFilename() {
    return tensorstore::internal::JoinPath(
        absl::GetFlag(FLAGS_tensorstore_test_data_dir), GetParam().filename);
  }

  tensorstore::Result<absl::Cord> ReadEntireFile(std::string filename) {
    absl::Cord file_data;
    TENSORSTORE_RETURN_IF_ERROR(
        riegeli::ReadAll(riegeli::FdReader(filename), file_data));
    return file_data;
  }

  std::unique_ptr<ImageReader> reader;
};

TEST_P(ReaderTest, ReadImage) {
  const auto& filename = GetParam().filename;
  ASSERT_FALSE(reader.get() == nullptr) << filename;
  ABSL_LOG(INFO) << filename;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(absl::Cord file_data,
                                   ReadEntireFile(GetFilename()));
  ASSERT_FALSE(file_data.empty());

  riegeli::CordReader cord_reader(&file_data);
  ASSERT_THAT(reader->Initialize(&cord_reader), ::tensorstore::IsOk())
      << filename;

  auto expected_info = GetParam().info;
  auto info = reader->GetImageInfo();
  EXPECT_EQ(info.width, expected_info.width) << filename;
  EXPECT_EQ(info.height, expected_info.height) << filename;
  EXPECT_EQ(info.num_components, expected_info.num_components) << filename;
  EXPECT_EQ(info.dtype, expected_info.dtype) << filename;

  const size_t image_bytes = ImageRequiredBytes(info);
  EXPECT_EQ(image_bytes, ImageRequiredBytes(expected_info));
  std::unique_ptr<unsigned char[]> image(new unsigned char[image_bytes]());
  EXPECT_THAT(reader->Decode(tensorstore::span(image.get(), image_bytes)),
              ::tensorstore::IsOk());
  EXPECT_TRUE(cord_reader.Close()) << cord_reader.status();

  /// Validate values.
  for (const V& v : GetParam().values) {
    ASSERT_LT(v.yx[0], expected_info.height)
        << " (" << v.yx[0] << "," << v.yx[1] << ")";
    ASSERT_LT(v.yx[1], expected_info.width)
        << " (" << v.yx[0] << "," << v.yx[1] << ")";
    size_t offset =
        expected_info.width * expected_info.num_components * v.yx[0] + v.yx[1];
    EXPECT_THAT(tensorstore::span<unsigned char>(image.get() + offset, 3),
                ::testing::ElementsAreArray(v.rgb))
        << " (" << v.yx[0] << "," << v.yx[1] << ") " << offset;
  }
}

TEST_P(ReaderTest, ReadImageTruncated) {
  const auto& filename = GetParam().filename;
  ASSERT_FALSE(reader.get() == nullptr) << filename;

  /// Skip this for some files.
  if (filename == "png/D75_01b.png") return;
  if (filename == "tiff/D75_01b.tiff") return;
  if (filename == "bmp/D75_08b_grey.bmp") return;
  if (IsWebP()) return;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(absl::Cord file_data,
                                   ReadEntireFile(GetFilename()));

  // Some file types don't need to read all the file bytes, though for the
  // examples here they require at least 90% of the original size.
  absl::Cord partial_file = file_data.Subcord(0, file_data.size() * 0.9);

  riegeli::CordReader cord_reader(&partial_file);
  absl::Status status = reader->Initialize(&cord_reader);
  if (status.ok()) {
    auto info = reader->GetImageInfo();
    auto expected_info = GetParam().info;
    if (info.width == expected_info.width) {
      EXPECT_EQ(info.width, expected_info.width) << filename;
      EXPECT_EQ(info.height, expected_info.height) << filename;
      EXPECT_EQ(info.num_components, expected_info.num_components) << filename;
      EXPECT_EQ(info.dtype, expected_info.dtype) << filename;
    }
    size_t image_bytes = ImageRequiredBytes(expected_info);
    std::unique_ptr<unsigned char[]> image(new unsigned char[image_bytes]());
    status.Update(reader->Decode(tensorstore::span(image.get(), image_bytes)));
  }

  if (status.ok()) {
    if (!cord_reader.Close()) {
      status.Update(cord_reader.status());
    }
  }
  EXPECT_FALSE(status.ok());
}

// Most images generated via tiffcp <source> <options> <dest>.
// Query image paramters using tiffinfo <image>
std ::vector<V> GetD75_08_Values() {
  return {
      // upper-left corner: hw=0,0   => 151,75,83
      V{{0, 0}, {151, 75, 83}},
      // lower-left corner: hw=171,0 => 255,250,251
      V{{171, 0}, {255, 250, 251}},
      // arbitrary point: hw=29,117  => 173,93,97
      V{{29, 117}, {173, 93, 97}},
  };
}

std ::vector<V> GetD75_08_Values_JPEG() {
  // JPEG values are inexact.
  return {
      V{{0, 0}, {152, 76, 88}},
      V{{171, 0}, {253, 247, 251}},
      V{{29, 117}, {174, 93, 99}},
  };
}

INSTANTIATE_TEST_SUITE_P(  //
    AvifFiles, ReaderTest,
    ::testing::Values(
        // lossless: avifenc -l <file.png> <file.avif>
        TestParam{"avif/D75_08b.avif", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        // lossy: avifenc -y 422 -a cq-level=1 --min 1 --max 1 <png> <avif>
        TestParam{"avif/D75_08b_cq1.avif", ImageInfo{172, 306, 3}},
        TestParam{"avif/D75_10b_cq1.avif",
                  ImageInfo{172, 306, 3, ::tensorstore::dtype_v<uint16_t>}},
        // avifenc -y 400  -a cq-level=1 --min 1 --max 1 <grey.png> <avif>
        TestParam{"avif/D75_08b_grey.avif",
                  ImageInfo{172, 306, 1},
                  {V{{29, 117}, {87, 87, 87}}}},
        // avifenc -y 400  -a cq-level=1 --min 1 --max 1 -d 12 <grey.png> <avif>
        TestParam{"avif/D75_12b_grey.avif",
                  ImageInfo{172, 306, 1, ::tensorstore::dtype_v<uint16_t>}} /**/
        ));

INSTANTIATE_TEST_SUITE_P(  //
    BmpFiles, ReaderTest,
    ::testing::Values(  //
        TestParam{"bmp/D75_08b.bmp", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"bmp/D75_08b_grey.bmp",
                  ImageInfo{172, 306, 1},
                  {V{{29, 117}, {87, 87, 87}}}} /**/
        ));

INSTANTIATE_TEST_SUITE_P(  //
    JpegFiles, ReaderTest,
    ::testing::Values(  //
        TestParam{"jpeg/D75_08b.jpeg", ImageInfo{172, 306, 3},
                  GetD75_08_Values_JPEG()} /**/
        ));

INSTANTIATE_TEST_SUITE_P(
    PngFiles, ReaderTest,
    ::testing::Values(
        TestParam{"png/D75_08b.png", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"png/D75_16b.png",
                  ImageInfo{172, 306, 3, ::tensorstore::dtype_v<uint16_t>}},
        TestParam{"png/D75_04b.png", ImageInfo{172, 306, 3}},
        TestParam{"png/D75_08b_grey.png",
                  ImageInfo{172, 306, 1},
                  {V{{29, 117}, {87, 87, 87}}}},
        TestParam{"png/D75_16b_grey.png",
                  ImageInfo{172, 306, 1, ::tensorstore::dtype_v<uint16_t>}},
        // convert D75_08b.png -depth 1 -threshold 50% D75_01b.png
        TestParam{"png/D75_01b.png",
                  ImageInfo{172, 306, 1, ::tensorstore::dtype_v<bool>}} /**/
        ));

// Use tiffcp -c/-t
INSTANTIATE_TEST_SUITE_P(
    TifFiles, ReaderTest,
    ::testing::Values(  //
        TestParam{"tiff/D75_08b.tiff", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"tiff/D75_08b_tiled.tiff", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"tiff/D75_08b_scanline.tiff", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"tiff/D75_08b_zip.tiff", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"tiff/D75_08b_lzw.tiff", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"tiff/D75_16b.tiff",
                  ImageInfo{172, 306, 3, ::tensorstore::dtype_v<uint16_t>}},
        // convert D75_08b.png -depth 1 -threshold 50% D75_01b.tiff
        TestParam{"tiff/D75_01b.tiff",
                  ImageInfo{172, 306, 1, ::tensorstore::dtype_v<bool>}},
        TestParam{"tiff/D75_08b_grey.tiff",
                  ImageInfo{172, 306, 1},
                  {V{{29, 117}, {87, 87, 87}}}},
        TestParam{"tiff/D75_16b_grey.tiff",
                  ImageInfo{172, 306, 1, ::tensorstore::dtype_v<uint16_t>}} /**/
        ));

INSTANTIATE_TEST_SUITE_P(  //
    WebPFiles, ReaderTest,
    ::testing::Values(  //
        TestParam{"webp/D75_08b.webp", ImageInfo{172, 306, 3},
                  GetD75_08_Values()},
        TestParam{"webp/D75_08b_q90.webp",
                  ImageInfo{172, 306, 3},
                  {V{{29, 117}, {166, 94, 91}}}} /**/
        ));

}  // namespace
