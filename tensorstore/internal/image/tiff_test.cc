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

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/tiff_reader.h"
#include "tensorstore/internal/image/tiff_writer.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

ABSL_FLAG(std::string, tensorstore_test_data_dir, ".",
          "Path to directory containing test data.");

namespace {

using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::TiffReader;
using ::tensorstore::internal_image::TiffWriter;
using ::tensorstore::internal_image::TiffWriterOptions;

class TiffTest : public ::testing::Test {
 public:
  TiffTest() {
  }
};

TEST_F(TiffTest, Decode) {
  static constexpr unsigned char data[] = {
      /*IFH*/ 0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08,
      /*DIR*/ 0x00, 0x04, /**/
      /*IFD*/ 0x01, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
      /*012*/ 0x00, 0x01, 0x00, 0x00, /*width*/
      /*IFD*/ 0x01, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
      /*...*/ 0x00, 0x01, 0x00, 0x00, /*length*/
      /*IFD*/ 0x01, 0x11, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
      /*...*/ 0x00, 0x00, 0x00, 0x00, /*strip offset*/
      /*IFD*/ 0x01, 0x17, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
      /*...*/ 0x00, 0x08, 0x00, 0x00, /*strip bytecount*/
      /*03A*/ 0x00, 0x00, 0x00, 0x00,
  };

  TiffReader decoder;
  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));
  ASSERT_THAT(decoder.Initialize(&string_reader), ::tensorstore::IsOk());

  // EXPECT_EQ(1, decoder.GetFrameCount());

  // See: http://www.libtiff.org/man/index.html
  const auto info = decoder.GetImageInfo();
  EXPECT_EQ(1, info.width);
  EXPECT_EQ(1, info.height);
  EXPECT_EQ(1, info.num_components);

  uint8_t pixel[1] = {};
  ASSERT_THAT(decoder.Decode(pixel), ::tensorstore::IsOk());
  EXPECT_THAT(pixel, ::testing::ElementsAre(0));
}

TEST_F(TiffTest, EncodeDecode) {
  static constexpr unsigned char le_data[] = {
      0x49, 0x49, 0x2a, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00,
      0x00, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x01, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x03, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x06, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x11, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x16, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x17, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x1c, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x31, 0x01, 0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x94, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x74,
      0x6f, 0x72, 0x65, 0x00,
  };
  static constexpr unsigned char be_data[] = {
      0x4d, 0x4d, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x0b,
      0x01, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x08, 0x00, 0x00,
      0x01, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x06, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x11, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08,
      0x01, 0x15, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x16, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x17, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x01, 0x1c, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
      0x01, 0x31, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x94,
      0x00, 0x00, 0x00, 0x00, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x74,
      0x6f, 0x72, 0x65, 0x00,
  };

  uint8_t pixels[1] = {};

  absl::Cord encoded;
  {
    riegeli::CordWriter riegeli_writer(&encoded);
    TiffWriter encoder;
    ASSERT_THAT(encoder.Initialize(&riegeli_writer, TiffWriterOptions{}),
                ::tensorstore::IsOk());
    ASSERT_THAT(encoder.Encode(ImageInfo{1, 1, 1}, pixels),
                ::tensorstore::IsOk());

    ASSERT_THAT(encoder.Done(), ::tensorstore::IsOk());
  }

  EXPECT_THAT(
      encoded,
      ::testing::AnyOf(
          ::testing::StrEq(std::string_view(
              reinterpret_cast<const char*>(le_data), sizeof(le_data))),
          ::testing::StrEq(std::string_view(
              reinterpret_cast<const char*>(be_data), sizeof(be_data)))));

  {
    TiffReader decoder;
    riegeli::CordReader cord_reader(&encoded);
    ASSERT_THAT(decoder.Initialize(&cord_reader), ::tensorstore::IsOk());

    // See: http://www.libtiff.org/man/index.html
    const auto& info = decoder.GetImageInfo();
    EXPECT_EQ(1, info.width);
    EXPECT_EQ(1, info.height);
    EXPECT_EQ(1, info.num_components);

    uint8_t new_pixels[1] = {};
    ASSERT_THAT(
        decoder.Decode(tensorstore::span(
            reinterpret_cast<unsigned char*>(&new_pixels), sizeof(new_pixels))),
        tensorstore::IsOk());

    EXPECT_THAT(new_pixels, ::testing::ElementsAre(0));
  }
}

TEST_F(TiffTest, ReadMultiPage) {
  absl::Cord file_data;
  {
    std::string filename = tensorstore::internal::JoinPath(
        absl::GetFlag(FLAGS_tensorstore_test_data_dir),
        "tiff/D75_08b_3page.tiff");
    TENSORSTORE_ASSERT_OK(
        riegeli::ReadAll(riegeli::FdReader(filename), file_data));
  }
  riegeli::CordReader cord_reader(&file_data);

  TiffReader decoder;

  ASSERT_THAT(decoder.Initialize(&cord_reader), ::tensorstore::IsOk());
  EXPECT_EQ(3, decoder.GetFrameCount());

  const ImageInfo expected_info{172, 306, 3};
  const size_t image_bytes = ImageRequiredBytes(expected_info);
  std::unique_ptr<unsigned char[]> image(new unsigned char[image_bytes]());

  for (int i = 0; i < decoder.GetFrameCount(); i++) {
    ASSERT_THAT(decoder.SeekFrame(i), ::tensorstore::IsOk());
    auto info = decoder.GetImageInfo();
    EXPECT_EQ(info.width, expected_info.width);
    EXPECT_EQ(info.height, expected_info.height);
    EXPECT_EQ(info.num_components, expected_info.num_components);
    EXPECT_EQ(info.dtype, expected_info.dtype);

    EXPECT_THAT(decoder.Decode(tensorstore::span(image.get(), image_bytes)),
                ::tensorstore::IsOk());
  }
}

TEST_F(TiffTest, CorruptData) {
  static constexpr unsigned char data[] = {
      0x49, 0x49, 0x2a, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));

  TiffReader decoder;
  EXPECT_THAT(decoder.Initialize(&string_reader),
              tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
