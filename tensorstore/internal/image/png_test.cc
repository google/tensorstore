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

#include <stdint.h>

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/png_reader.h"
#include "tensorstore/internal/image/png_writer.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::PngReader;
using ::tensorstore::internal_image::PngWriter;

TEST(PngTest, Decode) {
  // https://shoonia.github.io/1x1/#5f5f5fff
  static constexpr unsigned char data[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  // sig
      0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00,
      0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02,
      0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
      0xde,  // ihdr
      0x00, 0x00, 0x00, 0x01, 0x73, 0x52, 0x47, 0x42, 0x00,
      0xae, 0xce, 0x1c, 0xe9, 0x00, 0x00, 0x00, 0x0c, 0x49,
      0x44, 0x41, 0x54, 0x18, 0x57, 0x63, 0x88, 0x8f, 0x8f,
      0x07, 0x00, 0x02, 0x3e, 0x01, 0x1e, 0x78, 0xd8, 0x99,
      0x68,  // idat
      0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
      0x42, 0x60, 0x82,  // iend
  };
  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));

  PngReader decoder;
  ASSERT_THAT(decoder.Initialize(&string_reader), ::tensorstore::IsOk());

  // See: http://www.libtiff.org/man/index.html
  const auto info = decoder.GetImageInfo();
  EXPECT_EQ(1, info.width);
  EXPECT_EQ(1, info.height);
  EXPECT_EQ(3, info.num_components);

  uint8_t pixel[3] = {};
  ASSERT_THAT(decoder.Decode(pixel), ::tensorstore::IsOk());
  EXPECT_THAT(pixel, ::testing::ElementsAre(0x5f, 0x5f, 0x5f));
}

TEST(PngTest, EncodeDecode) {
  uint8_t pixels[1] = {0};
  absl::Cord encoded;

  {
    PngWriter encoder;
    riegeli::CordWriter cord_writer(&encoded);
    ASSERT_THAT(encoder.Initialize(&cord_writer), ::tensorstore::IsOk());
    ASSERT_THAT(encoder.Encode(ImageInfo{1, 1, 1}, pixels),
                ::tensorstore::IsOk());

    ASSERT_THAT(encoder.Done(), ::tensorstore::IsOk());
  }

  {
    PngReader decoder;
    riegeli::CordReader cord_reader(&encoded);
    ASSERT_THAT(decoder.Initialize(&cord_reader), ::tensorstore::IsOk());

    const auto& info = decoder.GetImageInfo();
    EXPECT_EQ(1, info.width);
    EXPECT_EQ(1, info.height);
    EXPECT_EQ(1, info.num_components);

    uint8_t new_pixels[1] = {1};
    ASSERT_THAT(
        decoder.Decode(tensorstore::span(
            reinterpret_cast<unsigned char*>(&new_pixels), sizeof(new_pixels))),
        tensorstore::IsOk());

    EXPECT_THAT(new_pixels, ::testing::ElementsAre(0));
  }
}

TEST(PngTest, CorruptData) {
  static constexpr unsigned char data[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  // sig
      0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
      0xde,  // ihdr
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));

  PngReader decoder;
  EXPECT_THAT(decoder.Initialize(&string_reader),
              tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
