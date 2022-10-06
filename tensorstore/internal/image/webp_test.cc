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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/webp_reader.h"
#include "tensorstore/internal/image/webp_writer.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::WebPReader;
using ::tensorstore::internal_image::WebPReaderOptions;
using ::tensorstore::internal_image::WebPWriter;
using ::tensorstore::internal_image::WebPWriterOptions;

TEST(WebPTest, Decode) {
  // https://shoonia.github.io/1x1/#405060ff
  static constexpr unsigned char data[] = {
      0x52, 0x49, 0x46, 0x46, 0x1e, 0x00, 0x00, 0x00, 0x57, 0x45,
      0x42, 0x50, 0x56, 0x50, 0x38, 0x4c, 0x11, 0x00, 0x00, 0x00,
      0x2f, 0x00, 0x00, 0x00, 0x00, 0x07, 0x50, 0xa8, 0x02, 0x15,
      0xac, 0xff, 0x81, 0x88, 0xe8, 0x7f, 0x00, 0x00,
  };

  WebPReader decoder;
  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));
  ASSERT_THAT(decoder.Initialize(&string_reader), ::tensorstore::IsOk());

  const auto info = decoder.GetImageInfo();
  EXPECT_EQ((ImageInfo{1, 1, 3}), info);

  uint8_t pixel[3] = {};
  ASSERT_THAT(decoder.Decode(pixel), tensorstore::IsOk());

  EXPECT_EQ(0x40, pixel[0]);
  EXPECT_EQ(0x50, pixel[1]);
  EXPECT_EQ(0x60, pixel[2]);
}

TEST(WebPTest, EncodeDecode) {
  static constexpr unsigned char raw[] = {
      0x52, 0x49, 0x46, 0x46, 0x20, 0x00, 0x00, 0x00, 0x57, 0x45,
      0x42, 0x50, 0x56, 0x50, 0x38, 0x4c, 0x14, 0x00, 0x00, 0x00,
      0x2f, 0x00, 0x00, 0x00, 0x00, 0x07, 0x50, 0x81, 0x54, 0x08,
      0x20, 0x00, 0x0a, 0x9a, 0xfe, 0xc7, 0x88, 0x88, 0xfe, 0x07,
  };

  uint8_t pixels[3] = {1, 2, 3};

  absl::Cord encoded;
  {
    WebPWriter encoder;
    riegeli::CordWriter cord_writer(&encoded);
    ASSERT_THAT(encoder.Initialize(&cord_writer), ::tensorstore::IsOk());

    ASSERT_THAT(encoder.Encode(ImageInfo{1, 1, 3}, pixels),
                ::tensorstore::IsOk());
    ASSERT_THAT(encoder.Done(), ::tensorstore::IsOk());
  }

  EXPECT_THAT(encoded, ::testing::StrEq(std::string_view(
                           reinterpret_cast<const char*>(raw), sizeof(raw))));

  {
    WebPReader decoder;
    riegeli::CordReader cord_reader(&encoded);
    ASSERT_THAT(decoder.Initialize(&cord_reader), ::tensorstore::IsOk());

    const auto& info = decoder.GetImageInfo();
    EXPECT_EQ((ImageInfo{1, 1, 3}), info);

    uint8_t new_pixels[3] = {};
    ASSERT_THAT(decoder.Decode(new_pixels), tensorstore::IsOk());
  }
}

TEST(WebPTest, CorruptData) {
  static constexpr unsigned char data[] = {
      0x52, 0x49, 0x46, 0x46, 0x20, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  };

  riegeli::StringReader string_reader(reinterpret_cast<const char*>(data),
                                      sizeof(data));

  WebPReader decoder;
  EXPECT_THAT(decoder.Initialize(&string_reader),
              tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
