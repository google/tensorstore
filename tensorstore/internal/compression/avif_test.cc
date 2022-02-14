// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/compression/avif.h"

#include <stddef.h>

#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::avif::Decode;
using tensorstore::avif::Encode;
using tensorstore::avif::EncodeOptions;

double ComputeRMSE(const std::vector<unsigned char>& a,
                   const std::vector<unsigned char>& b) {
  assert(a.size() == b.size());
  double squared_error = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const int diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    squared_error += diff * diff;
  }
  return std::sqrt(squared_error / a.size());
}

std::vector<unsigned char> MakeTestImage(const size_t width,
                                         const size_t height,
                                         const size_t num_components) {
  std::vector<unsigned char> image(width * height * num_components);
  uint64_t lcg = width * height * num_components;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      double gradient =
          static_cast<double>(x + y) / static_cast<double>(width + height);
      image[num_components * (y * width + x) + 0] =
          static_cast<unsigned char>(gradient * 255);
      if (num_components > 1) {
        // Channel 1 is essentially random.
        lcg = (lcg * 6364136223846793005) + 1;
        image[num_components * (y * width + x) + 1] =
            static_cast<unsigned char>(lcg);
      }
      if (num_components > 2) {
        // alternately gradient / constant.
        image[num_components * (y * width + x) + 2] =
            (y & 1) ? x : static_cast<unsigned char>((1.0 - gradient) * 255);
      }
      if (num_components > 3) {
        // alternately gradient / constant.
        image[num_components * (y * width + x) + 3] =
            (y & 1)
                ? x
                : static_cast<unsigned char>(std::abs(128 - gradient * 255));
      }
    }
  }
  return image;
}

struct P {
  EncodeOptions options;
  double rmse;
};

// Implements ::testing::PrintToStringParamName().
static std::string PrintToString(const P& p) {
  return absl::StrCat("q_", p.options.quantizer, "__s_", p.options.speed);
}

class AvifCodingTest : public ::testing::TestWithParam<P> {
 public:
};

INSTANTIATE_TEST_SUITE_P(AvifTests, AvifCodingTest,
                         testing::Values(P{EncodeOptions{0, 1}, 0},  //
                                         P{EncodeOptions{0, 6}, 0},
                                         P{EncodeOptions{25, 2}, 100},
                                         P{EncodeOptions{50, 6}, 100}  //
                                         ),
                         testing::PrintToStringParamName());

TEST_P(AvifCodingTest, OneComponent) {
  const size_t width = 100, height = 33, num_components = 1;

  std::vector<unsigned char> input_image =
      MakeTestImage(width, height, num_components);

  absl::Cord encoded;
  EncodeOptions options = GetParam().options;
  TENSORSTORE_ASSERT_OK(Encode(input_image.data(), width, height,
                               num_components, options, &encoded));

  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = Decode(encoded, [&](size_t w, size_t h, size_t num_c) {
    EXPECT_EQ(width, w);
    EXPECT_EQ(height, h);
    EXPECT_EQ(num_components, num_c);
    assert(width == w);
    assert(height == h);
    assert(num_components == num_c);
    return decoded.data();
  });
  TENSORSTORE_EXPECT_OK(decode_status);

  const double rmse = ComputeRMSE(input_image, decoded);
  EXPECT_GE(GetParam().rmse, rmse);
  if (GetParam().rmse == 0) {
    EXPECT_THAT(decoded, testing::Eq(input_image));
  }
}

TEST_P(AvifCodingTest, TwoComponents) {
  const size_t width = 100, height = 33, num_components = 2;

  std::vector<unsigned char> input_image =
      MakeTestImage(width, height, num_components);

  absl::Cord encoded;
  EncodeOptions options = GetParam().options;
  TENSORSTORE_ASSERT_OK(Encode(input_image.data(), width, height,
                               num_components, options, &encoded));

  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = Decode(encoded, [&](size_t w, size_t h, size_t num_c) {
    EXPECT_EQ(width, w);
    EXPECT_EQ(height, h);
    EXPECT_EQ(num_components, num_c);
    assert(width == w);
    assert(height == h);
    assert(num_components == num_c);
    return decoded.data();
  });
  TENSORSTORE_EXPECT_OK(decode_status);

  const double rmse = ComputeRMSE(input_image, decoded);
  EXPECT_GE(GetParam().rmse, rmse);
  if (GetParam().rmse == 0) {
    EXPECT_THAT(decoded, testing::Eq(input_image));
  }
}

TEST_P(AvifCodingTest, ThreeComponents) {
  const size_t width = 200, height = 33, num_components = 3;

  std::vector<unsigned char> input_image =
      MakeTestImage(width, height, num_components);

  absl::Cord encoded;
  EncodeOptions options = GetParam().options;
  TENSORSTORE_ASSERT_OK(Encode(input_image.data(), width, height,
                               num_components, options, &encoded));

  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = Decode(encoded, [&](size_t w, size_t h, size_t num_c) {
    EXPECT_EQ(width, w);
    EXPECT_EQ(height, h);
    EXPECT_EQ(num_components, num_c);
    assert(width == w);
    assert(height == h);
    assert(num_components == num_c);
    return decoded.data();
  });
  TENSORSTORE_EXPECT_OK(decode_status);

  const double rmse = ComputeRMSE(input_image, decoded);
  EXPECT_GE(GetParam().rmse, rmse);
  if (GetParam().rmse == 0) {
    EXPECT_THAT(decoded, testing::Eq(input_image));
  }
}

TEST_P(AvifCodingTest, FourComponents) {
  const size_t width = 200, height = 33, num_components = 4;

  std::vector<unsigned char> input_image =
      MakeTestImage(width, height, num_components);

  absl::Cord encoded;
  EncodeOptions options = GetParam().options;
  TENSORSTORE_ASSERT_OK(Encode(input_image.data(), width, height,
                               num_components, options, &encoded));

  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = Decode(encoded, [&](size_t w, size_t h, size_t num_c) {
    EXPECT_EQ(width, w);
    EXPECT_EQ(height, h);
    EXPECT_EQ(num_components, num_c);
    assert(width == w);
    assert(height == h);
    assert(num_components == num_c);
    return decoded.data();
  });
  TENSORSTORE_EXPECT_OK(decode_status);

  const double rmse = ComputeRMSE(input_image, decoded);
  EXPECT_GE(GetParam().rmse, rmse);
  if (GetParam().rmse == 0) {
    EXPECT_THAT(decoded, testing::Eq(input_image));
  }
}

TEST(AvifTest, EncodeInvalidNumComponents) {
  const size_t width = 100, height = 33, num_components = 5;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  absl::Cord encoded;
  EncodeOptions options;
  EXPECT_THAT(
      Encode(input_image.data(), width, height, num_components, options,
             &encoded),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*AVIF encoding requires between 1 and 4 components"));
}

TEST(AvifTest, SmallAvif) {
  static constexpr unsigned char data[] = {
      0x00, 0x00, 0x00, 0x1c, 0x66, 0x74, 0x79, 0x70, 0x6d, 0x69, 0x66, 0x31,
      0x00, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x66, 0x31, 0x61, 0x76, 0x69, 0x66,
      0x6d, 0x69, 0x61, 0x66, 0x00, 0x00, 0x00, 0xf3, 0x6d, 0x65, 0x74, 0x61,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x68, 0x64, 0x6c, 0x72,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x69, 0x63, 0x74,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x0e, 0x70, 0x69, 0x74, 0x6d, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1e, 0x69, 0x6c, 0x6f, 0x63, 0x00,
      0x00, 0x00, 0x00, 0x04, 0x40, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x17, 0x00, 0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00,
      0x28, 0x69, 0x69, 0x6e, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x1a, 0x69, 0x6e, 0x66, 0x65, 0x02, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x61, 0x76, 0x30, 0x31, 0x49, 0x6d, 0x61, 0x67, 0x65,
      0x00, 0x00, 0x00, 0x00, 0x72, 0x69, 0x70, 0x72, 0x70, 0x00, 0x00, 0x00,
      0x53, 0x69, 0x70, 0x63, 0x6f, 0x00, 0x00, 0x00, 0x14, 0x69, 0x73, 0x70,
      0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x00, 0x00, 0x10, 0x70, 0x61, 0x73, 0x70, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x61, 0x76, 0x31,
      0x43, 0x81, 0x20, 0x00, 0x00, 0x0a, 0x09, 0x38, 0x1d, 0xff, 0xff, 0xda,
      0x40, 0x43, 0x40, 0x08, 0x00, 0x00, 0x00, 0x10, 0x70, 0x69, 0x78, 0x69,
      0x00, 0x00, 0x00, 0x00, 0x03, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, 0x17,
      0x69, 0x70, 0x6d, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x01, 0x04, 0x01, 0x02, 0x83, 0x84, 0x00, 0x00, 0x00, 0x84, 0x6d,
      0x64, 0x61, 0x74, 0x0a, 0x09, 0x38, 0x1d, 0xff, 0xff, 0xda, 0x40, 0x43,
      0x40, 0x08, 0x32, 0x6f, 0x11, 0x90, 0x00, 0x00, 0x12, 0x5d, 0x60, 0x34,
      0x19, 0x59, 0x91, 0x94, 0xa0, 0x1e, 0xc3, 0xc8, 0xed, 0x0a, 0x11, 0xa0,
      0x57, 0x69, 0x46, 0xdf, 0x0a, 0xba, 0x02, 0x75, 0x9d, 0x7c, 0xa1, 0x38,
      0x79, 0x42, 0x29, 0x4a, 0xe2, 0x15, 0xd8, 0xdb, 0x9f, 0xf6, 0x57, 0xf3,
      0xa9, 0x38, 0xff, 0x39, 0x49, 0x78, 0x14, 0x47, 0xc7, 0xc5, 0x2d, 0x9c,
      0xd4, 0x7b, 0xde, 0xda, 0x53, 0x0c, 0x28, 0x31, 0x67, 0xe8, 0xa4, 0x0e,
      0xe7, 0x7a, 0xe9, 0xdd, 0xbd, 0x0f, 0x48, 0x83, 0x49, 0xc6, 0x1c, 0xb5,
      0xf7, 0x78, 0x65, 0xb1, 0x17, 0x2a, 0x67, 0x32, 0x54, 0xae, 0x0e, 0xa9,
      0x86, 0xc8, 0xea, 0xba, 0x38, 0xbe, 0x29, 0xb6, 0xc1, 0xc3, 0x93, 0x0a,
      0x06, 0x72, 0x4a, 0x09, 0x69, 0x41, 0x59,
  };

  std::vector<unsigned char> pixel_data(256 * 256 * 3);

  auto decode_status =
      Decode(absl::Cord(absl::string_view(reinterpret_cast<const char*>(data),
                                          sizeof(data))),
             [&](size_t w, size_t h, size_t num_c) {
               EXPECT_EQ(256, w);
               EXPECT_EQ(256, h);
               EXPECT_EQ(3, num_c);
               assert(256 == w);
               assert(256 == h);
               assert(3 == num_c);
               return pixel_data.data();
             });
  TENSORSTORE_EXPECT_OK(decode_status);
  EXPECT_EQ(1, pixel_data[0]);
  EXPECT_EQ(1, pixel_data[1]);
  EXPECT_EQ(0, pixel_data[2]);
}

}  // namespace
