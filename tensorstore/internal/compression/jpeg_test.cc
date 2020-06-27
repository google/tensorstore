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

#include "tensorstore/internal/compression/jpeg.h"

#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include "tensorstore/array.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Index;
using tensorstore::MatchesStatus;
using tensorstore::Status;

namespace jpeg = tensorstore::jpeg;

class JpegRoundtripTest : public ::testing::TestWithParam<bool> {
 public:
  /// Returns the square root of the mean squared error.
  double TestRoundTrip(unsigned char* input_image, size_t width, size_t height,
                       size_t num_components) {
    absl::Cord encoded;
    jpeg::EncodeOptions options;
    TENSORSTORE_EXPECT_OK(jpeg::Encode(input_image, width, height,
                                       num_components, options, &encoded));
    if (GetParam()) {
      std::vector<std::string> parts;
      constexpr size_t kFragmentSize = 100;
      for (size_t i = 0; i < encoded.size(); i += kFragmentSize) {
        parts.push_back(std::string(encoded.Subcord(i, kFragmentSize)));
      }
      encoded = absl::MakeFragmentedCord(parts);
    }
    std::vector<unsigned char> decoded(width * height * num_components);
    auto decode_status =
        jpeg::Decode(encoded, [&](size_t w, size_t h, size_t num_c) {
          EXPECT_EQ(width, w);
          EXPECT_EQ(height, h);
          EXPECT_EQ(num_components, num_c);
          return decoded.data();
        });
    TENSORSTORE_EXPECT_OK(decode_status);
    double squared_error = 0;
    for (size_t i = 0; i < decoded.size(); ++i) {
      const int diff =
          static_cast<int>(input_image[i]) - static_cast<int>(decoded[i]);
      squared_error += diff * diff;
    }
    const double rmse = std::sqrt(squared_error / decoded.size());
    return rmse;
  }
};

INSTANTIATE_TEST_SUITE_P(MaybeFragmented, JpegRoundtripTest, ::testing::Bool());

TEST_P(JpegRoundtripTest, OneComponentConstant) {
  const size_t width = 100, height = 33, num_components = 1;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  EXPECT_LT(TestRoundTrip(input_image.data(), width, height, num_components),
            1);
}

TEST_P(JpegRoundtripTest, OneComponentGradient) {
  const size_t width = 100, height = 33, num_components = 1;
  std::vector<unsigned char> input_image(width * height * num_components);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      input_image[y * width + x] =
          static_cast<unsigned char>(static_cast<double>(x + y) /
                                     static_cast<double>(width + height) * 255);
    }
  }
  EXPECT_LT(TestRoundTrip(input_image.data(), width, height, num_components),
            2);
}

TEST_P(JpegRoundtripTest, ThreeComponentGradient) {
  const size_t width = 200, height = 33, num_components = 3;
  std::vector<unsigned char> input_image(width * height * num_components);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      input_image[num_components * (y * width + x) + 0] =
          static_cast<unsigned char>(static_cast<double>(x + y) /
                                     static_cast<double>(width + height) * 255);
      input_image[num_components * (y * width + x) + 1] =
          static_cast<unsigned char>(
              255 - static_cast<double>(x + y) /
                        static_cast<double>(width + height) * 255);
      input_image[num_components * (y * width + x) + 2] =
          static_cast<unsigned char>(
              std::abs(128 - static_cast<double>(x + y) /
                                 static_cast<double>(width + height) * 255));
    }
  }
  EXPECT_LT(TestRoundTrip(input_image.data(), width, height, num_components),
            2);
}

TEST_P(JpegRoundtripTest, ThreeComponentConstant) {
  const size_t width = 100, height = 33, num_components = 3;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  EXPECT_LT(TestRoundTrip(input_image.data(), width, height, num_components),
            1);
}

TEST(JpegTest, EncodeInvalidNumComponents) {
  const size_t width = 100, height = 33, num_components = 2;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  EXPECT_THAT(jpeg::Encode(input_image.data(), width, height, num_components,
                           options, &encoded),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected 1 or 3 components, but received: 2"));
}

TEST(JpegTest, EncodeInvalidWidth) {
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  EXPECT_THAT(jpeg::Encode(nullptr, 10000000000, 17, 1, options, &encoded),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Image dimensions of .* exceed maximum size"));
}

TEST(JpegTest, EncodeInvalidHeight) {
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  EXPECT_THAT(jpeg::Encode(nullptr, 17, 10000000000, 1, options, &encoded),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Image dimensions of .* exceed maximum size"));
}

TEST(JpegTest, TruncateData) {
  const size_t width = 100, height = 33, num_components = 3;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  TENSORSTORE_EXPECT_OK(jpeg::Encode(input_image.data(), width, height,
                                     num_components, options, &encoded));
  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = jpeg::Decode(encoded.Subcord(0, encoded.size() - 1),
                                    [&](size_t w, size_t h, size_t num_c) {
                                      EXPECT_EQ(width, w);
                                      EXPECT_EQ(height, h);
                                      EXPECT_EQ(num_components, num_c);
                                      return decoded.data();
                                    });
  EXPECT_THAT(decode_status, MatchesStatus(absl::StatusCode::kInvalidArgument,
                                           "Error decoding JPEG: .*"));
}

TEST(JpegTest, ValidateSizeError) {
  const size_t width = 100, height = 33, num_components = 3;
  std::vector<unsigned char> input_image(width * height * num_components, 42);
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  TENSORSTORE_EXPECT_OK(jpeg::Encode(input_image.data(), width, height,
                                     num_components, options, &encoded));
  std::vector<unsigned char> decoded(width * height * num_components);
  auto decode_status = jpeg::Decode(encoded.Subcord(0, encoded.size() - 1),
                                    [&](size_t w, size_t h, size_t num_c) {
                                      return absl::UnknownError("Error");
                                    });
  EXPECT_THAT(decode_status, MatchesStatus(absl::StatusCode::kUnknown,
                                           "Error decoding JPEG: Error"));
}

// Tests resistance to libjpeg-turbo vulnerability LJT-01-003 DoS via
// progressive, arithmetic image decoding.
//
// https://cure53.de/pentest-report_libjpeg-turbo.pdf
TEST(JpegTest, LJT_01_003) {
  static constexpr unsigned char initial[] = {
      /*SOI*/ 0xFF, 0xD8,
      /*SOF10*/ 0xFF, 0xCA, 0x00, 0x0B, 0x08,
      /*dimension*/ 0x20, 0x00, /*dimension*/ 0x20, 0x00, 0x01, 0x00, 0x11,
      0x00,
      /*DQT*/ 0xFF, 0xDB, 0x00, 0x43, 0x00,
      /*quanttab*/
      16, 11, 10, 16, 24, 40, 51, 61,      //
      12, 12, 14, 19, 26, 58, 60, 55,      //
      14, 13, 16, 24, 40, 57, 69, 56,      //
      14, 17, 22, 29, 51, 87, 80, 62,      //
      18, 22, 37, 56, 68, 109, 103, 77,    //
      24, 35, 55, 64, 81, 104, 113, 92,    //
      49, 64, 78, 87, 103, 121, 120, 101,  //
      72, 92, 95, 98, 112, 100, 103, 99,   //
  };
  std::string encoded(std::begin(initial), std::end(initial));
  static constexpr unsigned char sos[] = {0xFF, 0xDA, 0x00, 0x08, 0x01,
                                          0x00, 0x00, 0x00, 0x00, 0x10};
  encoded.reserve(8000 * 1024);
  while (encoded.size() + std::size(sos) < 8000 * 1024) {
    encoded.append(std::begin(sos), std::end(sos));
  }
  // libjpeg-turbo reports a warning, and we treat all warnings as errors and
  // abort, which prevents the DoS.
  EXPECT_THAT(jpeg::Decode(absl::Cord(encoded),
                           [&](size_t w, size_t h, size_t num_c) {
                             // Never reached
                             return absl::UnknownError("");
                           }),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error decoding JPEG: .*"));
}

}  // namespace
