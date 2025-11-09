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

#include "tensorstore/internal/compression/xz_compressor.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include <lzma.h>
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::XzCompressor;

// Tests that a small input round trips, and that the result is appended to the
// output string without clearing the existing contents.
TEST(XzCompressorTest, SmallRoundtrip) {
  XzCompressor compressor;
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result("abc"), decode_result("def");
  TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result, 0));
  ASSERT_GE(encode_result.size(), 3);
  EXPECT_EQ("abc", encode_result.Subcord(0, 3));
  TENSORSTORE_ASSERT_OK(compressor.Decode(
      encode_result.Subcord(3, encode_result.size() - 3), &decode_result, 0));
  EXPECT_EQ("def" + std::string(input), decode_result);
}

// Same as above, but with fragmented input.
TEST(XzCompressorTest, SmallRoundtripFragmented) {
  XzCompressor compressor;
  const absl::Cord input = absl::MakeFragmentedCord(
      {"The quick", " brown fox", " jumped over", " ", "the lazy dog."});
  absl::Cord encode_result("abc"), decode_result("def");
  TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result, 0));
  ASSERT_GE(encode_result.size(), 3);
  EXPECT_EQ("abc", encode_result.Subcord(0, 3));
  std::vector<std::string> encode_result_fragments;
  for (size_t i = 3; i < encode_result.size(); ++i) {
    encode_result_fragments.push_back(std::string(encode_result.Subcord(i, 1)));
  }
  TENSORSTORE_ASSERT_OK(compressor.Decode(
      absl::MakeFragmentedCord(encode_result_fragments), &decode_result, 0));
  EXPECT_EQ("def" + std::string(input), decode_result);
}

// Tests that round tripping works for with an input that exceeds the 16KiB
// buffer size.
TEST(XzCompressorTest, LargeRoundtrip) {
  std::string input(100000, '\0');
  unsigned char x = 0;
  for (auto& v : input) {
    v = x;
    x += 7;
  }
  XzCompressor compressor;
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(
      compressor.Encode(absl::Cord(input), &encode_result, 0));
  TENSORSTORE_ASSERT_OK(compressor.Decode(encode_result, &decode_result, 0));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a level of 9 gives a result that is different from 6.
TEST(XzCompressorTest, NonDefaultLevel) {
  XzCompressor compressor;
  XzCompressor compressor2;
  compressor2.level = 9;

  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result1, 0));
  TENSORSTORE_ASSERT_OK(compressor2.Encode(input, &encode_result2, 0));
  EXPECT_NE(encode_result1, encode_result2);
  absl::Cord decode_result;
  TENSORSTORE_ASSERT_OK(compressor.Decode(encode_result2, &decode_result, 0));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a different integrity check gives a different result.
TEST(XzCompressorTest, NonDefaultCheck) {
  XzCompressor compressor;
  XzCompressor compressor2;
  compressor2.check = LZMA_CHECK_CRC32;

  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result1, 0));
  TENSORSTORE_ASSERT_OK(compressor2.Encode(input, &encode_result2, 0));
  EXPECT_NE(encode_result1, encode_result2);
  absl::Cord decode_result;
  TENSORSTORE_ASSERT_OK(compressor.Decode(encode_result2, &decode_result, 0));
  EXPECT_EQ(input, decode_result);
}

// Tests that decoding corrupt data gives an error.
TEST(XzCompressorTest, DecodeCorruptData) {
  XzCompressor compressor;
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");

  // Test corrupting the header.
  {
    absl::Cord encode_result, decode_result;
    TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result, 0));
    ASSERT_GE(encode_result.size(), 1);
    std::string corrupted(encode_result);
    corrupted[0] = 0;
    EXPECT_THAT(compressor.Decode(absl::Cord(corrupted), &decode_result, 0),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test corrupting the trailer.
  {
    absl::Cord encode_result, decode_result;
    TENSORSTORE_ASSERT_OK(compressor.Encode(input, &encode_result, 0));
    ASSERT_GE(encode_result.size(), 1);
    EXPECT_THAT(
        compressor.Decode(encode_result.Subcord(0, encode_result.size() - 1),
                          &decode_result, 0),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
