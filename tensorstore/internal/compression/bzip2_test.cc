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

#include "tensorstore/internal/compression/bzip2.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;

namespace bzip2 = tensorstore::bzip2;

// Tests that a small input round trips, and that the result is appended to the
// output string without clearing the existing contents.
TEST(Bzip2CompressorTest, SmallRoundtrip) {
  bzip2::Options options{6};
  const std::string input = "The quick brown fox jumped over the lazy dog.";
  std::string encode_result = "abc", decode_result = "def";
  bzip2::Encode(input, &encode_result, options);
  ASSERT_GE(encode_result.size(), 3);
  EXPECT_EQ("abc", encode_result.substr(0, 3));
  ASSERT_EQ(Status(), bzip2::Decode(encode_result.substr(3), &decode_result));
  EXPECT_EQ("def" + input, decode_result);
}

// Tests that round tripping works for with an input that exceeds the 16KiB
// buffer size.
TEST(Bzip2CompressorTest, LargeRoundtrip) {
  std::string input(100000, '\0');
  unsigned char x = 0;
  for (auto& v : input) {
    v = x;
    x += 7;
  }
  bzip2::Options options{6};
  std::string encode_result, decode_result;
  bzip2::Encode(input, &encode_result, options);
  ASSERT_EQ(Status(), bzip2::Decode(encode_result, &decode_result));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a level of 9 gives a result that is different from 6.
TEST(Bzip2CompressorTest, NonDefaultLevel) {
  bzip2::Options options1{6};
  bzip2::Options options2{9};
  const std::string input = "The quick brown fox jumped over the lazy dog.";
  std::string encode_result1, encode_result2;
  bzip2::Encode(input, &encode_result1, options1);
  bzip2::Encode(input, &encode_result2, options2);
  EXPECT_NE(encode_result1, encode_result2);
  std::string decode_result;
  ASSERT_EQ(Status(), bzip2::Decode(encode_result2, &decode_result));
  EXPECT_EQ(input, decode_result);
}

// Tests that decoding corrupt data gives an error.
TEST(Bzip2CompressorTest, DecodeCorruptData) {
  bzip2::Options options{6};
  const std::string input = "The quick brown fox jumped over the lazy dog.";

  // Test corrupting the header.
  {
    std::string encode_result, decode_result;
    bzip2::Encode(input, &encode_result, options);
    ASSERT_GE(encode_result.size(), 1);
    encode_result[0] = 0;
    EXPECT_THAT(bzip2::Decode(encode_result, &decode_result),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test corrupting the trailer.
  {
    std::string encode_result, decode_result;
    bzip2::Encode(input, &encode_result, options);
    ASSERT_GE(encode_result.size(), 1);
    encode_result.resize(encode_result.size() - 1);
    EXPECT_THAT(bzip2::Decode(encode_result, &decode_result),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
