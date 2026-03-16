// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/uri/percent_coder.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/uri/ascii_set.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOkAndHolds;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_uri::AsciiSet;
using ::tensorstore::internal_uri::PercentDecode;
using ::tensorstore::internal_uri::PercentDecodeChar;
using ::tensorstore::internal_uri::PercentEncode;
using ::tensorstore::internal_uri::PercentEncodeKvStoreUriPath;
using ::tensorstore::internal_uri::PercentEncodeUriComponent;
using ::tensorstore::internal_uri::PercentEncodeUriPath;
using ::tensorstore::internal_uri::RSplitPercentEncoded;
using ::testing::Eq;
using ::testing::StrEq;

TEST(PercentDecodeCharTest, Basic) {
  EXPECT_THAT(PercentDecodeChar("%20"), testing::Optional(' '));
  EXPECT_THAT(PercentDecodeChar("%FF"), testing::Optional('\xFF'));
  EXPECT_THAT(PercentDecodeChar("%41"), testing::Optional('A'));
  EXPECT_THAT(PercentDecodeChar("%41a"), Eq(std::nullopt));
  EXPECT_THAT(PercentDecodeChar("%4"), Eq(std::nullopt));
  EXPECT_THAT(PercentDecodeChar("ABC"), Eq(std::nullopt));
  EXPECT_THAT(PercentDecodeChar("%1G"), Eq(std::nullopt));
}

TEST(PercentDecodeTest, Basic) {
  EXPECT_THAT(PercentDecode("abc"), IsOkAndHolds("abc"));
  EXPECT_THAT(PercentDecode("a%20b%2Fc"), IsOkAndHolds("a b/c"));
  // Valid UTF-8 sequence.
  EXPECT_THAT(PercentDecode("%C3%A7"), IsOkAndHolds("\xc3\xa7"));
}

TEST(PercentDecodeTest, Invalid) {
  EXPECT_THAT(PercentDecode("abc%2ghi"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PercentDecode("abc%1"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Invalid UTF-8 sequence.
  EXPECT_THAT(PercentDecode("%fa"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PercentEncodeTest, Basic) {
  EXPECT_THAT(PercentEncode("abc", AsciiSet("abc")), StrEq("abc"));
  EXPECT_THAT(PercentEncode("a b/c", AsciiSet("abc")), StrEq("a%20b%2Fc"));
  EXPECT_THAT(PercentEncode("a b/c", AsciiSet("abc ")), StrEq("a b%2Fc"));
  EXPECT_THAT(PercentEncode("a b/c", ~AsciiSet()), StrEq("a b/c"));

  EXPECT_THAT(PercentEncode("\xc3\xa7", {}), StrEq("%C3%A7"));
}

TEST(PercentEncodeTest, Ascii) {
  std::string ascii;
  ascii.reserve(128);
  for (int i = 0; i < 128; i++) {
    ascii.push_back(static_cast<char>(i));
  }

  EXPECT_THAT(
      PercentEncodeUriPath(ascii),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20!%22%23$%25&'()*+,-./"
            "0123456789:;%3C=%3E%3F@ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%5D%5E_%"
            "60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));

  EXPECT_THAT(
      PercentEncodeKvStoreUriPath(ascii),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20!%22%23$%25&'()*+,-./"
            "0123456789:;%3C=%3E%3F%40ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%5D%5E_%"
            "60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));

  EXPECT_THAT(
      PercentEncodeUriComponent(ascii),
      StrEq("%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%"
            "15%16%17%18%19%1A%1B%1C%1D%1E%1F%20%21%22%23%24%25%26%27%28%29%2A%"
            "2B%2C-.%"
            "2F0123456789%3A%3B%3C%3D%3E%3F%40ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%"
            "5D%5E_%60abcdefghijklmnopqrstuvwxyz%7B%7C%7D~%7F"));
}

TEST(PercentEncodeTest, ValidUtf8Sequence) {
  EXPECT_THAT(PercentEncodeUriPath("\xc3\xa7"), StrEq("%C3%A7"));

  EXPECT_THAT(PercentEncodeKvStoreUriPath("\xc3\xa7"), StrEq("%C3%A7"));

  EXPECT_THAT(PercentEncodeUriComponent("\xc3\xa7"), StrEq("%C3%A7"));
}

TEST(RSplitPercentEncodedTest, Basic) {
  EXPECT_THAT(RSplitPercentEncoded("a%20b%2Fc", AsciiSet("/")),
              testing::Pair("a%20b%2F", "c"));
  EXPECT_THAT(RSplitPercentEncoded("a%20b%2Fc", AsciiSet(" ")),
              testing::Pair("a%20", "b%2Fc"));
  EXPECT_THAT(RSplitPercentEncoded("a%20b%2Fc", AsciiSet("c")),
              testing::Pair("", "a%20b%2Fc"));
}

}  // namespace
