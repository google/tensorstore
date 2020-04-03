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

#include "tensorstore/internal/utf8.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace {

using tensorstore::internal::IsValidUtf8;

// Basic sanity checks of `IsValidUtf8`.
//
// These tests are not exhaustive because the third-party utf8_decode.h
// implementation is presumed to already be correct.

// UTF-8 syntax from RFC 3629 Section 4:
//
// A UTF-8 string is a sequence of octets representing a sequence of UCS
// characters.  An octet sequence is valid UTF-8 only if it matches the
// following syntax, which is derived from the rules for encoding UTF-8
// and is expressed in the ABNF of [RFC2234].
// UTF8-octets = *( UTF8-char )
// UTF8-char   = UTF8-1 / UTF8-2 / UTF8-3 / UTF8-4
// UTF8-1      = %x00-7F
// UTF8-2      = %xC2-DF UTF8-tail
// UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
//               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
// UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
//               %xF4 %x80-8F 2( UTF8-tail )
// UTF8-tail   = %x80-BF

TEST(IsValidUtf8Test, Empty) {
  // Empty string
  EXPECT_TRUE(IsValidUtf8(""));
}

TEST(IsValidUtf8Test, Ascii) {
  EXPECT_TRUE(IsValidUtf8("ascii"));
  // Singe NUL byte
  EXPECT_TRUE(IsValidUtf8(absl::string_view("\0", 1)));
}

TEST(IsValidUtf8Test, TwoByte) {
  EXPECT_TRUE(IsValidUtf8("\xc2\x80"));
  EXPECT_TRUE(IsValidUtf8("\xc2\x80hello\xc2\xbf"));
}

TEST(IsValidUtf8Test, ThreeByte) {
  //
  EXPECT_TRUE(IsValidUtf8("\xe0\xa0\x80"));
}

TEST(IsValidUtf8Test, FourByte) {
  EXPECT_TRUE(IsValidUtf8("\xf0\x90\x80\x80"));
}

// Tests that surrogate code points are rejected.
TEST(IsValidUtf8Test, Surrogate) {
  // Lead surrogate
  EXPECT_FALSE(IsValidUtf8("\xed\xa0\x80"));
  // Trail surrogate
  EXPECT_FALSE(IsValidUtf8("\xed\xb0\x80"));
  // Surrogate pair
  EXPECT_FALSE(IsValidUtf8("\xed\xa0\x80\xed\xb0\x80"));
}

TEST(IsValidUtf8Test, IllFormedFirstByte) {
  EXPECT_FALSE(IsValidUtf8("\x80"));
  EXPECT_FALSE(IsValidUtf8("\xC1"));
  EXPECT_FALSE(IsValidUtf8("\xF5"));
  EXPECT_FALSE(IsValidUtf8("\xFF"));
}

TEST(IsValidUtf8Test, OverlongNul) {
  EXPECT_FALSE(IsValidUtf8("\xc0\x80"));
  EXPECT_FALSE(IsValidUtf8("\xe0\x80\x80"));
  EXPECT_FALSE(IsValidUtf8("\xf0\x80\x80\x80"));
  EXPECT_FALSE(IsValidUtf8("\xf8\x80\x80\x80\x80"));
  EXPECT_FALSE(IsValidUtf8("\xfc\x80\x80\x80\x80\x80"));
}

}  // namespace
