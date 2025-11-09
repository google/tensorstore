// Copyright 2023 The TensorStore Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/write.h"
#include "riegeli/digests/crc32c_digester.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"
#include "tensorstore/internal/riegeli/digest_suffixed_reader.h"
#include "tensorstore/internal/riegeli/digest_suffixed_writer.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::riegeli::Crc32cDigester;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::DigestSuffixedReader;
using ::tensorstore::internal::DigestSuffixedWriter;
using ::tensorstore::internal::LittleEndianDigestVerifier;
using ::tensorstore::internal::LittleEndianDigestWriter;

TEST(DigestSuffixedWriterTest, Basic) {
  std::string s;
  riegeli::StringWriter writer{&s};
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      "hello",
      DigestSuffixedWriter<Crc32cDigester, LittleEndianDigestWriter>{&writer}));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(
      s, ::testing::ElementsAre('h', 'e', 'l', 'l', 'o', 76, 187, 113, 154));
}

TEST(DigestSuffixedWriterTest, Empty) {
  std::string s;
  riegeli::StringWriter writer{&s};
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      "",
      DigestSuffixedWriter<Crc32cDigester, LittleEndianDigestWriter>{&writer}));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(s, ::testing::ElementsAre(0, 0, 0, 0));
}

TEST(DigestSuffixedReaderTest, Success) {
  std::string s{'h',
                'e',
                'l',
                'l',
                'o',
                static_cast<char>(76),
                static_cast<char>(187),
                static_cast<char>(113),
                static_cast<char>(154)};
  riegeli::StringReader reader{&s};
  std::string content;
  TENSORSTORE_ASSERT_OK(riegeli::ReadAll(
      DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{&reader},
      content));
  EXPECT_EQ("hello", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, Empty) {
  std::string s{0, 0, 0, 0};
  riegeli::StringReader reader{&s};
  std::string content;
  TENSORSTORE_ASSERT_OK(riegeli::ReadAll(
      DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{&reader},
      content));
  EXPECT_EQ("", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, InvalidDigest) {
  std::string s{'h', 'e', 'l', 'l', 'o', 0, 0, 0, 0};
  riegeli::StringReader reader{&s};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Digest mismatch, stored digest is 0x00000000 "
                    "but computed digest is 0x9a71bb4c"));
  EXPECT_EQ("hello", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, TooShort) {
  std::string s{
      'a',
      'b',
      'c',
  };
  riegeli::StringReader reader{&s};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Input size of 3 is less than digest size of 4"));
  EXPECT_EQ("", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, UnsizedSuccess) {
  std::string s{'h',
                'e',
                'l',
                'l',
                'o',
                static_cast<char>(76),
                static_cast<char>(187),
                static_cast<char>(113),
                static_cast<char>(154)};
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  TENSORSTORE_ASSERT_OK(riegeli::ReadAll(
      DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{&reader},
      content));
  EXPECT_EQ("hello", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, ExplicitSize) {
  std::string s{'h',
                'e',
                'l',
                'l',
                'o',
                static_cast<char>(76),
                static_cast<char>(187),
                static_cast<char>(113),
                static_cast<char>(154)};
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  TENSORSTORE_ASSERT_OK(riegeli::ReadAll(
      DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{&reader,
                                                                       5},
      content));
  EXPECT_EQ("hello", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, ExplicitSizeTooShort) {
  std::string s{'h',
                'e',
                'l',
                'l',
                'o',
                static_cast<char>(76),
                static_cast<char>(187),
                static_cast<char>(113),
                static_cast<char>(154)};
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader, 4},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss, "Digest mismatch.*"));
  EXPECT_EQ("hell", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, ExplicitSizeTooLong) {
  std::string s{'h',
                'e',
                'l',
                'l',
                'o',
                static_cast<char>(76),
                static_cast<char>(187),
                static_cast<char>(113),
                static_cast<char>(154)};
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader, 6},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss, "Unexpected end of input.*"));
  EXPECT_EQ("hello\x4c", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, UnsizedMismatch) {
  std::string s{'h', 'e', 'l', 'l', 'o', 0, 0, 0, 0};
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Digest mismatch, stored digest is 0x00000000 "
                    "but computed digest is 0x9a71bb4c"));
  EXPECT_EQ("hello", content);
  ASSERT_TRUE(reader.Close());
}

TEST(DigestSuffixedReaderTest, UnsizedTooShort) {
  std::string s{
      'a',
      'b',
      'c',
  };
  std::string compressed;
  TENSORSTORE_ASSERT_OK(riegeli::Write(
      s, riegeli::ZlibWriter{riegeli::StringWriter{&compressed}}));
  riegeli::ZlibReader reader{riegeli::StringReader{&compressed}};
  std::string content;
  EXPECT_THAT(
      riegeli::ReadAll(
          DigestSuffixedReader<Crc32cDigester, LittleEndianDigestVerifier>{
              &reader},
          content),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Input size of 3 is less than digest size of 4"));
  EXPECT_EQ("", content);
  ASSERT_TRUE(reader.Close());
}

}  // namespace
