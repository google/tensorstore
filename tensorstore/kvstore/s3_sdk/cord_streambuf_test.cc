// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/s3_sdk/cord_streambuf.h"

#include <iterator>
#include <string>

#include "absl/strings/cord.h"

#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/stream/ResponseStream.h>

#include <gtest/gtest.h>

using ::absl::Cord;
using ::absl::CordBuffer;
using ::Aws::Utils::Stream::DefaultUnderlyingStream;
using ::Aws::MakeUnique;
using ::tensorstore::internal_kvstore_s3::CordStreamBuf;

namespace {

static constexpr char kAwsTag[] = "AWS";
static constexpr int kNBuffers = 3;

CordStreamBuf & GetStreamBuf(DefaultUnderlyingStream & stream) {
  return *dynamic_cast<CordStreamBuf *>(stream.rdbuf());
}

TEST(CordStreamBufTest, Read) {
  auto cord = absl::Cord{"Hello World This is a test"};
  auto is = DefaultUnderlyingStream(MakeUnique<CordStreamBuf>(kAwsTag, std::move(cord)));
  std::istreambuf_iterator<char> in_it{is}, end;
  std::string s{in_it, end};
  EXPECT_EQ(s, "Hello World This is a test");
  EXPECT_TRUE(is.good());

  // eof triggered.
  char ch;
  EXPECT_FALSE(is.get(ch));
  EXPECT_FALSE(is.good());
}


TEST(CordStreamBufTest, Write) {
  auto os = DefaultUnderlyingStream(MakeUnique<CordStreamBuf>(kAwsTag));
  os << "Hello World";
  os << " ";
  os << "This is a test";
  EXPECT_TRUE(os.good());
  auto cord = GetStreamBuf(os).GetCord();
  EXPECT_EQ(cord, "Hello World This is a test");

  // Single Cord chunk
  auto it = cord.chunk_begin();
  EXPECT_EQ(*it, "Hello World This is a test");
  EXPECT_EQ(++it, cord.chunk_end());
}


/// Test seeking within the CordStreamBuf
TEST(CordSreamBufTest, ReadSeek) {
  absl::Cord cord;
  for(char ch = 0; ch < kNBuffers; ++ch) {
    cord.Append(std::string(CordBuffer::kDefaultLimit, '1' + ch));
  }
  EXPECT_EQ(std::distance(cord.Chunks().begin(), cord.Chunks().end()), 3);

  auto is = DefaultUnderlyingStream(
    MakeUnique<CordStreamBuf>(kAwsTag, std::move(cord)));

  for(char ch = 0; ch < kNBuffers; ++ch) {
    is.seekg(5 + CordBuffer::kDefaultLimit * ch);
    EXPECT_EQ(is.tellg(), 5 + CordBuffer::kDefaultLimit * ch);
    char result[6] = {0x00};
    is.read(result, sizeof(result));
    auto expected = std::string(sizeof(result), '1' + ch);
    EXPECT_EQ(std::string_view(result, sizeof(result)), expected);
    EXPECT_TRUE(is.good());
    EXPECT_EQ(is.tellg(), 5 + CordBuffer::kDefaultLimit * ch + sizeof(result));
  }
}

/// Test that reading the CordStreamBuf reads the Cord
TEST(CordStreamBufTest, GetEntireStreamBuf) {
  absl::Cord cord;
  for(char ch = 0; ch < kNBuffers; ++ch) {
    cord.Append(std::string(CordBuffer::kDefaultLimit, '1' + ch));
  }
  EXPECT_EQ(std::distance(cord.Chunks().begin(), cord.Chunks().end()), 3);

  auto is = DefaultUnderlyingStream(
    MakeUnique<CordStreamBuf>(kAwsTag, std::move(cord)));

  int count = 0;
  char ch;
  while(is.get(ch)) {
    EXPECT_EQ(ch, '1' + count / CordBuffer::kDefaultLimit);
    ++count;
  }
  EXPECT_EQ(count, CordBuffer::kDefaultLimit * kNBuffers);
}

} // namespace