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

using std::ios_base;

using ::absl::Cord;
using ::absl::CordBuffer;
using ::Aws::Utils::Stream::DefaultUnderlyingStream;
using ::Aws::MakeUnique;
using ::tensorstore::internal_kvstore_s3::CordStreamBuf;

namespace {

static constexpr char kAwsTag[] = "AWS";
static constexpr int kNBuffers = 3;
static constexpr auto kBufferSize = CordBuffer::kDefaultLimit;

absl::Cord ThreeBufferCord() {
  absl::Cord cord;
  for(char ch = 0; ch < kNBuffers; ++ch) {
    cord.Append(std::string(kBufferSize, '1' + ch));
  }
  assert(std::distance(cord.Chunks().begin(), cord.Chunks().end()) == 3);
  return cord;
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
  EXPECT_TRUE(is.eof());
}


TEST(CordStreamBufTest, Write) {
  auto os = DefaultUnderlyingStream(MakeUnique<CordStreamBuf>(kAwsTag));
  os << "Hello World";
  os << " ";
  os << "This is a test";
  EXPECT_TRUE(os.good());
  auto cord = dynamic_cast<CordStreamBuf *>(os.rdbuf())->GetCord();
  EXPECT_EQ(cord, "Hello World This is a test");

  // Single Cord chunk
  auto it = cord.chunk_begin();
  EXPECT_EQ(*it, "Hello World This is a test");
  EXPECT_EQ(++it, cord.chunk_end());
}


TEST(CordStreamBufTest, BufferSeek) {
  auto buffer = CordStreamBuf(ThreeBufferCord());

  // Seeks from beginning
  EXPECT_EQ(buffer.pubseekoff(0, ios_base::beg, ios_base::in), 0);
  EXPECT_EQ(buffer.pubseekoff(10, ios_base::beg, ios_base::in), 10);
  EXPECT_EQ(buffer.pubseekoff(10 + kBufferSize, ios_base::beg, ios_base::in), 10 + kBufferSize);
  EXPECT_EQ(buffer.pubseekoff(10 + 2*kBufferSize, ios_base::beg, ios_base::in), 10 + 2*kBufferSize);
  EXPECT_EQ(buffer.pubseekoff(10 + 3*kBufferSize, ios_base::beg, ios_base::in), -1);  // eof

  // Seeks from current position
  EXPECT_EQ(buffer.pubseekoff(0, ios_base::beg, ios_base::in), 0);
  EXPECT_EQ(buffer.pubseekoff(10, ios_base::cur, ios_base::in), 10);
  EXPECT_EQ(buffer.pubseekoff(kBufferSize, ios_base::cur, ios_base::in), 10 + kBufferSize);
  EXPECT_EQ(buffer.pubseekoff(kBufferSize, ios_base::cur, ios_base::in), 10 + 2*kBufferSize);
  EXPECT_EQ(buffer.pubseekoff(kBufferSize, ios_base::cur, ios_base::in), -1);  // eof
}

/// Test that reading the CordStreamBuf reads the Cord
TEST(CordStreamBufTest, GetEntireStreamBuf) {
  auto is = DefaultUnderlyingStream(
    MakeUnique<CordStreamBuf>(kAwsTag, ThreeBufferCord()));

  int count = 0;
  char ch;

  while(is.get(ch)) {
    EXPECT_EQ(ch, '1' + (count / kBufferSize));
    EXPECT_TRUE(is.good());
    EXPECT_FALSE(is.eof());
    ++count;
    EXPECT_EQ(is.tellg(), count < kBufferSize * kNBuffers ? count : -1);
  }
  EXPECT_EQ(count, kBufferSize * kNBuffers);
  EXPECT_FALSE(is.good());
  EXPECT_TRUE(is.eof());
}

/// Test seeking within the CordStreamBuf
TEST(CordStreamBufTest, ReadSeek) {
  auto is = DefaultUnderlyingStream(
    MakeUnique<CordStreamBuf>(kAwsTag, ThreeBufferCord()));

  for(char b = 0; b < kNBuffers; ++b) {
    is.seekg(5 + kBufferSize * b);
    EXPECT_EQ(is.tellg(), 5 + kBufferSize * b);
    char result[6] = {0x00};
    is.read(result, sizeof(result));
    auto expected = std::string(sizeof(result), '1' + b);
    EXPECT_EQ(std::string_view(result, sizeof(result)), expected);
    EXPECT_TRUE(is.good());
    EXPECT_EQ(is.tellg(), 5 + kBufferSize * b + sizeof(result));
  }

  is.seekg(kBufferSize * kNBuffers);
  EXPECT_EQ(is.tellg(), -1);
  EXPECT_FALSE(is.good());
}

} // namespace