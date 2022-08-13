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

#include "tensorstore/internal/compression/neuroglancer_compressed_segmentation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"

namespace {

using ::tensorstore::neuroglancer_compressed_segmentation::DecodeBlock;
using ::tensorstore::neuroglancer_compressed_segmentation::DecodeChannel;
using ::tensorstore::neuroglancer_compressed_segmentation::DecodeChannels;
using ::tensorstore::neuroglancer_compressed_segmentation::EncodeBlock;
using ::tensorstore::neuroglancer_compressed_segmentation::EncodeChannel;
using ::tensorstore::neuroglancer_compressed_segmentation::EncodeChannels;
using ::tensorstore::neuroglancer_compressed_segmentation::EncodedValueCache;

std::vector<std::uint32_t> AsVec(std::string_view s) {
  EXPECT_EQ(0, s.size() % 4);
  std::vector<std::uint32_t> out(s.size() / 4);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = absl::little_endian::Load32(s.data() + i * 4);
  }
  return out;
}

std::string FromVec(std::vector<std::uint32_t> v) {
  std::string s;
  s.resize(v.size() * 4);
  for (size_t i = 0; i < v.size(); ++i) {
    absl::little_endian::Store32(s.data() + i * 4, v[i]);
  }
  return s;
}

template <typename T>
void TestBlockRoundTrip(std::vector<T> input,
                        const std::ptrdiff_t (&input_shape)[3],
                        const std::ptrdiff_t (&block_shape)[3],
                        size_t expected_encoded_bits,
                        size_t expected_table_offset,
                        std::vector<uint32_t> expected_output,
                        EncodedValueCache<T> expected_cache) {
  // Use non-empty `output` to test that existing contents are preserved.
  std::string output{1, 2, 3};
  ASSERT_EQ(input_shape[0] * input_shape[1] * input_shape[2], input.size());
  constexpr std::ptrdiff_t s = sizeof(T);
  const std::ptrdiff_t input_byte_strides[3] = {
      input_shape[1] * input_shape[2] * s, input_shape[2] * s, s};
  size_t encoded_bits;
  size_t table_offset;
  EncodedValueCache<uint64_t> cache;
  const size_t initial_offset = output.size();
  EncodeBlock(input.data(), input_shape, input_byte_strides, block_shape,
              initial_offset, &encoded_bits, &table_offset, &cache, &output);
  ASSERT_THAT(output.substr(0, 3), ::testing::ElementsAre(1, 2, 3));
  EXPECT_EQ(expected_encoded_bits, encoded_bits);
  EXPECT_EQ(expected_table_offset, table_offset);
  EXPECT_EQ(expected_output, AsVec(output.substr(initial_offset)));
  EXPECT_EQ(expected_cache, cache);
  std::vector<T> decoded_output(input.size());
  EXPECT_TRUE(DecodeBlock(
      encoded_bits, output.data() + initial_offset,
      output.data() + initial_offset + table_offset * 4,
      (output.size() - (initial_offset + table_offset * 4)) / sizeof(T),
      block_shape, input_shape, input_byte_strides, decoded_output.data()));
  EXPECT_EQ(input, decoded_output);
}

template <typename T>
void TestSingleChannelRoundTrip(std::vector<T> input,
                                const std::ptrdiff_t (&input_shape)[3],
                                const std::ptrdiff_t (&block_shape)[3],
                                std::vector<uint32_t> expected_output) {
  // Use non-empty `output` to test that existing contents are preserved.
  std::string output{1, 2, 3};
  ASSERT_EQ(input_shape[0] * input_shape[1] * input_shape[2], input.size());
  constexpr std::ptrdiff_t s = sizeof(T);
  const std::ptrdiff_t input_byte_strides[3] = {
      input_shape[1] * input_shape[2] * s, input_shape[2] * s, s};
  const size_t initial_offset = output.size();
  EncodeChannel(input.data(), input_shape, input_byte_strides, block_shape,
                &output);
  ASSERT_THAT(output.substr(0, 3), ::testing::ElementsAre(1, 2, 3));
  EXPECT_EQ(expected_output, AsVec(output.substr(initial_offset)));
  std::vector<T> decoded_output(input.size());
  // Use a temporary std::vector rather than std::string as the input to
  // `DecodeChannels` in order to ensure AddressSanitizer catches
  // one-past-the-end accesses.
  std::vector<char> output_copy(output.begin() + initial_offset, output.end());
  EXPECT_TRUE(DecodeChannel(
      std::string_view(output_copy.data(), output_copy.size()), block_shape,
      input_shape, input_byte_strides, decoded_output.data()));
  EXPECT_EQ(input, decoded_output);
}

template <typename T>
void TestDecodeChannelError(std::string_view input,
                            const std::ptrdiff_t (&block_shape)[3],
                            const std::ptrdiff_t (&input_shape)[3]) {
  constexpr std::ptrdiff_t s = sizeof(T);
  const std::ptrdiff_t input_byte_strides[3] = {
      input_shape[1] * input_shape[2] * s, input_shape[2] * s, s};
  std::vector<T> decoded_output(input_shape[0] * input_shape[1] *
                                input_shape[2]);
  EXPECT_FALSE(DecodeChannel(input, block_shape, input_shape,
                             input_byte_strides, decoded_output.data()));
}

template <typename T>
void TestMultipleChannelsRoundTripBytes(
    std::vector<T> input, const std::ptrdiff_t (&input_shape)[4],
    const std::ptrdiff_t (&block_shape)[4],
    std::vector<unsigned char> expected_output) {
  // Use non-empty `output` to test that existing contents are preserved.
  std::string output{1, 2, 3};
  ASSERT_EQ(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3],
            input.size());
  constexpr std::ptrdiff_t s = sizeof(T);
  const std::ptrdiff_t input_byte_strides[4] = {
      input_shape[1] * input_shape[2] * input_shape[3] * s,
      input_shape[2] * input_shape[3] * s, input_shape[3] * s, s};
  const size_t initial_offset = output.size();
  EncodeChannels(input.data(), input_shape, input_byte_strides, block_shape,
                 &output);
  ASSERT_THAT(output.substr(0, 3), ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(output.substr(initial_offset),
              ::testing::ElementsAreArray(expected_output));
  std::vector<T> decoded_output(input.size());
  EXPECT_TRUE(DecodeChannels(output.substr(initial_offset), block_shape,
                             input_shape, input_byte_strides,
                             decoded_output.data()));
  EXPECT_EQ(input, decoded_output);
}

// Tests 0-bit encoding.
TEST(EncodeBlockTest, Basic0) {
  TestBlockRoundTrip<uint64_t>(/*input=*/{3, 3, 3, 3},
                               /*input_shape=*/{1, 2, 2},
                               /*block_shape=*/{1, 2, 2},
                               /*expected_encoded_bits=*/0,
                               /*expected_table_offset=*/0,
                               /*expected_output=*/{3, 0},
                               /*expected_cache=*/{{{3}, 0}});
}

// Tests 1-bit encoding.
TEST(EncodeBlockTest, Basic1) {
  TestBlockRoundTrip<uint64_t>(
      /*input=*/{4, 3, 4, 4},
      /*input_shape=*/{1, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_encoded_bits=*/1,
      /*expected_table_offset=*/1,
      /*expected_output=*/{0b1101, 3, 0, 4, 0},
      /*expected_cache=*/{{{3, 4}, 1}});
}

// Test 1-bit encoding, input_shape != block_shape.
TEST(EncodeBlockTest, SizeMismatch) {
  TestBlockRoundTrip<uint64_t>(
      /*input=*/{4, 3, 4, 3},
      /*input_shape=*/{1, 2, 2},
      /*block_shape=*/{1, 2, 3},
      /*expected_encoded_bits=*/1,
      /*expected_table_offset=*/1,
      /*expected_output=*/{0b001001, 3, 0, 4, 0},
      /*expected_cache=*/{{{3, 4}, 1}});
}

// Test 2-bit encoding.
TEST(EncodeBlockTest, Basic2) {
  TestBlockRoundTrip<uint64_t>(
      /*input=*/{4, 3, 5, 4},
      /*input_shape=*/{1, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_encoded_bits=*/2,
      /*expected_table_offset=*/1,
      /*expected_output=*/{0b01100001, 3, 0, 4, 0, 5, 0},
      /*expected_cache=*/{{{3, 4, 5}, 1}});
}

TEST(EncodeChannelTest, Basic) {
  TestSingleChannelRoundTrip<std::uint64_t>(
      /*input=*/{4, 3, 5, 4, 1, 3, 3, 3},
      /*input_shape=*/{2, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_output=*/
      {5 | (2 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0, 4, 0, 5, 0,
       0b1110, 1, 0, 3, 0});
}

TEST(EncodeChannelTest, BasicCached) {
  TestSingleChannelRoundTrip<std::uint64_t>(
      /*input=*/
      {
          4, 3, 5, 4,  //
          1, 3, 3, 3,  //
          3, 1, 1, 1,  //
          5, 5, 3, 4,  //
      },
      /*input_shape=*/{4, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_output=*/
      {
          9 | (2 << 24),
          8,
          16 | (1 << 24),
          15,
          16 | (1 << 24),
          20,
          9 | (2 << 24),
          21,
          0b01100001,
          3,
          0,
          4,
          0,
          5,
          0,
          0b1110,
          1,
          0,
          3,
          0,
          0b00000001,
          0b01001010,
      });
}

TEST(EncodeChannelTest, BasicCachedZeroBitsAtEnd) {
  TestSingleChannelRoundTrip<std::uint64_t>(
      /*input=*/
      {
          3, 3, 3, 3,  //
          3, 3, 3, 3,  //
          3, 3, 3, 3,  //
          3, 3, 3, 3,  //
      },
      /*input_shape=*/{4, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_output=*/
      {
          8 | (0 << 24),
          8,
          8 | (0 << 24),
          10,
          8 | (0 << 24),
          10,
          8 | (0 << 24),
          10,
          3,
          0,
      });
}

TEST(EncodeChannelTest, BasicCached32) {
  TestSingleChannelRoundTrip<std::uint32_t>(
      /*input=*/
      {
          4, 3, 5, 4,  //
          1, 3, 3, 3,  //
          3, 1, 1, 1,  //
          5, 5, 3, 4,  //
      },
      /*input_shape=*/{4, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_output=*/
      {
          9 | (2 << 24),
          8,
          13 | (1 << 24),
          12,
          13 | (1 << 24),
          15,
          9 | (2 << 24),
          16,
          0b01100001,
          3,
          4,
          5,
          0b1110,
          1,
          3,
          0b00000001,
          0b01001010,
      });
}

TEST(EncodeChannelsTest, Basic1Channel1Block) {
  TestMultipleChannelsRoundTripBytes<std::uint64_t>(
      /*input=*/{4, 0, 4, 0},
      /*input_shape=*/{1, 1, 2, 2},
      /*block_shape=*/{1, 2, 2},
      /*expected_output=*/
      {
          1,      0, 0, 0,              //
          3,      0, 0, 1, 2, 0, 0, 0,  //
          0b0101, 0, 0, 0,              //
          0,      0, 0, 0,              //
          0,      0, 0, 0,              //
          4,      0, 0, 0,              //
          0,      0, 0, 0,              //
      });
}

TEST(DecodeChannelTest, SizeNotMultipleOf4) {
  auto input = FromVec({5 | (2 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  input.resize(input.size() - 1);
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, Truncated) {
  auto input = FromVec({5 | (2 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  input.resize(input.size() - 4);
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, NonPowerOf2EncodedBits) {
  auto input = FromVec({5 | (3 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, MoreThan32EncodedBits) {
  auto input = FromVec({5 | (33 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, MissingBlockHeaders) {
  auto input = FromVec({5 | (3 << 24), 4, 12 | (1 << 24)});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, InvalidEncodedValueOffset) {
  auto input = FromVec({5 | (2 << 24), 16, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, InvalidTableOffset) {
  auto input = FromVec({16 | (2 << 24), 4, 12 | (1 << 24), 11, 0b01100001, 3, 0,
                        4, 0, 5, 0, 0b1110, 1, 0, 3, 0});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

TEST(DecodeChannelTest, MissingEncodedValues) {
  auto input = FromVec(
      {5 | (2 << 24), 4, 0 | (1 << 24), 11, 0b01100001, 3, 0, 4, 0, 5, 0});
  TestDecodeChannelError<std::uint64_t>(
      /*input=*/input,
      /*block_shape=*/{1, 2, 2},
      /*input_shape=*/{2, 2, 2});
}

template <typename T>
void RandomRoundTrip(size_t max_block_size, size_t max_input_size,
                     size_t max_channels, size_t max_distinct_ids,
                     size_t num_iterations) {
  absl::BitGen gen;
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    std::ptrdiff_t block_shape[3];
    std::ptrdiff_t input_shape[4];
    input_shape[0] = absl::Uniform(gen, 1u, max_channels + 1);
    for (int i = 0; i < 3; ++i) {
      block_shape[i] = absl::Uniform(gen, 1u, max_block_size + 1);
      input_shape[i + 1] = absl::Uniform(gen, 1u, max_input_size + 1);
    }
    std::vector<T> input(input_shape[0] * input_shape[1] * input_shape[2] *
                         input_shape[3]);
    std::vector<T> labels(max_distinct_ids);
    for (auto& label : labels) {
      label = absl::Uniform<T>(gen);
    }
    for (auto& label : input) {
      label = labels[absl::Uniform(gen, 0u, labels.size())];
    }
    constexpr std::ptrdiff_t s = sizeof(T);
    const std::ptrdiff_t input_byte_strides[4] = {
        input_shape[1] * input_shape[2] * input_shape[3] * s,
        input_shape[2] * input_shape[3] * s, input_shape[3] * s, s};
    std::string output;
    EncodeChannels(input.data(), input_shape, input_byte_strides, block_shape,
                   &output);
    std::vector<T> decoded_output(input.size());
    EXPECT_TRUE(DecodeChannels(output, block_shape, input_shape,
                               input_byte_strides, decoded_output.data()));
    EXPECT_EQ(input, decoded_output);
  }
}

void RandomRoundTripBothDataTypes(size_t max_block_size, size_t max_input_size,
                                  size_t max_channels, size_t max_distinct_ids,
                                  size_t num_iterations) {
  RandomRoundTrip<std::uint32_t>(max_block_size, max_input_size, max_channels,
                                 max_distinct_ids, num_iterations);
  RandomRoundTrip<std::uint64_t>(max_block_size, max_input_size, max_channels,
                                 max_distinct_ids, num_iterations);
}

TEST(RoundTripTest, Random) {
  RandomRoundTripBothDataTypes(/*max_block_size=*/4, /*max_input_size=*/10,
                               /*max_channels=*/3, /*max_distinct_ids=*/16,
                               /*num_iterations=*/100);
  RandomRoundTripBothDataTypes(/*max_block_size=*/10, /*max_input_size=*/16,
                               /*max_channels=*/3, /*max_distinct_ids=*/1000,
                               /*num_iterations=*/100);
}

}  // namespace
