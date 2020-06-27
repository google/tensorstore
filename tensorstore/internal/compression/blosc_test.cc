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

#include "tensorstore/internal/compression/blosc.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include <blosc.h>
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;

namespace blosc = tensorstore::blosc;

std::vector<blosc::Options> GetTestOptions() {
  // The element_size member is not set here.
  return {
      blosc::Options{"lz4", 5, -1, 0},
      blosc::Options{"lz4", 5, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4", 0, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4hc", 5, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4", 5, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4", 1, BLOSC_NOSHUFFLE, 0},
      blosc::Options{"lz4", 5, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4", 9, BLOSC_BITSHUFFLE, 0},
      blosc::Options{"zlib", 1, BLOSC_NOSHUFFLE, 0},
      blosc::Options{"zstd", 1, BLOSC_SHUFFLE, 0},
      blosc::Options{"blosclz", 1, BLOSC_BITSHUFFLE, 0},
      blosc::Options{"snappy", 1, BLOSC_NOSHUFFLE, 0},
      blosc::Options{"lz4", 5, BLOSC_SHUFFLE, 0},
      blosc::Options{"lz4", 5, BLOSC_SHUFFLE, 256},
      blosc::Options{"lz4", 1, BLOSC_NOSHUFFLE, 256},
  };
}

std::vector<absl::Cord> GetTestArrays() {
  std::vector<absl::Cord> arrays;

  // Add empty array.
  arrays.emplace_back();

  {
    std::string arr(100, '\0');
    unsigned char v = 0;
    for (auto& x : arr) {
      x = (v += 7);
    }
    arrays.emplace_back(std::move(arr));
  }

  arrays.emplace_back("The quick brown fox jumped over the lazy dog.");
  return arrays;
}

// Tests encoding and decoding, and verifies that the result is appended to the
// output string without clearing the existing contents.
TEST(BloscTest, EncodeDecode) {
  for (blosc::Options options : GetTestOptions()) {
    for (const auto& array : GetTestArrays()) {
      for (const std::size_t element_size : {1, 2, 10}) {
        options.element_size = element_size;
        absl::Cord encode_result("abc"), decode_result("def");
        TENSORSTORE_ASSERT_OK(blosc::Encode(array, &encode_result, options));
        ASSERT_GE(encode_result.size(), 3);
        EXPECT_EQ("abc", encode_result.Subcord(0, 3));
        TENSORSTORE_ASSERT_OK(
            blosc::Decode(encode_result.Subcord(3, encode_result.size() - 3),
                          &decode_result));
        EXPECT_EQ("def" + std::string(array), decode_result);
      }
    }
  }
}

// Tests that the compressed data has the expected blosc complib.
TEST(BloscTest, CheckComplib) {
  absl::Cord array("The quick brown fox jumped over the lazy dog.");
  const std::vector<std::pair<std::string, std::string>>
      cnames_and_complib_names{{BLOSC_BLOSCLZ_COMPNAME, BLOSC_BLOSCLZ_LIBNAME},
                               {BLOSC_LZ4_COMPNAME, BLOSC_LZ4_LIBNAME},
                               {BLOSC_LZ4HC_COMPNAME, BLOSC_LZ4_LIBNAME},
                               {BLOSC_SNAPPY_COMPNAME, BLOSC_SNAPPY_LIBNAME},
                               {BLOSC_ZLIB_COMPNAME, BLOSC_ZLIB_LIBNAME},
                               {BLOSC_ZSTD_COMPNAME, BLOSC_ZSTD_LIBNAME}};
  for (const auto& pair : cnames_and_complib_names) {
    blosc::Options options{/*.compressor==*/pair.first.c_str(), /*.clevel=*/5,
                           /*.shuffle=*/-1, /*.blocksize=*/0,
                           /*.element_size=*/1};
    absl::Cord encoded;
    TENSORSTORE_ASSERT_OK(blosc::Encode(array, &encoded, options));
    ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
    const char* complib = blosc_cbuffer_complib(encoded.Flatten().data());
    EXPECT_EQ(pair.second, complib);
  }
}

// Tests that the compressed data has the expected blosc shuffle and type size
// parameters.
TEST(BloscTest, CheckShuffleAndElementSize) {
  absl::Cord array("The quick brown fox jumped over the lazy dog.");
  for (int shuffle = -1; shuffle <= 2; ++shuffle) {
    for (const std::size_t element_size : {1, 2, 10}) {
      blosc::Options options{/*.compressor==*/"lz4", /*.clevel=*/5,
                             /*.shuffle=*/shuffle, /*.blocksize=*/0,
                             /*.element_size=*/element_size};
      absl::Cord encoded;
      TENSORSTORE_ASSERT_OK(blosc::Encode(array, &encoded, options));
      ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
      size_t typesize;
      int flags;
      blosc_cbuffer_metainfo(encoded.Flatten().data(), &typesize, &flags);
      EXPECT_EQ(element_size, typesize);
      const bool expected_byte_shuffle =
          shuffle == 1 || (shuffle == -1 && element_size != 1);
      const bool expected_bit_shuffle =
          shuffle == 2 || (shuffle == -1 && element_size == 1);
      EXPECT_EQ(expected_byte_shuffle,
                static_cast<bool>(flags & BLOSC_DOSHUFFLE));
      EXPECT_EQ(expected_bit_shuffle,
                static_cast<bool>(flags & BLOSC_DOBITSHUFFLE));
    }
  }
}

// Tests that the compressed data has the expected blosc blocksize.
TEST(BloscTest, CheckBlocksize) {
  absl::Cord array{std::string(100000, '\0')};
  for (std::size_t blocksize : {256, 512, 1024}) {
    // Set clevel to 0 to ensure our blocksize choice will be respected.
    // Otherwise blosc may choose a different blocksize.
    blosc::Options options{/*.compressor==*/"lz4", /*.clevel=*/0,
                           /*.shuffle=*/0, /*.blocksize=*/blocksize,
                           /*.element_size=*/1};
    absl::Cord encoded;
    TENSORSTORE_ASSERT_OK(blosc::Encode(array, &encoded, options));
    ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
    size_t nbytes, cbytes, bsize;
    blosc_cbuffer_sizes(encoded.Flatten().data(), &nbytes, &cbytes, &bsize);
    EXPECT_EQ(blocksize, bsize);
  }
}

// Tests that encoding a buffer longer than BLOSC_MAX_BUFFERSIZE bytes results
// in an error.
TEST(BloscTest, TooLong) {
  blosc::Options options{/*.compressor==*/"lz4", /*.clevel=*/5,
                         /*.shuffle=*/-1, /*.blocksize=*/0,
                         /*.element_size=*/1};
  absl::Cord encoded;
  EXPECT_THAT(blosc::Encode(tensorstore::internal::FlatCordBuilder(
                                BLOSC_MAX_BUFFERSIZE + 1)
                                .Build(),
                            &encoded, options),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that decoding data with a corrupted blosc header returns an error.
TEST(BloscTest, DecodeHeaderCorrupted) {
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(
      blosc::Encode(input, &encode_result,
                    blosc::Options{/*.compressor=*/"lz4", /*.clevel=*/1,
                                   /*.shuffle=*/-1, /*.blocksize=*/0,
                                   /*element_size=*/1}));
  ASSERT_GE(encode_result.size(), 1);
  std::string corrupted(encode_result);
  corrupted[0] = 0;
  EXPECT_THAT(blosc::Decode(absl::Cord(corrupted), &decode_result),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that decoding data with an incomplete blosc header returns an error.
TEST(BloscCompressorTest, DecodeHeaderTruncated) {
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(
      blosc::Encode(input, &encode_result,
                    blosc::Options{/*.compressor=*/"lz4", /*.clevel=*/1,
                                   /*.shuffle=*/-1, /*.blocksize=*/0,
                                   /*element_size=*/1}));
  ASSERT_GE(encode_result.size(), 5);
  EXPECT_THAT(blosc::Decode(encode_result.Subcord(0, 5), &decode_result),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that decoding truncated data returns an error.
TEST(BloscCompressorTest, DecodeDataTruncated) {
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(
      blosc::Encode(input, &encode_result,
                    blosc::Options{/*.compressor=*/"lz4", /*.clevel=*/1,
                                   /*.shuffle=*/-1, /*.blocksize=*/0,
                                   /*element_size=*/1}));
  EXPECT_THAT(blosc::Decode(encode_result.Subcord(0, BLOSC_MIN_HEADER_LENGTH),
                            &decode_result),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
