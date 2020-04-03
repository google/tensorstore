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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <blosc.h>
#include <nlohmann/json.hpp>
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::Compressor;

std::vector<::nlohmann::json> GetTestCompressorSpecs() {
  return {
      {{"id", "blosc"}},
      {{"id", "blosc"}, {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"}, {"clevel", 0}, {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"}, {"cname", "lz4hc"}, {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"}, {"cname", "lz4"}, {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"},
       {"cname", "lz4"},
       {"clevel", 1},
       {"shuffle", BLOSC_NOSHUFFLE}},
      {{"id", "blosc"},
       {"cname", "lz4"},
       {"clevel", 5},
       {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"},
       {"cname", "lz4"},
       {"clevel", 9},
       {"shuffle", BLOSC_BITSHUFFLE}},
      {{"id", "blosc"},
       {"cname", "zlib"},
       {"clevel", 1},
       {"shuffle", BLOSC_NOSHUFFLE}},
      {{"id", "blosc"},
       {"cname", "zstd"},
       {"clevel", 1},
       {"shuffle", BLOSC_SHUFFLE}},
      {{"id", "blosc"},
       {"cname", "blosclz"},
       {"clevel", 1},
       {"shuffle", BLOSC_BITSHUFFLE}},
      {{"id", "blosc"},
       {"cname", "snappy"},
       {"clevel", 1},
       {"shuffle", BLOSC_NOSHUFFLE}},
      {{"id", "blosc"}, {"shuffle", BLOSC_SHUFFLE}, {"blocksize", 0}},
      {{"id", "blosc"}, {"shuffle", BLOSC_SHUFFLE}, {"blocksize", 256}},
      {{"id", "blosc"},
       {"cname", "lz4"},
       {"clevel", 1},
       {"shuffle", BLOSC_NOSHUFFLE},
       {"blocksize", 256}},
  };
}

std::vector<std::string> GetTestArrays() {
  std::vector<std::string> arrays;

  // Add empty array.
  arrays.emplace_back(std::string());

  {
    std::string arr(100, '\0');
    unsigned char v = 0;
    for (auto& x : arr) {
      x = (v += 7);
    }
    arrays.push_back(std::move(arr));
  }

  arrays.emplace_back("The quick brown fox jumped over the lazy dog.");
  return arrays;
}

TEST(BloscCompressorTest, EncodeDecode) {
  for (const auto& spec : GetTestCompressorSpecs()) {
    auto compressor = Compressor::FromJson(spec).value();
    for (const auto& array : GetTestArrays()) {
      for (const std::size_t element_size : {1, 2, 10}) {
        std::string encode_result, decode_result;
        ASSERT_EQ(Status(),
                  compressor->Encode(array, &encode_result, element_size));
        ASSERT_EQ(Status(), compressor->Decode(encode_result, &decode_result,
                                               element_size));
        EXPECT_EQ(array, decode_result);
      }
    }
  }
}

// Tests that the compressed data has the expected blosc complib.
TEST(BloscCompressorTest, CheckComplib) {
  absl::string_view array = "The quick brown fox jumped over the lazy dog.";
  const std::vector<std::pair<std::string, std::string>>
      cnames_and_complib_names{{BLOSC_BLOSCLZ_COMPNAME, BLOSC_BLOSCLZ_LIBNAME},
                               {BLOSC_LZ4_COMPNAME, BLOSC_LZ4_LIBNAME},
                               {BLOSC_LZ4HC_COMPNAME, BLOSC_LZ4_LIBNAME},
                               {BLOSC_SNAPPY_COMPNAME, BLOSC_SNAPPY_LIBNAME},
                               {BLOSC_ZLIB_COMPNAME, BLOSC_ZLIB_LIBNAME},
                               {BLOSC_ZSTD_COMPNAME, BLOSC_ZSTD_LIBNAME}};
  for (const auto pair : cnames_and_complib_names) {
    auto compressor =
        Compressor::FromJson({{"id", "blosc"}, {"cname", pair.first}}).value();
    std::string encoded;
    ASSERT_EQ(Status(), compressor->Encode(array, &encoded, 1));
    ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
    const char* complib = blosc_cbuffer_complib(encoded.data());
    EXPECT_EQ(pair.second, complib);
  }
}

// Tests that the compressed data has the expected blosc shuffle and type size
// parameters.
TEST(BloscCompressorTest, CheckShuffleAndElementSize) {
  absl::string_view array = "The quick brown fox jumped over the lazy dog.";
  for (int shuffle = -1; shuffle <= 2; ++shuffle) {
    auto compressor =
        Compressor::FromJson({{"id", "blosc"}, {"shuffle", shuffle}}).value();
    for (const std::size_t element_size : {1, 2, 10}) {
      std::string encoded;
      ASSERT_EQ(Status(), compressor->Encode(array, &encoded, element_size));
      ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
      size_t typesize;
      int flags;
      blosc_cbuffer_metainfo(encoded.data(), &typesize, &flags);
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
TEST(BloscCompressorTest, CheckBlocksize) {
  std::string array(100000, '\0');
  for (std::size_t blocksize : {256, 512, 1024}) {
    // Set clevel to 0 to ensure our blocksize choice will be respected.
    // Otherwise blosc may choose a different blocksize.
    auto compressor = Compressor::FromJson({{"id", "blosc"},
                                            {"blocksize", blocksize},
                                            {"shuffle", 0},
                                            {"clevel", 0}})
                          .value();
    std::string encoded;
    ASSERT_EQ(Status(), compressor->Encode(array, &encoded, 1));
    ASSERT_GE(encoded.size(), BLOSC_MIN_HEADER_LENGTH);
    size_t nbytes, cbytes, bsize;
    blosc_cbuffer_sizes(encoded.data(), &nbytes, &cbytes, &bsize);
    EXPECT_EQ(blocksize, bsize);
  }
}

// Tests that encoding a buffer longer than BLOSC_MAX_BUFFERSIZE bytes results
// in an error.
TEST(BloscCompressorTest, TooLong) {
  std::string buf(BLOSC_MAX_BUFFERSIZE + 1, '\0');
  auto compressor = Compressor::FromJson({{"id", "blosc"}}).value();
  std::string encoded;
  EXPECT_THAT(compressor->Encode(buf, &encoded, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(BloscCompressorTest, InvalidParameters) {
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"cname", "foo"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"cname\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"cname", 5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"cname\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"clevel", "foo"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"clevel\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"clevel", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"clevel\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"clevel", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"clevel\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"shuffle", "foo"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"shuffle\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"shuffle", -2}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"shuffle\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"shuffle", 3}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"shuffle\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"blocksize", "foo"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"blocksize\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "blosc"}, {"foo", "foo"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"foo\""));
}

// Tests that decoding data with a corrupted blosc header returns an error.
TEST(BloscCompressorTest, DecodeHeaderCorrupted) {
  auto compressor = Compressor::FromJson({{"id", "blosc"}}).value();
  const std::string input = "The quick brown fox jumped over the lazy dog.";
  std::string encode_result, decode_result;
  ASSERT_EQ(Status(), compressor->Encode(input, &encode_result, 1));
  ASSERT_GE(encode_result.size(), 1);
  encode_result[0] = 0;
  EXPECT_THAT(compressor->Decode(encode_result, &decode_result, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that decoding data with an incomplete blosc header returns an error.
TEST(BloscCompressorTest, DecodeHeaderTruncated) {
  auto compressor = Compressor::FromJson({{"id", "blosc"}}).value();
  const std::string input = "The quick brown fox jumped over the lazy dog.";
  std::string encode_result, decode_result;
  ASSERT_EQ(Status(), compressor->Encode(input, &encode_result, 1));
  encode_result.resize(5);
  EXPECT_THAT(compressor->Decode(encode_result, &decode_result, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that decoding truncated data returns an error.
TEST(BloscCompressorTest, DecodeDataTruncated) {
  auto compressor = Compressor::FromJson({{"id", "blosc"}}).value();
  const std::string input = "The quick brown fox jumped over the lazy dog.";
  std::string encode_result, decode_result;
  ASSERT_EQ(Status(), compressor->Encode(input, &encode_result, 1));
  encode_result.resize(BLOSC_MIN_HEADER_LENGTH);
  EXPECT_THAT(compressor->Decode(encode_result, &decode_result, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(BloscCompressorTest, ToJson) {
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "lz4"},
                            {"clevel", 5},
                            {"shuffle", -1},
                            {"blocksize", 0}}),
            Compressor::FromJson({{"id", "blosc"}}).value().ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "zlib"},
                            {"clevel", 0},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "zlib"},
                                  {"clevel", 0},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "zstd"},
                            {"clevel", 9},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "zstd"},
                                  {"clevel", 9},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "lz4"},
                            {"clevel", 9},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "lz4"},
                                  {"clevel", 9},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "lz4hc"},
                            {"clevel", 9},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "lz4hc"},
                                  {"clevel", 9},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "snappy"},
                            {"clevel", 9},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "snappy"},
                                  {"clevel", 9},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
  EXPECT_EQ(nlohmann::json({{"id", "blosc"},
                            {"cname", "blosclz"},
                            {"clevel", 9},
                            {"shuffle", 2},
                            {"blocksize", 512}}),
            Compressor::FromJson({{"id", "blosc"},
                                  {"cname", "blosclz"},
                                  {"clevel", 9},
                                  {"shuffle", 2},
                                  {"blocksize", 512}})
                .value()
                .ToJson());
}

}  // namespace
