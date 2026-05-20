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

#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"

#include <stdint.h>

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::CodecSpec;
using ::tensorstore::MatchesJson;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_zarr3::ArrayCodecResolveParameters;
using ::tensorstore::internal_zarr3::BytesCodecResolveParameters;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecMerge;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;
using ::testing::HasSubstr;

TEST(CodecMergeTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto a,
      CodecSpec::FromJson({
          {"driver", "zarr3"},
          {"codecs",
           {{
               {"name", "sharding_indexed"},
               {"configuration",
                {
                    {"chunk_shape", {30, 40, 50}},
                    {"index_codecs",
                     {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
                    {"codecs",
                     {
                         {{"name", "transpose"},
                          {"configuration", {{"order", {2, 0, 1}}}}},
                         GetDefaultBytesCodecJson(),
                         {{"name", "gzip"}, {"configuration", {{"level", 6}}}},
                     }},
                }},
           }}},
      }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto b, CodecSpec::FromJson(
                  {{"driver", "zarr3"},
                   {"codecs",
                    {{{"name", "gzip"}, {"configuration", {{"level", 5}}}}}}}));
  EXPECT_THAT(a.MergeFrom(b),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Incompatible \"level\": 6 vs 5")));
}

TEST(CodecChainSpecTest, MissingArrayToBytes) {
  EXPECT_THAT(ZarrCodecChainSpec::FromJson(::nlohmann::json::array_t()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("array -> bytes codec must be specified")));
}

TEST(CodecChainSpecTest, MergeCodecNameMismatch) {
  EXPECT_THAT(TestCodecMerge({"gzip"}, {"crc32c"}, /*strict=*/true),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Cannot merge")));
}

TEST(CodecChainSpecTest, MergeArrayToBytes) {
  EXPECT_THAT(
      TestCodecMerge(
          {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}},
          ::nlohmann::json::array_t(), /*strict=*/true),
      ::testing::Optional(MatchesJson(
          {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}})));
}

TEST(CodecChainSpecTest, ExtraTranspose) {
  ::nlohmann::json a = {
      {{"name", "transpose"}, {"configuration", {{"order", {0, 2, 1}}}}},
      {{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
  };
  ::nlohmann::json b = {
      {{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
  };
  EXPECT_THAT(TestCodecMerge(a, b, /*strict=*/false),
              ::testing::Optional(MatchesJson(a)));
  EXPECT_THAT(
      TestCodecMerge(a, b, /*strict=*/true),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr(": Mismatch in number of array -> array codecs")));
}

TEST(CodecChainSpecTest, ExtraSharding) {
  ::nlohmann::json a = {{
      {"name", "sharding_indexed"},
      {"configuration",
       {
           {"chunk_shape", {30, 40, 50}},
           {"index_codecs", {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
           {"codecs",
            {
                {{"name", "transpose"},
                 {"configuration", {{"order", {2, 0, 1}}}}},
                GetDefaultBytesCodecJson(),
                {{"name", "gzip"}, {"configuration", {{"level", 6}}}},
            }},
       }},
  }};
  ::nlohmann::json b = {
      {{"name", "transpose"}, {"configuration", {{"order", {2, 0, 1}}}}},
      GetDefaultBytesCodecJson(),
      {{"name", "gzip"}, {"configuration", {{"level", 6}}}},
  };
  ::nlohmann::json c = {
      GetDefaultBytesCodecJson(),
      {{"name", "gzip"}, {"configuration", {{"level", 6}}}},
  };
  EXPECT_THAT(TestCodecMerge(a, b, /*strict=*/false),
              ::testing::Optional(MatchesJson(a)));
  EXPECT_THAT(TestCodecMerge(a, c, /*strict=*/false),
              ::testing::Optional(MatchesJson(a)));
  EXPECT_THAT(
      TestCodecMerge(a, b, /*strict=*/true),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr(": Mismatch in number of array -> array codecs")));
  EXPECT_THAT(TestCodecMerge(a, c, /*strict=*/true),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Cannot merge zarr codec constraints")));
}

TEST(CodecChainSpecTest, Crc32cItemBitsPropagation) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chain_spec,
      ZarrCodecChainSpec::FromJson({
          {{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
          {{"name", "crc32c"}},
      }));

  ArrayCodecResolveParameters decoded_array_params;
  decoded_array_params.dtype = ::tensorstore::dtype_v<uint16_t>;
  decoded_array_params.rank = 1;

  BytesCodecResolveParameters encoded_bytes_params;

  // crc32c codec sets item_bits to -1 because the data no longer matches the
  // original alignment.
  ZarrCodecChainSpec resolved_chain_spec;
  TENSORSTORE_ASSERT_OK(chain_spec.Resolve(std::move(decoded_array_params),
                                           encoded_bytes_params,
                                           &resolved_chain_spec));
  EXPECT_EQ(BytesCodecResolveParameters::kUnknownItemBits,
            encoded_bytes_params.item_bits);
}

}  // namespace
