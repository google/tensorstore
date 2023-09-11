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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::CodecSpec;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecMerge;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;

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
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*: Incompatible \"level\": 6 vs 5"));
}

TEST(CodecChainSpecTest, MissingArrayToBytes) {
  EXPECT_THAT(ZarrCodecChainSpec::FromJson(::nlohmann::json::array_t()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "array -> bytes codec must be specified"));
}

TEST(CodecChainSpecTest, MergeCodecNameMismatch) {
  EXPECT_THAT(
      TestCodecMerge({"gzip"}, {"crc32c"}, /*strict=*/true),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, "Cannot merge .*"));
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
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*: Mismatch in number of array -> array codecs.*"));
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
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*: Mismatch in number of array -> array codecs.*"));
  EXPECT_THAT(TestCodecMerge(a, c, /*strict=*/true),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Cannot merge zarr codec constraints .*"));
}

}  // namespace
