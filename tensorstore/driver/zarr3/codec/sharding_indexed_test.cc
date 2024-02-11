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

#include <stdint.h>

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_zarr3::ArrayCodecResolveParameters;
using ::tensorstore::internal_zarr3::BytesCodecResolveParameters;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;

TEST(ShardingIndexedTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec, ZarrCodecChainSpec::FromJson(
                      {{{"name", "sharding_indexed"},
                        {"configuration",
                         {
                             {"chunk_shape", {2, 3}},
                             {"codecs",
                              {{
                                  {"name", "bytes"},
                                  {"configuration", {{"endian", "little"}}},
                              }}},
                             {"index_codecs",
                              {
                                  {
                                      {"name", "bytes"},
                                      {"configuration", {{"endian", "little"}}},
                                  },
                                  {
                                      {"name", "crc32c"},
                                  },
                              }},
                         }}}}));
}

TEST(ShardingIndexedTest, InvalidBytesToBytes) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, ZarrCodecChainSpec::FromJson({
                     {{"name", "sharding_indexed"},
                      {"configuration",
                       {
                           {"chunk_shape", {2, 3}},
                           {"codecs",
                            {{
                                {"name", "bytes"},
                                {"configuration", {{"endian", "little"}}},
                            }}},
                           {"index_codecs",
                            {
                                {
                                    {"name", "bytes"},
                                    {"configuration", {{"endian", "little"}}},
                                },
                                {
                                    {"name", "crc32c"},
                                },
                            }},
                       }}},
                     {
                         {"name", "gzip"},
                         {"configuration", {{"level", 5}}},
                     },
                 }));
  ArrayCodecResolveParameters decoded_params;
  decoded_params.dtype = tensorstore::dtype_v<uint32_t>;
  decoded_params.rank = 2;
  decoded_params.fill_value = tensorstore::MakeScalarArray<uint32_t>(42);
  BytesCodecResolveParameters encoded_params;
  EXPECT_THAT(
      spec.Resolve(std::move(decoded_params), encoded_params, nullptr),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Sharding codec .* is not compatible with subsequent bytes "
                    "-> bytes .*"));
}

TEST(ShardingIndexedTest, DefaultIndexLocation) {
  CodecSpecRoundTripTestParams p;
  p.resolve_params.rank = 2;
  p.orig_spec = {
      {{"name", "sharding_indexed"},
       {"configuration",
        {
            {"chunk_shape", {2, 3}},
            {"codecs",
             {{
                 {"name", "bytes"},
                 {"configuration", {{"endian", "little"}}},
             }}},
            {"index_codecs",
             {
                 {
                     {"name", "bytes"},
                     {"configuration", {{"endian", "little"}}},
                 },
                 {
                     {"name", "crc32c"},
                 },
             }},
        }}},
  };
  p.expected_spec = {
      {{"name", "sharding_indexed"},
       {"configuration",
        {
            {"chunk_shape", {2, 3}},
            {"codecs",
             {{
                 {"name", "bytes"},
                 {"configuration", {{"endian", "little"}}},
             }}},
            {"index_location", "end"},
            {"index_codecs",
             {
                 {
                     {"name", "bytes"},
                     {"configuration", {{"endian", "little"}}},
                 },
                 {
                     {"name", "crc32c"},
                 },
             }},
        }}},
  };
  p.to_json_options.constraints = true;
  TestCodecSpecRoundTrip(p);

  p.expected_spec = {
      {{"name", "sharding_indexed"},
       {"configuration",
        {
            {"chunk_shape", {2, 3}},
            {"codecs",
             {{
                 {"name", "bytes"},
                 {"configuration", {{"endian", "little"}}},
             }}},
            {"index_codecs",
             {
                 {
                     {"name", "bytes"},
                     {"configuration", {{"endian", "little"}}},
                 },
                 {
                     {"name", "crc32c"},
                 },
             }},
        }}},
  };
  p.to_json_options.constraints = false;
  TestCodecSpecRoundTrip(p);
}

TEST(ShardingIndexedTest, IndexLocationEndNotStored) {
  ArrayCodecResolveParameters p;
  p.dtype = tensorstore::dtype_v<uint16_t>;
  p.rank = 2;
  EXPECT_THAT(TestCodecSpecResolve(
                  ::nlohmann::json::array_t{
                      {{"name", "sharding_indexed"},
                       {"configuration",
                        {
                            {"chunk_shape", {2, 3}},
                            {"codecs",
                             {{
                                 {"name", "bytes"},
                                 {"configuration", {{"endian", "little"}}},
                             }}},
                            {"index_codecs",
                             {
                                 {
                                     {"name", "bytes"},
                                     {"configuration", {{"endian", "little"}}},
                                 },
                                 {
                                     {"name", "crc32c"},
                                 },
                             }},
                            {"index_location", "end"},
                        }}}},
                  p,
                  /*constraints=*/false),
              ::testing::Optional(MatchesJson(::nlohmann::json::array_t{
                  {{"name", "sharding_indexed"},
                   {"configuration",
                    {
                        {"chunk_shape", {2, 3}},
                        {"codecs",
                         {{
                             {"name", "bytes"},
                             {"configuration", {{"endian", "little"}}},
                         }}},
                        {"index_codecs",
                         {
                             {
                                 {"name", "bytes"},
                                 {"configuration", {{"endian", "little"}}},
                             },
                             {
                                 {"name", "crc32c"},
                             },
                         }},
                    }}}})));
}

}  // namespace
