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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_zarr3::CodecRoundTripTestParams;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecMerge;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;

TEST(BloscTest, Precise) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "blosc"},
       {"configuration",
        {
            {"cname", "lz4"},
            {"clevel", 4},
            {"shuffle", "noshuffle"},
            {"blocksize", 128},
        }}},
  };
  p.expected_spec = {GetDefaultBytesCodecJson(),
                     {{"name", "blosc"},
                      {"configuration",
                       {
                           {"cname", "lz4"},
                           {"clevel", 4},
                           {"shuffle", "noshuffle"},
                           {"blocksize", 128},
                       }}}};
  TestCodecSpecRoundTrip(p);
}

TEST(BloscTest, DefaultsUint16) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {"blosc"};
  p.expected_spec = {GetDefaultBytesCodecJson(),
                     {{"name", "blosc"},
                      {"configuration",
                       {
                           {"cname", "lz4"},
                           {"clevel", 5},
                           {"shuffle", "shuffle"},
                           {"typesize", 2},
                           {"blocksize", 0},
                       }}}};
  TestCodecSpecRoundTrip(p);
}

TEST(BloscTest, DefaultsUint8) {
  CodecSpecRoundTripTestParams p;
  p.resolve_params.dtype = dtype_v<uint8_t>;
  p.orig_spec = {"blosc"};
  p.expected_spec = {{{"name", "bytes"}},
                     {{"name", "blosc"},
                      {"configuration",
                       {
                           {"cname", "lz4"},
                           {"clevel", 5},
                           {"shuffle", "bitshuffle"},
                           {"typesize", 1},
                           {"blocksize", 0},
                       }}}};
  TestCodecSpecRoundTrip(p);
}

TEST(BloscTest, RoundTrip) {
  CodecRoundTripTestParams p;
  p.spec = {"blosc"};
  TestCodecRoundTrip(p);
}

TEST(BloscTest, MergeCnameMismatch) {
  EXPECT_THAT(
      TestCodecMerge({{{"name", "blosc"},
                       {"configuration",
                        {
                            {"cname", "lz4"},
                        }}}},
                     {{{"name", "blosc"},
                       {"configuration",
                        {
                            {"cname", "zstd"},
                        }}}},
                     /*strict=*/true),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"cname\".*"));
}

TEST(BloscTest, MergeClevelMismatch) {
  EXPECT_THAT(
      TestCodecMerge({{{"name", "blosc"},
                       {"configuration",
                        {
                            {"clevel", 5},
                        }}}},
                     {{{"name", "blosc"},
                       {"configuration",
                        {
                            {"clevel", 6},
                        }}}},
                     /*strict=*/true),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"clevel\".*"));
}

TEST(BloscTest, MergeShuffleMismatch) {
  EXPECT_THAT(
      TestCodecMerge({{{"name", "blosc"},
                       {"configuration",
                        {
                            {"shuffle", "noshuffle"},
                        }}}},
                     {{{"name", "blosc"},
                       {"configuration",
                        {
                            {"shuffle", "shuffle"},
                        }}}},
                     /*strict=*/true),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"shuffle\".*"));
}

TEST(BloscTest, MergeTypesizeMismatch) {
  EXPECT_THAT(
      TestCodecMerge({{{"name", "blosc"},
                       {"configuration",
                        {
                            {"typesize", 2},
                        }}}},
                     {{{"name", "blosc"},
                       {"configuration",
                        {
                            {"typesize", 4},
                        }}}},
                     /*strict=*/true),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"typesize\".*"));
}

TEST(BloscTest, MergeBlocksizeMismatch) {
  EXPECT_THAT(TestCodecMerge({{{"name", "blosc"},
                               {"configuration",
                                {
                                    {"blocksize", 1000},
                                }}}},
                             {{{"name", "blosc"},
                               {"configuration",
                                {
                                    {"blocksize", 2000},
                                }}}},
                             /*strict=*/true),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*\"blocksize\".*"));
}

TEST(BloscTest, MergeSucces) {
  EXPECT_THAT(TestCodecMerge({{{"name", "blosc"},
                               {"configuration",
                                {
                                    {"cname", "lz4"},
                                    {"shuffle", "bitshuffle"},
                                    {"blocksize", 1000},
                                }}}},
                             {{{"name", "blosc"},
                               {"configuration",
                                {
                                    {"clevel", 5},
                                    {"typesize", 2},
                                    {"blocksize", 1000},
                                }}}},
                             /*strict=*/true),
              ::testing::Optional(MatchesJson({{{"name", "blosc"},
                                                {"configuration",
                                                 {
                                                     {"cname", "lz4"},
                                                     {"shuffle", "bitshuffle"},
                                                     {"blocksize", 1000},
                                                     {"clevel", 5},
                                                     {"typesize", 2},
                                                     {"blocksize", 2000},
                                                 }}}})));
}

TEST(BloscTest, InvalidCname) {
  EXPECT_THAT(ZarrCodecChainSpec::FromJson({{{"name", "blosc"},
                                             {"configuration",
                                              {
                                                  {"cname", "abc"},
                                                  {"shuffle", "bitshuffle"},
                                                  {"blocksize", 1000},
                                                  {"clevel", 5},
                                                  {"typesize", 2},
                                                  {"blocksize", 2000},
                                              }}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"cname\".*\"abc\".*"));
}

}  // namespace
