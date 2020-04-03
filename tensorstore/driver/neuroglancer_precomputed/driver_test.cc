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

/// End-to-end tests of the Neuroglancer precomputed driver.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/compression/jpeg.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/to_string.h"

namespace {

using tensorstore::Context;
using tensorstore::Index;
using tensorstore::KeyValueStore;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::StrCat;
using tensorstore::internal::GetMap;
using tensorstore::internal::ParseJsonMatches;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"data_type", "uint16"},
           {"num_channels", 4},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "raw"},
           {"chunk_size", {3, 2, 2}},
           {"size", {10, 99, 98}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
}

// Tests that `create` can be specified in the JSON spec.
TEST(DriverTest, CreateSpecifiedInJson) {
  auto context = Context::Default();

  ::nlohmann::json spec = GetJsonSpec();
  spec["create"] = true;
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open(context, spec, {tensorstore::ReadWriteMode::read_write})
          .result());
}

TEST(DriverTest, OpenNonExisting) {
  auto context = Context::Default();

  EXPECT_THAT(tensorstore::Open(context, GetJsonSpec(),
                                {tensorstore::OpenMode::open,
                                 tensorstore::ReadWriteMode::read_write})
                  .result(),
              MatchesStatus(absl::StatusCode::kNotFound,
                            ".*Metadata key \"prefix/info\" does not exist"));
}

TEST(DriverTest, OpenOrCreate) {
  auto context = Context::Default();

  EXPECT_EQ(Status(), GetStatus(tensorstore::Open(
                                    context, GetJsonSpec(),
                                    {tensorstore::OpenMode::open |
                                         tensorstore::OpenMode::create,
                                     tensorstore::ReadWriteMode::read_write})
                                    .result()));
}

TEST(DriverTest, Create) {
  ::nlohmann::json json_spec = GetJsonSpec();

  auto context = Context::Default();
  // Create the store.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(1, 2, 3, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(10, 99, 98, 4));
    EXPECT_THAT(store.domain().labels(),
                ::testing::ElementsAre("x", "y", "z", "channel"));
    EXPECT_THAT(store.domain().implicit_lower_bounds(),
                ::testing::ElementsAre(0, 0, 0, 0));
    EXPECT_THAT(store.domain().implicit_upper_bounds(),
                ::testing::ElementsAre(0, 0, 0, 0));

    // Test ResolveBounds.
    auto resolved = ResolveBounds(store).value();
    EXPECT_EQ(store.domain(), resolved.domain());

    // Issue a read to be filled with the fill value.
    EXPECT_EQ(tensorstore::MakeArray<std::uint16_t>({{{{0, 0, 0, 0}}}}),
              tensorstore::Read<tensorstore::zero_origin>(
                  ChainResult(store, tensorstore::AllDims().SizedInterval(
                                         {9, 7, 3, 0}, {1, 1, 1, 4})))
                  .value());

    // Issue an out-of-bounds read.
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           {11, 7, 3, 0}, {1, 1, 1, 1})))
                    .result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));

    // Issue a valid write.
    EXPECT_EQ(
        Status(),
        GetStatus(tensorstore::Write(
                      tensorstore::MakeArray<std::uint16_t>(
                          {{{{0x9871, 0x9872}, {0x9881, 0x9882}},
                            {{0x9971, 0x9972}, {0x9981, 0x9982}},
                            {{0x9A71, 0x9A72}, {0x9A81, 0x9A82}}},
                           {{{0xA871, 0xA872}, {0xA881, 0xA882}},
                            {{0xA971, 0xA972}, {0xA981, 0xA982}},
                            {{0xAA71, 0xAA72}, {0xAA81, 0xAA82}}}}),
                      ChainResult(store, tensorstore::AllDims().SizedInterval(
                                             {9, 8, 7, 1}, {2, 3, 2, 2})))
                      .commit_future.result()));

    // Issue an out-of-bounds write.
    EXPECT_THAT(
        tensorstore::Write(
            tensorstore::MakeArray<std::uint16_t>({{1, 2, 3}, {4, 5, 6}}),
            ChainResult(store,
                        tensorstore::Dims("z", "channel").IndexSlice({3, 0}),
                        tensorstore::AllDims().SizedInterval({10, 8}, {2, 3})))
            .commit_future.result(),
        MatchesStatus(absl::StatusCode::kOutOfRange));

    // Re-read and validate result.
    EXPECT_EQ(tensorstore::MakeArray<std::uint16_t>(
                  {{{{0x0000, 0x0000, 0x0000}, {0x0000, 0x0000, 0x0000}},
                    {{0x0000, 0x9871, 0x9872}, {0x0000, 0x9881, 0x9882}},
                    {{0x0000, 0x9971, 0x9972}, {0x0000, 0x9981, 0x9982}},
                    {{0x0000, 0x9A71, 0x9A72}, {0x0000, 0x9A81, 0x9A82}}},
                   {{{0x0000, 0x0000, 0x0000}, {0x0000, 0x0000, 0x0000}},
                    {{0x0000, 0xA871, 0xA872}, {0x0000, 0xA881, 0xA882}},
                    {{0x0000, 0xA971, 0xA972}, {0x0000, 0xA981, 0xA982}},
                    {{0x0000, 0xAA71, 0xAA72}, {0x0000, 0xAA81, 0xAA82}}}}),
              tensorstore::Read<tensorstore::zero_origin>(
                  ChainResult(store, tensorstore::AllDims().SizedInterval(
                                         {9, 7, 7, 0}, {2, 4, 2, 3})))
                  .value());
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, std::string>>>({
          Pair("prefix/info",  //
               ParseJsonMatches(
                   ::nlohmann::json{{"@type", "neuroglancer_multiscale_volume"},
                                    {"type", "image"},
                                    {"data_type", "uint16"},
                                    {"num_channels", 4},
                                    {"scales",
                                     {{
                                         {"resolution", {1, 1, 1}},
                                         {"encoding", "raw"},
                                         {"key", "1_1_1"},
                                         {"chunk_sizes", {{3, 2, 2}}},
                                         {"size", {10, 99, 98}},
                                         {"voxel_offset", {1, 2, 3}},
                                     }}}})),
          Pair("prefix/7-10_8-10_7-9",  //
               ElementsAreArray({
                   // x=7           8           9
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=8, z=7, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=9, z=7, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=8, z=8, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=9, z=8, channel=0

                   0x00, 0x00, 0x00, 0x00, 0x71, 0x98,  // y=8, z=7, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x71, 0x99,  // y=9, z=7, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x81, 0x98,  // y=8, z=8, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x81, 0x99,  // y=9, z=8, channel=1

                   0x00, 0x00, 0x00, 0x00, 0x72, 0x98,  // y=8, z=7, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x72, 0x99,  // y=9, z=7, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x82, 0x98,  // y=8, z=8, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x82, 0x99,  // y=9, z=8, channel=2

                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=8, z=7, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=9, z=7, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=8, z=8, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=9, z=8, channel=3
               })),
          Pair("prefix/10-11_8-10_7-9",  //
               ElementsAreArray({
                   // x=10
                   0x00, 0x00,  // y=8, z=7, channel=0
                   0x00, 0x00,  // y=9, z=7, channel=0
                   0x00, 0x00,  // y=8, z=8, channel=0
                   0x00, 0x00,  // y=9, z=8, channel=0

                   0x71, 0xA8,  // y=8, z=7, channel=1
                   0x71, 0xA9,  // y=9, z=7, channel=1
                   0x81, 0xA8,  // y=8, z=8, channel=1
                   0x81, 0xA9,  // y=9, z=8, channel=1

                   0x72, 0xA8,  // y=8, z=7, channel=2
                   0x72, 0xA9,  // y=9, z=7, channel=2
                   0x82, 0xA8,  // y=8, z=8, channel=2
                   0x82, 0xA9,  // y=9, z=8, channel=2

                   0x00, 0x00,  // y=8, z=7, channel=3
                   0x00, 0x00,  // y=9, z=7, channel=3
                   0x00, 0x00,  // y=8, z=8, channel=3
                   0x00, 0x00,  // y=9, z=8, channel=3
               })),

          Pair("prefix/7-10_10-12_7-9",  //
               ElementsAreArray({
                   // x=7           8           9
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=10, z=7, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=7, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=10, z=8, channel=0
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=8, channel=0

                   0x00, 0x00, 0x00, 0x00, 0x71, 0x9A,  // y=10, z=7, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=7, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x81, 0x9A,  // y=10, z=8, channel=1
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=8, channel=1

                   0x00, 0x00, 0x00, 0x00, 0x72, 0x9A,  // y=10, z=7, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=7, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x82, 0x9A,  // y=10, z=8, channel=2
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=8, channel=2

                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=10, z=7, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=7, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=10, z=8, channel=3
                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // y=11, z=8, channel=3
               })),
          Pair("prefix/10-11_10-12_7-9",  //
               ElementsAreArray({
                   // x=10
                   0x00, 0x00,  // y=10, z=7, channel=0
                   0x00, 0x00,  // y=11, z=7, channel=0
                   0x00, 0x00,  // y=10, z=8, channel=0
                   0x00, 0x00,  // y=11, z=8, channel=0

                   0x71, 0xAA,  // y=10, z=7, channel=1
                   0x00, 0x00,  // y=11, z=7, channel=1
                   0x81, 0xAA,  // y=10, z=8, channel=1
                   0x00, 0x00,  // y=11, z=8, channel=1

                   0x72, 0xAA,  // y=10, z=7, channel=2
                   0x00, 0x00,  // y=11, z=7, channel=2
                   0x82, 0xAA,  // y=10, z=8, channel=2
                   0x00, 0x00,  // y=11, z=8, channel=2

                   0x00, 0x00,  // y=10, z=7, channel=3
                   0x00, 0x00,  // y=11, z=7, channel=3
                   0x00, 0x00,  // y=10, z=8, channel=3
                   0x00, 0x00,  // y=11, z=8, channel=3
               })),
      })));

  // Check that attempting to create the store again fails.
  {
    EXPECT_THAT(tensorstore::Open(context, json_spec,
                                  {tensorstore::OpenMode::create,
                                   tensorstore::ReadWriteMode::read_write})
                    .result(),
                MatchesStatus(absl::StatusCode::kAlreadyExists));
  }

  // Check that create or open succeeds.
  {
    EXPECT_EQ(Status(), GetStatus(tensorstore::Open(
                                      context, json_spec,
                                      {tensorstore::OpenMode::create |
                                           tensorstore::OpenMode::open,
                                       tensorstore::ReadWriteMode::read_write})
                                      .result()));
  }

  // Check that open succeeds.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();
    EXPECT_EQ(tensorstore::MakeArray<std::uint16_t>(
                  {{{{0x0000, 0x0000, 0x0000}, {0x0000, 0x0000, 0x0000}},
                    {{0x0000, 0x9871, 0x9872}, {0x0000, 0x9881, 0x9882}},
                    {{0x0000, 0x9971, 0x9972}, {0x0000, 0x9981, 0x9982}},
                    {{0x0000, 0x9A71, 0x9A72}, {0x0000, 0x9A81, 0x9A82}}},
                   {{{0x0000, 0x0000, 0x0000}, {0x0000, 0x0000, 0x0000}},
                    {{0x0000, 0xA871, 0xA872}, {0x0000, 0xA881, 0xA882}},
                    {{0x0000, 0xA971, 0xA972}, {0x0000, 0xA981, 0xA982}},
                    {{0x0000, 0xAA71, 0xAA72}, {0x0000, 0xAA81, 0xAA82}}}}),
              tensorstore::Read<tensorstore::zero_origin>(
                  ChainResult(store, tensorstore::AllDims().SizedInterval(
                                         {9, 7, 7, 0}, {2, 4, 2, 3})))
                  .value());

    // Test corrupt "raw" chunk handling
    ::nlohmann::json storage_spec{{"driver", "memory"}};
    auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
    EXPECT_EQ(
        Status(),
        GetStatus(kv_store->Write("prefix/10-11_10-12_7-9", "junk").result()));
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           {9, 7, 7, 0}, {2, 4, 2, 3})))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Expected chunk length to be .*"));
  }

  // Check that delete_existing works.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create |
                                        tensorstore::OpenMode::delete_existing,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();

    EXPECT_EQ(tensorstore::AllocateArray<std::uint16_t>(
                  {2, 4, 2, 3}, tensorstore::c_order, tensorstore::value_init),
              tensorstore::Read<tensorstore::zero_origin>(
                  ChainResult(store, tensorstore::AllDims().SizedInterval(
                                         {9, 7, 7, 0}, {2, 4, 2, 3})))
                  .value());
    auto kv_store =
        KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
    EXPECT_THAT(ListFuture(kv_store.get()).value(),
                ::testing::UnorderedElementsAre("prefix/info"));
  }
}

TEST(DriverTest, SpecRaw) {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtrip(
      /*full_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}},
       {"path", "prefix"},
       {"multiscale_metadata",
        {
            {"num_channels", 4},
            {"type", "image"},
        }},
       {"scale_metadata",
        {
            {"key", "1_1_1"},
            {"resolution", {1.0, 1.0, 1.0}},
            {"encoding", "raw"},
            {"chunk_size", {3, 2, 2}},
            {"size", {10, 99, 98}},
            {"voxel_offset", {1, 2, 3}},
            {"sharding", nullptr},
        }},
       {"scale_index", 0},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}}},
      /*minimal_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"scale_index", 0},
       {"path", "prefix"},
       {"kvstore", {{"driver", "memory"}}},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}}});
}

TEST(DriverTest, ConvertSpec) {
  ::nlohmann::json spec{
      {"dtype", "uint16"},
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"num_channels", 4},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"key", "1_1_1"},
           {"resolution", {1.0, 1.0, 1.0}},
           {"encoding", "raw"},
           {"chunk_size", {3, 2, 2}},
           {"size", {10, 99, 98}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding", nullptr},
       }},
      {"scale_index", 0},
      {"transform",
       {{"input_labels", {"x", "y", "z", "channel"}},
        {"input_exclusive_max", {11, 101, 101, 4}},
        {"input_inclusive_min", {1, 2, 3, 0}}}},
      {"context", {{"cache_pool", {{"total_bytes_limit", 10000000}}}}}};
  // No-op conversion.  Verifies that `context` is retained.
  tensorstore::internal::TestTensorStoreDriverSpecConvert(
      /*orig_spec=*/spec,
      /*options=*/tensorstore::SpecRequestOptions{},
      /*expected_converted_spec=*/spec);

  // Convert to minimal spec.
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec.erase("multiscale_metadata");
    converted_spec.erase("scale_metadata");
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*options=*/tensorstore::MinimalSpec{true},
        /*expected_converted_spec=*/converted_spec);
  }

  // Convert to create+delete_existing spec
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec["create"] = true;
    converted_spec["delete_existing"] = true;
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*options=*/tensorstore::OpenMode::create |
            tensorstore::OpenMode::delete_existing,
        /*expected_converted_spec=*/converted_spec);
  }

  // Convert to open+create+allow_metadata_mismatch spec
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec["create"] = true;
    converted_spec["open"] = true;
    converted_spec["allow_metadata_mismatch"] = true;
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*options=*/tensorstore::OpenMode::create |
            tensorstore::OpenMode::open |
            tensorstore::OpenMode::allow_option_mismatch,
        /*expected_converted_spec=*/converted_spec);
  }

  // Convert `recheck_cached_data` and `recheck_cached_metadata`.
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec["recheck_cached_data"] = false;
    converted_spec["recheck_cached_metadata"] = false;
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*options=*/
        tensorstore::SpecRequestOptions{tensorstore::StalenessBounds{
            absl::InfinitePast(), absl::InfinitePast()}},
        /*expected_converted_spec=*/converted_spec);
  }
}

TEST(DriverTest, SpecRawCachePool) {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtrip(
      /*full_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}},
       {"path", "prefix"},
       {"multiscale_metadata",
        {
            {"num_channels", 4},
            {"type", "image"},
        }},
       {"scale_metadata",
        {
            {"key", "1_1_1"},
            {"resolution", {1.0, 1.0, 1.0}},
            {"encoding", "raw"},
            {"chunk_size", {3, 2, 2}},
            {"size", {10, 99, 98}},
            {"voxel_offset", {1, 2, 3}},
            {"sharding", nullptr},
        }},
       {"scale_index", 0},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}},
       {"context", {{"cache_pool", {{"total_bytes_limit", 10000000}}}}}},
      /*minimal_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"scale_index", 0},
       {"path", "prefix"},
       {"kvstore", {{"driver", "memory"}}},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}},
       {"context", {{"cache_pool", {{"total_bytes_limit", 10000000}}}}}});
}

TEST(DriverTest, SpecRawSharded) {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtrip(
      /*full_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}},
       {"path", "prefix"},
       {"multiscale_metadata",
        {
            {"num_channels", 4},
            {"type", "image"},
        }},
       {"scale_metadata",
        {
            {"key", "1_1_1"},
            {"resolution", {1.0, 1.0, 1.0}},
            {"encoding", "raw"},
            {"chunk_size", {3, 2, 2}},
            {"size", {10, 99, 98}},
            {"voxel_offset", {1, 2, 3}},
            {"sharding",
             {{"@type", "neuroglancer_uint64_sharded_v1"},
              {"preshift_bits", 1},
              {"minishard_bits", 2},
              {"shard_bits", 3},
              {"data_encoding", "raw"},
              {"minishard_index_encoding", "raw"},
              {"hash", "identity"}}},
        }},
       {"scale_index", 0},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}}},
      /*minimal_spec=*/
      {{"dtype", "uint16"},
       {"driver", "neuroglancer_precomputed"},
       {"scale_index", 0},
       {"path", "prefix"},
       {"kvstore", {{"driver", "memory"}}},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}}});
}

TEST(DriverTest, SpecCompressedSegmentation) {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtrip(
      /*full_spec=*/
      {{"dtype", "uint32"},
       {"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}},
       {"path", "prefix"},
       {"multiscale_metadata",
        {
            {"num_channels", 4},
            {"type", "segmentation"},
        }},
       {"scale_metadata",
        {
            {"key", "1_1_1"},
            {"resolution", {1.0, 1.0, 1.0}},
            {"encoding", "compressed_segmentation"},
            {"compressed_segmentation_block_size", {3, 2, 1}},
            {"chunk_size", {3, 2, 2}},
            {"size", {10, 99, 98}},
            {"voxel_offset", {1, 2, 3}},
            {"sharding", nullptr},
        }},
       {"scale_index", 0},
       {"transform",
        {{"input_labels", {"x", "y", "z", "channel"}},
         {"input_exclusive_max", {11, 101, 101, 4}},
         {"input_inclusive_min", {1, 2, 3, 0}}}}},
      /*minimal_spec=*/{{"dtype", "uint32"},
                        {"driver", "neuroglancer_precomputed"},
                        {"scale_index", 0},
                        {"path", "prefix"},
                        {"kvstore", {{"driver", "memory"}}},
                        {"transform",
                         {{"input_labels", {"x", "y", "z", "channel"}},
                          {"input_exclusive_max", {11, 101, 101, 4}},
                          {"input_inclusive_min", {1, 2, 3, 0}}}}});
}

TEST(DriverTest, UnsupportedDataTypeInSpec) {
  auto context = Context::Default();
  EXPECT_THAT(
      tensorstore::Open(context,
                        ::nlohmann::json{
                            {"dtype", "string"},
                            {"driver", "neuroglancer_precomputed"},
                            {"kvstore", {{"driver", "memory"}}},
                            {"path", "prefix"},
                            {"multiscale_metadata",
                             {
                                 {"num_channels", 1},
                                 {"type", "image"},
                             }},
                            {"scale_metadata",
                             {
                                 {"size", {100, 100, 100}},
                                 {"encoding", "raw"},
                                 {"resolution", {1, 1, 1}},
                                 {"chunk_size", {2, 3, 4}},
                             }},
                        },
                        {tensorstore::OpenMode::create,
                         tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "string data type is not one of the supported data types: .*"));
}

// Tests that constraints are allowed not to match only if
// `allow_option_mismatch` is specified.
TEST(DriverTest, OptionMismatch) {
  ::nlohmann::json json_spec = GetJsonSpec();
  auto context = Context::Default();
  // Create the store.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
          .result());

  {
    // Specify `IncludeContext{false}` since we need to re-open with the same
    // `memory_key_value_store` context resource from the parent `context`.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto modified_spec,
        store.spec().value().ToJson(tensorstore::IncludeContext{false}));
    modified_spec["multiscale_metadata"]["num_channels"] = 10;
    EXPECT_THAT(
        tensorstore::Open(context, modified_spec, tensorstore::OpenMode::open)
            .result(),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      ".*\"num_channels\".*"));
    TENSORSTORE_ASSERT_OK(
        tensorstore::Open(context, modified_spec,
                          tensorstore::OpenMode::open |
                              tensorstore::OpenMode::allow_option_mismatch)
            .result());
  }

  // Rank constraint must hold regardless of whether `allow_option_mismatch`.
  {
    auto modified_spec = json_spec;
    modified_spec["rank"] = 5;
    EXPECT_THAT(
        tensorstore::Open(context, modified_spec, tensorstore::OpenMode::open)
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*rank 5.*"));
    EXPECT_THAT(
        tensorstore::Open(context, modified_spec,
                          tensorstore::OpenMode::open |
                              tensorstore::OpenMode::allow_option_mismatch)
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*rank 5.*"));
  }
}

// Tests that the data type constraint still applies even with
// `allow_option_mismatch` specified.
TEST(DriverTest, DataTypeMismatch) {
  ::nlohmann::json json_spec = GetJsonSpec();
  auto context = Context::Default();
  // Create the store.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
          .result());
  // Specify `IncludeContext{false}` since we need to re-open with the same
  // `memory_key_value_store` context resource from the parent `context`.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ::nlohmann::json modified_spec,
      store.spec().value().ToJson(tensorstore::IncludeContext{false}));
  modified_spec["dtype"] = "uint32";
  EXPECT_THAT(
      tensorstore::Open(context, modified_spec,
                        tensorstore::OpenMode::open |
                            tensorstore::OpenMode::allow_option_mismatch)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*: Expected data type of uint32 but received: uint16"));
}

TEST(DriverTest, InvalidSpec) {
  auto context = Context::Default();

  {
    auto spec = GetJsonSpec();
    spec["extra_member"] = 5;
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Object includes extra members: \"extra_member\""));
  }

  // Verify that a missing "kvstore" member leads to an error.
  {
    auto spec = GetJsonSpec();
    spec.erase("kvstore");
    EXPECT_THAT(tensorstore::Open(context, spec,
                                  {tensorstore::OpenMode::create,
                                   tensorstore::ReadWriteMode::read_write})
                    .result(),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Error parsing object member \"kvstore\": "
                              "Expected object, but member is missing"));
  }

  for (const std::string& member_name :
       {"kvstore", "path", "scale_metadata", "multiscale_metadata",
        "scale_index"}) {
    auto spec = GetJsonSpec();
    spec[member_name] = nullptr;
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Error parsing object member \"", member_name,
                             "\": "
                             "Expected .*, but received: null")));
  }

  {
    auto spec = GetJsonSpec();
    spec["scale_metadata"].erase("size");
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"size\".*"));
  }
}

TEST(DriverTest, CompressedSegmentationEncodingUint32) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"data_type", "uint32"},
           {"num_channels", 1},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {3, 2, 1}},
           {"chunk_size", {3, 4, 2}},
           {"size", {100, 100, 3}},
       }},
  };

  {
    auto store =
        tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
            .value();
    tensorstore::Write(
        tensorstore::MakeArray<std::uint32_t>({
            // Each compressed segmentation block has a single distinct value.
            {{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}},
            {{3, 3, 3}, {3, 3, 3}, {4, 4, 4}, {4, 4, 4}},
            {{5, 5, 5}, {5, 5, 5}, {6, 6, 6}, {6, 6, 6}},
        }),
        ChainResult(store, tensorstore::Dims("channel").IndexSlice(0),
                    tensorstore::Dims("z", "y", "x")
                        .SizedInterval({0, 0, 0}, {3, 4, 3})
                        .Transpose()))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, std::string>>>({
          Pair("prefix/info",  //
               ParseJsonMatches(::nlohmann::json{
                   {"@type", "neuroglancer_multiscale_volume"},
                   {"type", "segmentation"},
                   {"data_type", "uint32"},
                   {"num_channels", 1},
                   {"scales",
                    {{
                        {"resolution", {1, 1, 1}},
                        {"encoding", "compressed_segmentation"},
                        {"compressed_segmentation_block_size", {3, 2, 1}},
                        {"key", "1_1_1"},
                        {"chunk_sizes", {{3, 4, 2}}},
                        {"size", {100, 100, 3}},
                        {"voxel_offset", {0, 0, 0}},
                    }}}})),
          Pair("prefix/0-3_0-4_0-2",  //
               ElementsAreArray({
                   0x01, 0x00, 0x00, 0x00,  // channel offset
                   0x08, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header0
                   0x08, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header1
                   0x09, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header0
                   0x09, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header1
                   0x0a, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) header0
                   0x0a, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) header1
                   0x0b, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) header0
                   0x0b, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) header1
                   0x01, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) table
                   0x02, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) table
                   0x03, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) table
                   0x04, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) table
               })),
          Pair("prefix/0-3_0-4_2-3",  //
               ElementsAreArray({
                   0x01, 0x00, 0x00, 0x00,  // channel offset
                   0x04, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header0
                   0x04, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header1
                   0x05, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header0
                   0x05, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header1
                   0x05, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) table
                   0x06, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) table
               })),
      })));

  // Verify that reading back has the expected result.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    EXPECT_EQ(
        tensorstore::MakeArray<std::uint32_t>({
            // Each compressed segmentation block has a single distinct value.
            {{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}},
            {{3, 3, 3}, {3, 3, 3}, {4, 4, 4}, {4, 4, 4}},
            {{5, 5, 5}, {5, 5, 5}, {6, 6, 6}, {6, 6, 6}},
        }),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::Dims("channel").IndexSlice(0),
                        tensorstore::Dims("z", "y", "x")
                            .SizedInterval({0, 0, 0}, {3, 4, 3})
                            .Transpose()))
            .value());
  }
}

TEST(DriverTest, CompressedSegmentationEncodingUint64) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 1},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {3, 2, 1}},
           {"chunk_size", {3, 4, 2}},
           {"size", {100, 100, 3}},
       }},
  };

  {
    auto store =
        tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
            .value();
    tensorstore::Write(
        tensorstore::MakeArray<std::uint64_t>({
            // Each compressed segmentation block has a single distinct value.
            {{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}},
            {{3, 3, 3}, {3, 3, 3}, {4, 4, 4}, {4, 4, 4}},
            {{5, 5, 5}, {5, 5, 5}, {6, 6, 6}, {6, 6, 6}},
        }),
        ChainResult(store, tensorstore::Dims("channel").IndexSlice(0),
                    tensorstore::Dims("z", "y", "x")
                        .SizedInterval({0, 0, 0}, {3, 4, 3})
                        .Transpose()))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, std::string>>>({
          Pair("prefix/info",  //
               ParseJsonMatches(::nlohmann::json{
                   {"@type", "neuroglancer_multiscale_volume"},
                   {"type", "segmentation"},
                   {"data_type", "uint64"},
                   {"num_channels", 1},
                   {"scales",
                    {{
                        {"resolution", {1, 1, 1}},
                        {"encoding", "compressed_segmentation"},
                        {"compressed_segmentation_block_size", {3, 2, 1}},
                        {"key", "1_1_1"},
                        {"chunk_sizes", {{3, 4, 2}}},
                        {"size", {100, 100, 3}},
                        {"voxel_offset", {0, 0, 0}},
                    }}}})),
          Pair("prefix/0-3_0-4_0-2",  //
               ElementsAreArray({
                   0x01, 0x00, 0x00, 0x00,  // channel offset
                   0x08, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header0
                   0x08, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header1
                   0x0a, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header0
                   0x0a, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header1
                   0x0c, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) header0
                   0x0c, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) header1
                   0x0e, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) header0
                   0x0e, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) header1
                   0x01, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) table
                   0x00, 0x00, 0x00, 0x00,
                   0x02, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) table
                   0x00, 0x00, 0x00, 0x00,
                   0x03, 0x00, 0x00, 0x00,  // block (z=1,y=0,x=0) table
                   0x00, 0x00, 0x00, 0x00,
                   0x04, 0x00, 0x00, 0x00,  // block (z=1,y=1,x=0) table
                   0x00, 0x00, 0x00, 0x00,
               })),
          Pair("prefix/0-3_0-4_2-3",  //
               ElementsAreArray({
                   0x01, 0x00, 0x00, 0x00,  // channel offset
                   0x04, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header0
                   0x04, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) header1
                   0x06, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header0
                   0x06, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) header1
                   0x05, 0x00, 0x00, 0x00,  // block (z=0,y=0,x=0) table
                   0x00, 0x00, 0x00, 0x00,
                   0x06, 0x00, 0x00, 0x00,  // block (z=0,y=1,x=0) table
                   0x00, 0x00, 0x00, 0x00,
               })),
      })));

  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    // Verify that reading back has the expected result.
    EXPECT_EQ(
        tensorstore::MakeArray<std::uint64_t>({
            // Each compressed segmentation block has a single distinct value.
            {{1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}},
            {{3, 3, 3}, {3, 3, 3}, {4, 4, 4}, {4, 4, 4}},
            {{5, 5, 5}, {5, 5, 5}, {6, 6, 6}, {6, 6, 6}},
        }),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::Dims("channel").IndexSlice(0),
                        tensorstore::Dims("z", "y", "x")
                            .SizedInterval({0, 0, 0}, {3, 4, 3})
                            .Transpose()))
            .value());

    // Test corrupt "compressed_segmentation" chunk handling
    ::nlohmann::json storage_spec{{"driver", "memory"}};
    auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
    EXPECT_EQ(
        Status(),
        GetStatus(kv_store->Write("prefix/0-3_0-4_0-2", "junk").result()));
    EXPECT_THAT(
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::Dims("channel").IndexSlice(0),
                        tensorstore::Dims("z", "y", "x")
                            .SizedInterval({0, 0, 0}, {3, 4, 3})
                            .Transpose()))
            .result(),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      ".*Corrupted Neuroglancer compressed segmentation.*"));
  }
}

double GetRootMeanSquaredError(
    tensorstore::ArrayView<const std::uint8_t> array_a,
    tensorstore::ArrayView<const std::uint8_t> array_b) {
  double mean_squared_error = 0;
  tensorstore::IterateOverArrays(
      [&](const std::uint8_t* a, const std::uint8_t* b) {
        double diff = static_cast<double>(*a) - static_cast<double>(*b);
        mean_squared_error += diff * diff;
      },
      /*constraints=*/{}, array_a, array_b);
  return std::sqrt(mean_squared_error / array_a.num_elements());
}

TEST(DriverTest, Jpeg1Channel) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"data_type", "uint8"},
           {"num_channels", 1},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "jpeg"},
           {"chunk_size", {3, 4, 2}},
           {"size", {5, 100, 100}},
       }},
  };

  auto array = tensorstore::AllocateArray<std::uint8_t>({5, 4, 2, 1});
  for (int x = 0; x < array.shape()[0]; ++x) {
    for (int y = 0; y < array.shape()[1]; ++y) {
      for (int z = 0; z < array.shape()[2]; ++z) {
        array(x, y, z, 0) = x * 20 + y * 5 + z * 3;
      }
    }
  }

  {
    auto store =
        tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
            .value();
    tensorstore::Write(array,
                       ChainResult(store, tensorstore::AllDims().SizedInterval(
                                              0, array.shape())))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, std::string>>>({
          Pair("prefix/info",  //
               ParseJsonMatches(
                   ::nlohmann::json{{"@type", "neuroglancer_multiscale_volume"},
                                    {"type", "image"},
                                    {"data_type", "uint8"},
                                    {"num_channels", 1},
                                    {"scales",
                                     {{
                                         {"resolution", {1, 1, 1}},
                                         {"encoding", "jpeg"},
                                         {"key", "1_1_1"},
                                         {"chunk_sizes", {{3, 4, 2}}},
                                         {"size", {5, 100, 100}},
                                         {"voxel_offset", {0, 0, 0}},
                                     }}}})),
          // 0xff 0xd8 0xff is the JPEG header
          Pair("prefix/0-3_0-4_0-2", ::testing::StartsWith("\xff\xd8\xff")),
          Pair("prefix/3-5_0-4_0-2", ::testing::StartsWith("\xff\xd8\xff")),
      })));

  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    // Verify that reading back has the expected result.
    auto read_array =
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store,
                        tensorstore::AllDims().SizedInterval(0, array.shape())))
            .value();
    EXPECT_LT(
        GetRootMeanSquaredError(
            tensorstore::StaticDataTypeCast<const std::uint8_t,
                                            tensorstore::unchecked>(read_array),
            array),
        5);

    // Test corrupt "jpeg" chunk handling
    ::nlohmann::json storage_spec{{"driver", "memory"}};
    auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
    // Write invalid jpeg
    EXPECT_EQ(
        Status(),
        GetStatus(kv_store->Write("prefix/0-3_0-4_0-2", "junk").result()));
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           0, array.shape())))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Error decoding chunk \"prefix/0-3_0-4_0-2\":"
                              ".*Not a JPEG file.*"));

    // Write valid JPEG with the wrong number of channels.
    {
      std::string jpeg_data;
      EXPECT_EQ(Status(),
                tensorstore::jpeg::Encode(
                    std::vector<unsigned char>(3 * 4 * 2 * 3).data(),
                    /*width=*/3, /*height=*/4 * 2, /*num_components=*/3,
                    tensorstore::jpeg::EncodeOptions{}, &jpeg_data));
      EXPECT_EQ(
          Status(),
          GetStatus(kv_store->Write("prefix/0-3_0-4_0-2", jpeg_data).result()));
    }
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           0, array.shape())))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Image dimensions .* are not compatible with "
                              "expected chunk shape.*"));

    // Write valid JPEG with the wrong dimensions.
    {
      std::string jpeg_data;
      EXPECT_EQ(Status(),
                tensorstore::jpeg::Encode(
                    std::vector<unsigned char>(3 * 5 * 2 * 1).data(),
                    /*width=*/3, /*height=*/5 * 2, /*num_components=*/1,
                    tensorstore::jpeg::EncodeOptions{}, &jpeg_data));
      EXPECT_EQ(
          Status(),
          GetStatus(kv_store->Write("prefix/0-3_0-4_0-2", jpeg_data).result()));
    }
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           0, array.shape())))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Image dimensions .* are not compatible with "
                              "expected chunk shape.*"));
  }
}

TEST(DriverTest, Jpeg3Channel) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"multiscale_metadata",
       {
           {"data_type", "uint8"},
           {"num_channels", 3},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "jpeg"},
           {"chunk_size", {3, 4, 2}},
           {"size", {5, 100, 100}},
       }},
  };

  auto array = tensorstore::AllocateArray<std::uint8_t>({5, 4, 2, 3});
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 4; ++y) {
      for (int z = 0; z < 2; ++z) {
        for (int c = 0; c < 3; ++c) {
          array(x, y, z, c) = x * 20 + y * 5 + z * 3 + c * 20;
        }
      }
    }
  }

  {
    auto store =
        tensorstore::Open(context, json_spec, tensorstore::OpenMode::create)
            .value();
    tensorstore::Write(array,
                       ChainResult(store, tensorstore::AllDims().SizedInterval(
                                              0, array.shape())))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, std::string>>>({
          Pair("prefix/info",  //
               ParseJsonMatches(
                   ::nlohmann::json{{"@type", "neuroglancer_multiscale_volume"},
                                    {"type", "image"},
                                    {"data_type", "uint8"},
                                    {"num_channels", 3},
                                    {"scales",
                                     {{
                                         {"resolution", {1, 1, 1}},
                                         {"encoding", "jpeg"},
                                         {"key", "1_1_1"},
                                         {"chunk_sizes", {{3, 4, 2}}},
                                         {"size", {5, 100, 100}},
                                         {"voxel_offset", {0, 0, 0}},
                                     }}}})),
          // 0xff 0xd8 0xff is the JPEG header
          Pair("prefix/0-3_0-4_0-2", ::testing::StartsWith("\xff\xd8\xff")),
          Pair("prefix/3-5_0-4_0-2", ::testing::StartsWith("\xff\xd8\xff")),
      })));

  // Verify that reading back has the expected result.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    auto read_array =
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store,
                        tensorstore::AllDims().SizedInterval(0, array.shape())))
            .value();
    EXPECT_LT(
        GetRootMeanSquaredError(
            tensorstore::StaticDataTypeCast<const std::uint8_t,
                                            tensorstore::unchecked>(read_array),
            array),
        9)
        << "read_array=" << read_array << ", array=" << array;
  }
}

TEST(DriverTest, CorruptMetadataTest) {
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();

  // Write invalid JSON
  EXPECT_EQ(Status(),
            GetStatus(kv_store->Write("prefix/info", "invalid").result()));

  auto json_spec = GetJsonSpec();
  EXPECT_THAT(
      tensorstore::Open(
          context, json_spec,
          {tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          ".*: Error decoding metadata from \"prefix/info\": Invalid JSON"));

  // Write valid JSON that is invalid metadata.
  EXPECT_EQ(Status(),
            GetStatus(kv_store->Write("prefix/info", "[1]").result()));

  EXPECT_THAT(tensorstore::Open(context, json_spec,
                                {tensorstore::OpenMode::open,
                                 tensorstore::ReadWriteMode::read_write})
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*: Error decoding metadata from \"prefix/info\": "
                            "Expected object, but received: \\[1\\]"));
}

// Tests basic read/write functionality, including concurrent writes, for both
// the unsharded and sharded formats.
TEST(BasicFunctionalityTest, Uint16Raw) {
  const auto DoTest = [&](const ::nlohmann::json& sharding_json) {
    tensorstore::internal::TestTensorStoreDriverBasicFunctionality(
        {
            {"driver", "neuroglancer_precomputed"},
            {"kvstore", {{"driver", "memory"}}},
            {"path", "prefix"},
            {"multiscale_metadata",
             {
                 {"data_type", "uint16"},
                 {"num_channels", 4},
                 {"type", "image"},
             }},
            {"scale_metadata",
             {
                 {"resolution", {1, 1, 1}},
                 {"encoding", "raw"},
                 {"chunk_size", {4, 5, 6}},
                 {"size", {10, 11, 12}},
                 {"voxel_offset", {1, 2, 3}},
                 {"sharding", sharding_json},
             }},
        },
        {"x", "y", "z", "channel"},
        tensorstore::AllocateArray<std::uint16_t>(
            tensorstore::BoxView({1, 2, 3, 0}, {10, 11, 12, 4}),
            tensorstore::c_order, tensorstore::value_init));
  };
  DoTest(/*sharding=*/{{"@type", "neuroglancer_uint64_sharded_v1"},
                       {"preshift_bits", 1},
                       {"minishard_bits", 2},
                       {"shard_bits", 3},
                       {"hash", "identity"}});
  DoTest(/*sharding=*/nullptr);
}

}  // namespace
