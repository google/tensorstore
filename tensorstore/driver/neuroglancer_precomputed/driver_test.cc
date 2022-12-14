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
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/jpeg_writer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::DimensionIndex;
using ::tensorstore::DimensionSet;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Schema;
using ::tensorstore::StorageGeneration;
using ::tensorstore::StrCat;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::Unit;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal::ParseJsonMatches;
using ::tensorstore::internal::ScopedTemporaryDirectory;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;
using ::tensorstore::internal::TestTensorStoreCreateWithSchema;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::JpegWriter;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

absl::Cord Bytes(std::vector<unsigned char> values) {
  return absl::Cord(std::string_view(
      reinterpret_cast<const char*>(values.data()), values.size()));
}

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "neuroglancer_precomputed"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
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
  ::nlohmann::json spec = GetJsonSpec();
  spec["create"] = true;
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open(spec, tensorstore::ReadWriteMode::read_write).result());
}

TEST(DriverTest, OpenNonExisting) {
  EXPECT_THAT(tensorstore::Open(GetJsonSpec(), tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kNotFound,
                            ".*Metadata at \"prefix/info\" does not exist"));
}

TEST(DriverTest, OpenOrCreate) {
  TENSORSTORE_EXPECT_OK(tensorstore::Open(
      GetJsonSpec(),
      tensorstore::OpenMode::open | tensorstore::OpenMode::create,
      tensorstore::ReadWriteMode::read_write));
}

TEST(DriverTest, Create) {
  ::nlohmann::json json_spec = GetJsonSpec();

  auto context = Context::Default();
  // Create the store.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result());
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(1, 2, 3, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(10, 99, 98, 4));
    EXPECT_THAT(store.domain().labels(),
                ::testing::ElementsAre("x", "y", "z", "channel"));
    EXPECT_THAT(store.domain().implicit_lower_bounds(),
                DimensionSet({0, 0, 0, 0}));
    EXPECT_THAT(store.domain().implicit_upper_bounds(),
                DimensionSet({0, 0, 0, 0}));

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
    TENSORSTORE_EXPECT_OK(tensorstore::Write(
        tensorstore::MakeArray<std::uint16_t>(
            {{{{0x9871, 0x9872}, {0x9881, 0x9882}},
              {{0x9971, 0x9972}, {0x9981, 0x9982}},
              {{0x9A71, 0x9A72}, {0x9A81, 0x9A82}}},
             {{{0xA871, 0xA872}, {0xA881, 0xA882}},
              {{0xA971, 0xA972}, {0xA981, 0xA982}},
              {{0xAA71, 0xAA72}, {0xAA81, 0xAA82}}}}),
        ChainResult(store, tensorstore::AllDims().SizedInterval(
                               {9, 8, 7, 1}, {2, 3, 2, 2}))));

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
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, absl::Cord>>>({
          Pair("prefix/info",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches(
                   {{"@type", "neuroglancer_multiscale_volume"},
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
                     }}}}))),
          Pair("prefix/1_1_1/7-10_8-10_7-9",  //
               Bytes({
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
          Pair("prefix/1_1_1/10-11_8-10_7-9",  //
               Bytes({
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

          Pair("prefix/1_1_1/7-10_10-12_7-9",  //
               Bytes({
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
          Pair("prefix/1_1_1/10-11_10-12_7-9",  //
               Bytes({
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
    EXPECT_THAT(
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result(),
        MatchesStatus(absl::StatusCode::kAlreadyExists));
  }

  // Check that create or open succeeds.
  TENSORSTORE_EXPECT_OK(tensorstore::Open(
      json_spec, context,
      tensorstore::OpenMode::create | tensorstore::OpenMode::open,
      tensorstore::ReadWriteMode::read_write));

  // Check that open succeeds.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read_write)
            .result());
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto kvs, kvstore::Open(storage_spec, context).result());
    TENSORSTORE_EXPECT_OK(kvstore::Write(kvs, "prefix/1_1_1/10-11_10-12_7-9",
                                         absl::Cord("junk")));
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           {9, 7, 7, 0}, {2, 4, 2, 3})))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Expected chunk length to be .*"));
  }

  // Check that delete_existing works.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context,
                          tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                          tensorstore::ReadWriteMode::read_write)
            .result());

    EXPECT_EQ(tensorstore::AllocateArray<std::uint16_t>(
                  {2, 4, 2, 3}, tensorstore::c_order, tensorstore::value_init),
              tensorstore::Read<tensorstore::zero_origin>(
                  ChainResult(store, tensorstore::AllDims().SizedInterval(
                                         {9, 7, 7, 0}, {2, 4, 2, 3})))
                  .value());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto kvs, kvstore::Open({{"driver", "memory"}}, context).result());
    EXPECT_THAT(
        ListFuture(kvs).result(),
        ::testing::Optional(::testing::UnorderedElementsAre("prefix/info")));
  }
}

TEST(DriverTest, ConvertSpec) {
  ::nlohmann::json spec{
      {"dtype", "uint16"},
      {"driver", "neuroglancer_precomputed"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
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
      /*expected_converted_spec=*/spec);

  // Convert to minimal spec.
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec.erase("multiscale_metadata");
    converted_spec.erase("scale_metadata");
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*expected_converted_spec=*/converted_spec,
        tensorstore::MinimalSpec{true});
  }

  // Convert to create+delete_existing spec
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec["create"] = true;
    converted_spec["delete_existing"] = true;
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*expected_converted_spec=*/converted_spec,
        tensorstore::OpenMode::create | tensorstore::OpenMode::delete_existing);
  }

  // Convert `recheck_cached_data` and `recheck_cached_metadata`.
  {
    ::nlohmann::json converted_spec = spec;
    converted_spec["recheck_cached_data"] = false;
    converted_spec["recheck_cached_metadata"] = false;
    tensorstore::internal::TestTensorStoreDriverSpecConvert(
        /*orig_spec=*/spec,
        /*expected_converted_spec=*/converted_spec,
        tensorstore::RecheckCachedData{false},
        tensorstore::RecheckCachedMetadata{false});
  }
}

TEST(DriverTest, UnsupportedDataTypeInSpec) {
  EXPECT_THAT(
      tensorstore::Open(
          {
              {"dtype", "string"},
              {"driver", "neuroglancer_precomputed"},
              {"kvstore",
               {
                   {"driver", "memory"},
                   {"path", "prefix/"},
               }},
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
          tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*string data type is not one of the supported data types: .*"));
}

TEST(DriverTest, OptionMismatch) {
  ::nlohmann::json json_spec = GetJsonSpec();
  auto context = Context::Default();
  // Create the store.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
          .result());

  // Rank constraint must hold.
  {
    auto modified_spec = json_spec;
    modified_spec["rank"] = 5;
    EXPECT_THAT(
        tensorstore::Open(modified_spec, context, tensorstore::OpenMode::open)
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*rank.*"));
  }
}

// Tests that the data type constraint applies.
TEST(DriverTest, DataTypeMismatchInSpec) {
  ::nlohmann::json json_spec = GetJsonSpec();
  auto context = Context::Default();
  // Create the store.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(::nlohmann::json modified_spec,
                                   store.spec().value().ToJson());
  modified_spec["dtype"] = "uint32";
  EXPECT_THAT(
      tensorstore::Open(modified_spec, context, tensorstore::OpenMode::open)
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          ".*: Expected \"data_type\" of \"uint32\" but received: \"uint16\""));
}

TEST(DriverTest, DataTypeMismatchInStoredMetadata) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        context, Schema::Shape({10, 20, 30, 1}),
                        dtype_v<uint8_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        context, dtype_v<uint32_t>, tensorstore::OpenMode::open)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*: data_type from metadata \\(uint8\\) does not match "
                    "dtype in schema \\(uint32\\)"));
}

TEST(DriverTest, InvalidSpecExtraMember) {
  auto spec = GetJsonSpec();
  spec["extra_member"] = 5;
  EXPECT_THAT(tensorstore::Open(spec, tensorstore::OpenMode::create,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"extra_member\""));
}

TEST(DriverTest, InvalidSpecMissingKvstore) {
  auto spec = GetJsonSpec();
  spec.erase("kvstore");
  EXPECT_THAT(
      tensorstore::Open(spec, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error opening \"neuroglancer_precomputed\" driver: "
                    "\"kvstore\" must be specified"));
}

TEST(DriverTest, InvalidSpecInvalidMemberType) {
  for (auto member_name : {"kvstore", "path", "scale_metadata",
                           "multiscale_metadata", "scale_index"}) {
    auto spec = GetJsonSpec();
    spec[member_name] = nullptr;
    EXPECT_THAT(
        tensorstore::Open(spec, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Error parsing object member \"", member_name,
                             "\": "
                             "Expected .*, but received: null")));
  }
}

TEST(DriverTest, InvalidSpecMissingDomain) {
  auto spec = GetJsonSpec();
  spec["scale_metadata"].erase("size");
  EXPECT_THAT(
      tensorstore::Open(spec, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"size\".*"));
}

TEST(DriverTest, CompressedSegmentationEncodingUint32) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
            .result());
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
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, absl::Cord>>>({
          Pair("prefix/info",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches(
                   {{"@type", "neuroglancer_multiscale_volume"},
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
                     }}}}))),
          Pair("prefix/1_1_1/0-3_0-4_0-2",  //
               Bytes({
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
          Pair("prefix/1_1_1/0-3_0-4_2-3",  //
               Bytes({
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read)
            .result());
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
      {"path", "prefix/"},
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
            .result());
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
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, absl::Cord>>>({
          Pair("prefix/info",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches(
                   {{"@type", "neuroglancer_multiscale_volume"},
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
                     }}}}))),
          Pair("prefix/1_1_1/0-3_0-4_0-2",  //
               Bytes({
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
          Pair("prefix/1_1_1/0-3_0-4_2-3",  //
               Bytes({
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read)
            .result());
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto kvs, kvstore::Open(storage_spec, context).result());
    TENSORSTORE_EXPECT_OK(
        kvstore::Write(kvs, "prefix/1_1_1/0-3_0-4_0-2", absl::Cord("junk")));
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
      {"path", "prefix/"},
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
            .result());
    tensorstore::Write(array,
                       ChainResult(store, tensorstore::AllDims().SizedInterval(
                                              0, array.shape())))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, absl::Cord>>>({
          Pair("prefix/info",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches(
                   {{"@type", "neuroglancer_multiscale_volume"},
                    {"type", "image"},
                    {"data_type", "uint8"},
                    {"num_channels", 1},
                    {"scales",
                     {{
                         {"resolution", {1, 1, 1}},
                         {"encoding", "jpeg"},
                         {"jpeg_quality", 75},
                         {"key", "1_1_1"},
                         {"chunk_sizes", {{3, 4, 2}}},
                         {"size", {5, 100, 100}},
                         {"voxel_offset", {0, 0, 0}},
                     }}}}))),
          // 0xff 0xd8 0xff is the JPEG header
          Pair("prefix/1_1_1/0-3_0-4_0-2",
               ::testing::MatcherCast<absl::Cord>(
                   ::testing::Matcher<std::string>(
                       ::testing::StartsWith("\xff\xd8\xff")))),
          Pair("prefix/1_1_1/3-5_0-4_0-2",
               ::testing::MatcherCast<absl::Cord>(
                   ::testing::Matcher<std::string>(
                       ::testing::StartsWith("\xff\xd8\xff")))),
      })));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read)
            .result());
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto kvs, kvstore::Open(storage_spec, context).result());
    // Write invalid jpeg
    TENSORSTORE_EXPECT_OK(
        kvstore::Write(kvs, "prefix/1_1_1/0-3_0-4_0-2", absl::Cord("junk")));
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    ChainResult(store, tensorstore::AllDims().SizedInterval(
                                           0, array.shape())))
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*Error reading \"prefix/1_1_1/0-3_0-4_0-2\":"
                              ".*Not a JPEG file.*"));

    // Write valid JPEG with the wrong number of channels.
    {
      absl::Cord jpeg_data;
      {
        std::vector<unsigned char> tmp(3 * 4 * 2 * 3);
        ImageInfo info{/*.height =*/4 * 2,
                       /*.width =*/3,
                       /*.num_components =*/3};
        JpegWriter writer;
        riegeli::CordWriter<> cord_writer(&jpeg_data);
        TENSORSTORE_EXPECT_OK(writer.Initialize(&cord_writer));
        TENSORSTORE_EXPECT_OK(writer.Encode(info, tmp));
        TENSORSTORE_EXPECT_OK(writer.Done());
      }
      TENSORSTORE_EXPECT_OK(
          kvstore::Write(kvs, "prefix/1_1_1/0-3_0-4_0-2", jpeg_data));
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
      absl::Cord jpeg_data;
      {
        std::vector<unsigned char> tmp(3 * 5 * 1);
        ImageInfo info{/*.height =*/5,
                       /*.width =*/3,
                       /*.num_components =*/1};

        JpegWriter writer;
        riegeli::CordWriter<> cord_writer(&jpeg_data);
        TENSORSTORE_EXPECT_OK(writer.Initialize(&cord_writer));
        TENSORSTORE_EXPECT_OK(writer.Encode(info, tmp));
        TENSORSTORE_EXPECT_OK(writer.Done());
      }
      TENSORSTORE_EXPECT_OK(
          kvstore::Write(kvs, "prefix/1_1_1/0-3_0-4_0-2", jpeg_data));
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

// Verify that jpeg quality has an effect.
TEST(DriverTest, JpegQuality) {
  std::vector<int> jpeg_quality_values{0, 50, 75, 100};
  std::vector<size_t> sizes;

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
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

  const auto get_size =
      [&](::nlohmann::json json_spec) -> tensorstore::Result<size_t> {
    auto context = Context::Default();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
            .result());
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::Write(array, store | tensorstore::AllDims().SizedInterval(
                                              0, array.shape()))
            .result());
    size_t size = 0;
    for (const auto& [key, value] :
         GetMap(kvstore::Open({{"driver", "memory"}}, context).value())
             .value()) {
      size += value.size();
    }
    return size;
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(size_t default_size, get_size(json_spec));
  for (int quality : jpeg_quality_values) {
    auto spec = json_spec;
    spec["scale_metadata"]["jpeg_quality"] = quality;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(size_t size, get_size(spec));
    if (!sizes.empty()) {
      EXPECT_LT(sizes.back(), size) << "quality=" << quality;
    }
    sizes.push_back(size);
    if (quality == 75) {
      EXPECT_EQ(default_size, size);
    }
  }
}

TEST(DriverTest, Jpeg3Channel) {
  auto context = Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
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
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create)
            .result());
    tensorstore::Write(array,
                       ChainResult(store, tensorstore::AllDims().SizedInterval(
                                              0, array.shape())))
        .commit_future.value();
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      (UnorderedElementsAreArray<
          ::testing::Matcher<std::pair<std::string, absl::Cord>>>({
          Pair("prefix/info",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches(
                   {{"@type", "neuroglancer_multiscale_volume"},
                    {"type", "image"},
                    {"data_type", "uint8"},
                    {"num_channels", 3},
                    {"scales",
                     {{
                         {"resolution", {1, 1, 1}},
                         {"encoding", "jpeg"},
                         {"jpeg_quality", 75},
                         {"key", "1_1_1"},
                         {"chunk_sizes", {{3, 4, 2}}},
                         {"size", {5, 100, 100}},
                         {"voxel_offset", {0, 0, 0}},
                     }}}}))),
          // 0xff 0xd8 0xff is the JPEG header
          Pair("prefix/1_1_1/0-3_0-4_0-2",
               ::testing::MatcherCast<absl::Cord>(
                   ::testing::Matcher<std::string>(
                       ::testing::StartsWith("\xff\xd8\xff")))),
          Pair("prefix/1_1_1/3-5_0-4_0-2",
               ::testing::MatcherCast<absl::Cord>(
                   ::testing::Matcher<std::string>(
                       ::testing::StartsWith("\xff\xd8\xff")))),
      })));

  // Verify that reading back has the expected result.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read)
            .result());
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
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, kvstore::Open(storage_spec, context).result());

  // Write invalid JSON
  TENSORSTORE_EXPECT_OK(
      kvstore::Write(kvs, "prefix/info", absl::Cord("invalid")));

  auto json_spec = GetJsonSpec();
  EXPECT_THAT(tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*: Error reading \"prefix/info\": Invalid JSON"));

  // Write valid JSON that is invalid metadata.
  TENSORSTORE_EXPECT_OK(kvstore::Write(kvs, "prefix/info", absl::Cord("[1]")));

  EXPECT_THAT(tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*: Error reading \"prefix/info\":.*"));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "neuroglancer_precomputed/raw";
  options.full_spec = {{"dtype", "uint16"},
                       {"driver", "neuroglancer_precomputed"},
                       {"kvstore",
                        {
                            {"driver", "memory"},
                            {"path", "prefix/"},
                        }},
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
                         {"input_inclusive_min", {1, 2, 3, 0}}}}};
  options.minimal_spec = {{"dtype", "uint16"},
                          {"driver", "neuroglancer_precomputed"},
                          {"scale_index", 0},
                          {"kvstore",
                           {
                               {"driver", "memory"},
                               {"path", "prefix/"},
                           }},
                          {"transform",
                           {{"input_labels", {"x", "y", "z", "channel"}},
                            {"input_exclusive_max", {11, 101, 101, 4}},
                            {"input_inclusive_min", {1, 2, 3, 0}}}}};
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "raw/sharded";
  options.full_spec = {
      {"dtype", "uint16"},
      {"driver", "neuroglancer_precomputed"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
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
        {"input_inclusive_min", {1, 2, 3, 0}}}},
  };
  options.minimal_spec = {
      {"dtype", "uint16"},
      {"driver", "neuroglancer_precomputed"},
      {"scale_index", 0},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
      {"transform",
       {{"input_labels", {"x", "y", "z", "channel"}},
        {"input_exclusive_max", {11, 101, 101, 4}},
        {"input_inclusive_min", {1, 2, 3, 0}}}},
  };
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "compressed_segmentation";
  options.full_spec = {
      {"dtype", "uint32"},
      {"driver", "neuroglancer_precomputed"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
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
        {"input_inclusive_min", {1, 2, 3, 0}}}},
  };
  options.minimal_spec = {
      {"dtype", "uint32"},
      {"driver", "neuroglancer_precomputed"},
      {"scale_index", 0},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
      {"transform",
       {{"input_labels", {"x", "y", "z", "channel"}},
        {"input_exclusive_max", {11, 101, 101, 4}},
        {"input_inclusive_min", {1, 2, 3, 0}}}},
  };
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

// Tests basic read/write functionality, including concurrent writes, for both
// the unsharded and sharded formats.
TENSORSTORE_GLOBAL_INITIALIZER {
  const auto RegisterShardingVariant = [](std::string sharding_name,
                                          const ::nlohmann::json&
                                              sharding_json) {
    const auto RegisterShapeVariant = [&](const Index(&shape)[4],
                                          std::vector<Index> chunk_size) {
      const auto [x_size, y_size, z_size, c_size] = shape;
      tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions
          options;
      options.test_name = tensorstore::StrCat(
          "neuroglancer_precomputed", "/sharding=", sharding_name,
          "/shape=", x_size, ",", y_size, ",", z_size, ",", c_size);
      options.create_spec = {
          {"driver", "neuroglancer_precomputed"},
          {"kvstore",
           {
               {"driver", "memory"},
               {"path", "prefix/"},
           }},
          {"multiscale_metadata",
           {
               {"data_type", "uint16"},
               {"num_channels", c_size},
               {"type", "image"},
           }},
          {"scale_metadata",
           {
               {"resolution", {1, 1, 1}},
               {"encoding", "raw"},
               {"chunk_size", chunk_size},
               {"size", {x_size, y_size, z_size}},
               {"voxel_offset", {1, 2, 3}},
               {"sharding", sharding_json},
           }},
      };
      options.expected_domain = tensorstore::IndexDomainBuilder(4)
                                    .origin({1, 2, 3, 0})
                                    .shape({x_size, y_size, z_size, c_size})
                                    .labels({"x", "y", "z", "channel"})
                                    .Finalize()
                                    .value();
      options.initial_value = tensorstore::AllocateArray<std::uint16_t>(
          tensorstore::BoxView({1, 2, 3, 0}, {x_size, y_size, z_size, c_size}),
          tensorstore::c_order, tensorstore::value_init);
      tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
          std::move(options));
    };
    RegisterShapeVariant({2, 2, 1, 1}, {2, 2, 1});
    RegisterShapeVariant({10, 11, 12, 4}, {4, 5, 6});
  };
  RegisterShardingVariant("null", nullptr);
  RegisterShardingVariant("single_shard_and_minishard",
                          {{"@type", "neuroglancer_uint64_sharded_v1"},
                           {"preshift_bits", 0},
                           {"minishard_bits", 0},
                           {"shard_bits", 0},
                           {"hash", "identity"}});
  RegisterShardingVariant("multiple_shards",
                          {{"@type", "neuroglancer_uint64_sharded_v1"},
                           {"preshift_bits", 1},
                           {"minishard_bits", 2},
                           {"shard_bits", 3},
                           {"hash", "identity"}});
}

TEST(ShardedWriteTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({
                            {"driver", "neuroglancer_precomputed"},
                            {"kvstore",
                             {
                                 {"driver", "memory"},
                                 {"path", "prefix/"},
                             }},
                            {"multiscale_metadata",
                             {
                                 {"data_type", "uint16"},
                                 {"num_channels", 2},
                                 {"type", "image"},
                             }},
                            {"scale_metadata",
                             {
                                 {"resolution", {1, 1, 1}},
                                 {"encoding", "raw"},
                                 {"chunk_size", {3, 4, 5}},
                                 {"size", {4, 5, 1}},
                                 {"voxel_offset", {0, 0, 0}},
                                 {"sharding",
                                  {{"@type", "neuroglancer_uint64_sharded_v1"},
                                   {"preshift_bits", 1},
                                   {"minishard_bits", 2},
                                   {"shard_bits", 3},
                                   {"hash", "identity"}}},
                             }},
                            {"create", true},
                        })
          .result());
  auto array = tensorstore::MakeArray<std::uint16_t>(
      {{{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}, {{9, 10}}},
       {{{11, 12}}, {{13, 14}}, {{15, 16}}, {{17, 18}}, {{19, 20}}},
       {{{21, 22}}, {{23, 24}}, {{25, 26}}, {{27, 28}}, {{29, 30}}},
       {{{31, 32}}, {{33, 34}}, {{35, 36}}, {{37, 38}}, {{39, 40}}}});
  TENSORSTORE_ASSERT_OK(tensorstore::Write(array, store).result());
  EXPECT_THAT(tensorstore::Read(store).result(), ::testing::Optional(array));
}

// Disable due to race condition whereby writeback of a shard may start while
// some chunks that have been modified are still being written back to it.
TEST(FullShardWriteTest, Basic) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_key_value_store = *mock_key_value_store_resource;

  ::nlohmann::json json_spec{
      // Use a cache to avoid early writeback of partial shard.
      {"context", {{"cache_pool", {{"total_bytes_limit", 10'000'000}}}}},
      {"driver", "neuroglancer_precomputed"},
      {"kvstore",
       {
           {"driver", "mock_key_value_store"},
           {"path", "prefix/"},
       }},
      {"create", true},
      {"multiscale_metadata",
       {
           {"data_type", "uint16"},
           {"num_channels", 1},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"key", "1_1_1"},
           {"resolution", {1, 1, 1}},
           {"encoding", "raw"},
           {"chunk_size", {2, 2, 2}},
           {"size", {4, 6, 10}},
           {"voxel_offset", {0, 0, 0}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 1},
             {"minishard_bits", 2},
             {"shard_bits", 3},
             {"data_encoding", "raw"},
             {"minishard_index_encoding", "raw"},
             {"hash", "identity"}}},
       }},
  };

  // Grid shape: {2, 3, 5}
  // Full shard shape is {2, 2, 2} in chunks.
  // Full shard shape is {4, 4, 4} in voxels.
  // Shard 0 origin: {0, 0, 0}
  // Shard 1 origin: {0, 4, 0}
  // Shard 2 origin: {0, 0, 4}
  // Shard 3 origin: {0, 4, 4}
  // Shard 4 origin: {0, 0, 8}
  // Shard 5 origin: {0, 4, 8}

  // Repeat the test to try to detect errors due to possible timing-dependent
  // behavior differences.
  for (int i = 0; i < 100; ++i) {
    auto store_future = tensorstore::Open(json_spec, context);
    store_future.Force();

    {
      auto req = mock_key_value_store->read_requests.pop();
      EXPECT_EQ("prefix/info", req.key);
      req.promise.SetResult(
          kvstore::ReadResult{kvstore::ReadResult::kMissing,
                              {},
                              {StorageGeneration::NoValue(), absl::Now()}});
    }

    {
      auto req = mock_key_value_store->write_requests.pop();
      EXPECT_EQ("prefix/info", req.key);
      EXPECT_EQ(StorageGeneration::NoValue(), req.options.if_equal);
      req.promise.SetResult(TimestampedStorageGeneration{
          StorageGeneration::FromString("g0"), absl::Now()});
    }

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, store_future.result());

    auto future = tensorstore::Write(
        tensorstore::MakeScalarArray<std::uint16_t>(42),
        tensorstore::ChainResult(
            store,
            tensorstore::Dims(0, 1, 2).SizedInterval({0, 4, 8}, {4, 2, 2})));

    // Ensure copying finishes before writeback starts.
    TENSORSTORE_ASSERT_OK(future.copy_future.result());
    ASSERT_FALSE(future.commit_future.ready());

    future.Force();

    {
      auto req = mock_key_value_store->write_requests.pop();
      ASSERT_EQ("prefix/1_1_1/5.shard", req.key);
      // Writeback is unconditional because the entire shard is being written.
      ASSERT_EQ(StorageGeneration::Unknown(), req.options.if_equal);
      req.promise.SetResult(TimestampedStorageGeneration{
          StorageGeneration::FromString("g0"), absl::Now()});
    }

    TENSORSTORE_ASSERT_OK(future.result());
  }
}

// Tests that an empty path is handled correctly.
TEST(DriverTest, NoPrefix) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint8"},
           {"num_channels", 1},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "raw"},
           {"chunk_size", {2, 3, 1}},
           {"size", {2, 3, 1}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());
  TENSORSTORE_EXPECT_OK(tensorstore::Write(
      tensorstore::MakeArray<std::uint8_t>({{1, 2, 3}, {4, 5, 6}}),
      store | tensorstore::Dims("z", "channel").IndexSlice({0, 0})));
  // Check that key value store has expected contents.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, kvstore::Open(storage_spec, context).result());
  EXPECT_THAT(  //
      GetMap(kvs).value(),
      ::testing::UnorderedElementsAre(
          Pair("info", ::testing::_),
          Pair("1_1_1/0-2_0-3_0-1", Bytes({1, 4, 2, 5, 3, 6}))));
}

TEST(DriverTest, ChunkLayoutUnshardedRaw) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"chunk_size", {5, 6, 7}},
           {"size", {10, 99, 98}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout, ChunkLayout::FromJson({
                                {"grid_origin", {1, 2, 3, 0}},
                                {"write_chunk", {{"shape", {5, 6, 7, 4}}}},
                                {"read_chunk", {{"shape", {5, 6, 7, 4}}}},
                                {"inner_order", {3, 2, 1, 0}},
                            }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(DriverTest, ChunkLayoutShardedRaw) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout,
      ChunkLayout::FromJson({
          {"grid_origin", {1, 2, 3, 0}},
          {"write_chunk", {{"shape", {2048, 2048, 2048, 4}}}},
          {"read_chunk", {{"shape", {64, 64, 64, 4}}}},
          {"inner_order", {3, 2, 1, 0}},
      }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(DriverTest, ChunkLayoutShardedRawNonUniform) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"chunk_size", {64, 65, 66}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout,
      ChunkLayout::FromJson({
          {"grid_origin", {1, 2, 3, 0}},
          {"write_chunk", {{"shape", {2048, 2080, 2112, 4}}}},
          {"read_chunk", {{"shape", {64, 65, 66, 4}}}},
          {"inner_order", {3, 2, 1, 0}},
      }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

// Tests obtaining the chunk layout in the case that each shard does not
// correspond to a single rectangular region.
TEST(DriverTest, ChunkLayoutShardedRawNonRectangular) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 5},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout,
      ChunkLayout::FromJson({
          {"grid_origin", {1, 2, 3, 0}},
          {"read_chunk", {{"shape", {64, 64, 64, 4}}}},
          // Write chunk shape is equal to the full shape of the domain, since
          // that is the finest granularity (for a rectangular region) at which
          // writes can be performed efficiently.  In practice this sharding
          // configuration should not be used.
          {"write_chunk", {{"shape", {34432, 39552, 51520, 4}}}},
          {"inner_order", {3, 2, 1, 0}},
      }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

// Tests obtaining the chunk layout in the case that a non-identity hash
// function is specified.
TEST(DriverTest, ChunkLayoutShardedRawNonIdentity) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "murmurhash3_x86_128"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout,
      ChunkLayout::FromJson({
          {"grid_origin", {1, 2, 3, 0}},
          {"read_chunk", {{"shape", {64, 64, 64, 4}}}},
          {"write_chunk", {{"shape", {34432, 39552, 51520, 4}}}},
          {"inner_order", {3, 2, 1, 0}},
      }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(DriverTest, ChunkLayoutUnshardedCompressedSegmentation) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 4},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"chunk_size", {5, 6, 7}},
           {"size", {10, 99, 98}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout, ChunkLayout::FromJson({
                                {"grid_origin", {1, 2, 3, 0}},
                                {"write_chunk", {{"shape", {5, 6, 7, 4}}}},
                                {"read_chunk", {{"shape", {5, 6, 7, 4}}}},
                                {"codec_chunk", {{"shape", {8, 9, 10, 1}}}},
                                {"inner_order", {3, 2, 1, 0}},
                            }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(DriverTest, ChunkLayoutShardedCompressedSegmentation) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 4},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_layout,
      ChunkLayout::FromJson({
          {"grid_origin", {1, 2, 3, 0}},
          {"write_chunk", {{"shape", {2048, 2048, 2048, 4}}}},
          {"read_chunk", {{"shape", {64, 64, 64, 4}}}},
          {"codec_chunk", {{"shape", {8, 9, 10, 1}}}},
          {"inner_order", {3, 2, 1, 0}},
      }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(CodecSpecTest, CompressedSegmentationShardedGzip) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 4},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "gzip"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  EXPECT_THAT(codec.ToJson(), ::testing::Optional(MatchesJson({
                                  {"driver", "neuroglancer_precomputed"},
                                  {"encoding", "compressed_segmentation"},
                                  {"shard_data_encoding", "gzip"},
                              })));
}

TEST(CodecSpecTest, CompressedSegmentationShardedRaw) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 4},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"preshift_bits", 9},
             {"minishard_bits", 6},
             {"shard_bits", 15},
             {"data_encoding", "raw"},
             {"minishard_index_encoding", "gzip"},
             {"hash", "identity"}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  EXPECT_THAT(codec.ToJson(), ::testing::Optional(MatchesJson({
                                  {"driver", "neuroglancer_precomputed"},
                                  {"encoding", "compressed_segmentation"},
                                  {"shard_data_encoding", "raw"},
                              })));
}

TEST(CodecSpecTest, CompressedSegmentationUnsharded) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 4},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  EXPECT_THAT(codec.ToJson(), ::testing::Optional(MatchesJson({
                                  {"driver", "neuroglancer_precomputed"},
                                  {"encoding", "compressed_segmentation"},
                              })));
}

TEST(CodecSpecTest, Jpeg) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
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
           {"jpeg_quality", 77},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  EXPECT_THAT(codec.ToJson(), ::testing::Optional(MatchesJson({
                                  {"driver", "neuroglancer_precomputed"},
                                  {"encoding", "jpeg"},
                                  {"jpeg_quality", 77},
                              })));
}

TEST(CodecSpecTest, Raw) {
  ::nlohmann::json json_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
      {"multiscale_metadata",
       {
           {"data_type", "uint8"},
           {"num_channels", 1},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"resolution", {1, 1, 1}},
           {"encoding", "raw"},
           {"chunk_size", {64, 64, 64}},
           {"size", {34432, 39552, 51508}},
           {"voxel_offset", {1, 2, 3}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  EXPECT_THAT(codec.ToJson(), ::testing::Optional(MatchesJson({
                                  {"driver", "neuroglancer_precomputed"},
                                  {"encoding", "raw"},
                              })));
}

TEST(DriverCreateWithSchemaTest, Dtypes) {
  constexpr tensorstore::DataType kSupportedDataTypes[] = {
      dtype_v<uint8_t>,  dtype_v<uint16_t>,
      dtype_v<uint32_t>, dtype_v<int8_t>,
      dtype_v<int16_t>,  dtype_v<int32_t>,
      dtype_v<uint64_t>, dtype_v<tensorstore::float32_t>,
  };
  for (auto dtype : kSupportedDataTypes) {
    TestTensorStoreCreateWithSchema({{"driver", "neuroglancer_precomputed"},
                                     {"kvstore", {{"driver", "memory"}}}},
                                    dtype, Schema::Shape({5, 6, 7, 2}));
  }
}

TEST(DriverCreateWithSchemaTest, DimensionUnits) {
  TestTensorStoreCreateWithSchema(
      {{"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}}},
      dtype_v<uint32_t>, Schema::Shape({5, 6, 7, 2}),
      ChunkLayout::ChunkShape({2, 3, 4, 0}),
      Schema::DimensionUnits({"3nm", "4nm", "5nm", std::nullopt}));
}

TEST(DriverCreateWithSchemaTest, ChunkShapeUnsharded) {
  TestTensorStoreCreateWithSchema({{"driver", "neuroglancer_precomputed"},
                                   {"kvstore", {{"driver", "memory"}}}},
                                  dtype_v<uint32_t>,
                                  Schema::Shape({5, 6, 7, 2}),
                                  ChunkLayout::ChunkShape({2, 3, 4, 0}));
}

TEST(DriverCreateWithSchemaTest, ChunkShapeSharded) {
  TestTensorStoreCreateWithSchema(
      {{"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}}},
      dtype_v<uint32_t>, Schema::Shape({1000, 1000, 1000, 2}),
      ChunkLayout::ReadChunkShape({30, 40, 50, 0}),
      ChunkLayout::WriteChunkShape({30 * 4, 40 * 4, 50 * 2, 0}));
}

TEST(DriverCreateWithSchemaTest, ChunkShapeShardedTargetElementsExact) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "neuroglancer_precomputed"},
          {"kvstore", {{"driver", "memory"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 1000, 1000, 1}}}},
               {"chunk_layout",
                {
                    {"read_chunk", {{"shape", {30, 40, 50, 0}}}},
                    {"write_chunk", {{"elements", 30 * 40 * 50 * 8}}},
                }},
               {"dimension_units", {"4nm", "5nm", "6nm", nullptr}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"domain",
           {{"shape", {1000, 1000, 1000, 1}},
            {"labels", {"x", "y", "z", "channel"}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {3, 2, 1, 0}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {30 * 2, 40 * 2, 50 * 2, 1}}}}}},
          {"codec",
           {{"driver", "neuroglancer_precomputed"},
            {"encoding", "raw"},
            {"shard_data_encoding", "gzip"}}},
          {"dimension_units", {"4nm", "5nm", "6nm", nullptr}},
      });
}

TEST(DriverCreateWithSchemaTest, ChunkShapeShardedWriteChunkSizeNegative1) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "neuroglancer_precomputed"},
          {"kvstore", {{"driver", "memory"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 1000, 1000, 1}}}},
               {"chunk_layout",
                {
                    {"read_chunk", {{"shape", {30, 40, 50, 0}}}},
                    {"write_chunk", {{"shape_soft_constraint", {0, 0, -1, 0}}}},
                }},
           }},
      },
      {
          {"dtype", "uint32"},
          {"domain",
           {{"shape", {1000, 1000, 1000, 1}},
            {"labels", {"x", "y", "z", "channel"}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {3, 2, 1, 0}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {960, 1000, 1000, 1}}}}}},
          {"codec",
           {{"driver", "neuroglancer_precomputed"},
            {"encoding", "raw"},
            {"shard_data_encoding", "gzip"}}},
          {"dimension_units", {"nm", "nm", "nm", nullptr}},
      });
}

TEST(DriverCreateWithSchemaTest, ChunkShapeShardedTargetElementsRoundDown) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "neuroglancer_precomputed"},
          {"kvstore", {{"driver", "memory"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 1000, 1000, 1}}}},
               {"chunk_layout",
                {
                    {"read_chunk", {{"shape", {30, 40, 50, 0}}}},
                    {"write_chunk", {{"elements", 30 * 40 * 50 * 9}}},
                }},
           }},
      },
      {
          {"dtype", "uint32"},
          {"domain",
           {{"shape", {1000, 1000, 1000, 1}},
            {"labels", {"x", "y", "z", "channel"}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {3, 2, 1, 0}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {30 * 2, 40 * 2, 50 * 2, 1}}}}}},
          {"codec",
           {{"driver", "neuroglancer_precomputed"},
            {"encoding", "raw"},
            {"shard_data_encoding", "gzip"}}},
          {"dimension_units", {"nm", "nm", "nm", nullptr}},
      });
}

TEST(DriverCreateWithSchemaTest, CompressedSegmentation) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "neuroglancer_precomputed"},
          {"kvstore", {{"driver", "memory"}}},
          {"scale_metadata", {{"encoding", "compressed_segmentation"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 1000, 1000, 2}}}},
               {"chunk_layout",
                {
                    {"chunk", {{"shape", {30, 40, 50, 0}}}},
                }},
           }},
      },
      {
          {"dtype", "uint32"},
          {"domain",
           {{"shape", {1000, 1000, 1000, 2}},
            {"labels", {"x", "y", "z", "channel"}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {3, 2, 1, 0}},
            {"chunk", {{"shape", {30, 40, 50, 2}}}},
            {"codec_chunk", {{"shape", {8, 8, 8, 1}}}}}},
          {"codec",
           {{"driver", "neuroglancer_precomputed"},
            {"encoding", "compressed_segmentation"}}},
          {"dimension_units", {"nm", "nm", "nm", nullptr}},
      });
}

TEST(DriverTest, InvalidCodec) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "neuroglancer_precomputed"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"schema",
                       {
                           {"dtype", "uint16"},
                           {"domain", {{"shape", {100, 200, 300, 1}}}},
                           {"codec", {{"driver", "zarr"}}},
                       }},
                  },
                  tensorstore::OpenMode::create)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Cannot merge codec spec .*"));
}

TEST(DriverTest, InvalidWriteChunkShape) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "neuroglancer_precomputed"},
                      {"kvstore", {{"driver", "memory"}}},
                  },
                  tensorstore::OpenMode::create, dtype_v<uint32_t>,
                  Schema::Shape({1000, 1000, 1000, 2}),
                  ChunkLayout::ReadChunkShape({30, 40, 50, 0}),
                  ChunkLayout::WriteChunkShape({30 * 4, 40 * 2, 50 * 4, 0}))
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Cannot satisfy write chunk shape constraint"));
}

TEST(DriverTest, NoDomain) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "neuroglancer_precomputed"},
                      {"kvstore", {{"driver", "memory"}}},
                  },
                  tensorstore::OpenMode::create, dtype_v<uint32_t>)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: domain must be specified"));
}

TEST(SpecSchemaTest, Domain) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata", {{"size", {100, 200, 300}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                   }},
                  {"domain",
                   {
                       {"labels", {"x", "y", "z", "channel"}},
                       {"shape", {100, 200, 300, {"+inf"}}},
                   }},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(SpecSchemaTest, DomainWithOffset) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {{"voxel_offset", {1, 2, 3}}, {"size", {100, 200, 300}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {1, 2, 3, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                   }},
                  {"domain",
                   {
                       {"labels", {"x", "y", "z", "channel"}},
                       {"inclusive_min", {1, 2, 3, 0}},
                       {"shape", {100, 200, 300, {"+inf"}}},
                   }},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(SpecSchemaTest, DomainWithNumChannels) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"multiscale_metadata", {{"num_channels", 3}}},
                  {"scale_metadata", {{"size", {100, 200, 300}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"chunk", {{"shape", {0, 0, 0, 3}}}},
                   }},
                  {"domain",
                   {
                       {"labels", {"x", "y", "z", "channel"}},
                       {"shape", {100, 200, 300, 3}},
                   }},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutShardingUnknown) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata", {{"chunk_size", {100, 200, 300}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {nullptr, nullptr, nullptr, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"read_chunk", {{"shape", {100, 200, 300, 0}}}},
                   }},
                  {"domain",
                   {{"inclusive_min", {{"-inf"}, {"-inf"}, {"-inf"}, 0}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutUnsharded) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {
                       {"chunk_size", {100, 200, 300}},
                       {"sharding", nullptr},
                   }}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {nullptr, nullptr, nullptr, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"chunk", {{"shape", {100, 200, 300, 0}}}},
                   }},
                  {"domain",
                   {{"inclusive_min", {{"-inf"}, {"-inf"}, {"-inf"}, 0}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutShardedWithoutVolumeSize) {
  // Even with the sharding parameters, without knowing the full volume size, we
  // cannot determine the write chunk shape.
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {
                       {"chunk_size", {100, 200, 300}},
                       {"sharding",
                        {{"@type", "neuroglancer_uint64_sharded_v1"},
                         {"preshift_bits", 1},
                         {"minishard_bits", 2},
                         {"shard_bits", 3},
                         {"data_encoding", "raw"},
                         {"minishard_index_encoding", "raw"},
                         {"hash", "identity"}}},
                   }}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {nullptr, nullptr, nullptr, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"read_chunk", {{"shape", {100, 200, 300, 0}}}},
                   }},
                  {"domain",
                   {{"inclusive_min", {{"-inf"}, {"-inf"}, {"-inf"}, 0}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec",
                   {{"driver", "neuroglancer_precomputed"},
                    {"shard_data_encoding", "raw"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutShardedWithVolumeSizeNonRectangular) {
  // Shard does not correspond to a rectangular region of the domain.  Therefore
  // the write_chunk_shape is still unspecified.
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {
                       {"chunk_size", {100, 200, 300}},
                       {"size", {10000, 20000, 30000}},
                       {"sharding",
                        {{"@type", "neuroglancer_uint64_sharded_v1"},
                         {"preshift_bits", 1},
                         {"minishard_bits", 2},
                         {"shard_bits", 3},
                         {"data_encoding", "raw"},
                         {"minishard_index_encoding", "raw"},
                         {"hash", "identity"}}},
                   }}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"read_chunk", {{"shape", {100, 200, 300, 0}}}},
                   }},
                  {"domain",
                   {{"shape", {10000, 20000, 30000, {"+inf"}}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec",
                   {{"driver", "neuroglancer_precomputed"},
                    {"shard_data_encoding", "raw"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutShardedWithVolumeSizeRectangular) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {
                       {"chunk_size", {100, 200, 300}},
                       {"size", {1000, 2000, 3000}},
                       {"sharding",
                        {{"@type", "neuroglancer_uint64_sharded_v1"},
                         {"preshift_bits", 1},
                         {"minishard_bits", 2},
                         {"shard_bits", 15},
                         {"data_encoding", "raw"},
                         {"minishard_index_encoding", "raw"},
                         {"hash", "identity"}}},
                   }}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                       {"read_chunk", {{"shape", {100, 200, 300, 0}}}},
                       {"write_chunk", {{"shape", {200, 400, 600, 0}}}},
                   }},
                  {"domain",
                   {{"shape", {1000, 2000, 3000, {"+inf"}}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec",
                   {{"driver", "neuroglancer_precomputed"},
                    {"shard_data_encoding", "raw"}}}});
}

TEST(SpecSchemaTest, ChunkLayoutShardedWithVolumeSizeRectangularNoChunkSize) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"scale_metadata",
                   {
                       {"size", {1000, 2000, 3000}},
                       {"sharding",
                        {{"@type", "neuroglancer_uint64_sharded_v1"},
                         {"preshift_bits", 1},
                         {"minishard_bits", 2},
                         {"shard_bits", 15},
                         {"data_encoding", "raw"},
                         {"minishard_index_encoding", "raw"},
                         {"hash", "identity"}}},
                   }}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                   }},
                  {"domain",
                   {{"shape", {1000, 2000, 3000, {"+inf"}}},
                    {"labels", {"x", "y", "z", "channel"}}}},
                  {"codec",
                   {{"driver", "neuroglancer_precomputed"},
                    {"shard_data_encoding", "raw"}}}});
}

TEST(SpecSchemaTest, Codec) {
  TestSpecSchema(
      {{"driver", "neuroglancer_precomputed"},
       {"kvstore", {{"driver", "memory"}}},
       {"scale_metadata", {{"encoding", "jpeg"}, {"jpeg_quality", 50}}}},
      {{"chunk_layout",
        {
            {"grid_origin", {nullptr, nullptr, nullptr, 0}},
            {"inner_order", {3, 2, 1, 0}},
        }},
       {"domain",
        {
            {"labels", {"x", "y", "z", "channel"}},
            {"inclusive_min", {{"-inf"}, {"-inf"}, {"-inf"}, 0}},
        }},
       {"codec",
        {{"driver", "neuroglancer_precomputed"},
         {"encoding", "jpeg"},
         {"jpeg_quality", 50}}}});
}

TEST(SpecSchemaTest, Dtype) {
  TestSpecSchema({{"driver", "neuroglancer_precomputed"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"multiscale_metadata", {{"data_type", "uint32"}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {nullptr, nullptr, nullptr, 0}},
                       {"inner_order", {3, 2, 1, 0}},
                   }},
                  {"dtype", "uint32"},
                  {"domain",
                   {
                       {"labels", {"x", "y", "z", "channel"}},
                       {"inclusive_min", {{"-inf"}, {"-inf"}, {"-inf"}, 0}},
                   }},
                  {"codec", {{"driver", "neuroglancer_precomputed"}}}});
}

TEST(DriverTest, ChunkLayoutMismatch) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        context, Schema::Shape({100, 200, 300, 1}),
                        ChunkLayout::ChunkShape({30, 40, 50, 1}),
                        dtype_v<uint32_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        context, ChunkLayout::ChunkShape({31, 40, 50, 1}))
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: chunk layout from metadata does not match chunk "
                    "layout in schema: .*"));
}

TEST(DriverTest, CodecMismatchEncoding) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"scale_metadata", {{"encoding", "raw"}}}},
                        context, Schema::Shape({100, 200, 300, 1}),
                        dtype_v<uint8_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"schema",
                          {{"codec",
                            {{"driver", "neuroglancer_precomputed"},
                             {"encoding", "jpeg"}}}}}},
                        context)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*: codec from metadata does not match codec in schema: .*"));
}

TEST(DriverTest, CodecChunkShapeInvalid) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"scale_metadata", {{"encoding", "raw"}}}},
                        context, Schema::Shape({100, 200, 300, 1}),
                        dtype_v<uint8_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(
      tensorstore::Open(
          {{"driver", "neuroglancer_precomputed"},
           {"kvstore", {{"driver", "memory"}}},
           {"schema",
            {{"chunk_layout", {{"codec_chunk", {{"shape", {8, 8, 8, 1}}}}}}}}},
          context)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: codec_chunk_shape not supported by raw encoding"));
}

TEST(DriverTest, CodecMismatchShardDataEncoding) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"scale_metadata", {{"encoding", "raw"}}}},
                        context, Schema::Shape({100, 200, 300, 1}),
                        dtype_v<uint8_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                                 {"kvstore", {{"driver", "memory"}}},
                                 {"schema",
                                  {{"codec",
                                    {{"driver", "neuroglancer_precomputed"},
                                     {"shard_data_encoding", "raw"}}}}}},
                                context)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: shard_data_encoding requires sharded format"));
}

TEST(DriverTest, FillValue) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        context, Schema::Shape({100, 200, 300, 1}),
                        dtype_v<uint8_t>, tensorstore::OpenMode::create)
          .result());
  EXPECT_THAT(
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"schema", {{"fill_value", 42}}}},
                        context)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*: fill_value not supported by neuroglancer_precomputed format"));
}

TEST(DriverTest, DimensionUnitsInvalidBaseUnit) {
  EXPECT_THAT(tensorstore::Open(
                  {{"driver", "neuroglancer_precomputed"},
                   {"kvstore", {{"driver", "memory"}}}},
                  Schema::Shape({100, 200, 300, 1}), dtype_v<uint8_t>,
                  Schema::DimensionUnits({"4nm", "4nm", "um", std::nullopt}),
                  tensorstore::OpenMode::create)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".* requires a base unit of \"nm\" .*"));
}

TEST(DriverTest, DimensionUnitsInvalidChannelUnit) {
  EXPECT_THAT(
      tensorstore::Open({{"driver", "neuroglancer_precomputed"},
                         {"kvstore", {{"driver", "memory"}}}},
                        Schema::Shape({100, 200, 300, 1}), dtype_v<uint8_t>,
                        Schema::DimensionUnits({"4nm", "4nm", "40nm", ""}),
                        tensorstore::OpenMode::create)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".* does not allow units to be specified for channel dimension"));
}

TEST(DriverTest, DimensionUnitsInvalidResolution) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "neuroglancer_precomputed"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"scale_metadata", {{"resolution", {4, 4, 50}}}},
                  },
                  Schema::Shape({100, 200, 300, 1}), dtype_v<uint8_t>,
                  Schema::DimensionUnits({"4nm", "4nm", "40nm", std::nullopt}),
                  tensorstore::OpenMode::create)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".* do not match \"resolution\" in metadata: .*"));
}

TEST(DriverTest, MultipleScales) {
  auto context = Context::Default();

  ::nlohmann::json base_spec{
      {"driver", "neuroglancer_precomputed"},
      {"kvstore", {{"driver", "memory"}}},
  };

  // Create first scale with 8x10x30nm resolution
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open(
          base_spec, context, Schema::Shape({100, 200, 300, 1}),
          dtype_v<uint8_t>,
          Schema::DimensionUnits({"8nm", "10nm", "30nm", std::nullopt}),
          tensorstore::OpenMode::create)
          .result());

  // Create second scale with 16x20x30nm resolution
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open(
          base_spec, context, Schema::Shape({50, 100, 300, 1}),
          dtype_v<uint8_t>,
          Schema::DimensionUnits({"16nm", "20nm", "30nm", std::nullopt}),
          tensorstore::OpenMode::create)
          .result());

  // Open 8x10x30nm scale via `DimensionUnits` constraint.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(base_spec, context,
                          Schema::DimensionUnits({"8nm", std::nullopt,
                                                  std::nullopt, std::nullopt}))
            .result());
    EXPECT_THAT(store.dimension_units(),
                ::testing::Optional(::testing::ElementsAre(
                    Unit("8nm"), Unit("10nm"), Unit("30nm"), std::nullopt)));
  }

  // Open 16x20x30nm scale via `DimensionUnits` constraint.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(base_spec, context,
                          Schema::DimensionUnits({"16nm", std::nullopt,
                                                  std::nullopt, std::nullopt}))
            .result());
    EXPECT_THAT(store.dimension_units(),
                ::testing::Optional(::testing::ElementsAre(
                    Unit("16nm"), Unit("20nm"), Unit("30nm"), std::nullopt)));
  }
}

TEST(DriverTest, SerializationRoundTrip) {
  ScopedTemporaryDirectory temp_dir;
  ::nlohmann::json json_spec;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "neuroglancer_precomputed"},
           {"kvstore", {{"driver", "file"}, {"path", temp_dir.path()}}}},
          tensorstore::OpenMode::create, tensorstore::dtype_v<uint8_t>,
          tensorstore::Schema::Shape({100, 200, 300, 1}),
          tensorstore::ReadWriteMode::read_write)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_spec, store.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_spec_json, store_spec.ToJson());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_copy,
                                   SerializationRoundTrip(store));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_copy_spec, store_copy.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_copy_spec_json,
                                   store_copy_spec.ToJson());
  EXPECT_THAT(store_copy_spec_json, MatchesJson(store_spec_json));
}

}  // namespace
