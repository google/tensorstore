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

/// End-to-end tests of the zarr driver.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/compression/blosc.h"
#include "tensorstore/internal/decoded_matches.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/open.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/to_string.h"

namespace {

using tensorstore::complex64_t;
using tensorstore::Context;
using tensorstore::Index;
using tensorstore::KeyValueStore;
using tensorstore::kImplicit;
using tensorstore::MatchesStatus;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StrCat;
using tensorstore::internal::DecodedMatches;
using tensorstore::internal::GetMap;
using tensorstore::internal::ParseJsonMatches;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", "<i2"},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
}

TEST(OpenTest, DeleteExistingWithoutCreate) {
  auto context = Context::Default();

  EXPECT_THAT(
      tensorstore::Open(
          context, GetJsonSpec(),
          {tensorstore::OpenMode::delete_existing | tensorstore::OpenMode::open,
           tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error opening \"zarr\" driver: "
          "Cannot specify an open mode of `delete_existing` without `create`"));
}

TEST(OpenTest, DeleteExistingWithOpen) {
  auto context = Context::Default();

  EXPECT_THAT(
      tensorstore::Open(
          context, GetJsonSpec(),
          {tensorstore::OpenMode::delete_existing |
               tensorstore::OpenMode::open | tensorstore::OpenMode::create,
           tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error opening \"zarr\" driver: "
          "Cannot specify an open mode of `delete_existing` with `open`"));
}

TEST(OpenTest, CreateWithoutWrite) {
  auto context = Context::Default();

  EXPECT_THAT(
      tensorstore::Open(
          context, GetJsonSpec(),
          {tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error opening \"zarr\" driver: "
                    "Cannot specify an open mode of `create` without `write`"));
}

TEST(ZarrDriverTest, OpenNonExisting) {
  auto context = Context::Default();

  EXPECT_THAT(
      tensorstore::Open(
          context, GetJsonSpec(),
          {tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(absl::StatusCode::kNotFound,
                    "Error opening \"zarr\" driver: "
                    "Metadata key \"prefix/\\.zarray\" does not exist"));
}

TEST(ZarrDriverTest, OpenOrCreate) {
  auto context = Context::Default();

  EXPECT_EQ(Status(), GetStatus(tensorstore::Open(
                                    context, GetJsonSpec(),
                                    {tensorstore::OpenMode::open |
                                         tensorstore::OpenMode::create,
                                     tensorstore::ReadWriteMode::read_write})
                                    .result()));
}

TEST(ZarrDriverTest, OpenInvalidRank) {
  auto context = Context::Default();
  auto spec = GetJsonSpec();
  spec["rank"] = 3;
  EXPECT_THAT(tensorstore::Open(
                  context, spec,
                  {tensorstore::OpenMode::open | tensorstore::OpenMode::create,
                   tensorstore::ReadWriteMode::read_write})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot compose transform of rank 2 -> 2 with "
                            "transform of rank 3 -> 3"));
}

TEST(ZarrDriverTest, Create) {
  ::nlohmann::json json_spec = GetJsonSpec();

  auto context = Context::Default();
  // Create the store.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(100, 100));
    EXPECT_THAT(store.domain().labels(), ::testing::ElementsAre("", ""));
    EXPECT_THAT(store.domain().implicit_lower_bounds(),
                ::testing::ElementsAre(0, 0));
    EXPECT_THAT(store.domain().implicit_upper_bounds(),
                ::testing::ElementsAre(1, 1));

    // Test ResolveBounds.
    auto resolved = ResolveBounds(store).value();
    EXPECT_EQ(store.domain(), resolved.domain());

    // Test ResolveBounds with a transform that swaps upper and lower bounds.
    auto reversed_dim0 = ChainResult(store, tensorstore::Dims(0).ClosedInterval(
                                                kImplicit, kImplicit, -1))
                             .value();
    auto resolved_reversed_dim0 = ResolveBounds(reversed_dim0).value();
    EXPECT_EQ(reversed_dim0.domain(), resolved_reversed_dim0.domain());

    // Issue a read to be filled with the fill value.
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>({{0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 7}, {1, 1})))
            .value());

    // Issue an out-of-bounds read.
    EXPECT_THAT(
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {100, 7}, {1, 1})))
            .result(),
        MatchesStatus(absl::StatusCode::kOutOfRange));

    // Issue a valid write.
    EXPECT_EQ(
        Status(),
        GetStatus(
            tensorstore::Write(
                tensorstore::MakeArray<std::int16_t>({{1, 2, 3}, {4, 5, 6}}),
                ChainResult(store,
                            tensorstore::AllDims().TranslateSizedInterval(
                                {9, 8}, {2, 3})))
                .commit_future.result()));

    // Issue an out-of-bounds write.
    EXPECT_THAT(
        tensorstore::Write(
            tensorstore::MakeArray<std::int16_t>({{1, 2, 3}, {4, 5, 6}}),
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {100, 8}, {2, 3})))
            .commit_future.result(),
        MatchesStatus(absl::StatusCode::kOutOfRange));

    // Re-read and validate result.
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>(
            {{0, 1, 2, 3, 0}, {0, 4, 5, 6, 0}, {0, 0, 0, 0, 0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 7}, {3, 5})))
            .value());
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      UnorderedElementsAreArray({
          Pair("prefix/.zarray",  //
               ParseJsonMatches(::nlohmann::json{
                   {"zarr_format", 2},
                   {"order", "C"},
                   {"filters", nullptr},
                   {"fill_value", nullptr},
                   {"compressor",
                    {{"id", "blosc"},
                     {"blocksize", 0},
                     {"clevel", 5},
                     {"cname", "lz4"},
                     {"shuffle", -1}}},
                   {"dtype", "<i2"},
                   {"shape", {100, 100}},
                   {"chunks", {3, 2}},
               })),
          Pair("prefix/3.4",    //
               DecodedMatches(  //
                   ElementsAreArray({1, 0, 2, 0, 4, 0, 5, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
          Pair("prefix/3.5",    //
               DecodedMatches(  //
                   ElementsAreArray({3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
      }));

  // Check that attempting to create the store again fails.
  {
    EXPECT_THAT(
        tensorstore::Open(context, json_spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(
            absl::StatusCode::kAlreadyExists,
            "Error opening \"zarr\" driver: "
            "Error creating array with metadata key \"prefix/\\.zarray\""));
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
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>(
            {{0, 1, 2, 3, 0}, {0, 4, 5, 6, 0}, {0, 0, 0, 0, 0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 7}, {3, 5})))
            .value());
  }

  // Check that delete_existing works.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create |
                                        tensorstore::OpenMode::delete_existing,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();

    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>(
            {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 7}, {3, 5})))
            .value());
    auto kv_store =
        KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
    EXPECT_THAT(ListFuture(kv_store.get()).value(),
                ::testing::UnorderedElementsAre("prefix/.zarray"));
  }
}

TEST(ZarrDriverTest, Spec) {
  const ::nlohmann::json create_spec = GetJsonSpec();
  const ::nlohmann::json full_spec  //
      {{"dtype", "int16"},
       {"driver", "zarr"},
       {"path", "prefix"},
       {"allow_metadata_mismatch", false},
       {"delete_existing", false},
       {"recheck_cached_data", true},
       {"recheck_cached_metadata", "open"},
       {"key_encoding", "."},
       {"metadata",
        {{"chunks", {3, 2}},
         {"compressor",
          {{"blocksize", 0},
           {"clevel", 5},
           {"cname", "lz4"},
           {"id", "blosc"},
           {"shuffle", -1}}},
         {"dtype", "<i2"},
         {"fill_value", nullptr},
         {"filters", nullptr},
         {"order", "C"},
         {"shape", {100, 100}},
         {"zarr_format", 2}}},
       {"kvstore", {{"driver", "memory"}}},
       {"transform",
        {{"input_exclusive_max", {{100}, {100}}},
         {"input_inclusive_min", {0, 0}}}}};

  const ::nlohmann::json minimal_spec  //
      {{"dtype", "int16"},
       {"driver", "zarr"},
       {"path", "prefix"},
       {"allow_metadata_mismatch", false},
       {"delete_existing", false},
       {"recheck_cached_data", true},
       {"recheck_cached_metadata", "open"},
       {"key_encoding", "."},
       {"kvstore", {{"driver", "memory"}}},
       {"transform",
        {{"input_exclusive_max", {{100}, {100}}},
         {"input_inclusive_min", {0, 0}}}}};

  {
    auto context = Context::Default();
    auto store =
        tensorstore::Open(context, create_spec, tensorstore::OpenMode::create)
            .value();

    // Test retrieving the full and minimal Spec.
    EXPECT_EQ(full_spec, store.spec()
                             .value()
                             .ToJson(tensorstore::IncludeContext{false})
                             .value());
    EXPECT_EQ(minimal_spec, store.spec(tensorstore::MinimalSpec{true})
                                .value()
                                .ToJson(tensorstore::IncludeContext{false})
                                .value());

    // Test that the minimal spec round trips for opening existing TensorStore.
    auto store2 =
        tensorstore::Open(context, minimal_spec, tensorstore::OpenMode::open)
            .value();
    EXPECT_EQ(full_spec, store2.spec()
                             .value()
                             .ToJson(tensorstore::IncludeContext{false})
                             .value());
    EXPECT_EQ(minimal_spec, store2.spec(tensorstore::MinimalSpec{true})
                                .value()
                                .ToJson(tensorstore::IncludeContext{false})
                                .value());
  }

  // Test that the full Spec round trips for creating a new TensorStore.
  {
    auto context = Context::Default();
    auto store =
        tensorstore::Open(context, full_spec, tensorstore::OpenMode::create)
            .value();
    EXPECT_EQ(full_spec,
              store.spec().value().ToJson(tensorstore::IncludeContext{false}));
  }
}

void TestCreateWriteRead(Context context, ::nlohmann::json json_spec) {
  // Create the store.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();
    EXPECT_EQ(
        Status(),
        GetStatus(
            tensorstore::Write(
                tensorstore::MakeArray<std::int16_t>({{1, 2, 3}, {4, 5, 6}}),
                ChainResult(store,
                            tensorstore::AllDims().TranslateSizedInterval(
                                {9, 8}, {2, 3})))
                .commit_future.result()));
  }

  // Reopen the store.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>(
            {{0, 1, 2, 3, 0}, {0, 4, 5, 6, 0}, {0, 0, 0, 0, 0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 7}, {3, 5})))
            .value());
  }
}

TEST(ZarrDriverTest, CreateBigEndian) {
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", ">i2"},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
  auto context = Context::Default();
  TestCreateWriteRead(context, json_spec);
  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      UnorderedElementsAreArray({
          Pair("prefix/.zarray",  //
               ParseJsonMatches(::nlohmann::json{
                   {"zarr_format", 2},
                   {"order", "C"},
                   {"filters", nullptr},
                   {"fill_value", nullptr},
                   {"compressor",
                    {{"id", "blosc"},
                     {"blocksize", 0},
                     {"clevel", 5},
                     {"cname", "lz4"},
                     {"shuffle", -1}}},
                   {"dtype", ">i2"},
                   {"shape", {100, 100}},
                   {"chunks", {3, 2}},
               })),
          Pair("prefix/3.4",    //
               DecodedMatches(  //
                   ElementsAreArray({0, 1, 0, 2, 0, 4, 0, 5, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
          Pair("prefix/3.5",    //
               DecodedMatches(  //
                   ElementsAreArray({0, 3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
      }));
}

TEST(ZarrDriverTest, CreateBigEndianUnaligned) {
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"field", "y"},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", ::nlohmann::json::array_t{{"x", "|b1"}, {"y", ">i2"}}},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
  auto context = Context::Default();
  TestCreateWriteRead(context, json_spec);

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      UnorderedElementsAreArray({
          Pair("prefix/.zarray",
               ParseJsonMatches(::nlohmann::json{
                   {"zarr_format", 2},
                   {"order", "C"},
                   {"filters", nullptr},
                   {"fill_value", nullptr},
                   {"compressor",
                    {{"id", "blosc"},
                     {"blocksize", 0},
                     {"clevel", 5},
                     {"cname", "lz4"},
                     {"shuffle", -1}}},
                   {"dtype",
                    ::nlohmann::json::array_t{{"x", "|b1"}, {"y", ">i2"}}},
                   {"shape", {100, 100}},
                   {"chunks", {3, 2}},
               })),
          Pair("prefix/3.4",    //
               DecodedMatches(  //
                   ElementsAreArray(
                       {0, 0, 1, 0, 0, 2, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
          Pair("prefix/3.5",    //
               DecodedMatches(  //
                   ElementsAreArray(
                       {0, 0, 3, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
      }));
}

TEST(ZarrDriverTest, CreateLittleEndianUnaligned) {
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"field", "y"},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", ::nlohmann::json::array_t{{"x", "|b1"}, {"y", "<i2"}}},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
  auto context = Context::Default();
  TestCreateWriteRead(context, json_spec);

  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read})
                     .value();
    EXPECT_EQ(::nlohmann::json({{"dtype", "int16"},
                                {"driver", "zarr"},
                                {"field", "y"},
                                {"path", "prefix"},
                                {"kvstore", {{"driver", "memory"}}},
                                {"transform",
                                 {{"input_exclusive_max", {{100}, {100}}},
                                  {"input_inclusive_min", {0, 0}}}}}),
              store.spec(tensorstore::MinimalSpec{true})
                  .value()
                  .ToJson(tensorstore::IncludeDefaults{false}));
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      UnorderedElementsAreArray({
          Pair("prefix/.zarray",
               ParseJsonMatches(::nlohmann::json{
                   {"zarr_format", 2},
                   {"order", "C"},
                   {"filters", nullptr},
                   {"fill_value", nullptr},
                   {"compressor",
                    {{"id", "blosc"},
                     {"blocksize", 0},
                     {"clevel", 5},
                     {"cname", "lz4"},
                     {"shuffle", -1}}},
                   {"dtype",
                    ::nlohmann::json::array_t{{"x", "|b1"}, {"y", "<i2"}}},
                   {"shape", {100, 100}},
                   {"chunks", {3, 2}},
               })),
          Pair("prefix/3.4",    //
               DecodedMatches(  //
                   ElementsAreArray(
                       {0, 1, 0, 0, 2, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
          Pair("prefix/3.5",    //
               DecodedMatches(  //
                   ElementsAreArray(
                       {0, 3, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
      }));
}

TEST(ZarrDriverTest, CreateComplexWithFillValue) {
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", "<c8"},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
           {"fill_value", {1, 2}},
       }},
  };
  auto context = Context::Default();
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();

  EXPECT_EQ(tensorstore::MakeScalarArray<complex64_t>(complex64_t{1, 2}),
            tensorstore::Read(
                ChainResult(store, tensorstore::Dims(0, 1).IndexSlice(4)))
                .value());
}

::nlohmann::json GetBasicResizeMetadata() {
  return {
      {"zarr_format", 2},      {"order", "C"},          {"filters", nullptr},
      {"fill_value", nullptr}, {"compressor", nullptr}, {"dtype", "|i1"},
      {"shape", {100, 100}},   {"chunks", {3, 2}},
  };
}

TEST(ZarrDriverTest, KeyEncodingWithSlash) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"key_encoding", "/"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_EQ(
      Status(),
      GetStatus(
          tensorstore::Write(
              tensorstore::MakeArray<std::int8_t>({{1, 2, 3}, {4, 5, 6}}),
              ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                     {2, 1}, {2, 3})))
              .commit_future.result()));
  // Check that key value store has expected contents.
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/.zarray", ParseJsonMatches(zarr_metadata_json)),
          Pair("prefix/0/0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
          Pair("prefix/0/1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
          Pair("prefix/1/0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
          Pair("prefix/1/1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));
}

TEST(ZarrDriverTest, Resize) {
  for (bool enable_cache : {false, true}) {
    for (const auto resize_mode :
         {tensorstore::ResizeMode(), tensorstore::shrink_only}) {
      Context context(
          Context::Spec::FromJson(
              {{"cache_pool",
                {{"total_bytes_limit", enable_cache ? 10000000 : 0}}}})
              .value());
      SCOPED_TRACE(StrCat("resize_mode=", resize_mode));
      // Create the store.
      ::nlohmann::json storage_spec{{"driver", "memory"}};
      ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
      ::nlohmann::json json_spec{
          {"driver", "zarr"},
          {"kvstore", storage_spec},
          {"path", "prefix"},
          {"metadata", zarr_metadata_json},
      };
      auto store = tensorstore::Open(context, json_spec,
                                     {tensorstore::OpenMode::create,
                                      tensorstore::ReadWriteMode::read_write})
                       .value();
      EXPECT_EQ(
          Status(),
          GetStatus(
              tensorstore::Write(
                  tensorstore::MakeArray<std::int8_t>({{1, 2, 3}, {4, 5, 6}}),
                  ChainResult(store,
                              tensorstore::AllDims().TranslateSizedInterval(
                                  {2, 1}, {2, 3})))
                  .commit_future.result()));
      // Check that key value store has expected contents.
      auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
      EXPECT_THAT(  //
          GetMap(kv_store).value(),
          UnorderedElementsAre(
              Pair("prefix/.zarray", ParseJsonMatches(zarr_metadata_json)),
              Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
              Pair("prefix/0.1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
              Pair("prefix/1.0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
              Pair("prefix/1.1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));

      auto resize_future =
          Resize(store, span<const Index>({kImplicit, kImplicit}),
                 span<const Index>({3, 2}), resize_mode);
      ASSERT_EQ(Status(), GetStatus(resize_future.result()));
      EXPECT_EQ(tensorstore::BoxView({3, 2}),
                resize_future.value().domain().box());

      ::nlohmann::json resized_zarr_metadata_json = zarr_metadata_json;
      resized_zarr_metadata_json["shape"] = {3, 2};
      EXPECT_THAT(  //
          GetMap(kv_store).value(),
          UnorderedElementsAre(
              Pair("prefix/.zarray",
                   ParseJsonMatches(resized_zarr_metadata_json)),
              Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1}))));
    }
  }
}

// Tests that zero-size resizable dimensions are handled correctly.
//
// `op...` should be a pack of functions that can be applied to a `TensorStore`,
// which when composed have the effect of selecting a region of shape `{2, 3}`.
template <typename... Op>
void TestResizeToZeroAndBack(Op... op) {
  Context context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  // Resize to shape of {0, 0}
  auto resize_future = Resize(store, span<const Index>({kImplicit, kImplicit}),
                              span<const Index>({0, 0}));
  ASSERT_EQ(Status(), GetStatus(resize_future.result()));
  EXPECT_EQ(tensorstore::BoxView({0, 0}), resize_future.value().domain().box());

  // Resize back to non-zero shape of {10, 20}.
  auto resize_future2 = Resize(store, span<const Index>({kImplicit, kImplicit}),
                               span<const Index>({10, 20}));
  ASSERT_EQ(Status(), GetStatus(resize_future2.result()));
  EXPECT_EQ(tensorstore::BoxView({10, 20}),
            resize_future2.value().domain().box());

  auto transformed_store = ChainResult(resize_future.value(), op...);
  EXPECT_EQ(Status(), GetStatus(transformed_store));

  // Should be able to write using `resize_future.value()`.  Use
  // `IndexArraySlice` to ensure that edge cases of `ComposeTransforms` are
  // tested.
  EXPECT_EQ(Status(),
            GetStatus(tensorstore::Write(tensorstore::MakeArray<std::int8_t>(
                                             {{1, 2, 3}, {4, 5, 6}}),
                                         transformed_store)
                          .commit_future.result()));

  // Test that reading back yields the correct result.
  EXPECT_THAT(tensorstore::Read(transformed_store).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<std::int8_t>({{1, 2, 3}, {4, 5, 6}})));
}

// Tests that zero-size resizable dimensions are handled correctly.
//
// After resizing, domain is:
// [0, 0*), [0, 0*)
//
// The `IndexArraySlice` operation results in a domain of:
// [0, 2), [0, 0)*
//
// The subsequent `TranslateSizedInterval` operation results in a domain of:
// [0, 2), [0, 3)
//
// This test verifies that the intermediate transform with a domain of
// `[0, 2), [0, 0*)` is handled correctly.
TEST(ZarrDriverTest, ResizeToZeroAndBackIndexArray) {
  TestResizeToZeroAndBack(tensorstore::Dims(0).IndexArraySlice(
                              tensorstore::MakeArray<Index>({0, 1})),
                          tensorstore::Dims(1).TranslateSizedInterval(0, 3));
}

// Same as above, but using an `IndexTransform` explicitly rather than a
// `DimExpression`.
TEST(ZarrDriverTest, ResizeToZeroAndBackIndexTransform) {
  TestResizeToZeroAndBack(tensorstore::IndexTransformBuilder<>(2, 2)
                              .input_shape({2, 0})
                              .implicit_upper_bounds({0, 1})
                              .output_single_input_dimension(0, 0)
                              .output_single_input_dimension(1, 1)
                              .Finalize()
                              .value(),
                          tensorstore::IndexTransformBuilder<>(2, 2)
                              .input_shape({2, 3})
                              .output_single_input_dimension(0, 0)
                              .output_single_input_dimension(1, 1)
                              .Finalize()
                              .value());
}

TEST(ZarrDriverTest, ResizeMetadataOnly) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_EQ(
      Status(),
      GetStatus(
          tensorstore::Write(
              tensorstore::MakeArray<std::int8_t>({{1, 2, 3}, {4, 5, 6}}),
              ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                     {2, 1}, {2, 3})))
              .commit_future.result()));
  // Check that key value store has expected contents.
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/.zarray", ParseJsonMatches(zarr_metadata_json)),
          Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
          Pair("prefix/0.1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
          Pair("prefix/1.0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
          Pair("prefix/1.1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));

  auto resize_future =
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({3, 2}), tensorstore::resize_metadata_only);
  ASSERT_EQ(Status(), GetStatus(resize_future.result()));
  EXPECT_EQ(tensorstore::BoxView({3, 2}), resize_future.value().domain().box());

  ::nlohmann::json resized_zarr_metadata_json = zarr_metadata_json;
  resized_zarr_metadata_json["shape"] = {3, 2};
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/.zarray", ParseJsonMatches(resized_zarr_metadata_json)),
          Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
          Pair("prefix/0.1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
          Pair("prefix/1.0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
          Pair("prefix/1.1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));
}

TEST(ZarrDriverTest, ResizeExpandOnly) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_EQ(
      Status(),
      GetStatus(
          tensorstore::Write(
              tensorstore::MakeArray<std::int8_t>({{1, 2, 3}, {4, 5, 6}}),
              ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                     {2, 1}, {2, 3})))
              .commit_future.result()));
  // Check that key value store has expected contents.
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/.zarray", ParseJsonMatches(zarr_metadata_json)),
          Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
          Pair("prefix/0.1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
          Pair("prefix/1.0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
          Pair("prefix/1.1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));

  auto resize_future =
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({150, 200}), tensorstore::expand_only);
  ASSERT_EQ(Status(), GetStatus(resize_future.result()));
  EXPECT_EQ(tensorstore::BoxView({150, 200}),
            resize_future.value().domain().box());

  ::nlohmann::json resized_zarr_metadata_json = zarr_metadata_json;
  resized_zarr_metadata_json["shape"] = {150, 200};
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/.zarray", ParseJsonMatches(resized_zarr_metadata_json)),
          Pair("prefix/0.0", ElementsAreArray({0, 0, 0, 0, 0, 1})),
          Pair("prefix/0.1", ElementsAreArray({0, 0, 0, 0, 2, 3})),
          Pair("prefix/1.0", ElementsAreArray({0, 4, 0, 0, 0, 0})),
          Pair("prefix/1.1", ElementsAreArray({5, 6, 0, 0, 0, 0}))));
}

TEST(ZarrDriverTest, InvalidResize) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_THAT(
      Resize(
          ChainResult(store, tensorstore::Dims(0).SizedInterval(0, 10)).value(),
          span<const Index>({kImplicit, kImplicit}),
          span<const Index>({kImplicit, 2}))
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would also affect output dimension 0 "
                    "over the interval \\[10, 100\\) but `resize_tied_bounds` "
                    "was not specified"));

  EXPECT_THAT(
      Resize(ChainResult(store, tensorstore::Dims(0).HalfOpenInterval(5, 100))
                 .value(),
             span<const Index>({kImplicit, kImplicit}),
             span<const Index>({kImplicit, 2}))
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would also affect output dimension 0 "
                    "over the interval \\[0, 5\\) but `resize_tied_bounds` "
                    "was not specified"));

  EXPECT_THAT(
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({kImplicit, 10}), tensorstore::expand_only)
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Resize operation would shrink output dimension 1 from "
          "\\[0, 100\\) to \\[0, 10\\) but `expand_only` was specified"));

  EXPECT_THAT(
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({kImplicit, 200}), tensorstore::shrink_only)
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Resize operation would expand output dimension 1 from "
          "\\[0, 100\\) to \\[0, 200\\) but `shrink_only` was specified"));
}

TEST(ZarrDriverTest, InvalidResizeConcurrentModification) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();

  // Make bounds of dimension 0 explicit.
  auto store_slice =
      ChainResult(store, tensorstore::Dims(0).HalfOpenInterval(0, 100)).value();

  EXPECT_EQ(Status(),
            GetStatus(Resize(store, span<const Index>({kImplicit, kImplicit}),
                             span<const Index>({50, kImplicit}))
                          .result()));

  EXPECT_THAT(
      Resize(store_slice, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({kImplicit, 50}))
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would also affect output dimension 0 "
                    "over the out-of-bounds interval \\[50, 100\\)"));
}

TEST(ZarrDriverTest, InvalidResizeLowerBound) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();

  EXPECT_THAT(
      Resize(ChainResult(store, tensorstore::Dims(0).UnsafeMarkBoundsImplicit())
                 .value(),
             span<const Index>({10, kImplicit}),
             span<const Index>({kImplicit, kImplicit}))
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Cannot change inclusive lower bound of output dimension 0, "
          "which is fixed at 0, to 10"));
}

TEST(ZarrDriverTest, InvalidResizeDueToOtherFields) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  zarr_metadata_json["dtype"] =
      ::nlohmann::json::array_t{{"x", "<u2"}, {"y", "<i2"}};
  ::nlohmann::json json_spec{
      {"driver", "zarr"}, {"kvstore", storage_spec},        {"path", "prefix"},
      {"field", "x"},     {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_THAT(Resize(store, span<const Index>({kImplicit, kImplicit}),
                     span<const Index>({kImplicit, 2}))
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Resize operation would affect other fields but "
                            "`resize_tied_bounds` was not specified"));
}

TEST(ZarrDriverTest, InvalidResizeDueToFieldShapeConstraints) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  zarr_metadata_json["dtype"] = ::nlohmann::json::array_t{{"x", "<u2", {2, 3}}};
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_THAT(
      Resize(ChainResult(store, tensorstore::Dims(3).UnsafeMarkBoundsImplicit())
                 .value(),
             span<const Index>({kImplicit, kImplicit, kImplicit, 0}),
             span<const Index>({kImplicit, kImplicit, kImplicit, 2}))
          .result(),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Cannot change exclusive upper bound of output dimension 3, "
          "which is fixed at 3, to 2"));

  EXPECT_THAT(
      Resize(
          ChainResult(store, tensorstore::Dims(3).SizedInterval(0, 2)).value(),
          span<const Index>({kImplicit, kImplicit, kImplicit, kImplicit}),
          span<const Index>({kImplicit, 2, kImplicit, kImplicit}))
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would also affect output dimension 3 "
                    "over the interval \\[2, 3\\) but `resize_tied_bounds` was "
                    "not specified"));
}

TEST(ZarrDriverTest, InvalidResizeTooManyChunksToDelete) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  zarr_metadata_json["shape"] = ::nlohmann::json::array_t(100, 10);
  zarr_metadata_json["chunks"] = ::nlohmann::json::array_t(100, 1);
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_THAT(Resize(store, std::vector<Index>(100, kImplicit),
                     std::vector<Index>(100, 5))
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            StrCat("Resize would require more than ",
                                   std::numeric_limits<Index>::max(),
                                   " chunk regions to be deleted")));
}

TEST(ZarrDriverTest, InvalidResizeIncompatibleMetadata) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  json_spec["metadata"]["chunks"] = {5, 5};
  auto store2 = tensorstore::Open(context, json_spec,
                                  {tensorstore::OpenMode::create |
                                       tensorstore::OpenMode::delete_existing,
                                   tensorstore::ReadWriteMode::read_write})
                    .value();
  EXPECT_THAT(
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({5, 5}), tensorstore::resize_metadata_only)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Updated zarr metadata .* is incompatible with "
                    "existing metadata .*"));
}

TEST(ZarrDriverTest, InvalidResizeDeletedMetadata) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
  kv_store->Delete("prefix/.zarray").value();
  EXPECT_THAT(
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({5, 5}), tensorstore::resize_metadata_only)
          .result(),
      MatchesStatus(absl::StatusCode::kNotFound, "Metadata was deleted"));
}

TEST(ZarrDriverTest, InvalidSpec) {
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
       {"kvstore", "path", "field", "key_encoding", "metadata"}) {
    auto spec = GetJsonSpec();
    spec[member_name] = 5;
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Error parsing object member \"", member_name,
                             "\": "
                             "Expected .*, but received: 5")));
  }

  {
    auto spec = GetJsonSpec();
    spec["key_encoding"] = "-";
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Error parsing object member \"key_encoding\": "
                      "Expected \"\\.\" or \"/\", but received: \"-\""));
  }

  {
    auto spec = GetJsonSpec();
    spec["metadata"].erase("shape");
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      ".*: "
                      "Cannot create array from specified \"metadata\": "
                      "\"shape\" must be specified"));
  }
}

TEST(ZarrDriverTest, OpenInvalidMetadata) {
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();

  {
    // Write invalid JSON
    EXPECT_EQ(Status(),
              GetStatus(kv_store->Write("prefix/.zarray", "invalid").result()));

    EXPECT_THAT(
        tensorstore::Open(context, json_spec,
                          {tensorstore::OpenMode::open,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(
            absl::StatusCode::kFailedPrecondition,
            "Error opening \"zarr\" driver: "
            "Error decoding metadata from \"prefix/.zarray\": Invalid JSON"));
  }

  {
    auto invalid_json = zarr_metadata_json;
    invalid_json.erase("zarr_format");

    // Write invalid metadata JSON
    EXPECT_EQ(
        Status(),
        GetStatus(
            kv_store->Write("prefix/.zarray", invalid_json.dump()).result()));

    EXPECT_THAT(
        tensorstore::Open(context, json_spec,
                          {tensorstore::OpenMode::open,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      "Error opening \"zarr\" driver: "
                      "Error decoding metadata from \"prefix/.zarray\": "
                      "Missing object member \"zarr_format\""));
  }
}

TEST(ZarrDriverTest, ResolveBoundsIncompatibleMetadata) {
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json zarr_metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();

  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();

  // Overwrite metadata
  zarr_metadata_json["chunks"] = {3, 3};
  json_spec = {
      {"driver", "zarr"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", zarr_metadata_json},
  };

  auto store_new =
      tensorstore::Open(context, json_spec,
                        {tensorstore::OpenMode::create |
                             tensorstore::OpenMode::delete_existing,
                         tensorstore::ReadWriteMode::read_write})
          .value();

  EXPECT_THAT(ResolveBounds(store).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Updated zarr metadata .* is incompatible with "
                            "existing metadata .*"));
}

/// Policies used by `TestDataCaching` and `TestMetadataCaching`.
enum class RecheckOption {
  /// Specify explicit time bound equal to a timestamp saved prior to
  /// modification.
  kExplicitBeforeModifyBound,
  /// Specify explicit time bound equal to the time just before the TensorStore
  /// is opened.
  kExplicitOpenTimeBound,
  /// Specify explicit time bound many seconds in the future.
  kExplicitFutureBound,
  /// Specify constant of `false`, which indicates never to recheck.
  kNeverRecheck,
  /// Specify constant of `true`, which indicates always to check.
  kAlwaysRecheck,
  /// Specify constant of `"open"`, which indicates a time bound equal to the
  /// time just before the TensorStore is opened.
  kOpen,
  /// Specify constant of `0`, which indicates the unix epoch (should behave
  /// like `kNeverRecheck`).
  kExplicitEpochBound,
};

std::ostream& operator<<(std::ostream& os, RecheckOption recheck_option) {
  switch (recheck_option) {
    case RecheckOption::kExplicitBeforeModifyBound:
      return os << "kExplicitBeforeModifyBound";
    case RecheckOption::kExplicitOpenTimeBound:
      return os << "kExplicitOpenTimeBound";
    case RecheckOption::kExplicitFutureBound:
      return os << "kExplicitFutureBound";
    case RecheckOption::kNeverRecheck:
      return os << "kNeverRecheck";
    case RecheckOption::kAlwaysRecheck:
      return os << "kAlwaysRecheck";
    case RecheckOption::kOpen:
      return os << "kOpen";
    case RecheckOption::kExplicitEpochBound:
      return os << "kExplicitEpochBound";
  }
  TENSORSTORE_UNREACHABLE;
}

::nlohmann::json GetRecheckBound(absl::Time before_modify_time,
                                 RecheckOption recheck_option) {
  switch (recheck_option) {
    case RecheckOption::kExplicitBeforeModifyBound:
      return absl::ToDoubleSeconds(before_modify_time - absl::UnixEpoch());
    case RecheckOption::kExplicitOpenTimeBound:
      return absl::ToDoubleSeconds(absl::Now() - absl::UnixEpoch());
    case RecheckOption::kExplicitFutureBound:
      return absl::ToDoubleSeconds(absl::Now() + absl::Seconds(100000) -
                                   absl::UnixEpoch());
    case RecheckOption::kNeverRecheck:
      return false;
    case RecheckOption::kAlwaysRecheck:
      return true;
    case RecheckOption::kOpen:
      return "open";
    case RecheckOption::kExplicitEpochBound:
      return 0;
  }
  TENSORSTORE_UNREACHABLE;
}

/// Performs a sequence of reads and modifications to test the behavior of the
/// `recheck_cached_data` policy specified by `recheck_option`.
///
/// 1. The initial (fill) value is 0.
///
/// 2. Records `before_modify_time` as current timestamp.
///
/// 3. If `modify_before_reopen == true`, writes 1 (without cache coherency).
///
/// 4. Reopens using `recheck_cached_data` of
///    `GetRecheckBound(before_modify_time, recheck_option)`.
///
/// 5. Checks read result against `expected_value1`.
///
/// 6. If `modify_after_reopen == true`, writes 2 (without cache coherency).
///
/// 7. Checks read result against `expected_value2`.
void TestDataCaching(RecheckOption recheck_option, bool modify_before_reopen,
                     bool modify_after_reopen, std::int16_t expected_value1,
                     std::int16_t expected_value2) {
  SCOPED_TRACE(tensorstore::StrCat(
      "recheck_option=", recheck_option, ", modify_before_open=",
      modify_before_reopen, ", modify_after_open=", modify_after_reopen));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context_spec,
      Context::Spec::FromJson(
          {{"cache_pool", {{"total_bytes_limit", 10000000}}}}));

  Context base_context(context_spec);
  auto base_spec = GetJsonSpec();
  base_spec["transform"] = ::nlohmann::json{
      {"input_rank", 0}, {"output", {{{"offset", 0}}, {{"offset", 0}}}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto initial_store,
      tensorstore::Open(base_context, base_spec, tensorstore::OpenMode::create)
          .result());

  auto before_modify_time = absl::Now();

  // Populate the cache.
  TENSORSTORE_ASSERT_OK(tensorstore::Read(initial_store).result());

  const auto modify = [&](std::int16_t new_value) {
    // Create new context that shares the same `memory_key_value_store` but does
    // not share the cache pool.
    auto new_cache_context = Context(context_spec, base_context);

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto new_arr, tensorstore::Open(new_cache_context, base_spec).result());

    // Fill with `new_value`.
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(
            tensorstore::MakeScalarArray<std::int16_t>(new_value), new_arr)
            .result());
  };

  if (modify_before_reopen) {
    modify(1);
  }

  auto new_json_spec = base_spec;
  new_json_spec["recheck_cached_data"] =
      GetRecheckBound(before_modify_time, recheck_option);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_store, tensorstore::Open(base_context, new_json_spec).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto value1,
                                   tensorstore::Read(new_store).result());

  EXPECT_EQ(tensorstore::MakeScalarArray<std::int16_t>(expected_value1),
            value1);

  if (modify_after_reopen) {
    modify(2);
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto value2,
                                   tensorstore::Read(new_store).result());

  EXPECT_EQ(tensorstore::MakeScalarArray<std::int16_t>(expected_value2),
            value2);
}

TEST(ZarrDriverTest, RecheckCachedData) {
  // Test the case where stale cached data is always used.
  for (const auto recheck_option : {
           RecheckOption::kExplicitBeforeModifyBound,
           RecheckOption::kNeverRecheck,
           RecheckOption::kExplicitEpochBound,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestDataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_value1=*/0,
            /*expected_value2=*/0);
      }
    }
  }

  // Test the case where only modifications prior to opening are seen.
  for (const auto recheck_option : {
           RecheckOption::kExplicitOpenTimeBound,
           RecheckOption::kOpen,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestDataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_value1=*/modify_before_reopen ? 1 : 0,
            /*expected_value2=*/modify_before_reopen ? 1 : 0);
      }
    }
  }

  // Test the case where all modifications are seen.
  for (const auto recheck_option : {
           RecheckOption::kAlwaysRecheck,
           RecheckOption::kExplicitFutureBound,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestDataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_value1=*/modify_before_reopen ? 1 : 0,
            /*expected_value2=*/
            modify_after_reopen ? 2 : (modify_before_reopen ? 1 : 0));
      }
    }
  }
}

/// Performs a sequence of metadata reads (`Open` or `ResolveBounds`) and
/// modifications (`Resize`) to test the behavior of the
/// `recheck_cached_metadata` policy specified by `recheck_option`.
///
/// 1. Records `before_modify_time` as current timestamp.
///
/// 2. The initial size is [100, 100].
///
/// 3. If `modify_before_reopen == true`, resizes dim 0 to 200 (without cache
///    coherency).
///
/// 4. Reopens using `recheck_cached_metadata` of
///    `GetRecheckBound(before_modify_time, recheck_option)`.
///
/// 5. Checks dim 0 against `expected_dim0`.
///
/// 6. If `modify_after_reopen == true`, resizes dim 1 to 200 (without cache
///    coherency).
///
/// 7. Checks dim 1 against `expected_dim1`.
void TestMetadataCaching(RecheckOption recheck_option,
                         bool modify_before_reopen, bool modify_after_reopen,
                         Index expected_dim0, Index expected_dim1) {
  SCOPED_TRACE(tensorstore::StrCat(
      "recheck_option=", recheck_option, ", modify_before_open=",
      modify_before_reopen, ", modify_after_open=", modify_after_reopen));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context_spec,
      Context::Spec::FromJson(
          {{"cache_pool", {{"total_bytes_limit", 10000000}}}}));

  Context base_context(context_spec);
  auto base_spec = GetJsonSpec();

  auto before_modify_time = absl::Now();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto initial_store,
      tensorstore::Open(base_context, base_spec, tensorstore::OpenMode::create)
          .result());

  base_spec.erase("metadata");

  const auto modify = [&](tensorstore::DimensionIndex dim) {
    // Create new context that shares the same `memory_key_value_store` but does
    // not share the cache pool.
    const Index new_inclusive_min[2] = {kImplicit, kImplicit};
    Index new_exclusive_max[2] = {kImplicit, kImplicit};
    new_exclusive_max[dim] = 200;

    auto new_cache_context = Context(context_spec, base_context);

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto new_arr, tensorstore::Open(new_cache_context, base_spec).result());

    TENSORSTORE_ASSERT_OK(tensorstore::Resize(new_arr, new_inclusive_min,
                                              new_exclusive_max,
                                              tensorstore::resize_metadata_only)
                              .result());
  };

  if (modify_before_reopen) {
    modify(0);
  }

  auto new_json_spec = base_spec;
  new_json_spec["recheck_cached_metadata"] =
      GetRecheckBound(before_modify_time, recheck_option);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_store, tensorstore::Open(base_context, new_json_spec).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      new_store, tensorstore::ResolveBounds(new_store).result());

  EXPECT_EQ(expected_dim0, new_store.domain().shape()[0]);

  if (modify_after_reopen) {
    modify(1);
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      new_store, tensorstore::ResolveBounds(new_store).result());

  EXPECT_EQ(expected_dim1, new_store.domain().shape()[1]);
}

TEST(ZarrDriverTest, RecheckCachedMetadata) {
  // Test the case where stale cached data is always used.
  for (const auto recheck_option : {
           RecheckOption::kExplicitBeforeModifyBound,
           RecheckOption::kNeverRecheck,
           RecheckOption::kExplicitEpochBound,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestMetadataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_dim0=*/100,
            /*expected_dim1=*/100);
      }
    }
  }

  // Test the case where only modifications prior to opening are seen.
  for (const auto recheck_option : {
           RecheckOption::kExplicitOpenTimeBound,
           RecheckOption::kOpen,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestMetadataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_dim0=*/modify_before_reopen ? 200 : 100,
            /*expected_dim1=*/100);
      }
    }
  }

  // Test the case where all modifications are seen.
  for (const auto recheck_option : {
           RecheckOption::kAlwaysRecheck,
           RecheckOption::kExplicitFutureBound,
       }) {
    for (const bool modify_before_reopen : {false, true}) {
      for (const bool modify_after_reopen : {false, true}) {
        TestMetadataCaching(
            /*recheck_option=*/recheck_option,
            /*modify_before_reopen=*/modify_before_reopen,
            /*modify_after_reopen=*/modify_after_reopen,
            /*expected_dim0=*/modify_before_reopen ? 200 : 100,
            /*expected_dim1=*/modify_after_reopen ? 200 : 100);
      }
    }
  }
}

}  // namespace
