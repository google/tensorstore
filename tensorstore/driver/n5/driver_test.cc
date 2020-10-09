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

/// End-to-end tests of the n5 driver.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::Context;
using tensorstore::Index;
using tensorstore::KeyValueStore;
using tensorstore::kImplicit;
using tensorstore::MatchesStatus;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StrCat;
using tensorstore::internal::GetMap;
using tensorstore::internal::ParseJsonMatches;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "n5"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"metadata",
       {
           {"compression", {{"type", "raw"}}},
           {"dataType", "int16"},
           {"dimensions", {10, 11}},
           {"blockSize", {3, 2}},
       }},
  };
}

TEST(N5DriverTest, OpenNonExisting) {
  auto context = Context::Default();

  EXPECT_THAT(
      tensorstore::Open(
          context, GetJsonSpec(),
          {tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(absl::StatusCode::kNotFound,
                    "Error opening \"n5\" driver: "
                    "Metadata at \"prefix/attributes\\.json\" does not exist"));
}

TEST(N5DriverTest, OpenOrCreate) {
  auto context = Context::Default();

  EXPECT_EQ(Status(), GetStatus(tensorstore::Open(
                                    context, GetJsonSpec(),
                                    {tensorstore::OpenMode::open |
                                         tensorstore::OpenMode::create,
                                     tensorstore::ReadWriteMode::read_write})
                                    .result()));
}

::testing::Matcher<absl::Cord> MatchesRawChunk(std::vector<Index> shape,
                                               std::vector<char> data) {
  std::string out(shape.size() * 4 + 4 + data.size(), '\0');
  out[3] = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    out[7 + i * 4] = shape[i];
  }
  std::copy(data.begin(), data.end(), out.data() + shape.size() * 4 + 4);
  return absl::Cord(out);
}

// Sanity check of `MatchesRawChunk`.
TEST(MatchesRawChunkTest, Basic) {
  std::string chunk{
      0, 0,        // mode
      0, 2,        // rank
      0, 0, 0, 3,  // chunk_shape[0]
      0, 0, 0, 2,  // chunk_shape[1]
      0, 1, 0, 2,  // int16be data
      0, 4, 0, 5,  // int16be data
      0, 0, 0, 0,  // int16be data
  };
  EXPECT_THAT(absl::Cord(chunk), MatchesRawChunk(  //
                                     {3, 2},       //
                                     {
                                         0, 1, 0, 2,  // int16be data
                                         0, 4, 0, 5,  // int16be data
                                         0, 0, 0, 0,  // int16be data
                                     }));
  // Change to invalid "mode" value as trivial test that matcher is doing a
  // comparison.
  chunk[0] = 1;
  EXPECT_THAT(absl::Cord(chunk), ::testing::Not(MatchesRawChunk(  //
                                     {3, 2},                      //
                                     {
                                         0, 1, 0, 2,  // int16be data
                                         0, 4, 0, 5,  // int16be data
                                         0, 0, 0, 0,  // int16be data
                                     })));
}

TEST(N5DriverTest, Create) {
  ::nlohmann::json json_spec = GetJsonSpec();

  json_spec["metadata"].emplace("extra", "attribute");

  auto context = Context::Default();
  // Create the store.
  {
    auto store = tensorstore::Open(context, json_spec,
                                   {tensorstore::OpenMode::create,
                                    tensorstore::ReadWriteMode::read_write})
                     .value();
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(10, 11));
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
    EXPECT_THAT(reversed_dim0.domain().origin(), ::testing::ElementsAre(-9, 0));
    EXPECT_THAT(reversed_dim0.domain().shape(), ::testing::ElementsAre(10, 11));
    EXPECT_THAT(reversed_dim0.domain().labels(),
                ::testing::ElementsAre("", ""));
    EXPECT_THAT(reversed_dim0.domain().implicit_lower_bounds(),
                ::testing::ElementsAre(1, 0));
    EXPECT_THAT(reversed_dim0.domain().implicit_upper_bounds(),
                ::testing::ElementsAre(0, 1));
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
                                   {10, 7}, {1, 1})))
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
                                {8, 8}, {2, 3})))
                .commit_future.result()));

    // Issue an out-of-bounds write.
    EXPECT_THAT(
        tensorstore::Write(
            tensorstore::MakeArray<std::int16_t>({{61, 62, 63}, {64, 65, 66}}),
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {9, 8}, {2, 3})))
            .commit_future.result(),
        MatchesStatus(absl::StatusCode::kOutOfRange));

    // Re-read and validate result.  This verifies that the read/write
    // encoding/decoding paths round trip.
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>({
            {0, 0, 0, 0},
            {0, 1, 2, 3},
            {0, 4, 5, 6},
        }),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {7, 7}, {3, 4})))
            .value());
  }

  // Check that key value store has expected contents.  This verifies that the
  // encoding path works as expected.
  EXPECT_THAT(
      GetMap(KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value())
          .value(),
      UnorderedElementsAreArray({
          Pair("prefix/attributes.json",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches({
                   {"compression", {{"type", "raw"}}},
                   {"dataType", "int16"},
                   {"dimensions", {10, 11}},
                   {"blockSize", {3, 2}},
                   {"extra", "attribute"},
               }))),
          Pair("prefix/2/4",  // chunk starting at: 6, 8
               MatchesRawChunk({3, 2},
                               {
                                   0, 0, 0, 0, 0, 1,  // int16be data
                                   0, 0, 0, 0, 0, 2,  // int16be data
                               })),
          Pair("prefix/2/5",  // chunk starting at 6, 10
               MatchesRawChunk({3, 1},
                               {
                                   0, 0,  // int16be data
                                   0, 0,  // int16be data
                                   0, 3,  // int16be data
                               })),
          Pair("prefix/3/4",  // chunk starting at 9, 8
               MatchesRawChunk({1, 2},
                               {
                                   0, 4, 0, 5,  // int16be data
                               })),
          Pair("prefix/3/5",  // chunk starting at 9, 10
               MatchesRawChunk({1, 1},
                               {
                                   0, 6,  // int16be data
                               })),
      }));

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
    EXPECT_EQ(
        tensorstore::MakeArray<std::int16_t>({
            {0, 0, 0, 0},
            {0, 1, 2, 3},
            {0, 4, 5, 6},
        }),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {7, 7}, {3, 4})))
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
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}),
        tensorstore::Read<tensorstore::zero_origin>(
            ChainResult(store, tensorstore::AllDims().TranslateSizedInterval(
                                   {7, 7}, {3, 4})))
            .value());
    auto kv_store =
        KeyValueStore::Open(context, {{"driver", "memory"}}, {}).value();
    EXPECT_THAT(ListFuture(kv_store.get()).value(),
                ::testing::UnorderedElementsAre("prefix/attributes.json"));
  }
}

::nlohmann::json GetBasicResizeMetadata() {
  return {
      {"compression", {{"type", "raw"}}},
      {"dataType", "int8"},
      {"dimensions", {100, 100}},
      {"blockSize", {3, 2}},
  };
}

TEST(N5DriverTest, Resize) {
  for (bool enable_cache : {false, true}) {
    for (const auto resize_mode :
         {tensorstore::ResizeMode(), tensorstore::shrink_only}) {
      Context context(
          Context::Spec::FromJson(
              {{"cache_pool",
                {{"total_bytes_limit", enable_cache ? 10000000 : 0}}}})
              .value());
      SCOPED_TRACE(StrCat("resize_mode=", resize_mode));
      // Create the store.i
      ::nlohmann::json storage_spec{{"driver", "memory"}};
      ::nlohmann::json metadata_json = GetBasicResizeMetadata();
      ::nlohmann::json json_spec{
          {"driver", "n5"},
          {"kvstore", storage_spec},
          {"path", "prefix"},
          {"metadata", metadata_json},
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
              Pair("prefix/attributes.json",
                   ::testing::MatcherCast<absl::Cord>(
                       ParseJsonMatches(metadata_json))),
              Pair("prefix/0/0", MatchesRawChunk({3, 2}, {0, 0, 0, 0, 0, 1})),
              Pair("prefix/0/1", MatchesRawChunk({3, 2}, {0, 0, 2, 0, 0, 3})),
              Pair("prefix/1/0", MatchesRawChunk({3, 2}, {0, 0, 0, 4, 0, 0})),
              Pair("prefix/1/1", MatchesRawChunk({3, 2}, {5, 0, 0, 6, 0, 0}))));

      auto resize_future =
          Resize(store, span<const Index>({kImplicit, kImplicit}),
                 span<const Index>({3, 2}), resize_mode);
      ASSERT_EQ(Status(), GetStatus(resize_future.result()));
      EXPECT_EQ(tensorstore::BoxView({3, 2}),
                resize_future.value().domain().box());

      ::nlohmann::json resized_metadata_json = metadata_json;
      resized_metadata_json["dimensions"] = {3, 2};
      EXPECT_THAT(  //
          GetMap(kv_store).value(),
          UnorderedElementsAre(
              Pair("prefix/attributes.json",
                   ::testing::MatcherCast<absl::Cord>(
                       ParseJsonMatches(resized_metadata_json))),
              Pair("prefix/0/0", MatchesRawChunk({3, 2}, {0, 0, 0, 0, 0, 1}))));
    }
  }
}

TEST(N5DriverTest, ResizeMetadataOnly) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
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
          Pair("prefix/attributes.json", ::testing::MatcherCast<absl::Cord>(
                                             ParseJsonMatches(metadata_json))),
          Pair("prefix/0/0", MatchesRawChunk({3, 2}, {0, 0, 0, 0, 0, 1})),
          Pair("prefix/0/1", MatchesRawChunk({3, 2}, {0, 0, 2, 0, 0, 3})),
          Pair("prefix/1/0", MatchesRawChunk({3, 2}, {0, 0, 0, 4, 0, 0})),
          Pair("prefix/1/1", MatchesRawChunk({3, 2}, {5, 0, 0, 6, 0, 0}))));

  auto resize_future =
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({3, 2}), tensorstore::resize_metadata_only);
  ASSERT_EQ(Status(), GetStatus(resize_future.result()));
  EXPECT_EQ(tensorstore::BoxView({3, 2}), resize_future.value().domain().box());

  ::nlohmann::json resized_metadata_json = metadata_json;
  resized_metadata_json["dimensions"] = {3, 2};
  EXPECT_THAT(  //
      GetMap(kv_store).value(),
      UnorderedElementsAre(
          Pair("prefix/attributes.json",
               ::testing::MatcherCast<absl::Cord>(
                   ParseJsonMatches(resized_metadata_json))),
          Pair("prefix/0/0", MatchesRawChunk({3, 2}, {0, 0, 0, 0, 0, 1})),
          Pair("prefix/0/1", MatchesRawChunk({3, 2}, {0, 0, 2, 0, 0, 3})),
          Pair("prefix/1/0", MatchesRawChunk({3, 2}, {0, 0, 0, 4, 0, 0})),
          Pair("prefix/1/1", MatchesRawChunk({3, 2}, {5, 0, 0, 6, 0, 0}))));
}

TEST(N5DriverTest, InvalidResize) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
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
          "Error writing \"prefix/attributes\\.json\": "
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

TEST(N5DriverTest, InvalidResizeConcurrentModification) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
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

TEST(N5DriverTest, InvalidResizeLowerBound) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
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

TEST(N5DriverTest, InvalidResizeIncompatibleMetadata) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  json_spec["metadata"]["blockSize"] = {5, 5};
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
                    "Error writing \"prefix/attributes\\.json\": "
                    "Updated N5 metadata .* is incompatible with "
                    "existing metadata .*"));
}

TEST(N5DriverTest, InvalidResizeDeletedMetadata) {
  auto context = Context::Default();
  // Create the store.
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
  };
  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();
  kv_store->Delete("prefix/attributes.json").value();
  // To avoid risk of accidental data loss, no recovery of this TensorStore
  // object is possible after the metadata is modified in an unexpected way.
  EXPECT_THAT(
      Resize(store, span<const Index>({kImplicit, kImplicit}),
             span<const Index>({5, 5}), tensorstore::resize_metadata_only)
          .result(),
      MatchesStatus(absl::StatusCode::kNotFound,
                    "Error writing \"prefix/attributes\\.json\": "
                    "Metadata was deleted"));
}

TEST(N5DriverTest, UnsupportedDataTypeInSpec) {
  auto context = Context::Default();
  EXPECT_THAT(
      tensorstore::Open(context,
                        ::nlohmann::json{
                            {"dtype", "string"},
                            {"driver", "n5"},
                            {"kvstore", {{"driver", "memory"}}},
                            {"path", "prefix"},
                            {"metadata",
                             {
                                 {"compression", {{"type", "raw"}}},
                                 {"dimensions", {100, 100}},
                                 {"blockSize", {3, 2}},
                             }},
                        },
                        {tensorstore::OpenMode::create,
                         tensorstore::ReadWriteMode::read_write})
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "string data type is not one of the supported data types: .*"));
}

TEST(N5DriverTest, DataTypeMismatch) {
  auto context = Context::Default();
  auto store = tensorstore::Open(context,
                                 ::nlohmann::json{
                                     {"dtype", "int8"},
                                     {"driver", "n5"},
                                     {"kvstore", {{"driver", "memory"}}},
                                     {"path", "prefix"},
                                     {"metadata",
                                      {
                                          {"compression", {{"type", "raw"}}},
                                          {"dimensions", {100, 100}},
                                          {"blockSize", {3, 2}},
                                      }},
                                 },
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();
  EXPECT_EQ(tensorstore::DataTypeOf<std::int8_t>(), store.data_type());
  EXPECT_THAT(tensorstore::Open(context,
                                ::nlohmann::json{
                                    {"dtype", "uint8"},
                                    {"driver", "n5"},
                                    {"kvstore", {{"driver", "memory"}}},
                                    {"path", "prefix"},
                                },
                                {tensorstore::OpenMode::open,
                                 tensorstore::ReadWriteMode::read_write})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error opening \"n5\" driver: "
                            "Expected data type of uint8 but received: int8"));
}

TEST(N5DriverTest, InvalidSpec) {
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

  for (const std::string& member_name : {"kvstore", "path", "metadata"}) {
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
    spec["metadata"].erase("dimensions");
    EXPECT_THAT(
        tensorstore::Open(context, spec,
                          {tensorstore::OpenMode::create,
                           tensorstore::ReadWriteMode::read_write})
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      ".*: "
                      "Cannot create array from specified \"metadata\": "
                      "\"dimensions\" must be specified"));
  }
}

TEST(N5DriverTest, OpenInvalidMetadata) {
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
  };
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();

  {
    // Write invalid JSON
    TENSORSTORE_EXPECT_OK(
        kv_store->Write("prefix/attributes.json", absl::Cord("invalid")));

    EXPECT_THAT(tensorstore::Open(context, json_spec,
                                  {tensorstore::OpenMode::open,
                                   tensorstore::ReadWriteMode::read_write})
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              "Error opening \"n5\" driver: "
                              "Error reading "
                              "\"prefix/attributes.json\": Invalid JSON"));
  }

  {
    auto invalid_json = metadata_json;
    invalid_json.erase("dimensions");

    // Write invalid metadata JSON
    TENSORSTORE_EXPECT_OK(kv_store->Write("prefix/attributes.json",
                                          absl::Cord(invalid_json.dump())));

    EXPECT_THAT(tensorstore::Open(context, json_spec,
                                  {tensorstore::OpenMode::open,
                                   tensorstore::ReadWriteMode::read_write})
                    .result(),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              "Error opening \"n5\" driver: "
                              "Error reading \"prefix/attributes.json\": "
                              "Missing object member \"dimensions\""));
  }
}

TEST(N5DriverTest, ResolveBoundsIncompatibleMetadata) {
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json metadata_json = GetBasicResizeMetadata();
  ::nlohmann::json json_spec{
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
  };
  auto kv_store = KeyValueStore::Open(context, storage_spec, {}).value();

  auto store = tensorstore::Open(context, json_spec,
                                 {tensorstore::OpenMode::create,
                                  tensorstore::ReadWriteMode::read_write})
                   .value();

  // Overwrite metadata: change blockSize from {3, 2} -> {3, 3}.
  metadata_json["blockSize"] = {3, 3};
  json_spec = {
      {"driver", "n5"},
      {"kvstore", storage_spec},
      {"path", "prefix"},
      {"metadata", metadata_json},
  };

  auto store_new =
      tensorstore::Open(context, json_spec,
                        {tensorstore::OpenMode::create |
                             tensorstore::OpenMode::delete_existing,
                         tensorstore::ReadWriteMode::read_write})
          .value();

  EXPECT_THAT(ResolveBounds(store).result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Updated N5 metadata .* is incompatible with "
                            "existing metadata .*"));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "n5";
  options.create_spec = GetJsonSpec();
  options.full_spec = {
      {"dtype", "int16"},
      {"driver", "n5"},
      {"path", "prefix"},
      {"metadata",
       {{"blockSize", {3, 2}},
        {"compression", {{"type", "raw"}}},
        {"dataType", "int16"},
        {"dimensions", {10, 11}}}},
      {"kvstore", {{"driver", "memory"}}},
      {"transform",
       {{"input_exclusive_max", {{10}, {11}}},
        {"input_inclusive_min", {0, 0}}}},
  };

  options.minimal_spec = {
      {"dtype", "int16"},
      {"driver", "n5"},
      {"path", "prefix"},
      {"kvstore", {{"driver", "memory"}}},
      {"transform",
       {{"input_exclusive_max", {{10}, {11}}},
        {"input_inclusive_min", {0, 0}}}},
  };
  options.to_json_options = tensorstore::IncludeDefaults{false};
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "n5";
  options.create_spec = {
      {"driver", "n5"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix"},
      {"metadata",
       {
           {"compression", {{"type", "raw"}}},
           {"dataType", "uint16"},
           {"dimensions", {10, 11}},
           {"blockSize", {4, 5}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape({10, 11})
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<std::uint16_t>(
      tensorstore::BoxView({10, 11}), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverResizeOptions options;
  options.test_name = "n5";
  options.get_create_spec = [](tensorstore::BoxView<> bounds) {
    return ::nlohmann::json{
        {"driver", "n5"},
        {"kvstore", {{"driver", "memory"}}},
        {"path", "prefix"},
        {"dtype", "uint16"},
        {"metadata",
         {
             {"dataType", "uint16"},
             {"dimensions", bounds.shape()},
             {"blockSize", {4, 5}},
             {"compression", {{"type", "raw"}}},
         }},
        {"transform",
         {
             {"input_inclusive_min", {0, 0}},
             {"input_exclusive_max",
              {{bounds.shape()[0]}, {bounds.shape()[1]}}},
         }},
    };
  };
  options.initial_bounds = tensorstore::Box<>({0, 0}, {10, 11});
  tensorstore::internal::RegisterTensorStoreDriverResizeTest(
      std::move(options));
}

}  // namespace
