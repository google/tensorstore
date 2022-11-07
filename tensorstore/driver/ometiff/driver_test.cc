

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/parse_json_matches.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/open.h"
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
using ::tensorstore::kImplicit;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Schema;
using ::tensorstore::span;
using ::tensorstore::StrCat;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal::ParseJsonMatches;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "ometiff"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
      {"metadata",
       {
           {"dataType", "int16"},
           {"dimensions", {10, 11}},
           {"blockSize", {3, 2}},
       }},
  };
}

TEST(OmeTiffDriverTest, OpenNonExisting) {
  EXPECT_THAT(
      tensorstore::Open(GetJsonSpec(), tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kNotFound,
                    "Error opening \"ometiff\" driver: "
                    "Metadata at \"prefix/__TAG__/IMAGE_DESCRIPTION\" does not exist"));
}

TEST(OmeTiffDriverTest, OpenOrCreate) {
  TENSORSTORE_EXPECT_OK(tensorstore::Open(
      GetJsonSpec(),
      tensorstore::OpenMode::open | tensorstore::OpenMode::create,
      tensorstore::ReadWriteMode::read_write));
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

TEST(OmeTiffDriverTest, Create) {
  ::nlohmann::json json_spec = GetJsonSpec();

  json_spec["metadata"].emplace("extra", "attribute");

  auto context = Context::Default();
  // Create the store.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result());
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(10, 11));
    EXPECT_THAT(store.domain().labels(), ::testing::ElementsAre("", ""));
    EXPECT_THAT(store.domain().implicit_lower_bounds(), DimensionSet({0, 0}));
    EXPECT_THAT(store.domain().implicit_upper_bounds(), DimensionSet({1, 1}));

    // Test ResolveBounds.
    // TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved,
    //                                  ResolveBounds(store).result());
    // EXPECT_EQ(store.domain(), resolved.domain());

    // Test ResolveBounds with a transform that swaps upper and lower bounds.
    // TENSORSTORE_ASSERT_OK_AND_ASSIGN(
    //     auto reversed_dim0,
    //     store | tensorstore::Dims(0).ClosedInterval(kImplicit, kImplicit, -1));
    // EXPECT_THAT(reversed_dim0.domain().origin(), ::testing::ElementsAre(-9, 0));
    // EXPECT_THAT(reversed_dim0.domain().shape(), ::testing::ElementsAre(10, 11));
    // EXPECT_THAT(reversed_dim0.domain().labels(),
    //             ::testing::ElementsAre("", ""));
    // EXPECT_THAT(reversed_dim0.domain().implicit_lower_bounds(),
    //             DimensionSet({1, 0}));
    // EXPECT_THAT(reversed_dim0.domain().implicit_upper_bounds(),
    //             DimensionSet({0, 1}));
    // TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved_reversed_dim0,
    //                                  ResolveBounds(reversed_dim0).result());
    // EXPECT_EQ(reversed_dim0.domain(), resolved_reversed_dim0.domain());

    // Issue a read to be filled with the fill value.
    // EXPECT_THAT(
    //     tensorstore::Read<tensorstore::zero_origin>(
    //         store |
    //         tensorstore::AllDims().TranslateSizedInterval({9, 7}, {1, 1}))
    //         .result(),
    //     ::testing::Optional(tensorstore::MakeArray<std::int16_t>({{0}})));

    // Issue an out-of-bounds read.
    // EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
    //                 store | tensorstore::AllDims().TranslateSizedInterval(
    //                             {10, 7}, {1, 1}))
    //                 .result(),
    //             MatchesStatus(absl::StatusCode::kOutOfRange));

    // Issue a valid write.
    // TENSORSTORE_EXPECT_OK(tensorstore::Write(
    //     tensorstore::MakeArray<std::int16_t>({{1, 2, 3}, {4, 5, 6}}),
    //     store | tensorstore::AllDims().TranslateSizedInterval({8, 8}, {2, 3})));

    // Issue an out-of-bounds write.
    // EXPECT_THAT(
    //     tensorstore::Write(
    //         tensorstore::MakeArray<std::int16_t>({{61, 62, 63}, {64, 65, 66}}),
    //         store |
    //             tensorstore::AllDims().TranslateSizedInterval({9, 8}, {2, 3}))
    //         .commit_future.result(),
    //     MatchesStatus(absl::StatusCode::kOutOfRange));

    // Re-read and validate result.  This verifies that the read/write
    // encoding/decoding paths round trip.
    // EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
    //                 store | tensorstore::AllDims().TranslateSizedInterval(
    //                             {7, 7}, {3, 4}))
    //                 .result(),
    //             ::testing::Optional(tensorstore::MakeArray<std::int16_t>({
    //                 {0, 0, 0, 0},
    //                 {0, 1, 2, 3},
    //                 {0, 4, 5, 6},
    //             })));
  }

  // Check that key value store has expected contents.  This verifies that the
  // encoding path works as expected.
//   EXPECT_THAT(
//       GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
//       UnorderedElementsAreArray({
//           Pair("prefix/attributes.json",  //
//                ::testing::MatcherCast<absl::Cord>(ParseJsonMatches({
//                    {"compression", {{"type", "raw"}}},
//                    {"dataType", "int16"},
//                    {"dimensions", {10, 11}},
//                    {"blockSize", {3, 2}},
//                    {"extra", "attribute"},
//                }))),
//           Pair("prefix/2/4",  // chunk starting at: 6, 8
//                MatchesRawChunk({3, 2},
//                                {
//                                    0, 0, 0, 0, 0, 1,  // int16be data
//                                    0, 0, 0, 0, 0, 2,  // int16be data
//                                })),
//           Pair("prefix/2/5",  // chunk starting at 6, 10
//                MatchesRawChunk({3, 2},
//                                {
//                                    0, 0, 0, 0, 0, 3,  // int16be data
//                                    0, 0, 0, 0, 0, 0,  // int16be data
//                                })),
//           Pair("prefix/3/4",  // chunk starting at 9, 8
//                MatchesRawChunk({3, 2},
//                                {
//                                    0, 4, 0, 0, 0, 0,  // int16be data
//                                    0, 5, 0, 0, 0, 0,  // int16be data
//                                })),
//           Pair("prefix/3/5",  // chunk starting at 9, 10
//                MatchesRawChunk({3, 2},
//                                {
//                                    0, 6, 0, 0, 0, 0,  // int16be data
//                                    0, 0, 0, 0, 0, 0,  // int16be data
//                                })),
//       }));

  // Check that attempting to create the store again fails.
  EXPECT_THAT(
      tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kAlreadyExists));

  // Check that create or open succeeds.
  TENSORSTORE_EXPECT_OK(tensorstore::Open(
      json_spec, context,
      tensorstore::OpenMode::create | tensorstore::OpenMode::open,
      tensorstore::ReadWriteMode::read_write));

}

TEST(OmeTiffDriverTest, CreateRank0) {
  ::nlohmann::json json_spec{
      {"driver", "ometiff"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
      {"metadata",
       {
           {"dataType", "int16"},
           {"dimensions", ::nlohmann::json::array_t()},
           {"blockSize", ::nlohmann::json::array_t()},
       }},
      {"create", true},
  };
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ts, tensorstore::Open(json_spec, context).result());
//   TENSORSTORE_ASSERT_OK(
//       tensorstore::Write(tensorstore::MakeScalarArray<int16_t>(42), ts));
  // Check that key value store has expected contents.
//   EXPECT_THAT(
//       GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
//       UnorderedElementsAreArray({
//           Pair("prefix/",  //
//                ::testing::MatcherCast<absl::Cord>(ParseJsonMatches({
//                    {"dataType", "int16"},
//                    {"dimensions", ::nlohmann::json::array_t()},
//                    {"blockSize", ::nlohmann::json::array_t()},
//                }))),
//           Pair("prefix/__TAG__/IMAGE_DESCRIPTION", MatchesRawChunk({}, {0, 42})),
//       }));
}

::nlohmann::json GetBasicResizeMetadata() {
  return {
      {"dataType", "int8"},
      {"dimensions", {100, 100}},
      {"blockSize", {3, 2}},
      {"dimOrder", 1},
  };
}


TEST(OmeTiffDriverTest, UnsupportedDataTypeInSpec) {
  EXPECT_THAT(
      tensorstore::Open(
          {
              {"dtype", "string"},
              {"driver", "ometiff"},
              {"kvstore", {{"driver", "memory"}}},
              {"path", "prefix/"},
              {"metadata",
               {
                   {"dimOrder", 1},
                   {"dimensions", {100, 100}},
                   {"blockSize", {3, 2}},
               }},
          },
          tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "string data type is not one of the supported data types: .*"));
}

TEST(OmeTiffDriverTest, DataTypeMismatch) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(
                      {
                          {"dtype", "int8"},
                          {"driver", "ometiff"},
                          {"kvstore", {{"driver", "memory"}}},
                          {"path", "prefix/"},
                          {"metadata",
                           {
                               {"dimOrder", 0},
                               {"dimensions", {100, 100}},
                               {"blockSize", {3, 2}},
                           }},
                      },
                      context, tensorstore::OpenMode::create,
                      tensorstore::ReadWriteMode::read_write)
                      .result());
  EXPECT_EQ(tensorstore::dtype_v<std::int8_t>, store.dtype());
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"dtype", "uint8"},
                      {"driver", "ometiff"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"path", "prefix/"},
                  },
                  context, tensorstore::OpenMode::open,
                  tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Error opening \"ometiff\" driver: "
                            ".*dtype.*"));
}

TEST(OmeTiffDriverTest, InvalidSpecExtraMember) {
  auto spec = GetJsonSpec();
  spec["extra_member"] = 5;
  EXPECT_THAT(tensorstore::Open(spec, tensorstore::OpenMode::create,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"extra_member\""));
}

TEST(OmeTiffDriverTest, InvalidSpecMissingKvstore) {
  auto spec = GetJsonSpec();
  spec.erase("kvstore");
  EXPECT_THAT(tensorstore::Open(spec, tensorstore::OpenMode::create,
                                tensorstore::ReadWriteMode::read_write)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error opening \"ometiff\" driver: "
                            "\"kvstore\" must be specified"));
}

TEST(OmeTiffDriverTest, InvalidSpecInvalidMemberType) {
  for (auto member_name : {"kvstore", "path", "metadata"}) {
    auto spec = GetJsonSpec();
    spec[member_name] = 5;
    EXPECT_THAT(
        tensorstore::Open(spec, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result(),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Error parsing object member \"", member_name,
                             "\": "
                             "Expected .*, but received: 5")));
  }
}

TEST(OmeTiffDriverTest, InvalidSpecMissingDomain) {
  auto spec = GetJsonSpec();
  spec["metadata"].erase("dimensions");
  EXPECT_THAT(
      tensorstore::Open(spec, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: "
                    "Cannot create using specified \"metadata\" and schema: "
                    "domain must be specified"));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverResizeOptions options;
  options.test_name = "ometiff";
  options.get_create_spec = [](tensorstore::BoxView<> bounds) {
    return ::nlohmann::json{
        {"driver", "ometiff"},
        {"kvstore",
         {
             {"driver", "memory"},
             {"path", "prefix/"},
         }},
        {"dtype", "uint16"},
        {"metadata",
         {
             {"dataType", "uint16"},
             {"dimensions", bounds.shape()},
             {"blockSize", {4, 5}},
             {"dimOrder", 0},
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

TEST(DriverTest, ChunkLayout) {
  ::nlohmann::json json_spec{
      {"driver", "ometiff"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"dataType", "int16"},
           {"dimensions", {10, 11}},
           {"blockSize", {3, 2}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout,
                                   ChunkLayout::FromJson({
                                       {"write_chunk", {{"shape", {3, 2}}}},
                                       {"read_chunk", {{"shape", {3, 2}}}},
                                       {"inner_order", {1, 0}},
                                       {"grid_origin", {0, 0}},
                                   }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(DriverTest, ChunkLayoutRank0) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "ometiff"}, {"kvstore", "memory://"}},
                        tensorstore::dtype_v<int32_t>,
                        tensorstore::RankConstraint{0},
                        tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout, ChunkLayout::FromJson({
                                                             {"rank", 0},
                                                         }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto chunk_layout, store.chunk_layout());
  EXPECT_EQ(expected_layout, chunk_layout);
  tensorstore::Box<> box(0);
  TENSORSTORE_EXPECT_OK(chunk_layout.GetReadChunkTemplate(box));
}

TEST(SpecTest, ChunkLayoutRank0) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, tensorstore::Spec::FromJson(
                     {{"driver", "ometiff"},
                      {"kvstore", "memory://"},
                      {"metadata",
                       {{"dimensions", ::nlohmann::json::array_t()},
                        {"blockSize", ::nlohmann::json::array_t()},
                        {"dataType", "uint16"}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout, ChunkLayout::FromJson({
                                                             {"rank", 0},
                                                         }));
  EXPECT_THAT(spec.chunk_layout(), ::testing::Optional(expected_layout));
}



template <typename... Option>
void TestCreateMetadata(::nlohmann::json base_spec,
                        ::nlohmann::json expected_metadata,
                        Option&&... option) {
  base_spec["driver"] = "ometiff";
  base_spec["kvstore"] = {{"driver", "memory"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(base_spec));
  TENSORSTORE_ASSERT_OK(
      spec.Set(std::forward<Option>(option)..., tensorstore::OpenMode::create));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_spec, store.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto json_spec, new_spec.ToJson());
  EXPECT_THAT(json_spec["metadata"],
              tensorstore::MatchesJson(expected_metadata));
}

TEST(DriverTest, SchemaUniformChunkAspectRatioWithTargetElements) {
  TestCreateMetadata({},
                     {
                         {"dimensions", {1000, 2000, 3000}},
                         {"dimOrder", 0},
                         {"blockSize", {64, 64, 64}},
                         {"dataType", "uint32"},
                     },
                     tensorstore::dtype_v<uint32_t>,
                     tensorstore::IndexDomainBuilder(3)
                         .shape({1000, 2000, 3000})
                         .labels({"x", "y", "z"})
                         .Finalize()
                         .value(),
                     tensorstore::ChunkLayout::ChunkElements{64 * 64 * 64});
}

TEST(DriverTest, SchemaObjectUniformChunkAspectRatioWithTargetElements) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::dtype_v<uint32_t>));
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::IndexDomainBuilder(3)
                                       .shape({1000, 2000, 3000})
                                       .labels({"x", "y", "z"})
                                       .Finalize()
                                       .value()));
  TENSORSTORE_ASSERT_OK(
      schema.Set(tensorstore::ChunkLayout::ChunkElements{64 * 64 * 64}));
  TestCreateMetadata({},
                     {
                         {"dimensions", {1000, 2000, 3000}},
                         {"blockSize", {64, 64, 64}},
                         {"dataType", "uint32"},
                         {"dimOrder", 0},
                     },
                     schema);
}

TEST(DriverTest, SchemaUniformChunkAspectRatio) {
  TestCreateMetadata({},
                     {
                         {"dimensions", {1000, 2000, 3000}},
                         {"blockSize", {102, 102, 102}},
                         {"dataType", "uint32"},
                         {"dimOrder", 0},
                     },
                     tensorstore::dtype_v<uint32_t>,
                     tensorstore::IndexDomainBuilder(3)
                         .shape({1000, 2000, 3000})
                         .labels({"x", "y", "z"})
                         .Finalize()
                         .value());
}

TEST(DriverTest, SchemaNonUniformChunkAspectRatio) {
  TestCreateMetadata({},
                     {
                         {"dimensions", {1000, 2000, 3000}},
                         {"blockSize", {64, 128, 128}},
                         {"dataType", "uint32"},
                         {"dimOrder", 0},
                     },
                     tensorstore::dtype_v<uint32_t>,
                     tensorstore::IndexDomainBuilder(3)
                         .shape({1000, 2000, 3000})
                         .labels({"x", "y", "z"})
                         .Finalize()
                         .value(),
                     tensorstore::ChunkLayout::ChunkAspectRatio{{1, 2, 2}},
                     tensorstore::ChunkLayout::ChunkElements{64 * 128 * 128});
}


TEST(DriverTest, MissingDtype) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "ometiff"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"schema",
                       {
                           {"domain", {{"shape", {100}}}},
                       }},
                  },
                  tensorstore::OpenMode::create)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: dtype must be specified"));
}


TEST(DriverTest, MetadataMismatch) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "ometiff"},
                                     {"kvstore", {{"driver", "memory"}}},
                                     {"metadata",
                                      {
                                          {"dimensions", {100}},
                                          {"dataType", "uint32"},
                                          {"blockSize", {10}},
                                      }}},
                                    context, tensorstore::OpenMode::create)
                      .result());
  // Opening without metadata succeeds
  TENSORSTORE_EXPECT_OK(
      tensorstore::Open({{"driver", "ometiff"}, {"kvstore", {{"driver", "memory"}}}},
                        context, tensorstore::OpenMode::open)
          .result());

  // Mismatched "dimensions"
  EXPECT_THAT(tensorstore::Open({{"driver", "ometiff"},
                                 {"kvstore", {{"driver", "memory"}}},
                                 {"metadata",
                                  {
                                      {"dimensions", {200}},
                                  }}},
                                context, tensorstore::OpenMode::open)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*\"dimensions\".*"));



  // Mismatched "dataType"
  EXPECT_THAT(
      tensorstore::Open({{"driver", "ometiff"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"metadata",
                          {
                              {"dataType", "int32"},
                          }}},
                        context, tensorstore::OpenMode::open)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"dataType\".*"));

  // Mismatched "blockSize"
  EXPECT_THAT(tensorstore::Open({{"driver", "ometiff"},
                                 {"kvstore", {{"driver", "memory"}}},
                                 {"metadata",
                                  {
                                      {"blockSize", {20}},
                                  }}},
                                context, tensorstore::OpenMode::open)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*\"blockSize\".*"));

  // Mismatched "rank"
  EXPECT_THAT(
      tensorstore::Open({{"driver", "ometiff"},
                         {"kvstore", {{"driver", "memory"}}},
                         {"schema", {{"rank", 2}}}},
                        context, tensorstore::OpenMode::open)
          .result(),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*Rank specified by schema \\(2\\) "
                    "does not match rank specified by metadata \\(1\\)"));
}

TEST(DriverTest, SchemaMismatch) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "ometiff"},
                                     {"kvstore", {{"driver", "memory"}}},
                                     {"metadata",
                                      {
                                          {"dimensions", {100}},
                                          {"dataType", "uint32"},
                                          {"blockSize", {10}},
                                      }}},
                                    context, tensorstore::OpenMode::create)
                      .result());

  // Mismatched rank
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "ometiff"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"schema", {{"rank", 2}}},
                  },
                  context, tensorstore::OpenMode::open)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Rank specified by schema \\(2\\) does not "
                            "match rank specified by metadata \\(1\\)"));

  // Mismatched dtype
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "ometiff"},
                      {"kvstore", {{"driver", "memory"}}},
                      {"schema", {{"dtype", "int32"}}},
                  },
                  context, tensorstore::OpenMode::open)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*dtype from metadata \\(uint32\\) does not "
                            "match dtype in schema \\(int32\\)"));

  
}

TEST(SpecSchemaTest, Domain) {
  TestSpecSchema({{"driver", "ometiff"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"dimensions", {3, 4, 5}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0}},
                       {"inner_order", {2, 1, 0}},
                   }},
                  {"domain", {{"shape", {{3}, {4}, {5}}}}},
                  });
}

TEST(SpecSchemaTest, SchemaDomain) {
  TestSpecSchema({{"driver", "ometiff"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"schema", {{"domain", {{"shape", {3, 4, 5}}}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0}},
                       {"inner_order", {2, 1, 0}},
                   }},
                  {"domain", {{"shape", {{3}, {4}, {5}}}}},
                  });
}

TEST(SpecSchemaTest, ChunkLayout) {
  TestSpecSchema({{"driver", "ometiff"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"blockSize", {3, 4, 5}}}}},
                 {{"chunk_layout",
                   {
                       {"grid_origin", {0, 0, 0}},
                       {"chunk", {{"shape", {3, 4, 5}}}},
                       {"inner_order", {2, 1, 0}},
                   }}});
}

TEST(SpecSchemaTest, Dtype) {
  TestSpecSchema({{"driver", "ometiff"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"dataType", "uint16"}}}},
                 {
                     {"dtype", "uint16"},
                 });
}


TEST(DriverCreateCheckSchemaTest, Simple) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "ometiff"},
          {"kvstore", {{"driver", "memory"}}},
          {"metadata", {{"dimOrder", 1}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain",
                {
                    {"shape", {1000, 2000, 3000}},
                }},
               {"chunk_layout",
                {
                    {"chunk", {{"shape", {30, 40, 50}}}},
                }},
           }},
      },
      {
          {"dtype", "uint32"},
          {"domain",
           {
               {"shape", {{1000}, {2000}, {3000}}},
           }},
          {"chunk_layout",
           {
               {"grid_origin", {0, 0, 0}},
               {"inner_order", {2, 1, 0}},
               {"chunk", {{"shape", {30, 40, 50}}}},
           }},
      });
}



}  // namespace
