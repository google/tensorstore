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

/// End-to-end tests of the zarr3 driver.

#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/batch.h"
#include "tensorstore/box.h"
#include "tensorstore/cast.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::JsonSubValuesMatch;
using ::tensorstore::MatchesJson;
using ::tensorstore::Result;
using ::tensorstore::Schema;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::internal::GetMap;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;
using ::tensorstore::internal::TestTensorStoreCreateWithSchema;
using ::tensorstore::internal::TestTensorStoreSpecRoundtripNormalize;
using ::tensorstore::internal::TestTensorStoreUrlRoundtrip;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", "prefix/"},
       }},
      {"metadata",
       {
           {"data_type", "int16"},
           {"shape", {10, 11}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {3, 2}}}}}},
           {"chunk_key_encoding", {{"name", "default"}}},
       }},
  };
}

TEST(ZarrDriverTest, OpenNonExisting) {
  EXPECT_THAT(
      tensorstore::Open(GetJsonSpec(), tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("Error opening \"zarr3\" driver: "
                         "Metadata at \"prefix/zarr.json\" does not exist")));
}

TEST(ZarrDriverTest, OpenTransactional) {
  auto context = Context::Default();
  auto transaction = tensorstore::Transaction(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open({{"driver", "zarr3"}, {"kvstore", "memory://"}},
                          dtype_v<uint8_t>, tensorstore::Schema::Shape({1}),
                          transaction, context, tensorstore::OpenMode::create)
            .result());
    // Reading `zarr.json` directly revokes the metadata cache entry.
    EXPECT_THAT(
        tensorstore::kvstore::Read(store.kvstore(), "zarr.json").result(),
        MatchesKvsReadResult(::testing::Matcher<absl::Cord>(::testing::_)));
  }
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open({{"driver", "zarr3"}, {"kvstore", "memory://"}},
                          transaction, context)
            .result());
  }
}

TEST(ZarrDriverTest, OpenWithOpenKvStore) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, tensorstore::kvstore::Open("memory://").result());
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto txn_kvs, kvs | tensorstore::Transaction(tensorstore::isolated));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                                      tensorstore::Schema::Shape({1}), txn_kvs,
                                      tensorstore::OpenMode::create)
                        .result());
    EXPECT_THAT(
        tensorstore::kvstore::Read(txn_kvs, "zarr.json").result(),
        MatchesKvsReadResult(::testing::Matcher<absl::Cord>(::testing::_)));
    EXPECT_THAT(tensorstore::kvstore::Read(kvs, "zarr.json").result(),
                MatchesKvsReadResultNotFound());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto txn_kvs, kvs | tensorstore::Transaction(tensorstore::isolated));
    // Can specify both transactional KvStore and transaction as long
    // as they are consistent.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                          tensorstore::Schema::Shape({1}), txn_kvs,
                          txn_kvs.transaction, tensorstore::OpenMode::create)
            .result());
    EXPECT_THAT(
        tensorstore::kvstore::Read(txn_kvs, "zarr.json").result(),
        MatchesKvsReadResult(::testing::Matcher<absl::Cord>(::testing::_)));
    EXPECT_THAT(tensorstore::kvstore::Read(kvs, "zarr.json").result(),
                MatchesKvsReadResultNotFound());
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                                    tensorstore::Schema::Shape({1}), kvs,
                                    tensorstore::OpenMode::create)
                      .result());

  EXPECT_THAT(tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                                tensorstore::Schema::Shape({1}), kvs, kvs,
                                tensorstore::OpenMode::create)
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("KvStore already specified")));
  EXPECT_THAT(tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                                tensorstore::Schema::Shape({1}), kvs,
                                tensorstore::Transaction(tensorstore::isolated),
                                tensorstore::Transaction(tensorstore::isolated),
                                tensorstore::OpenMode::create)
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Inconsistent transactions specified")));
  EXPECT_THAT(tensorstore::Open({{"driver", "zarr3"}}, dtype_v<uint8_t>,
                                tensorstore::Schema::Shape({1}), kvs,
                                tensorstore::Transaction(tensorstore::isolated),
                                tensorstore::Transaction(tensorstore::isolated),
                                tensorstore::OpenMode::create)
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Inconsistent transactions specified")));
  EXPECT_THAT(
      tensorstore::kvstore::Read(kvs, "zarr.json").result(),
      MatchesKvsReadResult(::testing::Matcher<absl::Cord>(::testing::_)));
}

TEST(ZarrDriverTest, ShardedTranspose) {
  std::vector<Index> shape{10, 11};
  auto array = tensorstore::AllocateArray<uint16_t>(shape);
  for (Index i = 0; i < shape[0]; ++i) {
    for (Index j = 0; j < shape[1]; ++j) {
      array(i, j) = static_cast<uint16_t>(i * 100 + j);
    }
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {
              {"driver", "zarr3"},
              {"kvstore", "memory://"},
              {"metadata",
               {
                   {"data_type", "uint16"},
                   {"shape", shape},
                   {"chunk_grid",
                    {{"name", "regular"},
                     {"configuration", {{"chunk_shape", {8, 9}}}}}},
                   {"codecs",
                    {
                        {{"name", "transpose"},
                         {"configuration", {{"order", {1, 0}}}}},
                        {{"name", "sharding_indexed"},
                         {"configuration", {{"chunk_shape", {3, 2}}}}},
                    }},
               }},
          },
          tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK(tensorstore::Write(array, store));
  EXPECT_THAT(tensorstore::Read(store).result(), ::testing::Optional(array));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3";
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 5}}}}}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape({10, 11})
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView({10, 11}), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3/sharding";
  std::vector<Index> shape{10, 11};
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", shape},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 9}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration", {{"chunk_shape", {2, 3}}}}}}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape(shape)
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView<>(shape), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3/sharding_index_at_start";
  std::vector<Index> shape{10, 11};
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", shape},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 9}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"index_location", "start"}, {"chunk_shape", {2, 3}}}}}}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape(shape)
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView<>(shape), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3/sharding_nested";
  std::vector<Index> shape{10, 11};
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", shape},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 9}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {
                   {"chunk_shape", {2, 3}},
                   {"codecs",
                    {{{"name", "sharding_indexed"},
                      {"configuration", {{"chunk_shape", {1, 3}}}}}}},
               }}}}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape(shape)
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView<>(shape), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3/transpose_sharding";
  std::vector<Index> shape{10, 11};
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", shape},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 9}}}}}},
           {"codecs",
            {
                {{"name", "transpose"}, {"configuration", {{"order", {1, 0}}}}},
                {{"name", "sharding_indexed"},
                 {"configuration", {{"chunk_shape", {3, 2}}}}},
            }},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape(shape)
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView<>(shape), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TensorStoreDriverBasicFunctionalityTestOptions options;
  options.test_name = "zarr3/with_dimension_names";
  options.create_spec = {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "prefix/"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11}},
           {"dimension_names", {"x", "y"}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 5}}}}}},
       }},
  };
  options.expected_domain = tensorstore::IndexDomainBuilder(2)
                                .shape({10, 11})
                                .labels({"x", "y"})
                                .implicit_upper_bounds({1, 1})
                                .Finalize()
                                .value();
  options.initial_value = tensorstore::AllocateArray<uint16_t>(
      tensorstore::BoxView({10, 11}), tensorstore::c_order,
      tensorstore::value_init);
  tensorstore::internal::RegisterTensorStoreDriverBasicFunctionalityTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverResizeOptions options;
  options.test_name = "zarr3/metadata";
  options.get_create_spec = [](tensorstore::BoxView<> bounds) {
    return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "memory"},
             {"path", "prefix/"},
         }},
        {"dtype", "uint16"},
        {"metadata",
         {
             {"zarr_format", 3},
             {"node_type", "array"},
             {"data_type", "uint16"},
             {"fill_value", 0},
             {"shape", bounds.shape()},
             {"dimension_names", {"", ""}},
             {"chunk_grid",
              {{"name", "regular"},
               {"configuration", {{"chunk_shape", {4, 5}}}}}},
             {"chunk_key_encoding", {{"name", "default"}}},
             {"codecs", {GetDefaultBytesCodecJson()}},
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

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverResizeOptions options;
  options.test_name = "zarr3/data";
  options.get_create_spec = [](tensorstore::BoxView<> bounds) {
    return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "memory"},
             {"path", "prefix/"},
         }},
        {"dtype", "uint16"},
        {"metadata",
         {
             {"zarr_format", 3},
             {"node_type", "array"},
             {"data_type", "uint16"},
             {"fill_value", 0},
             {"shape", bounds.shape()},
             {"dimension_names", {"", ""}},
             {"chunk_grid",
              {{"name", "regular"},
               {"configuration", {{"chunk_shape", {4, 5}}}}}},
             {"chunk_key_encoding", {{"name", "default"}}},
             {"codecs", {GetDefaultBytesCodecJson()}},
         }},
        {"transform",
         {
             {"input_inclusive_min", {0, 0}},
             {"input_exclusive_max",
              {{bounds.shape()[0]}, {bounds.shape()[1]}}},
         }},
    };
  };
  options.initial_bounds = tensorstore::Box<>({0, 0}, {20, 30});
  options.test_data = true;
  tensorstore::internal::RegisterTensorStoreDriverResizeTest(
      std::move(options));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverResizeOptions options;
  options.test_name = "zarr3/data/sharded";
  options.get_create_spec = [](tensorstore::BoxView<> bounds) {
    return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "memory"},
             {"path", "prefix/"},
         }},
        {"dtype", "uint16"},
        {"metadata",
         {
             {"zarr_format", 3},
             {"node_type", "array"},
             {"data_type", "uint16"},
             {"fill_value", 0},
             {"shape", bounds.shape()},
             {"dimension_names", {"", ""}},
             {"chunk_grid",
              {{"name", "regular"},
               {"configuration", {{"chunk_shape", {4, 6}}}}}},
             {"chunk_key_encoding", {{"name", "default"}}},
             {"codecs",
              {
                  {{"name", "transpose"},
                   {"configuration", {{"order", {1, 0}}}}},
                  {{"name", "sharding_indexed"},
                   {"configuration",
                    {
                        {"chunk_shape", {3, 2}},
                        {"codecs", {GetDefaultBytesCodecJson()}},
                        {"index_location", "end"},
                        {"index_codecs",
                         {
                             GetDefaultBytesCodecJson(),
                             {{"name", "crc32c"}},
                         }},
                    }}},
              }},
         }},
        {"transform",
         {
             {"input_inclusive_min", {0, 0}},
             {"input_exclusive_max",
              {{bounds.shape()[0]}, {bounds.shape()[1]}}},
         }},
    };
  };
  options.initial_bounds = tensorstore::Box<>({0, 0}, {20, 30});
  options.test_data = true;
  tensorstore::internal::RegisterTensorStoreDriverResizeTest(
      std::move(options));
}

TEST(ZarrDriverTest, Codec) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 5}}}}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_codec,
                                   tensorstore::CodecSpec::FromJson({
                                       {"driver", "zarr3"},
                                       {"codecs", {GetDefaultBytesCodecJson()}},
                                   }));
  EXPECT_THAT(store.codec(), ::testing::Optional(expected_codec));
}

TEST(ZarrDriverTest, ChunkLayoutCOrder) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11, 12}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 5, 6}}}}}},
       }},
  };
  // Open with default inner order (C order)
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto expected_layout, ChunkLayout::FromJson({
                                  {"grid_origin", {0, 0, 0}},
                                  {"write_chunk", {{"shape", {4, 5, 6}}}},
                                  {"read_chunk", {{"shape", {4, 5, 6}}}},
                                  {"inner_order", {0, 1, 2}},
                              }));
    EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
  }
}

TEST(ZarrDriverTest, ChunkLayoutFOrder) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11, 12}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 5, 6}}}}}},
       }},
  };
  // Open with transposed order
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, ChunkLayout::InnerOrder({2, 1, 0}),
                          tensorstore::OpenMode::create)
            .result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto expected_layout, ChunkLayout::FromJson({
                                  {"grid_origin", {0, 0, 0}},
                                  {"write_chunk", {{"shape", {4, 5, 6}}}},
                                  {"read_chunk", {{"shape", {4, 5, 6}}}},
                                  {"inner_order", {2, 1, 0}},
                              }));
    EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
  }
}

TEST(ZarrDriverTest, ChunkLayoutSharding) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11, 12}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 6, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration", {{"chunk_shape", {2, 3, 4}}}}}}},
       }},
  };
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto expected_layout, ChunkLayout::FromJson({
                                  {"grid_origin", {0, 0, 0}},
                                  {"write_chunk", {{"shape", {4, 6, 8}}}},
                                  {"read_chunk", {{"shape", {2, 3, 4}}}},
                                  {"inner_order", {0, 1, 2}},
                              }));
    EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
  }
}

TEST(ZarrDriverTest, ChunkLayoutShardingTranspose) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11, 12}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 6, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {
                   {"chunk_shape", {2, 3, 4}},
                   {"codecs",
                    {
                        {{"name", "transpose"},
                         {"configuration", {{"order", {1, 2, 0}}}}},
                    }},
               }}}}},
       }},
  };
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto expected_layout, ChunkLayout::FromJson({
                                  {"grid_origin", {0, 0, 0}},
                                  {"write_chunk", {{"shape", {4, 6, 8}}}},
                                  {"read_chunk", {{"shape", {2, 3, 4}}}},
                                  {"inner_order", {1, 2, 0}},
                              }));
    EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
  }
}

TEST(ZarrDriverTest, ChunkLayoutTransposeSharding) {
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata",
       {
           {"data_type", "uint16"},
           {"shape", {10, 11, 12}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {4, 6, 8}}}}}},
           {"codecs",
            {{{"name", "transpose"}, {"configuration", {{"order", {1, 2, 0}}}}},
             {{"name", "sharding_indexed"},
              {"configuration",
               {
                   {"chunk_shape", {3, 4, 2}},
               }}}}},
       }},
  };
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, tensorstore::OpenMode::create).result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto expected_layout, ChunkLayout::FromJson({
                                  {"grid_origin", {0, 0, 0}},
                                  {"write_chunk", {{"shape", {4, 6, 8}}}},
                                  {"read_chunk", {{"shape", {2, 3, 4}}}},
                                  {"inner_order", {1, 2, 0}},
                              }));
    EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
  }
}

TEST(SpecSchemaTest, ChunkLayout) {
  TestSpecSchema(
      {
          {"driver", "zarr3"},
          {"kvstore", {{"driver", "memory"}}},
          {"metadata",
           {
               {"data_type", "uint16"},
               {"chunk_grid",
                {{"name", "regular"},
                 {"configuration", {{"chunk_shape", {3, 4, 5}}}}}},
           }},
      },
      {{"dtype", "uint16"},
       {"chunk_layout",
        {
            {"grid_origin", {0, 0, 0}},
            {"write_chunk", {{"shape", {3, 4, 5}}}},
        }},
       {"codec", {{"driver", "zarr3"}}}});
}

TEST(SpecSchemaTest, Codec) {
  TestSpecSchema(
      {{"driver", "zarr3"},
       {"kvstore", {{"driver", "memory"}}},
       {"metadata",
        {{"data_type", "uint16"}, {"codecs", ::nlohmann::json::array_t{}}}}},
      {{"dtype", "uint16"},
       {"codec",
        {{"driver", "zarr3"}, {"codecs", ::nlohmann::json::array_t{}}}}});
}

TEST(SpecSchemaTest, FillValue) {
  TestSpecSchema({{"driver", "zarr3"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"data_type", "uint16"}, {"fill_value", 42}}}},
                 {{"dtype", "uint16"},
                  {"fill_value", 42},
                  {"codec", {{"driver", "zarr3"}}}});
}

TEST(SpecSchemaTest, FillValueWithTransform) {
  TestSpecSchema({{"driver", "zarr3"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"data_type", "uint16"}, {"fill_value", 42}}},
                  {"transform", {{"input_rank", 3}}}},
                 {{"dtype", "uint16"},
                  {"fill_value", 42},
                  {"chunk_layout", {{"grid_origin", {0, 0, 0}}}},
                  {"rank", 3},
                  {"domain", {{"rank", 3}}},
                  {"codec", {{"driver", "zarr3"}}}});
}

TEST(SpecSchemaTest, DimensionUnits) {
  TestSpecSchema({{"driver", "zarr3"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata",
                   {
                       {"data_type", "uint16"},
                       {"fill_value", 42},
                       {"attributes",
                        {
                            {"dimension_units", {"m", "m", "s"}},
                        }},
                   }}},
                 {{"dtype", "uint16"},
                  {"fill_value", 42},
                  {"dimension_units", {"m", "m", "s"}},
                  {"chunk_layout", {{"grid_origin", {0, 0, 0}}}},
                  {"codec", {{"driver", "zarr3"}}}});
}

TEST(SpecSchemaTest, NoRank) {
  TestSpecSchema({{"driver", "zarr3"},
                  {"kvstore", {{"driver", "memory"}}},
                  {"metadata", {{"data_type", "uint16"}}}},
                 {
                     {"dtype", "uint16"},
                     {"codec", {{"driver", "zarr3"}}},
                 });
}

TEST(DriverCreateWithSchemaTest, Dtypes) {
  constexpr tensorstore::DataType kSupportedDataTypes[] = {
      dtype_v<tensorstore::dtypes::bool_t>,
      dtype_v<tensorstore::dtypes::uint8_t>,
      dtype_v<tensorstore::dtypes::uint16_t>,
      dtype_v<tensorstore::dtypes::uint32_t>,
      dtype_v<tensorstore::dtypes::int8_t>,
      dtype_v<tensorstore::dtypes::int16_t>,
      dtype_v<tensorstore::dtypes::int32_t>,
      dtype_v<tensorstore::dtypes::uint64_t>,
      dtype_v<tensorstore::dtypes::float32_t>,
      dtype_v<tensorstore::dtypes::float64_t>,
      dtype_v<tensorstore::dtypes::complex64_t>,
      dtype_v<tensorstore::dtypes::complex128_t>,
  };
  for (auto dtype : kSupportedDataTypes) {
    TestTensorStoreCreateWithSchema(
        {{"driver", "zarr3"}, {"kvstore", {{"driver", "memory"}}}}, dtype,
        Schema::Shape({5, 6, 7}));
  }
}

TEST(DriverCreateWithSchemaTest, ChunkShapeUnsharded) {
  TestTensorStoreCreateWithSchema(
      {{"driver", "zarr3"}, {"kvstore", {{"driver", "memory"}}}},
      dtype_v<uint32_t>, Schema::Shape({5, 6, 7, 2}),
      ChunkLayout::ChunkShape({2, 3, 4, 0}));
}

TEST(DriverCreateWithSchemaTest, ChunkShapeSharded) {
  TestTensorStoreCreateWithSchema(
      {{"driver", "zarr3"}, {"kvstore", {{"driver", "memory"}}}},
      dtype_v<uint32_t>, Schema::Shape({1000, 1000, 1000, 2}),
      ChunkLayout::ReadChunkShape({30, 40, 50, 0}),
      ChunkLayout::WriteChunkShape({30 * 4, 40 * 4, 50 * 2, 0}));
}

TEST(DriverCreateWithSchemaTest, ChunkShapeShardedTargetElements) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "zarr3"},
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
           }},
      },
      {
          {"dtype", "uint32"},
          {"fill_value", 0},
          {"domain", {{"shape", {{1000}, {1000}, {1000}, {1}}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {0, 1, 2, 3}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {30 * 2, 40 * 2, 50, 1}}}}}},
          {"codec",
           {
               {"driver", "zarr3"},
               {"codecs",
                {{
                    {"name", "sharding_indexed"},
                    {"configuration",
                     {
                         {"chunk_shape", {30, 40, 50, 1}},
                         {"index_location", "end"},
                         {"index_codecs",
                          {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
                         {"codecs", {GetDefaultBytesCodecJson()}},
                     }},
                }}},
           }},
      });
}

TEST(DriverCreateWithSchemaTest, ShardingAndInnerOrder) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "zarr3"},
          {"kvstore", {{"driver", "memory"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 1000, 1000, 1}}}},
               {"chunk_layout",
                {
                    {"inner_order", {3, 1, 0, 2}},
                    {"read_chunk", {{"shape", {30, 40, 50, 0}}}},
                    {"write_chunk", {{"elements", 30 * 40 * 50 * 8}}},
                }},
           }},
      },
      {
          {"dtype", "uint32"},
          {"fill_value", 0},
          {"domain", {{"shape", {{1000}, {1000}, {1000}, {1}}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {3, 1, 0, 2}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {30 * 2, 40 * 2, 50, 1}}}}}},
          {"codec",
           {
               {"driver", "zarr3"},
               {"codecs",
                {{
                    {"name", "sharding_indexed"},
                    {"configuration",
                     {
                         {"chunk_shape", {30, 40, 50, 1}},
                         {"index_location", "end"},
                         {"index_codecs",
                          {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
                         {"codecs",
                          {{{"name", "transpose"},
                            {"configuration", {{"order", {3, 1, 0, 2}}}}},
                           GetDefaultBytesCodecJson()}},
                     }},
                }}},
           }},
      });
}

TEST(DriverCreateWithSchemaTest, ChunkShapeShardedWriteChunkSizeNegative1) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "zarr3"},
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
          {"fill_value", 0},
          {"domain", {{"shape", {{1000}, {1000}, {1000}, {1}}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0, 0}},
            {"inner_order", {0, 1, 2, 3}},
            {"read_chunk", {{"shape", {30, 40, 50, 1}}}},
            {"write_chunk", {{"shape", {30, 40, 1000, 1}}}}}},
          {"codec",
           {
               {"driver", "zarr3"},
               {"codecs",
                {{
                    {"name", "sharding_indexed"},
                    {"configuration",
                     {
                         {"chunk_shape", {30, 40, 50, 1}},
                         {"index_location", "end"},
                         {"index_codecs",
                          {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
                         {"codecs", {GetDefaultBytesCodecJson()}},
                     }},
                }}},
           }},
      });
}

TEST(DriverCreateWithSchemaTest, TransposeGzip) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "zarr3"},
          {"kvstore", {{"driver", "memory"}}},
          {"schema",
           {
               {"dtype", "uint32"},
               {"domain", {{"shape", {1000, 2000, 3000}}}},
               {"chunk_layout",
                {
                    {"inner_order", {2, 0, 1}},
                    {"read_chunk", {{"shape", {30, 40, 50}}}},
                    {"write_chunk",
                     {{"shape_soft_constraint", {200, 300, 400}}}},
                }},
               {"fill_value", 42},
               {"codec", {{"driver", "zarr3"}, {"codecs", {"gzip"}}}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"fill_value", 42},
          {"domain", {{"shape", {{1000}, {2000}, {3000}}}}},
          {"chunk_layout",
           {{"grid_origin", {0, 0, 0}},
            {"inner_order", {2, 0, 1}},
            {"read_chunk", {{"shape", {30, 40, 50}}}},
            {"write_chunk", {{"shape", {210, 280, 400}}}}}},
          {"codec",
           {
               {"driver", "zarr3"},
               {"codecs",
                {{
                    {"name", "sharding_indexed"},
                    {"configuration",
                     {
                         {"chunk_shape", {30, 40, 50}},
                         {"index_location", "end"},
                         {"index_codecs",
                          {GetDefaultBytesCodecJson(), {{"name", "crc32c"}}}},
                         {"codecs",
                          {
                              {{"name", "transpose"},
                               {"configuration", {{"order", {2, 0, 1}}}}},
                              GetDefaultBytesCodecJson(),
                              {{"name", "gzip"},
                               {"configuration", {{"level", 6}}}},
                          }},
                     }},
                }}},
           }},
      });
}

TEST(DriverCreateWithSchemaTest, DimensionUnits) {
  TestTensorStoreCreateWithSchema(
      {{"driver", "zarr3"}, {"kvstore", {{"driver", "memory"}}}},
      dtype_v<uint32_t>, Schema::Shape({5, 6, 7, 2}),
      Schema::DimensionUnits({"m", "m", "s", "s"}));
}

template <typename... Option>
Result<std::vector<std::string>> GetKeysForChunkKeyEncoding(
    ::nlohmann::json encoding, Option&&... option) {
  ::nlohmann::json spec{
      {"driver", "zarr3"},
      {"kvstore", "memory://"},
      {"metadata", {{"chunk_key_encoding", encoding}}},
  };
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto store,
      tensorstore::Open(spec, dtype_v<uint8_t>, tensorstore::OpenMode::create,
                        std::forward<Option>(option)...)
          .result());
  TENSORSTORE_RETURN_IF_ERROR(
      tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(42), store));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto entries, tensorstore::kvstore::ListFuture(store.kvstore()).result());

  std::vector<std::string> keys_without_zarr_json;
  keys_without_zarr_json.reserve(entries.size());
  for (const auto& e : entries) {
    if (e.key != "zarr.json") {
      keys_without_zarr_json.push_back(std::move(e.key));
    }
  }
  return keys_without_zarr_json;
}

TEST(ChunkKeyEncodingTest, DefaultRank0) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding({{"name", "default"}}, Schema::Shape{{}}),
      ::testing::Optional(::testing::ElementsAre("c")));
}

TEST(ChunkKeyEncodingTest, DefaultRank1) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding({{"name", "default"}}, Schema::Shape{{5}},
                                 ChunkLayout::ChunkShape({3})),
      ::testing::Optional(::testing::ElementsAre("c/0", "c/1")));
}

TEST(ChunkKeyEncodingTest, DefaultRank1Dot) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding(
          {{"name", "default"}, {"configuration", {{"separator", "."}}}},
          Schema::Shape{{5}}, ChunkLayout::ChunkShape({3})),
      ::testing::Optional(::testing::ElementsAre("c.0", "c.1")));
}

TEST(ChunkKeyEncodingTest, DefaultRank2) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding({{"name", "default"}}, Schema::Shape{{5, 6}},
                                 ChunkLayout::ChunkShape({3, 3})),
      ::testing::Optional(
          ::testing::ElementsAre("c/0/0", "c/0/1", "c/1/0", "c/1/1")));
}

TEST(ChunkKeyEncodingTest, V2Rank0) {
  EXPECT_THAT(GetKeysForChunkKeyEncoding({{"name", "v2"}}, Schema::Shape{{}}),
              ::testing::Optional(::testing::ElementsAre("0")));
}

TEST(ChunkKeyEncodingTest, V2Rank1) {
  EXPECT_THAT(GetKeysForChunkKeyEncoding({{"name", "v2"}}, Schema::Shape{{5}},
                                         ChunkLayout::ChunkShape({3})),
              ::testing::Optional(::testing::ElementsAre("0", "1")));
}

TEST(ChunkKeyEncodingTest, V2Rank2) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding({{"name", "v2"}}, Schema::Shape{{5, 6}},
                                 ChunkLayout::ChunkShape({3, 3})),
      ::testing::Optional(::testing::ElementsAre("0.0", "0.1", "1.0", "1.1")));
}

TEST(ChunkKeyEncodingTest, V2Rank2Slash) {
  EXPECT_THAT(
      GetKeysForChunkKeyEncoding(
          {{"name", "v2"}, {"configuration", {{"separator", "/"}}}},
          Schema::Shape{{5, 6}}, ChunkLayout::ChunkShape({3, 3})),
      ::testing::Optional(::testing::ElementsAre("0/0", "0/1", "1/0", "1/1")));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "zarr3";
  options.create_spec = {
      {"driver", "zarr3"},
      {"metadata",
       {
           {"zarr_format", 3},
           {"node_type", "array"},
           {"shape", {10, 11, 12}},
           {"data_type", "int16"},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
           {"codecs",
            {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
           {"attributes", {{"a", "b"}, {"c", "d"}}},
       }},
      {"kvstore", "file://${TEMPDIR}/prefix/"},
  };
  options.full_spec = {
      {"dtype", "int16"},
      {"driver", "zarr3"},
      {"metadata",
       {
           {"zarr_format", 3},
           {"node_type", "array"},
           {"shape", {10, 11, 12}},
           {"data_type", "int16"},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {1, 2, 3}}}}}},
           {"chunk_key_encoding", {{"name", "default"}}},
           {"fill_value", 0},
           {"codecs",
            {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}},
           {"attributes", {{"a", "b"}, {"c", "d"}}},
           {"dimension_names", {"", "", ""}},
       }},
      {"kvstore",
       {
           {"driver", "file"},
           {"path", "${TEMPDIR}/prefix/"},
       }},
      {"transform",
       {{"input_exclusive_max", {{10}, {11}, {12}}},
        {"input_inclusive_min", {0, 0, 0}}}},
  };
  options.minimal_spec = {
      {"dtype", "int16"},
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "file"},
           {"path", "${TEMPDIR}/prefix/"},
       }},
      {"transform",
       {{"input_exclusive_max", {{10}, {11}, {12}}},
        {"input_inclusive_min", {0, 0, 0}}}},
  };
  options.check_serialization = true;
  options.url = "file://${TEMPDIR}/prefix/|zarr3:";
  options.check_auto_detect = true;
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

TEST(ZarrDriverTest, OpenCorruptMetadata) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, tensorstore::kvstore::Open("memory://prefix/zarr.json", context)
                    .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::kvstore::Write(kvs, "", absl::Cord("")).result());
  EXPECT_THAT(
      tensorstore::Open(GetJsonSpec(), tensorstore::OpenMode::open, context)
          .result(),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("Invalid JSON")));
}

TEST(ZarrDriverTest, IncompatibleMetadata) {
  auto context = Context::Default();
  ::nlohmann::json json_spec{{"driver", "zarr3"}, {"kvstore", "memory://"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1,
      tensorstore::Open(json_spec, dtype_v<uint8_t>, Schema::Shape({100}),
                        tensorstore::OpenMode::create,
                        tensorstore::RecheckCachedMetadata{true}, context)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2,
      tensorstore::Open(json_spec, dtype_v<uint16_t>, Schema::Shape({100}),
                        tensorstore::OpenMode::create,
                        tensorstore::OpenMode::delete_existing,
                        tensorstore::RecheckCachedMetadata{true}, context)
          .result());
  EXPECT_THAT(tensorstore::ResolveBounds(store1).result(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Updated zarr metadata")));
}

TEST(ZarrDriverTest, DeleteExisting) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvs, tensorstore::kvstore::Open("memory://prefix/zarr.json", context)
                    .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::kvstore::Write(kvs, "", absl::Cord("")).result());
  EXPECT_THAT(
      tensorstore::Open(GetJsonSpec(), tensorstore::OpenMode::open, context)
          .result(),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("Invalid JSON")));
}

TEST(ZarrDriverTest, ShardingBatchRead) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_kvstore_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_kvstore = *mock_kvstore_resource;
  mock_kvstore->forward_to = tensorstore::GetMemoryKeyValueStore();
  mock_kvstore->log_requests = true;
  mock_kvstore->handle_batch_requests = true;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "zarr3"},
                         {"kvstore", {{"driver", "mock_key_value_store"}}}},
                        tensorstore::OpenMode::create, context,
                        dtype_v<uint16_t>, Schema::Shape({8, 8}),
                        ChunkLayout::ReadChunkShape({2, 2}),
                        ChunkLayout::WriteChunkShape({4, 4}))
          .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeScalarArray<uint16_t>(42), store));
  mock_kvstore->request_log.pop_all();

  TENSORSTORE_ASSERT_OK(tensorstore::Read(store));

  EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(4));
}

TEST(ZarrDriverTest, CodecLifetime) {
  tensorstore::internal_testing::ScopedTemporaryDirectory tempdir;
  tensorstore::Future<const void> future;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(
            {{"driver", "zarr3"},
             {"kvstore", {{"driver", "file"}, {"path", tempdir.path()}}},
             {"metadata",
              {{"shape", {10}},
               {"chunk_grid",
                {{"name", "regular"},
                 {"configuration", {{"chunk_shape", {10}}}}}},
               {"codecs",
                ::nlohmann::json::array_t{
                    {{"name", "sharding_indexed"},
                     {"configuration", {{"chunk_shape", {10}}}}}}}}},
             {"create", true},
             {"dtype", "float32"}})
            .result());
    future = tensorstore::Write(tensorstore::MakeScalarArray<float>(42), store)
                 .commit_future;
  }
  TENSORSTORE_ASSERT_OK(future.status());
}

TEST(ZarrDriverTest, CanReferenceSourceDataIndefinitely) {
  // Dimension 0 is chunked with a size of 64.  Need chunk to be at least 256
  // bytes due to riegeli size limits for non copying cords.
  for (bool reference_source_data : {false, true}) {
    SCOPED_TRACE(
        absl::StrFormat("reference_source_data=%d", reference_source_data));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open({{"driver", "zarr3"}, {"kvstore", "memory://"}},
                          dtype_v<uint32_t>, Schema::Shape({64}),
                          tensorstore::OpenMode::create)
            .result());
    auto array = tensorstore::AllocateArray<uint32_t>({64});
    std::fill_n(array.data(), 64, 1);
    TENSORSTORE_ASSERT_OK(tensorstore::Write(
        array, store,
        reference_source_data
            ? tensorstore::can_reference_source_data_indefinitely
            : tensorstore::cannot_reference_source_data));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto read_result,
        tensorstore::kvstore::Read(store.kvstore(), "c/0").result());
    EXPECT_EQ(reference_source_data,
              read_result.value.Flatten().data() ==
                  reinterpret_cast<const char*>(array.data()));
  }
}

TEST(ZarrDriverTest, CanReferenceSourceDataIndefinitelyWithCast) {
  // Dimension 0 is chunked with a size of 64.  Need chunk to be at least 256
  // bytes due to riegeli size limits for non copying cords.
  for (bool reference_source_data : {false, true}) {
    for (bool can_reinterpret_cast : {false, true}) {
      SCOPED_TRACE(
          absl::StrFormat("reference_source_data=%d", reference_source_data));
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto store,
          tensorstore::Open({{"driver", "zarr3"}, {"kvstore", "memory://"}},
                            can_reinterpret_cast ? DataType(dtype_v<int32_t>)
                                                 : DataType(dtype_v<float>),
                            Schema::Shape({64}), tensorstore::OpenMode::create)
              .result());
      auto cast_store = tensorstore::Cast(store, dtype_v<uint32_t>);
      auto array = tensorstore::AllocateArray<uint32_t>({64});
      std::fill_n(array.data(), 64, 1);
      TENSORSTORE_ASSERT_OK(tensorstore::Write(
          array, cast_store,
          reference_source_data
              ? tensorstore::can_reference_source_data_indefinitely
              : tensorstore::cannot_reference_source_data));
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto read_result,
          tensorstore::kvstore::Read(store.kvstore(), "c/0").result());
      EXPECT_EQ(reference_source_data && can_reinterpret_cast,
                read_result.value.Flatten().data() ==
                    reinterpret_cast<const char*>(array.data()));
    }
  }
}

TEST(FullShardWriteTest, WithoutTransaction) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_key_value_store = *mock_key_value_store_resource;

  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "mock_key_value_store"},
           {"path", "prefix/"},
       }},
      {"create", true},
      {"schema",
       {
           {"chunk_layout",
            {{"read_chunk", {{"shape", {2, 2, 2}}}},
             {"write_chunk", {{"shape", {4, 4, 4}}}}}},
           {"domain", {{"shape", {4, 8, 12}}}},
           {"dtype", "uint16"},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  auto store_future = tensorstore::Open(spec, context);
  store_future.Force();

  {
    auto req = mock_key_value_store->read_requests.pop();
    EXPECT_EQ("prefix/zarr.json", req.key);
    req.promise.SetResult(
        tensorstore::kvstore::ReadResult::Missing(absl::Now()));
  }

  {
    auto req = mock_key_value_store->write_requests.pop();
    EXPECT_EQ("prefix/zarr.json", req.key);
    EXPECT_EQ(StorageGeneration::NoValue(),
              req.options.generation_conditions.if_equal);
    req.promise.SetResult(TimestampedStorageGeneration{
        StorageGeneration::FromString("g0"), absl::Now()});
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store, store_future.result());

  auto future = tensorstore::Write(
      tensorstore::MakeScalarArray<uint16_t>(42),
      store | tensorstore::Dims(0, 1, 2).SizedInterval({0, 4, 8}, {4, 4, 4}));

  future.Force();

  {
    auto req = mock_key_value_store->write_requests.pop();
    ASSERT_EQ("prefix/c/0/1/2", req.key);
    // Writeback is unconditional because the entire shard is being written.
    ASSERT_EQ(StorageGeneration::Unknown(),
              req.options.generation_conditions.if_equal);
    req.promise.SetResult(TimestampedStorageGeneration{
        StorageGeneration::FromString("g0"), absl::Now()});
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(ZarrDriverTest, MetadataCache) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson(
          {{"cache_pool", {{"total_bytes_limit", 1024 * 1024 * 10}}}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_kvstore = *mock_key_value_store_resource;
  mock_kvstore->forward_to = tensorstore::GetMemoryKeyValueStore();
  mock_kvstore->log_requests = true;

  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "mock_key_value_store"},
           {"path", "prefix/"},
       }},
      {"schema",
       {
           {"domain", {{"shape", {4}}}},
           {"dtype", "uint16"},
       }},
      {"recheck_cached_metadata", false},
      {"recheck_cached_data", false},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(spec, context, tensorstore::OpenMode::create)
            .result());
    mock_kvstore->request_log.pop_all();
  }

  {
    // Reopening uses cached metadata without revalidation.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(spec, context, tensorstore::OpenMode::open).result());
    EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(0));
  }
}

TEST(ZarrDriverTest, SeparateMetadataCache) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson({{"cache_pool#metadata",
                          {{"total_bytes_limit", 1024 * 1024 * 10}}}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_kvstore = *mock_key_value_store_resource;
  mock_kvstore->forward_to = tensorstore::GetMemoryKeyValueStore();
  mock_kvstore->log_requests = true;

  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "mock_key_value_store"},
           {"path", "prefix/"},
       }},
      {"metadata_cache_pool", "cache_pool#metadata"},
      {"schema",
       {
           {"chunk_layout",
            {{"read_chunk", {{"shape", {1}}}},
             {"write_chunk", {{"shape", {2}}}}}},
           {"domain", {{"shape", {4}}}},
           {"dtype", "uint16"},
       }},
      {"recheck_cached_metadata", false},
      {"recheck_cached_data", false},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(spec, context, tensorstore::OpenMode::create)
            .result());
    // Write initial data.
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeScalarArray<uint16_t>(42),
                           store | tensorstore::Dims(0).IndexSlice(0)));

    mock_kvstore->request_log.pop_all();

    // Read value from chunk that was written.
    //
    // The shard index will be read and cached.
    EXPECT_THAT(
        tensorstore::Read(store | tensorstore::Dims(0).IndexSlice(0)).result(),
        tensorstore::MakeScalarArray<uint16_t>(42));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(2));

    // Read same value again.
    //
    // The data chunk will be re-read but the shard index will not be.
    EXPECT_THAT(
        tensorstore::Read(store | tensorstore::Dims(0).IndexSlice(0)).result(),
        tensorstore::MakeScalarArray<uint16_t>(42));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(1));

    // Read value from missing chunk in same shard. No kvstore reads are
    // performed because the shard index is cached.
    EXPECT_THAT(
        tensorstore::Read(store | tensorstore::Dims(0).IndexSlice(1)).result(),
        tensorstore::MakeScalarArray<uint16_t>(0));
    EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(0));
  }

  {
    // Reopening uses cached metadata without revalidation.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(spec, context, tensorstore::OpenMode::open).result());
    EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::SizeIs(0));
  }
}

TEST(DriverTest, FillMissingDataReads) {
  for (bool fill_missing_data_reads : {false, true}) {
    SCOPED_TRACE(
        absl::StrCat("fill_missing_data_reads=", fill_missing_data_reads));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(
            {
                {"driver", "zarr3"},
                {"kvstore", "memory://"},
                {"fill_missing_data_reads", fill_missing_data_reads},
            },
            dtype_v<int16_t>, Schema::Shape({1}), tensorstore::OpenMode::create)
            .result());
    {
      auto read_result = tensorstore::Read(store).result();
      if (fill_missing_data_reads) {
        EXPECT_THAT(read_result,
                    ::testing::Optional(tensorstore::MakeArray<int16_t>({0})));
      } else {
        EXPECT_THAT(
            read_result,
            StatusIs(absl::StatusCode::kNotFound,
                     HasSubstr("chunk {0} stored at \"c/0\" is missing")));
      }
    }
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeArray<int16_t>({1}), store)
            .result());
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>({1})));
  }
}

TEST(DriverTest, FillMissingDataReadsSharding) {
  for (bool fill_missing_data_reads : {false, true}) {
    SCOPED_TRACE(
        absl::StrCat("fill_missing_data_reads=", fill_missing_data_reads));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(
            {
                {"driver", "zarr3"},
                {"kvstore", "memory://"},
                {"fill_missing_data_reads", fill_missing_data_reads},
            },
            dtype_v<int16_t>, Schema::Shape({2}),
            tensorstore::ChunkLayout::ReadChunkShape({1}),
            tensorstore::ChunkLayout::WriteChunkShape({2}),
            tensorstore::OpenMode::create)
            .result());
    {
      auto read_result = tensorstore::Read(store).result();
      if (fill_missing_data_reads) {
        EXPECT_THAT(read_result, ::testing::Optional(
                                     tensorstore::MakeArray<int16_t>({0, 0})));
      } else {
        EXPECT_THAT(read_result,
                    StatusIs(absl::StatusCode::kNotFound,
                             MatchesRegex(".*chunk .* is missing.*")));
      }
    }
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeArray<int16_t>({1}), store)
            .result());
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>({1, 1})));
  }
}

// Tests that all-zero chunks are written if
// `store_data_equal_to_fill_value=true`.
TEST(DriverTest, StoreDataEqualToFillValue) {
  for (bool store_data_equal_to_fill_value : {false, true}) {
    SCOPED_TRACE(absl::StrCat("store_data_equal_to_fill_value=",
                              store_data_equal_to_fill_value));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Open({{"driver", "zarr3"},
                                       {"kvstore", "memory://"},
                                       {"store_data_equal_to_fill_value",
                                        store_data_equal_to_fill_value}},
                                      tensorstore::dtype_v<uint8_t>,
                                      tensorstore::RankConstraint{0},
                                      tensorstore::OpenMode::create)
                        .result());
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(0), store));
    if (store_data_equal_to_fill_value) {
      EXPECT_THAT(GetMap(store.kvstore()),
                  ::testing::Optional(::testing::UnorderedElementsAre(
                      ::testing::Pair("zarr.json", ::testing::_),
                      ::testing::Pair("c", ::testing::_))));
    } else {
      EXPECT_THAT(GetMap(store.kvstore()),
                  ::testing::Optional(::testing::UnorderedElementsAre(
                      ::testing::Pair("zarr.json", ::testing::_))));
    }
  }
}

TEST(DriverTest, StoreDataEqualToFillValueSharding) {
  for (bool store_data_equal_to_fill_value : {false, true}) {
    SCOPED_TRACE(absl::StrCat("store_data_equal_to_fill_value=",
                              store_data_equal_to_fill_value));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open({{"driver", "zarr3"},
                           {"kvstore", "memory://"},
                           {"store_data_equal_to_fill_value",
                            store_data_equal_to_fill_value}},
                          tensorstore::dtype_v<uint8_t>,
                          tensorstore::Schema::Shape({2}),
                          tensorstore::ChunkLayout::ReadChunkShape({1}),
                          tensorstore::ChunkLayout::WriteChunkShape({2}),
                          tensorstore::OpenMode::create)
            .result());
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(0), store));
    if (store_data_equal_to_fill_value) {
      EXPECT_THAT(GetMap(store.kvstore()),
                  ::testing::Optional(::testing::UnorderedElementsAre(
                      ::testing::Pair("zarr.json", ::testing::_),
                      ::testing::Pair("c/0", ::testing::_))));
    } else {
      EXPECT_THAT(GetMap(store.kvstore()),
                  ::testing::Optional(::testing::UnorderedElementsAre(
                      ::testing::Pair("zarr.json", ::testing::_))));
    }
  }
}

TEST(DriverTest, TransactionalZeroByteReadAfterWritingChunk) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_kvstore = *mock_key_value_store_resource;
  mock_kvstore->forward_to = tensorstore::GetMemoryKeyValueStore();
  mock_kvstore->log_requests = true;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "zarr3"},
                         {"kvstore", {{"driver", "mock_key_value_store"}}},
                         {"store_data_equal_to_fill_value", true}},
                        tensorstore::dtype_v<uint8_t>,
                        tensorstore::Schema::Shape({}),
                        tensorstore::OpenMode::create, context)
          .result());

  mock_kvstore->request_log.pop_all();

  tensorstore::Transaction txn(tensorstore::isolated);
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      tensorstore::MakeScalarArray<uint8_t>(42), store | txn));

  {
    tensorstore::kvstore::ReadOptions options;
    options.byte_range = tensorstore::OptionalByteRangeRequest::Stat();
    EXPECT_THAT(tensorstore::kvstore::Read((store | txn)->kvstore(), "c",
                                           std::move(options))
                    .result(),
                MatchesKvsReadResult(absl::Cord()));
  }

  EXPECT_THAT(mock_kvstore->request_log.pop_all(), ::testing::ElementsAre());
}

TEST(DriverTest, WriteCoalescingWhenCopyingFromChunkedTensorStore) {
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto mock_key_value_store_resource,
      context.GetResource<tensorstore::internal::MockKeyValueStoreResource>());
  auto mock_kvstore = *mock_key_value_store_resource;
  mock_kvstore->forward_to = tensorstore::GetMemoryKeyValueStore();
  mock_kvstore->log_requests = true;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto source,
      tensorstore::Open(
          {
              {"driver", "zarr3"},
              {"kvstore",
               {{"driver", "mock_key_value_store"}, {"path", "source/"}}},
          },
          tensorstore::dtype_v<uint8_t>, tensorstore::Schema::Shape({6}),
          tensorstore::ChunkLayout::ReadChunkShape({1}),
          tensorstore::ChunkLayout::WriteChunkShape({3}),
          tensorstore::OpenMode::create, context)
          .result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeScalarArray<uint8_t>(42), source));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dest,
      tensorstore::Open(
          {
              {"driver", "zarr3"},
              {"kvstore",
               {{"driver", "mock_key_value_store"}, {"path", "dest/"}}},
          },
          tensorstore::dtype_v<uint8_t>, tensorstore::Schema::Shape({6}),
          tensorstore::ChunkLayout::ReadChunkShape({1}),
          tensorstore::ChunkLayout::WriteChunkShape({3}),
          tensorstore::OpenMode::create, context)
          .result());

  mock_kvstore->request_log.pop_all();
  auto copy_future = tensorstore::Copy(source, dest, tensorstore::Batch::New());
  TENSORSTORE_ASSERT_OK(copy_future.result());
  EXPECT_THAT(
      mock_kvstore->request_log.pop_all(),
      ::testing::UnorderedElementsAreArray({
          JsonSubValuesMatch({{"/type", "read"}, {"/key", "source/c/0"}}),
          JsonSubValuesMatch({{"/type", "read"}, {"/key", "source/c/1"}}),
          JsonSubValuesMatch({{"/type", "write"}, {"/key", "dest/c/0"}}),
          JsonSubValuesMatch({{"/type", "write"}, {"/key", "dest/c/1"}}),
      }));
}

TEST(DriverTest, UrlSchemeRoundtrip) {
  TestTensorStoreUrlRoundtrip(
      {{"driver", "zarr3"},
       {"kvstore", {{"driver", "memory"}, {"path", "abc.zarr3/"}}}},
      "memory://abc.zarr3/|zarr3:");
  TestTensorStoreSpecRoundtripNormalize(
      "memory://abc.zarr3|zarr3:def",
      {{"driver", "zarr3"},
       {"kvstore", {{"driver", "memory"}, {"path", "abc.zarr3/def/"}}}});
}

// Tests for open_as_void functionality

// Helper functions to reduce duplication in open_as_void tests.

// Returns the JSON for a structured data type with fields x (uint8) and y
// (int16). Total size: 3 bytes per element.
::nlohmann::json GetStructuredDataTypeJson() {
  return {{"name", "struct"},
          {"configuration",
           {{"fields",
             ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                      {{"name", "y"}, {"data_type", "int16"}}})}}}};
}

// Returns a create spec for a structured type array with field selection.
// The structured type has fields x (uint8) and y (int16).
::nlohmann::json GetStructuredCreateSpec(
    std::string_view kvstore_path,
    std::vector<Index> shape,
    std::vector<Index> chunk_shape,
    std::optional<::nlohmann::json> codecs = std::nullopt) {
  ::nlohmann::json metadata = {
      {"data_type", GetStructuredDataTypeJson()},
      {"shape", shape},
      {"chunk_grid",
       {{"name", "regular"}, {"configuration", {{"chunk_shape", chunk_shape}}}}},
  };
  if (codecs.has_value()) {
    metadata["codecs"] = *codecs;
  }
  return {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", kvstore_path}}},
      {"field", "y"},
      {"metadata", metadata},
  };
}

// Returns a spec for opening a zarr3 array with open_as_void=true.
::nlohmann::json GetOpenAsVoidSpec(std::string_view kvstore_path) {
  return {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", kvstore_path}}},
      {"open_as_void", true},
  };
}

TEST(Zarr3OpenAsVoidTest, SimpleType) {
  auto context = Context::Default();

  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"data_type", "int16"},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };

  EXPECT_THAT(tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read)
                  .result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".*open_as_void is only supported for structured dtypes.*"));
}

TEST(Zarr3OpenAsVoidTest, StructuredType) {
  // Test open_as_void with a structured data type
  auto context = Context::Default();

  // Create and write the array using a structured dtype (with field)
  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes total
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(GetStructuredCreateSpec("prefix/", {4, 4}, {2, 2}),
                        context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Write some data to field y (int16)
  // int16 100 = 0x0064 in little endian = [0x64, 0x00]
  // int16 200 = 0x00C8 in little endian = [0xC8, 0x00]
  auto data = tensorstore::MakeArray<int16_t>({{100, 200}, {300, 400}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(data, store | tensorstore::Dims(0, 1).SizedInterval(
                                           {0, 0}, {2, 2}))
          .result());

  // Close store to ensure data is flushed
  store = tensorstore::TensorStore<int16_t>();

  // Open with open_as_void=true, specifying byte_t element type.
  // This gives us a TensorStore<byte_t> which supports indexed array access.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(GetOpenAsVoidSpec("prefix/"), context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // The store should have rank = original_rank + 1 (for bytes dimension)
  EXPECT_EQ(3, byte_store.rank());

  // The last dimension should be 3 bytes (1 byte for uint8 + 2 bytes for int16)
  EXPECT_EQ(3, byte_store.domain().shape()[2]);

  // The data type should be byte
  EXPECT_EQ(tensorstore::dtype_v<tensorstore::dtypes::byte_t>,
            byte_store.dtype());

  // Read and verify byte content for field y only
  // Since we only wrote to field y, field x will be zeros (fill value)
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(
          byte_store | tensorstore::Dims(0, 1, 2).SizedInterval({0, 0, 0},
                                                                {2, 2, 3}))
          .result());

  EXPECT_THAT(byte_array.shape(), ::testing::ElementsAre(2, 2, 3));

  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // y values: 100=0x0064, 200=0x00C8, 300=0x012C, 400=0x0190 (little-endian)
  // x is fill value (0) since we only wrote to field y
  EXPECT_EQ(tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0}, std::byte{0x64}, std::byte{0x00}},
                  {std::byte{0}, std::byte{0xC8}, std::byte{0x00}}},
                 {{std::byte{0}, std::byte{0x2C}, std::byte{0x01}},
                  {std::byte{0}, std::byte{0x90}, std::byte{0x01}}}}),
            byte_array);
}

TEST(Zarr3OpenAsVoidTest, WithCompression) {
  auto context = Context::Default();
  ::nlohmann::json gzip_codecs = {{{"name", "bytes"}}, {{"name", "gzip"}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          GetStructuredCreateSpec("prefix/", {4, 4}, {2, 2}, gzip_codecs),
          context, tensorstore::OpenMode::create,
          tensorstore::ReadWriteMode::read_write)
          .result());

  // Write some data to field y (int16)
  // Using values with distinct byte patterns for verification
  auto data = tensorstore::MakeArray<int16_t>(
      {{0x0102, 0x0304}, {0x0506, 0x0708}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(data, store | tensorstore::Dims(0, 1).SizedInterval(
                                           {0, 0}, {2, 2}))
          .result());

  // Now open with open_as_void=true, specifying byte_t element type
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(GetOpenAsVoidSpec("prefix/"), context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // The store should have rank = original_rank + 1 (for bytes dimension)
  EXPECT_EQ(3, byte_store.rank());

  // The last dimension should be 3 bytes (1 for uint8 + 2 for int16)
  EXPECT_EQ(3, byte_store.domain().shape()[2]);

  // The data type should be byte
  EXPECT_EQ(tensorstore::dtype_v<tensorstore::dtypes::byte_t>,
            byte_store.dtype());

  // Read the raw bytes and verify decompression works
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1).SizedInterval(
                                         {0, 0}, {2, 2}))
          .result());
  EXPECT_THAT(byte_array.shape(), ::testing::ElementsAre(2, 2, 3));

  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // y values: 0x0102, 0x0304, 0x0506, 0x0708 (little-endian)
  // x is fill value (0) since we only wrote to field y
  EXPECT_EQ(tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0x00}, std::byte{0x02}, std::byte{0x01}},
                  {std::byte{0x00}, std::byte{0x04}, std::byte{0x03}}},
                 {{std::byte{0x00}, std::byte{0x06}, std::byte{0x05}},
                  {std::byte{0x00}, std::byte{0x08}, std::byte{0x07}}}}),
            byte_array);
}

TEST(Zarr3OpenAsVoidTest, SpecRoundtrip) {
  // Test that open_as_void is properly preserved in spec round-trips
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type", "int16"},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto json_result, spec.ToJson());

  EXPECT_EQ(true, json_result.value("open_as_void", false));
}

TEST(Zarr3OpenAsVoidTest, GetBoundSpecData) {
  // Test that open_as_void is correctly preserved when getting spec from an
  // opened void store. This tests ZarrDataCache::GetBoundSpecData.
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(GetStructuredCreateSpec("prefix/", {4, 4}, {2, 2}),
                        context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Now open with open_as_void=true
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(GetOpenAsVoidSpec("prefix/"), context,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // Get the spec from the opened void store - this invokes GetBoundSpecData
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto obtained_spec, void_store.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto obtained_json, obtained_spec.ToJson());

  // Verify open_as_void is true in the obtained spec
  EXPECT_EQ(true, obtained_json.value("open_as_void", false));

  // Also verify metadata was correctly populated
  EXPECT_TRUE(obtained_json.contains("metadata"));
  auto& metadata = obtained_json["metadata"];
  EXPECT_THAT(metadata["data_type"], MatchesJson(GetStructuredDataTypeJson()));
}

TEST(Zarr3OpenAsVoidTest, CannotUseWithField) {
  // Test that specifying both open_as_void and field is rejected as they are
  // mutually exclusive options.
  ::nlohmann::json spec_with_both{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"data_type", GetStructuredDataTypeJson()},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
      {"field", "x"},
      {"open_as_void", true},
  };

  // Specifying both field and open_as_void should fail at spec parsing
  EXPECT_THAT(
      tensorstore::Spec::FromJson(spec_with_both),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("\"field\" and \"open_as_void\" are mutually "
                         "exclusive")));
}

TEST(Zarr3OpenAsVoidTest, UrlNotSupported) {
  // Test that open_as_void is not supported with URL syntax
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type", "int16"},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // ToUrl should fail when open_as_void is specified
  EXPECT_THAT(spec.ToUrl(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Zarr3OpenAsVoidTest, ReadWrite) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(GetStructuredCreateSpec("prefix/", {2, 2}, {2, 2}),
                        context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Write data to field y (int16)
  auto data =
      tensorstore::MakeArray<int16_t>({{0x0102, 0x0304}, {0x0506, 0x0708}});
  TENSORSTORE_EXPECT_OK(tensorstore::Write(data, store).result());

  // Open as void with byte_t type and read
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(GetOpenAsVoidSpec("prefix/"), context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Read the raw bytes
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto byte_array,
                                   tensorstore::Read(byte_store).result());

  EXPECT_THAT(byte_array.shape(), ::testing::ElementsAre(2, 2, 3));

  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // y values: 0x0102, 0x0304, 0x0506, 0x0708 (little-endian)
  // x is fill value (0) since we only wrote to field y
  EXPECT_EQ(tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0x00}, std::byte{0x02}, std::byte{0x01}},
                  {std::byte{0x00}, std::byte{0x04}, std::byte{0x03}}},
                 {{std::byte{0x00}, std::byte{0x06}, std::byte{0x05}},
                  {std::byte{0x00}, std::byte{0x08}, std::byte{0x07}}}}),
            byte_array);
}


TEST(Zarr3OpenAsVoidTest, WriteWithCompression) {
  // Test writing through open_as_void with compression enabled.
  // Verifies that the EncodeChunk method correctly compresses data.
  auto context = Context::Default();
  ::nlohmann::json gzip_codecs_with_endian = ::nlohmann::json::array(
      {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
       {{"name", "gzip"}, {"configuration", {{"level", 5}}}}});
  auto create_spec =
      GetStructuredCreateSpec("prefix/", {4, 4}, {4, 4}, gzip_codecs_with_endian);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Initialize with zeros
  auto zeros = tensorstore::MakeArray<int16_t>(
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  TENSORSTORE_EXPECT_OK(tensorstore::Write(zeros, store).result());

  // Open as void for writing
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(GetOpenAsVoidSpec("prefix/"), context,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Verify the void store has the expected shape: [4, 4, 3] (3 bytes per element)
  EXPECT_EQ(3, void_store.rank());
  EXPECT_EQ(4, void_store.domain().shape()[0]);
  EXPECT_EQ(4, void_store.domain().shape()[1]);
  EXPECT_EQ(3, void_store.domain().shape()[2]);

  // Create raw bytes for the structured type
  auto raw_bytes = tensorstore::AllocateArray<tensorstore::dtypes::byte_t>(
      {4, 4, 3}, tensorstore::c_order, tensorstore::value_init);

  // Set first element: x=0x11, y=0x0304 (little endian: 04 03)
  auto raw_bytes_ptr = static_cast<unsigned char*>(
      const_cast<void*>(static_cast<const void*>(raw_bytes.data())));
  raw_bytes_ptr[0] = 0x11;  // x field
  raw_bytes_ptr[1] = 0x04;  // y low byte
  raw_bytes_ptr[2] = 0x03;  // y high byte

  // Write raw bytes through void access (triggers compression)
  TENSORSTORE_EXPECT_OK(tensorstore::Write(raw_bytes, void_store).result());

  // Verify the write worked by reading back through void access
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto void_read,
                                   tensorstore::Read(void_store).result());
  auto void_read_ptr = static_cast<const unsigned char*>(void_read.data());
  // First 3 bytes should be our pattern
  EXPECT_EQ(void_read_ptr[0], 0x11);  // x field
  EXPECT_EQ(void_read_ptr[1], 0x04);  // y low byte
  EXPECT_EQ(void_read_ptr[2], 0x03);  // y high byte

  // Read back through typed access to field y
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto typed_store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto typed_read,
                                   tensorstore::Read(typed_store).result());
  auto typed_ptr = static_cast<const int16_t*>(typed_read.data());

  // First element y field should be 0x0304
  EXPECT_EQ(typed_ptr[0], 0x0304);
  // Rest should be zeros
  EXPECT_EQ(typed_ptr[1], 0);
}

TEST(Zarr3DriverTest, FieldSelectionUrlNotSupported) {
  // Test that field selection is not supported with URL syntax
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "x"},
      {"metadata",
       {
           {"data_type", GetStructuredDataTypeJson()},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // ToUrl should fail when field is specified
  EXPECT_THAT(spec.ToUrl(), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("selected_field")));
}

TEST(Zarr3DriverTest, StructuredFieldWithFieldShape) {
  // Test reading and writing with a structured field that has a field_shape.
  // Fields with field_shape (like r16) add extra dimensions to the store.
  auto context = Context::Default();

  // Struct layout: a (int32, 4 bytes) + b (r16, 2 bytes with shape [2]) = 6 bytes
  // Create directly with field "b" which has field_shape [2]
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix_field_shape/"}}},
      {"field", "b"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields", ::nlohmann::json::array(
                              {{{"name", "a"}, {"data_type", "int32"}},
                               {{"name", "b"}, {"data_type", "r16"}}})}}}}},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Field 'b' (r16) should have rank original_rank + field_shape_rank = 2 + 1 = 3
  ASSERT_EQ(3, store.rank());
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(4, 4, 2));

  // Write some data to field b
  auto data = tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
      {{{std::byte{1}, std::byte{2}}, {std::byte{3}, std::byte{4}}},
       {{std::byte{5}, std::byte{6}}, {std::byte{7}, std::byte{8}}}});
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(data, store | tensorstore::Dims(0, 1).SizedInterval(
                                           {0, 0}, {2, 2}))
          .result());

  // Read it back
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_data,
      tensorstore::Read(store |
                        tensorstore::Dims(0, 1).SizedInterval({0, 0}, {2, 2}))
          .result());

  EXPECT_EQ(data, read_data);
}

// Tests for GetSpecInfo() with open_as_void (mirroring v2 tests)

TEST(Zarr3OpenAsVoidTest, GetSpecInfoWithKnownRank) {
  // Test that GetSpecInfo correctly computes rank when open_as_void=true
  // and dtype is specified with known chunked_rank.
  // Expected: full_rank = chunked_rank + 1 (for bytes dimension)
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type", "int32"},  // 4-byte integer
           {"shape", {10, 20}},     // 2D array, so chunked_rank=2
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {5, 10}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // With open_as_void and dtype specified, rank should be chunked_rank + 1
  // chunked_rank = 2 (from shape), so full_rank = 3
  EXPECT_EQ(3, spec.rank());
}

TEST(Zarr3OpenAsVoidTest, GetSpecInfoWithStructuredDtype) {
  // Test GetSpecInfo with open_as_void=true and a structured dtype.
  // The bytes dimension should reflect the full struct size.
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type",
            {{"name", "structured"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{"x", "int32"}, {"y", "uint16"}})}}}}},
           {"shape", {8}},  // 1D array
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {4}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // chunked_rank = 1, so full_rank = 2
  EXPECT_EQ(2, spec.rank());
}

TEST(Zarr3OpenAsVoidTest, GetSpecInfoWithDynamicRank) {
  // Test GetSpecInfo when open_as_void=true with dtype but no shape/chunks
  // (i.e., chunked_rank is dynamic). In this case, full_rank should remain
  // dynamic until metadata is loaded.
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type", "int16"},
           // No shape or chunks specified, so chunked_rank is dynamic
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // When chunked_rank is dynamic, full_rank remains dynamic
  EXPECT_EQ(tensorstore::dynamic_rank, spec.rank());
}

TEST(Zarr3OpenAsVoidTest, GetSpecInfoWithoutDtype) {
  // Test that when open_as_void=true but dtype is not specified,
  // GetSpecInfo falls through to normal GetSpecRankAndFieldInfo behavior.
  ::nlohmann::json json_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
      // No metadata.data_type specified
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  // Without dtype, rank should be dynamic (normal behavior)
  EXPECT_EQ(tensorstore::dynamic_rank, spec.rank());
}

TEST(Zarr3OpenAsVoidTest, GetSpecInfoRankConsistency) {
  // Verify that the rank computed by GetSpecInfo matches what we get when
  // actually opening the store with a structured dtype.
  auto context = Context::Default();
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                         {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {3, 4, 5}},  // 3D array
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {3, 4, 5}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Open the store with open_as_void - don't specify metadata so it's read
  // from the existing store
  ::nlohmann::json void_spec_json{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec_json, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // Opened store rank should be chunked_rank + 1 = 3 + 1 = 4
  EXPECT_EQ(4, void_store.rank());

  // Verify bytes dimension size - the domain is valid on an opened store
  auto store_domain = void_store.domain();
  EXPECT_TRUE(store_domain.valid());
  EXPECT_EQ(3, store_domain.shape()[3]);  // 3 bytes (1 for uint8 + 2 for int16)

  // Now test the spec parsing with known structured metadata also sets rank correctly
  ::nlohmann::json void_spec_with_metadata{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix2/"}}},
      {"open_as_void", true},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                         {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {3, 4, 5}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {3, 4, 5}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_spec, tensorstore::Spec::FromJson(void_spec_with_metadata));

  // Spec rank should be 4 (3D chunked + 1 bytes dimension)
  // This verifies GetSpecInfo computes full_rank = chunked_rank + 1
  EXPECT_EQ(4, void_spec.rank());
}

TEST(Zarr3OpenAsVoidTest, FillValue) {  // TODO: We need to define behavior for whether fill_value is required always for struct dtype.
  // Test that fill_value is correctly obtained from metadata when using
  // open_as_void. The void access should get the fill_value representing
  // the raw bytes of the original fill_value.
  auto context = Context::Default();

  // Create an array with structured dtype and explicit fill_value
  // Per zarr-extensions spec, fill_value for struct must be a JSON object
  // mapping field names to their fill values.
  // Struct: x (uint8) = 0x12, y (int16) = 0x3456
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                         {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
           {"fill_value", {{"x", 0x12}, {"y", 0x3456}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Verify the normal store has the expected fill_value for field "y"
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto normal_fill, store.fill_value());
  EXPECT_TRUE(normal_fill.valid());
  EXPECT_EQ(tensorstore::MakeScalarArray<int16_t>(0x3456), normal_fill);

  // Open with open_as_void=true
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // Verify void store has a valid fill_value derived from the original
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto void_fill, void_store.fill_value());
  EXPECT_TRUE(void_fill.valid());

  // The void fill_value should have shape {3} (3 bytes for struct: 1 "x" + 2
  // "y")
  EXPECT_THAT(void_fill.shape(), ::testing::ElementsAre(3));

  // The fill_value bytes should represent the struct in little endian:
  // x = 0x12 (1 byte), y = 0x3456 -> bytes 0x56, 0x34 (little endian)
  EXPECT_THAT(void_fill, tensorstore::MatchesArray(
                             tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                                 {std::byte{0x12}, std::byte{0x56}, std::byte{0x34}})));
}

TEST(Zarr3OpenAsVoidTest, IncompatibleMetadata) {
  // Test that open_as_void correctly rejects incompatible metadata when the
  // underlying storage is modified to have a different bytes_per_outer_element.
  auto context = Context::Default();
  ::nlohmann::json storage_spec{{"driver", "memory"}};
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", storage_spec},
      {"path", "prefix/"},
      {"field", "y"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{{"name", "x"}, {"data_type", "uint8"}},
                                         {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {2, 2}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Write some data to field y
  auto data = tensorstore::MakeArray<int16_t>({{1, 2}, {3, 4}});
  TENSORSTORE_EXPECT_OK(tensorstore::Write(data, store).result());

  // Open with open_as_void
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", storage_spec},
      {"path", "prefix/"},
      {"open_as_void", true},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // Now overwrite the underlying storage with incompatible metadata
  // (different bytes_per_outer_element: 5 bytes instead of 3)
  ::nlohmann::json incompatible_spec{
      {"driver", "zarr3"},
      {"kvstore", storage_spec},
      {"path", "prefix/"},
      {"field", "b"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array({{{"name", "a"}, {"data_type", "uint8"}},
                                         {{"name", "b"}, {"data_type", "int32"}}})}}}}},  // 5 bytes - incompatible
           {"shape", {2, 2}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto incompatible_store,
      tensorstore::Open(incompatible_spec, context,
                        tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // ResolveBounds on the original void store should fail because the
  // underlying metadata changed to an incompatible dtype
  EXPECT_THAT(ResolveBounds(void_store).result(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(Zarr3OpenAsVoidTest, WithShardingRejectsSimpleType) {
  // Test that open_as_void with sharding correctly rejects simple dtypes.
  auto context = Context::Default();

  // Create a sharded array with simple dtype
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"data_type", "int32"},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Write some data
  auto data = tensorstore::MakeArray<int32_t>(
      {{1, 2, 0, 0, 0, 0, 0, 0},
       {3, 4, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0}});
  TENSORSTORE_EXPECT_OK(tensorstore::Write(data, store).result());

  // Attempt to open with open_as_void=true - should fail for simple dtype
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };

  EXPECT_THAT(tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                                tensorstore::ReadWriteMode::read)
                  .result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".*open_as_void is only supported for structured dtypes.*"));
}

TEST(Zarr3OpenAsVoidTest, InvalidSchema) {
  // Test that schema constraints are properly validated when using open_as_void.
  auto context = Context::Default();

  // Create a structured array with shape {4, 4}.
  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes total.
  // With open_as_void, this becomes a 3D uint8 array with shape {4, 4, 3}.
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix_void_invalid/"}}},
      {"field", "x"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields", ::nlohmann::json::array(
                              {{{"name", "x"}, {"data_type", "uint8"}},
                               {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {2, 2}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix_void_invalid/"}}},
      {"open_as_void", true},
  };

  // Test rank mismatch: RankConstraint should be 3 for this open_as_void
  // (2D array + 1 dimension for struct bytes).
  EXPECT_THAT(
      tensorstore::Open(void_spec, context, tensorstore::RankConstraint(2))
          .result(),
      tensorstore::MatchesStatus(absl::StatusCode::kFailedPrecondition,
                                 ".*Rank specified by schema \\(2\\) does not "
                                 "match rank specified by metadata \\(3\\)"));

  // Test dtype mismatch: open_as_void dtype must be byte.
  EXPECT_THAT(tensorstore::Open(void_spec, context, dtype_v<int16_t>).result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  ".*data_type from metadata \\(byte\\) does not match dtype "
                  "in schema \\(int16\\)"));
}

TEST(Zarr3OpenAsVoidTest, StructBigEndian) {
  auto context = Context::Default();

  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields", ::nlohmann::json::array(
                              {{{"name", "x"}, {"data_type", "uint8"}},
                               {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {4, 4}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {2, 2}}}}}},
           {"codecs",
            {{{"name", "bytes"}, {"configuration", {{"endian", "big"}}}},
             {{"name", "gzip"}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Write 0x1234 to field y. In big-endian, this is [0x12, 0x34].
  auto data = tensorstore::MakeArray<int16_t>({{0x1234}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(
          data, store | tensorstore::Dims(0, 1).SizedInterval({0, 0}, {1, 1}))
          .result());

  // Open as void
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store, tensorstore::Open(void_spec, context,
                                         dtype_v<tensorstore::dtypes::byte_t>,
                                         tensorstore::OpenMode::open,
                                         tensorstore::ReadWriteMode::read)
                           .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {1, 1, 3}))
          .result());

  // Struct: x (uint8) = 0, + y (int16) = 0x1234.
  // open_as_void doesn't handle endianness conversion so it remains in native
  // endianness.
  auto expected_array =
      tensorstore::endian::native == tensorstore::endian::little
          ? tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0}, std::byte{0x34}, std::byte{0x12}}}})
          : tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0}, std::byte{0x12}, std::byte{0x34}}}});

  EXPECT_THAT(byte_array, MatchesArray(expected_array));
}

TEST(Zarr3OpenAsVoidTest, ShardedSubChunkShapeExtension) {
  auto context = Context::Default();

  // Create sharded structured array.
  // Struct: x (uint8), y (int16) -> 3 bytes.
  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix_shard/"}}},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields", ::nlohmann::json::array(
                              {{{"name", "x"}, {"data_type", "uint8"}},
                               {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}}, {"codecs", {{{"name", "bytes"}}}}}}}}},
       }},
      {"field", "x"},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Open as void.
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix_shard/"}}},
      {"open_as_void", true},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  // Spec should have sub_chunk_shape extended with the bytes dimension [4, 4,
  // 3]
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, void_store.spec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_json, spec.ToJson());
  EXPECT_THAT(
      spec_json["metadata"]["codecs"],
      ::testing::Contains(JsonSubValuesMatch(
          {{"/name", "sharding_indexed"},
           {"/configuration/chunk_shape", ::nlohmann::json({4, 4, 3})}})));
}

// Helper: returns a JSON spec for creating a sharded structured array.
// Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes total.
::nlohmann::json ShardedStructSpec(const std::string& field,
                                   const std::string& path = "prefix/") {
  ::nlohmann::json spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", path}}},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array(
                    {{{"name", "x"}, {"data_type", "uint8"}},
                     {{"name", "y"}, {"data_type", "int16"}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };
  if (!field.empty()) {
    spec["field"] = field;
  }
  return spec;
}

TEST(Zarr3StructuredTest, ShardedFieldYWriteRead) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(ShardedStructSpec("y"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  EXPECT_EQ(tensorstore::dtype_v<int16_t>, store.dtype());
  EXPECT_EQ(2, store.rank());

  auto data = tensorstore::MakeArray<int16_t>({{100, 200}, {300, 400}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(data, store | tensorstore::Dims(0, 1).SizedInterval(
                                           {0, 0}, {2, 2}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_back,
      tensorstore::Read(store | tensorstore::Dims(0, 1).SizedInterval(
                                    {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(data, read_back);
}

TEST(Zarr3StructuredTest, ShardedFieldXWriteRead) {
  auto context = Context::Default();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(ShardedStructSpec("x"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  EXPECT_EQ(tensorstore::dtype_v<uint8_t>, store.dtype());
  EXPECT_EQ(2, store.rank());

  auto data = tensorstore::MakeArray<uint8_t>({{10, 20}, {30, 40}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(data, store | tensorstore::Dims(0, 1).SizedInterval(
                                           {0, 0}, {2, 2}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_back,
      tensorstore::Read(store | tensorstore::Dims(0, 1).SizedInterval(
                                    {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(data, read_back);
}

TEST(Zarr3StructuredTest, ShardedFieldYWriteThenVoidRead) {
  auto context = Context::Default();

  // Write typed data to field "y"
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(ShardedStructSpec("y"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // int16 100 = 0x0064 LE = [0x64, 0x00]
  auto data = tensorstore::MakeArray<int16_t>({{100, 200}, {300, 400}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(data, y_store | tensorstore::Dims(0, 1).SizedInterval(
                                             {0, 0}, {2, 2}))
          .result());

  // Open with byte_t type for void access and verify byte layout
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(void_spec, context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  EXPECT_EQ(3, byte_store.rank());
  EXPECT_EQ(3, byte_store.domain().shape()[2]);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {2, 2, 3}))
          .result());

  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // y values: 100=0x0064, 200=0x00C8, 300=0x012C, 400=0x0190 (little-endian)
  EXPECT_EQ(tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{0}, std::byte{0x64}, std::byte{0x00}},
                  {std::byte{0}, std::byte{0xC8}, std::byte{0x00}}},
                 {{std::byte{0}, std::byte{0x2C}, std::byte{0x01}},
                  {std::byte{0}, std::byte{0x90}, std::byte{0x01}}}}),
            byte_array);
}

struct ShardedVoidWriteFieldReadParam {
  std::string name;
  std::string y_dtype;
  int struct_size;
  std::vector<unsigned char> y_bytes;
  std::function<void(const tensorstore::SharedOffsetArray<const void>&)>
      verify_result;

  friend std::ostream& operator<<(std::ostream& os,
                                  const ShardedVoidWriteFieldReadParam& p) {
    return os << p.name;
  }
};

class ShardedVoidWriteThenFieldReadTest
    : public ::testing::TestWithParam<ShardedVoidWriteFieldReadParam> {};

TEST_P(ShardedVoidWriteThenFieldReadTest, VoidWriteThenFieldRead) {
  const auto& param = GetParam();
  auto context = Context::Default();

  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "x"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array(
                    {{{"name", "x"}, {"data_type", "uint8"}},
                     {{"name", "y"}, {"data_type", param.y_dtype}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto write_bytes = tensorstore::AllocateArray<tensorstore::dtypes::byte_t>(
      {1, 1, param.struct_size}, tensorstore::c_order, tensorstore::value_init);
  auto ptr = static_cast<unsigned char*>(
      const_cast<void*>(static_cast<const void*>(write_bytes.data())));
  ptr[0] = 0xAA;  // x value
  std::memcpy(ptr + 1, param.y_bytes.data(), param.y_bytes.size());

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(write_bytes,
                         void_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                          {0, 0, 0}, {1, 1, param.struct_size}))
          .result());

  ::nlohmann::json y_open_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_open_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_data,
      tensorstore::Read(y_store |
                        tensorstore::Dims(0, 1).SizedInterval({0, 0}, {1, 1}))
          .result());

  param.verify_result(y_data);
}

INSTANTIATE_TEST_SUITE_P(
    Zarr3StructuredTest, ShardedVoidWriteThenFieldReadTest,
    ::testing::Values(
        ShardedVoidWriteFieldReadParam{
            "int16",
            "int16",
            3,
            {0xC8, 0x00},
            [](const tensorstore::SharedOffsetArray<const void>& y_data) {
              auto expected = tensorstore::MakeArray<int16_t>({{200}});
              EXPECT_EQ(expected, y_data);
            }},
        ShardedVoidWriteFieldReadParam{
            "r16",
            "r16",
            3,
            {0xC8, 0x00},
            [](const tensorstore::SharedOffsetArray<const void>& y_data) {
              auto expected =
                  tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                      {{{std::byte{0xC8}, std::byte{0x00}}}});
              EXPECT_EQ(expected, y_data);
            }},
        ShardedVoidWriteFieldReadParam{
            "r64",
            "r64",
            9,
            {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08},
            [](const tensorstore::SharedOffsetArray<const void>& y_data) {
              auto expected =
                  tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                      {{{std::byte{0x01}, std::byte{0x02}, std::byte{0x03},
                         std::byte{0x04}, std::byte{0x05}, std::byte{0x06},
                         std::byte{0x07}, std::byte{0x08}}}});
              EXPECT_EQ(expected, y_data);
            }}),
    [](const ::testing::TestParamInfo<ShardedVoidWriteFieldReadParam>& info) {
      return info.param.name;
    });

TEST(Zarr3StructuredTest, ShardedFieldR16WriteRead) {
  auto context = Context::Default();

  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "x"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array(
                    {{{"name", "x"}, {"data_type", "uint8"}},
                     {{"name", "y"}, {"data_type", "r16"}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  ::nlohmann::json y_open_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_open_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto y_data = tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
      {{{std::byte{0xC8}, std::byte{0x00}}, {std::byte{0x90}, std::byte{0x01}}},
       {{std::byte{0x58}, std::byte{0x02}}, {std::byte{0x20}, std::byte{0x03}}}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(y_data, y_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_read,
      tensorstore::Read(y_store |
                        tensorstore::Dims(0, 1).SizedInterval({0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(y_data, y_read);
}

TEST(Zarr3StructuredTest, ShardedFieldR64WriteRead) {
  auto context = Context::Default();

  ::nlohmann::json create_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "x"},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array(
                    {{{"name", "x"}, {"data_type", "uint8"}},
                     {{"name", "y"}, {"data_type", "r64"}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(create_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  ::nlohmann::json y_open_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_open_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto y_data = tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
      {{{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
         std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}}}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(y_data, y_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {1, 1}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_read,
      tensorstore::Read(y_store |
                        tensorstore::Dims(0, 1).SizedInterval({0, 0}, {1, 1}))
          .result());
  EXPECT_EQ(y_data, y_read);
}

TEST(Zarr3StructuredTest, ShardedMultiFieldRoundtrip) {
  auto context = Context::Default();

  // Create and write to field "x"
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(ShardedStructSpec("x"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto x_data = tensorstore::MakeArray<uint8_t>({{10, 20}, {30, 40}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(x_data, x_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  // Open and write to field "y"
  ::nlohmann::json y_open_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_open_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto y_data = tensorstore::MakeArray<int16_t>({{1000, 2000}, {3000, 4000}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(y_data, y_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  // Read back both fields independently
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_read,
      tensorstore::Read(x_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(x_data, x_read);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_read,
      tensorstore::Read(y_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(y_data, y_read);

  // Verify via byte_t access that bytes are correctly interleaved
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(void_spec, context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {2, 2, 3}))
          .result());

  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // x values: 10, 20, 30, 40
  // y values: 1000=0x03E8, 2000=0x07D0, 3000=0x0BB8, 4000=0x0FA0 (little-endian)
  EXPECT_EQ(tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
                {{{std::byte{10}, std::byte{0xE8}, std::byte{0x03}},
                  {std::byte{20}, std::byte{0xD0}, std::byte{0x07}}},
                 {{std::byte{30}, std::byte{0xB8}, std::byte{0x0B}},
                  {std::byte{40}, std::byte{0xA0}, std::byte{0x0F}}}}),
            byte_array);
}

TEST(Zarr3StructuredTest, ShardedReopenWithDifferentField) {
  auto context = Context::Default();

  // Create with field "x" and write some data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(ShardedStructSpec("x"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  auto x_data = tensorstore::MakeArray<uint8_t>({{42, 99}, {7, 13}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(x_data, x_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  // Reopen with field "y" -- should see fill values (zeros)
  ::nlohmann::json y_open_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_open_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  EXPECT_EQ(tensorstore::dtype_v<int16_t>, y_store.dtype());
  EXPECT_EQ(2, y_store.rank());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_data,
      tensorstore::Read(y_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());

  // Field "y" was never written, so it should be all zeros (fill value)
  auto expected_y = tensorstore::MakeArray<int16_t>({{0, 0}, {0, 0}});
  EXPECT_EQ(expected_y, y_data);

  // Now write to field "y" and verify field "x" is still intact
  auto new_y = tensorstore::MakeArray<int16_t>({{500, 600}, {700, 800}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(new_y, y_store | tensorstore::Dims(0, 1).SizedInterval(
                                              {0, 0}, {2, 2}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_read,
      tensorstore::Read(x_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(x_data, x_read);
}

TEST(Zarr3StructuredTest, ShardedVoidAccessRoundtrip) {
  auto context = Context::Default();

  // Create with field "x" to establish metadata
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(ShardedStructSpec("x"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Open with void access
  ::nlohmann::json void_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"open_as_void", true},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto void_store,
      tensorstore::Open(void_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  EXPECT_EQ(3, void_store.rank());
  EXPECT_EQ(3, void_store.domain().shape()[2]);

  // Write bytes and read them back
  auto write_bytes = tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
      {{{std::byte{0xAA}, std::byte{0xCC}, std::byte{0xBB}}}});

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(write_bytes,
                         void_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                          {0, 0, 0}, {1, 1, 3}))
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bytes_read,
      tensorstore::Read(void_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {1, 1, 3}))
          .result());

  EXPECT_EQ(write_bytes, bytes_read);
}

TEST(Zarr3StructuredTest, ShardedNoFieldRejectsOpen) {
  auto context = Context::Default();

  // Opening a sharded structured dtype without specifying a field (and without
  // open_as_void) must fail, just like the non-sharded case.
  EXPECT_THAT(
      tensorstore::Open(ShardedStructSpec(""), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      tensorstore::MatchesStatus(absl::StatusCode::kFailedPrecondition,
                                 ".*Must specify a \"field\".*"));
}

TEST(Zarr3StructuredTest, ShardedMissingFieldRejectsOpen) {
  auto context = Context::Default();

  // First create the array with field "x" to establish metadata (has fields "x"
  // and "y").
  TENSORSTORE_ASSERT_OK(
      tensorstore::Open(ShardedStructSpec("x"), context,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Opening with a non-existent field must fail.
  EXPECT_THAT(
      tensorstore::Open(ShardedStructSpec("missing_field"), context,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      tensorstore::MatchesStatus(absl::StatusCode::kFailedPrecondition,
                                 ".*Requested field.*missing_field.*"
                                 "is not one of.*"));
}

TEST(Zarr3StructuredTest, ShardedOpenAsVoidNoFieldCreate) {
  auto context = Context::Default();

  // open_as_void with no field on a sharded structured dtype must work: creates
  // the array with an extra bytes dimension, just like the non-sharded case.
  auto spec = ShardedStructSpec("");
  spec["open_as_void"] = true;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(spec, context, dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  EXPECT_EQ(tensorstore::dtype_v<tensorstore::dtypes::byte_t>,
            byte_store.dtype());
  EXPECT_EQ(3, byte_store.rank());
  EXPECT_EQ(3, byte_store.domain().shape()[2]);

  // Write some raw bytes and read them back.
  // Struct layout: x (uint8, 1 byte) + y (int16, 2 bytes) = 3 bytes
  // x values: 0x11, 0x22, 0x33, 0x44
  // y values: 100=0x0064, 200=0x00C8, 300=0x012C, 400=0x0190 (little-endian)
  auto write_bytes = tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
      {{{std::byte{0x11}, std::byte{0x64}, std::byte{0x00}},
        {std::byte{0x22}, std::byte{0xC8}, std::byte{0x00}}},
       {{std::byte{0x33}, std::byte{0x2C}, std::byte{0x01}},
        {std::byte{0x44}, std::byte{0x90}, std::byte{0x01}}}});

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(write_bytes,
                         byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                          {0, 0, 0}, {2, 2, 3}))
          .result());

  // Read back through byte_t store and verify the data.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {2, 2, 3}))
          .result());

  EXPECT_EQ(write_bytes, byte_array);

  // Reopen with field "y" and verify the typed int16 values.
  ::nlohmann::json y_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "y"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_store,
      tensorstore::Open(y_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto y_data,
      tensorstore::Read(y_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  auto expected_y = tensorstore::MakeArray<int16_t>({{100, 200}, {300, 400}});
  EXPECT_EQ(expected_y, y_data);

  // Reopen with field "x" and verify the typed uint8 values.
  ::nlohmann::json x_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"field", "x"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_store,
      tensorstore::Open(x_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto x_data,
      tensorstore::Read(x_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  auto expected_x =
      tensorstore::MakeArray<uint8_t>({{0x11, 0x22}, {0x33, 0x44}});
  EXPECT_EQ(expected_x, x_data);
}

TEST(Zarr3StructuredTest, ShardedWiderFieldRoundtrip) {
  // Use {uint8, int32} so bytes_per_element=5, which is distinct from rank=3.
  // This breaks the coincidental alignment in other tests where
  // bytes_per_element=3 and the void rank is also 3.
  auto context = Context::Default();

  ::nlohmann::json base_spec{
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"data_type",
            {{"name", "struct"},
             {"configuration",
              {{"fields",
                ::nlohmann::json::array(
                    {{{"name", "a"}, {"data_type", "uint8"}},
                     {{"name", "b"}, {"data_type", "int32"}}})}}}}},
           {"shape", {8, 8}},
           {"chunk_grid",
            {{"name", "regular"},
             {"configuration", {{"chunk_shape", {8, 8}}}}}},
           {"codecs",
            {{{"name", "sharding_indexed"},
              {"configuration",
               {{"chunk_shape", {4, 4}},
                {"codecs", {{{"name", "bytes"}}}},
                {"index_codecs",
                 {{{"name", "bytes"}}, {{"name", "crc32c"}}}}}}}}},
       }},
  };

  // Write via field "b" (int32).
  auto b_spec = base_spec;
  b_spec["field"] = "b";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto b_store,
      tensorstore::Open(b_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());
  EXPECT_EQ(2, b_store.rank());
  auto b_data = tensorstore::MakeArray<int32_t>({{100000, 200000},
                                                  {300000, 400000}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(b_data, b_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  // Write via field "a" (uint8).
  auto a_spec = base_spec;
  a_spec["field"] = "a";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto a_store,
      tensorstore::Open(a_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read_write)
          .result());
  EXPECT_EQ(2, a_store.rank());
  auto a_data = tensorstore::MakeArray<uint8_t>({{0xAA, 0xBB},
                                                  {0xCC, 0xDD}});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(a_data, a_store | tensorstore::Dims(0, 1).SizedInterval(
                                               {0, 0}, {2, 2}))
          .result());

  // Reopen as void with byte_t type -- rank should be 3, last dim should be 5 (1+4 bytes).
  auto void_spec = base_spec;
  void_spec.erase("metadata");
  void_spec["open_as_void"] = true;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_store,
      tensorstore::Open(void_spec, context,
                        dtype_v<tensorstore::dtypes::byte_t>,
                        tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());
  EXPECT_EQ(3, byte_store.rank());
  EXPECT_EQ(5, byte_store.domain().shape()[2]);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto byte_array,
      tensorstore::Read(byte_store | tensorstore::Dims(0, 1, 2).SizedInterval(
                                         {0, 0, 0}, {2, 2, 5}))
          .result());
  // Struct layout: a (uint8, 1 byte) + b (int32, 4 bytes) = 5 bytes
  // a values: 0xAA, 0xBB, 0xCC, 0xDD
  // b values: 100000=0x000186A0, 200000=0x00030D40, 300000=0x000493E0, 400000=0x00061A80
  EXPECT_EQ(
      tensorstore::MakeArray<tensorstore::dtypes::byte_t>(
          {{{std::byte{0xAA}, std::byte{0xA0}, std::byte{0x86}, std::byte{0x01},
             std::byte{0x00}},
            {std::byte{0xBB}, std::byte{0x40}, std::byte{0x0D}, std::byte{0x03},
             std::byte{0x00}}},
           {{std::byte{0xCC}, std::byte{0xE0}, std::byte{0x93}, std::byte{0x04},
             std::byte{0x00}},
            {std::byte{0xDD}, std::byte{0x80}, std::byte{0x1A}, std::byte{0x06},
             std::byte{0x00}}}}),
      byte_array);

  // Read back through typed fields to confirm consistency.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto a_read,
      tensorstore::Read(a_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(a_data, a_read);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto b_read,
      tensorstore::Read(b_store | tensorstore::Dims(0, 1).SizedInterval(
                                      {0, 0}, {2, 2}))
          .result());
  EXPECT_EQ(b_data, b_read);
}

}  // namespace
