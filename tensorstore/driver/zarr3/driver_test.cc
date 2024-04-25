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

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::Schema;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;
using ::tensorstore::internal::TestTensorStoreCreateWithSchema;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;

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
      MatchesStatus(absl::StatusCode::kNotFound,
                    "Error opening \"zarr3\" driver: "
                    "Metadata at \"prefix/zarr\\.json\" does not exist"));
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
      MatchesStatus(absl::StatusCode::kDataLoss, ".*: Invalid JSON"));
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
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Updated zarr metadata .*"));
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
      MatchesStatus(absl::StatusCode::kDataLoss, ".*: Invalid JSON"));
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

}  // namespace
