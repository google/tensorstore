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

#include "tensorstore/downsample.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/open.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::BoxView;
using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::DimensionIndex;
using ::tensorstore::DownsampleMethod;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ReadWriteMode;
using ::tensorstore::Spec;
using ::tensorstore::TensorStore;
using ::tensorstore::internal::CollectReadChunks;
using ::tensorstore::internal::MakeArrayBackedReadChunk;
using ::tensorstore::internal::MockDriver;
using ::tensorstore::internal::ReadAsIndividualChunks;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;
using ::testing::Optional;
using ::testing::Pair;

TEST(DownsampleTest, Rank1Mean) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(),
                                         MakeArray<float>({1, 2, 5, 7})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2}, DownsampleMethod::kMean));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({1.5, 6})));
}

TEST(DownsampleTest, Rank1Median) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(),
                                         MakeArray<float>({1, 2, 5, 7})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2}, DownsampleMethod::kMin));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({1, 5})));
}

TEST(DownsampleTest, Rank1Empty) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::FromArray(Context::Default(),
                             tensorstore::AllocateArray<float>({2, 0, 3})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2, 3, 2}, DownsampleMethod::kMean));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(tensorstore::AllocateArray<float>({1, 0, 2})));
}

TEST(DownsampleTest, Rank1MeanTranslated) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::FromArray(Context::Default(),
                             MakeOffsetArray<float>({1}, {1, 2, 5, 7})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2}, DownsampleMethod::kMean));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({1, 3.5, 7})));
}

TEST(DownsampleTest, Rank1Stride) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(),
                                         MakeArray<float>({1, 2, 5, 7})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2}, DownsampleMethod::kStride));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({1, 5})));
}

TEST(DownsampleTest, Rank1MeanChunked) {
  ::nlohmann::json base_spec{{"driver", "n5"},
                             {"kvstore", {{"driver", "memory"}}},
                             {"metadata",
                              {{"dataType", "uint8"},
                               {"dimensions", {11}},
                               {"blockSize", {3}},
                               {"compression", {{"type", "raw"}}}}}};
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(base_spec, context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      MakeArray<uint8_t>({0, 2, 3, 9, 1, 5, 7, 3, 4, 0, 5}), base_store));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store, tensorstore::Open({{"driver", "downsample"},
                                                 {"base", base_spec},
                                                 {"downsample_factors", {2}},
                                                 {"downsample_method", "mean"}},
                                                context)
                                  .result());
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<uint8_t>({1, 6, 3, 5, 2, 5})));
}

TEST(DownsampleTest, Rank1MeanChunkedTranslated) {
  ::nlohmann::json base_spec{{"driver", "n5"},
                             {"kvstore", {{"driver", "memory"}}},
                             {"metadata",
                              {{"dataType", "uint8"},
                               {"dimensions", {11}},
                               {"blockSize", {3}},
                               {"compression", {{"type", "raw"}}}}},
                             {"transform",
                              {
                                  {"input_inclusive_min", {1}},
                                  {"input_exclusive_max", {12}},
                                  {"output",
                                   {
                                       {{"input_dimension", 0}, {"offset", -1}},
                                   }},
                              }}};
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(base_spec, context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      MakeArray<uint8_t>({0, 2, 3, 9, 1, 5, 7, 3, 4, 0, 5}), base_store));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store, tensorstore::Open({{"driver", "downsample"},
                                                 {"base", base_spec},
                                                 {"downsample_factors", {2}},
                                                 {"downsample_method", "mean"}},
                                                context)
                                  .result());
  EXPECT_THAT(ReadAsIndividualChunks(downsampled_store).result(),
              Optional(::testing::UnorderedElementsAre(
                  Pair(MakeOffsetArray<uint8_t>({0}, {0, 2}),
                       IdentityTransform(BoxView({0}, {2}))),
                  Pair(MakeOffsetArray<uint8_t>({5}, {2}),
                       IdentityTransform(BoxView({5}, {1}))),
                  Pair(MakeOffsetArray<uint8_t>({2}, {5, 6, 4}),
                       IdentityTransform(BoxView({2}, {3}))))));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<uint8_t>({0, 2, 5, 6, 4, 2})));
}

TEST(DownsampleTest, Rank1MeanChunkedIndexArray) {
  ::nlohmann::json base_spec{{"driver", "n5"},
                             {"kvstore", {{"driver", "memory"}}},
                             {"metadata",
                              {{"dataType", "uint8"},
                               {"dimensions", {11}},
                               {"blockSize", {3}},
                               {"compression", {{"type", "raw"}}}}}};
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(base_spec, context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      MakeArray<uint8_t>({0, 2, 3, 9, 1, 5, 7, 3, 4, 0, 5}), base_store));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store, tensorstore::Open({{"driver", "downsample"},
                                                 {"base", base_spec},
                                                 {"downsample_factors", {2}},
                                                 {"downsample_method", "mean"}},
                                                context)
                                  .result());
  EXPECT_THAT(tensorstore::Read(downsampled_store |
                                tensorstore::Dims(0).IndexArraySlice(
                                    MakeArray<Index>({0, 3, 2})))
                  .result(),
              Optional(MakeArray<uint8_t>({1, 5, 3})));
}

TEST(DownsampleTest, JsonSpecArray) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "downsample"},
                                     {"base", base_spec},
                                     {"downsample_factors", {2}},
                                     {"downsample_method", "mean"}})
                      .result());
  EXPECT_THAT(tensorstore::Read(store).result(),
              Optional(MakeArray<float>({1.5, 3.5})));
}

TEST(DownsampleTest, JsonSpecArrayRank0) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", 42},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open({{"driver", "downsample"},
                         {"base", base_spec},
                         {"downsample_factors", ::nlohmann::json::array_t{}},
                         {"downsample_method", "mean"}})
          .result());
  EXPECT_THAT(tensorstore::Read(store).result(),
              Optional(tensorstore::MakeScalarArray<float>(42)));
}

TEST(DownsampleTest, JsonSpecErrorMissingBase) {
  EXPECT_THAT(
      tensorstore::Open({
                            {"driver", "downsample"},
                            {"downsample_factors", {2}},
                            {"downsample_method", "mean"},
                        })
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"base\".*"));
}

TEST(DownsampleTest, JsonSpecErrorMissingDownsampleFactors) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "downsample"},
                                    {"base", base_spec},
                                    {"downsample_method", "mean"},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"downsample_factors\".*"));
}

TEST(DownsampleTest, JsonSpecErrorDownsampleFactorsInvalidRank) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "downsample"},
                                    {"base", base_spec},
                                    {"downsample_method", "mean"},
                                    {"downsample_factors", {2, 3}},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"downsample_factors\": .*rank.*"));
}

TEST(DownsampleTest, JsonSpecErrorDownsampleFactorsZero) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(
      tensorstore::Open({
                            {"driver", "downsample"},
                            {"base", base_spec},
                            {"downsample_method", "mean"},
                            {"downsample_factors", {0}},
                        })
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"downsample_factors\":.*Expected .*, but received: 0"));
}

TEST(DownsampleTest, JsonSpecErrorDownsampleFactorsNegative) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "downsample"},
                                    {"base", base_spec},
                                    {"downsample_method", "mean"},
                                    {"downsample_factors", {-2}},
                                })
                  .result(),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".*\"downsample_factors\":.*Expected .*, but received: -2"));
}

TEST(DownsampleTest, JsonSpecErrorMissingDownsampleMethod) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "downsample"},
                                    {"base", base_spec},
                                    {"downsample_factors", {2}},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"downsample_method\".*"));
}

TEST(DownsampleTest, JsonSpecErrorInvalidDownsampleMethod) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "downsample"},
                                    {"base", base_spec},
                                    {"downsample_factors", {2}},
                                    {"downsample_method", 42},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"downsample_method\".*42.*"));
}

TEST(DownsampleTest, ErrorOpenWriteOnly) {
  ::nlohmann::json base_spec{
      {"driver", "array"},
      {"dtype", "float32"},
      {"array", {1, 2, 3, 4}},
  };
  for (auto mode : {ReadWriteMode::write, ReadWriteMode::read_write}) {
    SCOPED_TRACE(tensorstore::StrCat("mode=", mode));
    EXPECT_THAT(tensorstore::Open(
                    {
                        {"driver", "downsample"},
                        {"base", base_spec},
                        {"downsample_factors", {2}},
                        {"downsample_method", "mean"},
                    },
                    mode)
                    .result(),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*: only reading is supported"));
  }
}

TEST(DownsampleTest, AdapterErrorNegativeDownsampleFactor) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(),
                                         MakeArray<float>({1, 2, 5, 7})));
  EXPECT_THAT(
      tensorstore::Downsample(store, {-2}, DownsampleMethod::kMean),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Downsample factors \\{-2\\} are not all positive"));
}

TEST(DownsampleTest, AdapterErrorZeroDownsampleFactor) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(),
                                         MakeArray<float>({1, 2, 5, 7})));
  EXPECT_THAT(tensorstore::Downsample(store, {0}, DownsampleMethod::kMean),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Downsample factors \\{0\\} are not all positive"));
}

TEST(DownsampleTest, AdapterErrorDownsampleFactorsRankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      TensorStore<float> store,
      tensorstore::FromArray(Context::Default(),
                             MakeArray<float>({1, 2, 5, 7})));
  EXPECT_THAT(
      tensorstore::Downsample(store, {2, 2}, DownsampleMethod::kMean),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Number of downsample factors \\(2\\) does not match "
                    "TensorStore rank \\(1\\)"));
}

TEST(DownsampleTest, AdapterErrorDataType) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::FromArray(Context::Default(),
                             MakeArray<std::string>({"a", "b", "c"})));
  TENSORSTORE_EXPECT_OK(
      tensorstore::Downsample(store, {2}, DownsampleMethod::kStride));
  EXPECT_THAT(tensorstore::Downsample(store, {2}, DownsampleMethod::kMean),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Downsample method \"mean\" does not support "
                            "data type \"string\""));
}

TEST(DownsampleTest, AdapterErrorWriteOnly) {
  tensorstore::TensorStore<float, 1> store;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      store,
      tensorstore::FromArray(Context::Default(), MakeArray<float>({1, 2, 3})));
  store = tensorstore::ModeCast<ReadWriteMode::write, tensorstore::unchecked>(
      std::move(store));
  EXPECT_THAT(tensorstore::Downsample(store, {2}, DownsampleMethod::kMean),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot downsample write-only TensorStore"));
}

// Tests that a read error from the base TensorStore is handled correctly.
TEST(DownsampleTest, ReadError) {
  auto mock_driver = MockDriver::Make(tensorstore::ReadWriteMode::dynamic,
                                      tensorstore::dtype_v<float>, 1);
  auto mock_store = mock_driver->Wrap(tensorstore::IdentityTransform<1>({10}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(mock_store, {2}, DownsampleMethod::kMean));
  auto read_future = tensorstore::Read(downsampled_store);
  {
    auto read_req = mock_driver->read_requests.pop();
    EXPECT_EQ(tensorstore::IdentityTransform<1>({10}), read_req.transform);
    tensorstore::execution::set_error(
        tensorstore::FlowSingleReceiver{std::move(read_req.receiver)},
        absl::UnknownError("read error"));
  }
  EXPECT_THAT(read_future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "read error"));
}

TEST(DownsampleTest, CancelRead) {
  auto mock_driver = MockDriver::Make(tensorstore::ReadWriteMode::dynamic,
                                      tensorstore::dtype_v<float>, 1);
  auto mock_store = mock_driver->Wrap(tensorstore::IdentityTransform<1>({10}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(mock_store, {2}, DownsampleMethod::kMean));
  auto read_future = tensorstore::Read(downsampled_store);
  auto canceled = std::make_shared<bool>(false);
  {
    auto read_req = mock_driver->read_requests.pop();
    tensorstore::execution::set_starting(read_req.receiver,
                                         [canceled] { *canceled = true; });
    read_future = {};
    EXPECT_EQ(true, *canceled);
    tensorstore::execution::set_done(read_req.receiver);
    tensorstore::execution::set_stopping(read_req.receiver);
  }
}

// Tests the case where an independently-emitted chunk is the final chunk when a
// `data_buffer_` was previously allocated, and causes buffered chunks to be
// emitted.
TEST(DownsampleTest, IndependentChunkCompletesBufferedChunk) {
  auto mock_driver = MockDriver::Make(tensorstore::ReadWriteMode::dynamic,
                                      tensorstore::dtype_v<float>, 1);
  auto mock_store = mock_driver->Wrap(tensorstore::IdentityTransform<1>({4}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(mock_store, {2}, DownsampleMethod::kMean));
  auto read_future = tensorstore::Read(downsampled_store);
  {
    auto read_req = mock_driver->read_requests.pop();
    tensorstore::execution::set_starting(read_req.receiver, [] {});
    // Send chunk with index transform that won't be downsampled independently.
    tensorstore::execution::set_value(
        read_req.receiver, MakeArrayBackedReadChunk(MakeArray<float>({0, 1})),
        (tensorstore::IdentityTransform(1) |
         tensorstore::Dims(0).IndexArraySlice(MakeArray<Index>({0, 1})))
            .value());
    // Send chunk that can be downsampled independently.
    tensorstore::execution::set_value(
        read_req.receiver,
        MakeArrayBackedReadChunk(MakeOffsetArray<float>({2}, {2, 3})),
        tensorstore::IdentityTransform(BoxView<1>({2}, {2})));
    tensorstore::execution::set_done(read_req.receiver);
    tensorstore::execution::set_stopping(read_req.receiver);
  }
  ASSERT_TRUE(read_future.ready());
  EXPECT_THAT(read_future.result(), Optional(MakeArray<float>({0.5, 2.5})));
}

// Tests that a read error from the base TensorStore is handled correctly.
TEST(DownsampleTest, EmptyChunk) {
  auto mock_driver = MockDriver::Make(tensorstore::ReadWriteMode::dynamic,
                                      tensorstore::dtype_v<float>, 1);
  auto mock_store = mock_driver->Wrap(tensorstore::IdentityTransform<1>({10}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(mock_store, {2}, DownsampleMethod::kMean));
  auto read_future = tensorstore::Read(downsampled_store);
  {
    auto read_req = mock_driver->read_requests.pop();
    EXPECT_EQ(tensorstore::IdentityTransform<1>({10}), read_req.transform);
    tensorstore::execution::set_error(
        tensorstore::FlowSingleReceiver{std::move(read_req.receiver)},
        absl::UnknownError("read error"));
  }
  EXPECT_THAT(read_future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "read error"));
}

// Tests reading part of a `ReadChunk` returned by the downsample driver using
// an index transform.  This results in additional synthetic dimensions from
// `PropagateIndexTransformDownsampling` that have to be handled by
// `DownsampledNDIterator`.
TEST(DownsampleTest, ReadChunkWithIndexTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::FromArray(Context::Default(), MakeArray<float>({
                                                     {1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10},
                                                     {11, 12, 13, 14, 15},
                                                 })));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Downsample(store, {2, 3}, DownsampleMethod::kMean));
  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({
                  {4.5, 7},
                  {12, 14.5},
              })));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chunks, CollectReadChunks(downsampled_store).result());
  ASSERT_THAT(chunks,
              ::testing::ElementsAre(Pair(
                  ::testing::_, tensorstore::IdentityTransform<2>({2, 2}))));
  auto& entry = chunks[0];
  // Test with an index transform that adds one synthetic dimension.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto transform, entry.second | tensorstore::Dims(0).IndexArraySlice(
                                           MakeArray<Index>({0, 1, 1, 0})));
    auto target_array = tensorstore::AllocateArray<float>({4, 2});
    TENSORSTORE_ASSERT_OK(tensorstore::internal::CopyReadChunk(
        entry.first.impl, transform,
        tensorstore::TransformedArray(target_array)));
    EXPECT_EQ(MakeArray<float>({
                  {4.5, 7},
                  {12, 14.5},
                  {12, 14.5},
                  {4.5, 7},
              }),
              target_array);
  }

  // Test with an index transform that adds two synthetic dimensions.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto transform, entry.second | tensorstore::Dims(0, 1).IndexArraySlice(
                                           MakeArray<Index>({0, 1, 1}),
                                           MakeArray<Index>({0, 0, 1})));
    auto target_array = tensorstore::AllocateArray<float>({3});
    TENSORSTORE_ASSERT_OK(tensorstore::internal::CopyReadChunk(
        entry.first.impl, transform,
        tensorstore::TransformedArray(target_array)));
    // Note that the last entry is the average of {14, 15, 15} because of how
    // `PropagateIndexTransformDownsampling` handles index arrays.
    EXPECT_EQ(MakeArray<float>({4.5, 12, 14.666666666666666}), target_array);
  }
}

// Tests that an error from the base `NDIterable` is handled correctly.
TEST(DownsampleTest, ConvertError) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store,
      tensorstore::Open({
                            {"driver", "downsample"},
                            {"base",
                             {
                                 {"driver", "cast"},
                                 {"base",
                                  {
                                      {"driver", "array"},
                                      {"dtype", "json"},
                                      {"array", {1, "abc", 2}},
                                  }},
                                 {"dtype", "uint8"},
                             }},
                            {"downsample_method", "mean"},
                            {"downsample_factors", {2}},
                        })
          .result());
  auto dest = tensorstore::MakeArray<uint8_t>({0, 0});
  EXPECT_THAT(
      tensorstore::Read(downsampled_store, dest).result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected integer in the range \\[0, 255\\], but received: \"abc\""));
  EXPECT_EQ(dest, MakeArray<uint8_t>({0, 0}));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "downsample";
  options.create_spec = {
      {"driver", "downsample"},
      {"base",
       {
           {"driver", "array"},
           {"dtype", "float32"},
           {"array", {{1, 2, 3}, {4, 5, 6}}},
       }},
      {"downsample_method", "mean"},
      {"downsample_factors", {1, 2}},
  };
  options.full_spec = {
      {"driver", "downsample"},
      {"base",
       {
           {"driver", "array"},
           {"array", {{1, 2, 3}, {4, 5, 6}}},
           {"transform",
            {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {2, 3}}}},
       }},
      {"dtype", "float32"},
      {"downsample_method", "mean"},
      {"downsample_factors", {1, 2}},
      {"transform",
       {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {2, 2}}}},
  };
  options.minimal_spec = options.full_spec;
  options.check_not_found_before_create = false;
  options.check_not_found_before_commit = false;
  options.supported_transaction_modes = {};
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

TEST(DownsampleTest, Spec) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, Spec::FromJson({
                                                  {"driver", "array"},
                                                  {"dtype", "float32"},
                                                  {"array", {1, 2, 3, 4}},
                                              }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_spec,
      tensorstore::Downsample(spec, {2}, DownsampleMethod::kMean));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_store, tensorstore::Open(downsampled_spec).result());

  EXPECT_THAT(tensorstore::Read(downsampled_store).result(),
              Optional(MakeArray<float>({1.5, 3.5})));

  EXPECT_THAT(
      downsampled_spec.ToJson(),
      Optional(MatchesJson(::nlohmann::json({
          {"driver", "downsample"},
          {"dtype", "float32"},
          {"base",
           {
               {"driver", "array"},
               {"array", {1, 2, 3, 4}},
               {"transform",
                {{"input_inclusive_min", {0}}, {"input_exclusive_max", {4}}}},
           }},
          {"downsample_factors", {2}},
          {"downsample_method", "mean"},
          {"transform",
           {{"input_inclusive_min", {0}}, {"input_exclusive_max", {2}}}},
      }))));
}

TEST(DownsampleTest, ChunkLayout) {
  ::nlohmann::json base_spec{
      {"driver", "n5"},
      {"kvstore", {{"driver", "memory"}}},
      {"metadata",
       {{"dataType", "uint8"},
        {"dimensions", {100, 200}},
        {"blockSize", {10, 21}},
        {"compression", {{"type", "raw"}}}}},
  };
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(base_spec, context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "downsample"},
                                     {"base", base_spec},
                                     {"downsample_factors", {2, 3}},
                                     {"downsample_method", "mean"}},
                                    context)
                      .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout,
                                   ChunkLayout::FromJson({
                                       {"write_chunk", {{"shape", {5, 7}}}},
                                       {"read_chunk", {{"shape", {5, 7}}}},
                                       {"grid_origin", {0, 0}},
                                       {"inner_order", {1, 0}},
                                   }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(SpecSchemaTest, Basic) {
  TestSpecSchema(
      {
          {"driver", "downsample"},
          {"downsample_method", "mean"},
          {"downsample_factors", {1, 2}},
          {"base",
           {
               {"driver", "array"},
               {"array", {{1, 2, 3, 4}, {5, 6, 7, 8}}},
               {"dtype", "float32"},
           }},
          {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
      },
      {
          {"rank", 2},
          {"dtype", "float32"},
          {"domain", {{"shape", {2, 2}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(TensorStoreCreateCheckSchemaTest, Basic) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "downsample"},
          {"downsample_method", "mean"},
          {"downsample_factors", {1, 2}},
          {"base",
           {
               {"driver", "array"},
               {"array", {{1, 2, 3, 4}, {5, 6, 7, 8}}},
               {"dtype", "float32"},
           }},
          {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
      },
      {
          {"rank", 2},
          {"dtype", "float32"},
          {"domain", {{"shape", {2, 2}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(DownsampleTest, DomainSpecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_spec,
                                   tensorstore::Spec::FromJson({
                                       {"driver", "zarr"},
                                       {"kvstore", {{"driver", "memory"}}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_spec,
      tensorstore::Downsample(base_spec, {2, 1}, DownsampleMethod::kMean));
  TENSORSTORE_ASSERT_OK(
      downsampled_spec.Set(tensorstore::Schema::Shape({10, 10})));
  EXPECT_THAT(downsampled_spec.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"driver", "downsample"},
                  {"base",
                   {
                       {"driver", "zarr"},
                       {"kvstore", {{"driver", "memory"}}},
                       {"schema",
                        {
                            {"domain",
                             {{"inclusive_min", {{"-inf"}, 0}},
                              {"exclusive_max", {{"+inf"}, 10}}}},
                        }},
                       {"transform",
                        {
                            {"input_exclusive_max", {{"+inf"}, {10}}},
                            {"input_inclusive_min", {0, 0}},
                        }},
                   }},
                  {"downsample_factors", {2, 1}},
                  {"downsample_method", "mean"},
                  {"schema",
                   {
                       {"domain",
                        {
                            {"inclusive_min", {0, 0}},
                            {"exclusive_max", {10, 10}},
                        }},
                   }},
                  {"transform",
                   {{"input_exclusive_max", {10, 10}},
                    {"input_inclusive_min", {0, 0}}}},
              })));
}

TEST(DownsampleTest, FillValueNotSpecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(
          {
              {"driver", "zarr"},
              {"kvstore", {{"driver", "memory"}}},
              {"metadata", {{"dtype", {{"x", "<u4", {4, 3}}}}}},
          },
          tensorstore::OpenMode::create,
          tensorstore::Schema::Shape({100, 4, 3}))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Downsample(base_store, {1, 2, 1},
                              tensorstore::DownsampleMethod::kMean));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto fill_value, store.fill_value());
  EXPECT_FALSE(fill_value.valid());
}

TEST(DownsampleTest, FillValueSpecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(
          {
              {"driver", "zarr"},
              {"kvstore", {{"driver", "memory"}}},
              {"metadata", {{"dtype", {{"x", "<u4", {4, 3}}}}}},
          },
          tensorstore::OpenMode::create,
          tensorstore::Schema::Shape({100, 4, 3}),
          tensorstore::Schema::FillValue(tensorstore::MakeArray<uint32_t>(
              {{1, 2, 3}, {40, 50, 60}, {7, 8, 9}, {100, 110, 120}})))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Downsample(base_store, {1, 2, 1},
                              tensorstore::DownsampleMethod::kMean));
  EXPECT_THAT(store.fill_value(),
              ::testing::Optional(tensorstore::MakeArray<uint32_t>(
                  {{20, 26, 32}, {54, 59, 64}})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transformed, store | tensorstore::Dims(2).SizedInterval(1, 2));
  EXPECT_THAT(transformed.fill_value(),
              ::testing::Optional(
                  tensorstore::MakeArray<uint32_t>({{26, 32}, {59, 64}})));
}

TEST(DownsampleTest, DimensionUnits) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::FromArray(
          tensorstore::Context::Default(),
          tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}}), {"4nm", "5nm"}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Downsample(base_store, {1, 2},
                              tensorstore::DownsampleMethod::kMean));
  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("10nm"))));
}

}  // namespace
