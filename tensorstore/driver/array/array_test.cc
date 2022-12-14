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

#include "tensorstore/driver/array/array.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ChunkLayout;
using ::tensorstore::Context;
using ::tensorstore::CopyProgress;
using ::tensorstore::CopyProgressFunction;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::offset_origin;
using ::tensorstore::ReadProgress;
using ::tensorstore::ReadProgressFunction;
using ::tensorstore::ReadWriteMode;
using ::tensorstore::TensorStore;
using ::tensorstore::WriteProgress;
using ::tensorstore::WriteProgressFunction;
using ::tensorstore::zero_origin;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;

constexpr const char kMismatchRE[] = ".* mismatch with target dimension .*";
constexpr const char kOutsideValidRangeRE[] = ".* is outside valid range .*";

namespace driver_tests {

TEST(ArrayDriverTest, Read) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto transformed_driver =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<ReadProgress> read_progress;
  auto dest_array = tensorstore::AllocateArray<int>(array.domain());
  TENSORSTORE_ASSERT_OK(tensorstore::internal::DriverRead(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver,
      /*target=*/dest_array,
      {/*.progress_function=*/[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }}));
  EXPECT_EQ(array, dest_array);
  EXPECT_THAT(read_progress, ::testing::ElementsAre(ReadProgress{6, 6}));
}

TEST(ArrayDriverTest, ReadIntoNewArray) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto transformed_driver =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<ReadProgress> read_progress;
  EXPECT_THAT(
      tensorstore::internal::DriverReadIntoNewArray(
          /*executor=*/tensorstore::InlineExecutor{},
          /*source=*/transformed_driver,
          /*target_dtype=*/array.dtype(),
          /*target_layout_order=*/tensorstore::c_order,
          {/*.progress_function=*/[&read_progress](ReadProgress progress) {
            read_progress.push_back(progress);
          }})
          .result(),
      ::testing::Optional(array));
  EXPECT_THAT(read_progress, ::testing::ElementsAre(ReadProgress{6, 6}));
}

/// Tests calling Read with a source domain that does not match the destination
/// domain.
TEST(ArrayDriverTest, ReadDomainMismatch) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto transformed_driver =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<ReadProgress> read_progress;
  auto dest_array =
      tensorstore::AllocateArray<int>(tensorstore::BoxView({1, 2}, {3, 3}));
  auto future = tensorstore::internal::DriverRead(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver,
      /*target=*/dest_array,
      {/*.progress_function=*/[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }});
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(read_progress, ::testing::ElementsAre());
}

/// Tests calling Read with an index array containing out-of-bounds indices.
TEST(ArrayDriverTest, ReadCopyTransformError) {
  auto array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto context = Context::Default();
  auto transformed_driver =
      tensorstore::internal::MakeArrayDriver<zero_origin>(context, array)
          .value();
  std::vector<ReadProgress> read_progress;
  auto dest_array =
      ChainResult(tensorstore::AllocateArray<int>({1}),
                  tensorstore::Dims(0).IndexArraySlice(
                      tensorstore::MakeArray<Index>({0, 1, 2, 3})))
          .value();

  auto future = tensorstore::internal::DriverRead(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver,
      /*target=*/dest_array,
      {/*.progress_function=*/[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }});
  // Error occurs due to the invalid index of 1 in the index array, which is
  // validated when copying from the ReadChunk to the target array.
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kOutOfRange,
                                             ".* is outside valid range .*"));
  EXPECT_THAT(read_progress, ::testing::ElementsAre());
}

TEST(ArrayDriverTest, Write) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto [driver, transform, transaction] =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<WriteProgress> write_progress;
  auto write_result = tensorstore::internal::DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/tensorstore::MakeOffsetArray<int>({2, 3}, {{7, 8}}),
      /*target=*/
      {driver, ChainResult(transform, tensorstore::Dims(0, 1).SizedInterval(
                                          {2, 3}, {1, 2}))
                   .value()},
      {/*.progress_function=*/[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 7, 8}}),
            array);
  EXPECT_THAT(write_progress, ::testing::ElementsAre(WriteProgress{2, 2, 0},
                                                     WriteProgress{2, 2, 2}));
}

/// Tests calling Write with a source domain that does not match the destination
/// domain.
TEST(ArrayDriverTest, WriteDomainMismatch) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto [driver, transform, transaction] =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<WriteProgress> write_progress;
  auto write_result = tensorstore::internal::DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/tensorstore::MakeArray<int>({{7, 8, 9, 10, 11}}),
      /*target=*/
      {driver, ChainResult(transform, tensorstore::Dims(0, 1).SizedInterval(
                                          {2, 3}, {1, 2}))
                   .value()},
      {/*.progress_function=*/[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  EXPECT_THAT(write_result.copy_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(write_result.commit_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(write_progress, ::testing::ElementsAre());
}

TEST(ArrayDriverTest, Copy) {
  auto context = Context::Default();
  auto array_a =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto [driver_a, transform_a, transaction_a] =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array_a)
          .value();
  auto array_b =
      tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 7, 7}, {7, 7, 7}});
  auto [driver_b, transform_b, transaction_b] =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array_b)
          .value();
  std::vector<CopyProgress> progress;
  auto write_result = tensorstore::internal::DriverCopy(
      /*executor=*/tensorstore::InlineExecutor{}, /*source=*/
      {driver_a,
       ChainResult(transform_a, tensorstore::Dims(0, 1).TranslateSizedInterval(
                                    {1, 2}, {2, 2}, {1, 2}))
           .value()},
      /*target=*/
      {driver_b,
       ChainResult(transform_b, tensorstore::Dims(0, 1).TranslateSizedInterval(
                                    {1, 5}, {2, 2}))
           .value()},
      {/*.progress_function=*/[&progress](CopyProgress p) {
        progress.push_back(p);
      }});
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 1, 3}, {7, 4, 6}}),
            array_b);
  EXPECT_THAT(progress, ::testing::ElementsAre(CopyProgress{4, 4, 0, 0},
                                               CopyProgress{4, 4, 4, 0},
                                               CopyProgress{4, 4, 4, 4}));
}

TEST(ArrayDriverTest, CopyDomainMismatch) {
  auto context = Context::Default();
  auto array_a =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto transformed_driver_a =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array_a)
          .value();
  auto array_b =
      tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 7, 7, 7}, {7, 7, 7, 7}});
  auto transformed_driver_b =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array_b)
          .value();
  std::vector<CopyProgress> progress;
  auto write_result = tensorstore::internal::DriverCopy(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver_a,
      /*target=*/transformed_driver_b,
      {/*.progress_function=*/[&progress](CopyProgress p) {
        progress.push_back(p);
      }});
  EXPECT_THAT(write_result.copy_future,
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_EQ(write_result.copy_future.status(),
            write_result.commit_future.status());
  EXPECT_THAT(progress, ::testing::ElementsAre());
}

}  // namespace driver_tests

namespace frontend_tests {

TEST(ApplyIndexTransformTest, Basic) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();
  auto store2 =
      ApplyIndexTransform(tensorstore::Dims(0, 1).SizedInterval({1, 2}, {2, 2}),
                          store)
          .value();
  EXPECT_EQ(tensorstore::BoxView({1, 2}, {2, 2}), store2.domain().box());
}

TEST(FromArrayTest, ResolveBounds) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();

  auto store2 = ResolveBounds(store).value();
  EXPECT_EQ(array.domain(), store2.domain().box());
  EXPECT_EQ(tensorstore::dtype_v<int>, store2.dtype());
  EXPECT_THAT(store2.domain().labels(), ::testing::ElementsAre("", ""));
}

TEST(FromArrayTest, Read) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();
  static_assert(std::is_same_v<TensorStore<int, 2, ReadWriteMode::read_write>,
                               decltype(store)>);
  std::vector<ReadProgress> read_progress;
  auto dest_array = tensorstore::AllocateArray<int>(array.domain());
  TENSORSTORE_ASSERT_OK(
      Read(store, dest_array,
           ReadProgressFunction{[&read_progress](ReadProgress progress) {
             read_progress.push_back(progress);
           }}));
  EXPECT_EQ(array, dest_array);
  EXPECT_THAT(read_progress, ::testing::ElementsAre(ReadProgress{6, 6}));
}

TEST(FromArrayTest, ReadBroadcast) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();
  auto dest_array = tensorstore::AllocateArray<int>({2, 2, 3});
  auto future = Read(store, dest_array);
  TENSORSTORE_EXPECT_OK(future);
  EXPECT_EQ(tensorstore::MakeArray<int>(
                {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}}),
            dest_array);
}

TEST(FromArrayTest, ReadAlignByLabel) {
  //     | x=1 | x=2 | x=3
  // y=2 |  1  |  2  |  3
  // y=3 |  4  |  5  |  6
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();
  auto dest_array = tensorstore::AllocateArray<int>({3, 2});
  tensorstore::Future<void> future =
      Read(ChainResult(store, tensorstore::AllDims().Label("x", "y")),
           ChainResult(dest_array, tensorstore::AllDims().Label("y", "x")));
  TENSORSTORE_EXPECT_OK(future);
  EXPECT_EQ(tensorstore::MakeArray<int>({{1, 4}, {2, 5}, {3, 6}}), dest_array);
}

TEST(FromArrayTest, ReadIntoNewArray) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<ReadProgress> read_progress;
  auto future = tensorstore::Read(
      store, /*options=*/{
          tensorstore::c_order,
          ReadProgressFunction{[&read_progress](ReadProgress progress) {
            read_progress.push_back(progress);
          }}});
  TENSORSTORE_EXPECT_OK(future);
  auto read_array = future.value();
  EXPECT_EQ(array, read_array);
  EXPECT_THAT(read_progress, ::testing::ElementsAre(ReadProgress{6, 6}));
}

/// Tests calling Read with a source domain that does not match the destination
/// domain.
TEST(FromArrayTest, ReadDomainMismatch) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<ReadProgress> read_progress;
  auto dest_array =
      tensorstore::AllocateArray<int>(tensorstore::BoxView({1, 2}, {3, 3}));
  auto future =
      Read(store, dest_array,
           ReadProgressFunction{[&read_progress](ReadProgress progress) {
             read_progress.push_back(progress);
           }});
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(read_progress, ::testing::ElementsAre());
}

/// Tests calling Read with an index array containing out-of-bounds indices.
TEST(FromArrayTest, ReadCopyTransformError) {
  auto array = tensorstore::MakeArray<int>({1, 2, 3, 4});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<ReadProgress> read_progress;
  auto future = tensorstore::Read(
      store,
      ChainResult(tensorstore::AllocateArray<int>({1}),
                  tensorstore::Dims(0).IndexArraySlice(
                      tensorstore::MakeArray<Index>({0, 1, 2, 3}))),
      ReadProgressFunction{[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }});
  EXPECT_THAT(future.result(), MatchesStatus(absl::StatusCode::kOutOfRange,
                                             kOutsideValidRangeRE));
  EXPECT_THAT(read_progress, ::testing::ElementsAre());
}

TEST(FromArrayTest, Write) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<WriteProgress> write_progress;
  auto write_result = tensorstore::Write(
      tensorstore::MakeOffsetArray<int>({2, 3}, {{7, 8}}),
      ChainResult(store, tensorstore::Dims(0, 1).SizedInterval({2, 3}, {1, 2})),
      WriteProgressFunction{[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 7, 8}}),
            array);
  EXPECT_THAT(write_progress, ::testing::ElementsAre(WriteProgress{2, 2, 0},
                                                     WriteProgress{2, 2, 2}));
}

TEST(FromArrayTest, WriteBroadcast) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  auto write_result = tensorstore::Write(
      tensorstore::MakeScalarArray<int>(42),
      ChainResult(store,
                  tensorstore::Dims(0, 1).SizedInterval({2, 3}, {1, 2})));
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 42, 42}}),
            array);
}

/// Tests calling Write with a source domain that does not match the destination
/// domain.
TEST(FromArrayTest, WriteDomainMismatch) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<WriteProgress> write_progress;
  auto write_result = tensorstore::Write(
      tensorstore::MakeOffsetArray<int>({1, 3}, {{7, 8, 9, 10}}),
      ChainResult(store, tensorstore::Dims(0, 1).SizedInterval({2, 3}, {1, 2})),
      WriteProgressFunction{[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  EXPECT_THAT(write_result.copy_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(write_result.commit_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(write_progress, ::testing::ElementsAre());
}

TEST(FromArrayTest, Copy) {
  auto context = Context::Default();
  auto array_a =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto store_a = tensorstore::FromArray(context, array_a);
  auto array_b =
      tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 7, 7}, {7, 7, 7}});
  auto store_b = tensorstore::FromArray(context, array_b);
  std::vector<CopyProgress> progress;
  auto write_result = tensorstore::Copy(
      ChainResult(store_a, tensorstore::Dims(0, 1).TranslateSizedInterval(
                               {1, 2}, {2, 2}, {1, 2})),
      ChainResult(store_b, tensorstore::Dims(0, 1).TranslateSizedInterval(
                               {1, 5}, {2, 2})),
      CopyProgressFunction{
          [&progress](CopyProgress p) { progress.push_back(p); }});
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 1, 3}, {7, 4, 6}}),
            array_b);
  EXPECT_THAT(progress, ::testing::ElementsAre(CopyProgress{4, 4, 0, 0},
                                               CopyProgress{4, 4, 4, 0},
                                               CopyProgress{4, 4, 4, 4}));
}

TEST(FromArrayTest, CopyBroadcast) {
  auto context = Context::Default();
  auto array_a =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto store_a = tensorstore::FromArray(context, array_a);
  auto array_b =
      tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 7, 7}, {7, 7, 7}});
  auto store_b = tensorstore::FromArray(context, array_b);
  auto write_result = tensorstore::Copy(
      ChainResult(store_a, tensorstore::Dims(1).SizedInterval(2, 2, 2),
                  tensorstore::Dims(0).IndexSlice(1)),
      ChainResult(store_b,
                  tensorstore::Dims(0, 1).SizedInterval({1, 5}, {2, 2})));
  TENSORSTORE_EXPECT_OK(write_result.copy_future);
  TENSORSTORE_EXPECT_OK(write_result.commit_future);
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 4}, {{7, 1, 3}, {7, 1, 3}}),
            array_b);
}

TEST(FromArrayTest, CopyDomainMismatch) {
  auto context = Context::Default();
  auto array_a =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto store_a = tensorstore::FromArray(context, array_a);
  auto array_b = tensorstore::MakeOffsetArray<int>(
      {1, 4}, {{7, 7, 7, 7, 7}, {7, 7, 7, 7, 7}});
  auto store_b = tensorstore::FromArray(context, array_b);
  std::vector<CopyProgress> progress;
  auto write_result = tensorstore::Copy(
      store_a, store_b, CopyProgressFunction{[&progress](CopyProgress p) {
        progress.push_back(p);
      }});
  EXPECT_THAT(write_result.copy_future.result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_EQ(write_result.copy_future.status(),
            write_result.commit_future.status());
  EXPECT_THAT(progress, ::testing::ElementsAre());
}

TEST(FromArrayTest, ReadDataTypeConversion) {
  auto context = Context::Default();
  auto source = tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  auto dest = tensorstore::AllocateArray<std::int64_t>({3});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Read(tensorstore::FromArray(context, source), dest));
  EXPECT_EQ(dest, tensorstore::MakeArray<std::int64_t>({1, 2, 3}));
}

TEST(FromArrayTest, ReadInvalidDataTypeConversion) {
  auto context = Context::Default();
  tensorstore::SharedArray<void> source =
      tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  tensorstore::SharedArray<void> dest =
      tensorstore::AllocateArray<std::int16_t>({3});
  EXPECT_THAT(
      tensorstore::Read(tensorstore::FromArray(context, source), dest).result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Explicit data type conversion required to convert int32 -> int16"));
}

TEST(FromArrayTest, WriteDataTypeConversion) {
  auto context = Context::Default();
  auto source = tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  auto dest = tensorstore::AllocateArray<std::int64_t>({3});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(source, tensorstore::FromArray(context, dest)));
  EXPECT_EQ(dest, tensorstore::MakeArray<std::int64_t>({1, 2, 3}));
}

TEST(FromArrayTest, WriteInvalidDataTypeConversion) {
  auto context = Context::Default();
  tensorstore::SharedArray<void> source =
      tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  tensorstore::SharedArray<void> dest =
      tensorstore::AllocateArray<std::int16_t>({3});
  EXPECT_THAT(
      tensorstore::Write(source, tensorstore::FromArray(context, dest))
          .commit_future.result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Explicit data type conversion required to convert int32 -> int16"));
}

TEST(FromArrayTest, CopyDataTypeConversion) {
  auto context = Context::Default();
  auto source = tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  auto dest = tensorstore::AllocateArray<std::int64_t>({3});
  TENSORSTORE_EXPECT_OK(
      tensorstore::Copy(tensorstore::FromArray(context, source),
                        tensorstore::FromArray(context, dest)));
  EXPECT_EQ(dest, tensorstore::MakeArray<std::int64_t>({1, 2, 3}));
}

TEST(FromArrayTest, CopyInvalidDataTypeConversion) {
  auto context = Context::Default();
  tensorstore::SharedArray<void> source =
      tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  tensorstore::SharedArray<void> dest =
      tensorstore::AllocateArray<std::int16_t>({3});
  EXPECT_THAT(
      tensorstore::Copy(tensorstore::FromArray(context, source),
                        tensorstore::FromArray(context, dest))
          .commit_future.result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Explicit data type conversion required to convert int32 -> int16"));
}

TEST(FromArrayTest, ChunkLayoutCOrder) {
  auto array =
      tensorstore::AllocateArray<float>({2, 3, 4}, tensorstore::c_order);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(), array));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout,
                                   ChunkLayout::FromJson({
                                       {"grid_origin", {0, 0, 0}},
                                       {"inner_order", {0, 1, 2}},
                                   }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(FromArrayTest, ChunkLayoutFortranOrder) {
  auto array =
      tensorstore::AllocateArray<float>({2, 3, 4}, tensorstore::fortran_order);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(Context::Default(), array));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout,
                                   ChunkLayout::FromJson({
                                       {"grid_origin", {0, 0, 0}},
                                       {"inner_order", {2, 1, 0}},
                                   }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(FromArrayTest, DimensionUnits) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::FromArray(context, array, {"4nm", "5nm"}));
  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("5nm"))));
}

}  // namespace frontend_tests

namespace open_tests {

// Tests that tensorstore::Open can successfully open an "array"-backed
// tensorstore and that the spec round-trips.

template <typename T>
class OpenNumericTest : public ::testing::Test {};

using OpenNumericTestTypes =
    ::testing::Types<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                     std::int32_t, std::uint32_t, std::int64_t, std::uint64_t,
                     float, double>;

TYPED_TEST_SUITE(OpenNumericTest, OpenNumericTestTypes);

TYPED_TEST(OpenNumericTest, Roundtrip) {
  using T = TypeParam;
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array",
       {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
        {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)}}},
      {"dtype", std::string(tensorstore::dtype_v<T>.name())},
      {"transform",
       {{"input_inclusive_min", {1, 2}},
        {"input_exclusive_max", {3, 5}},
        {
            "output",
            {
                {{"offset", -1}, {"input_dimension", 0}},
                {{"offset", -2}, {"input_dimension", 1}},
            },
        }}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(MatchesJson(json_spec)));
  EXPECT_EQ(
      tensorstore::MakeOffsetArray<TypeParam>({1, 2}, {{1, 2, 3}, {4, 5, 6}}),
      tensorstore::Read(store).value());
}

TEST(OpenTest, RoundtripString) {
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", {{"a", "b", "c"}, {"d", "e", "f"}}},
      {"dtype", "string"},
      {"transform",
       {{"input_exclusive_max", {2, 3}}, {"input_inclusive_min", {0, 0}}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(json_spec));
  EXPECT_THAT(tensorstore::Read(store).result(),
              ::testing::Optional(tensorstore::MakeArray<std::string>(
                  {{"a", "b", "c"}, {"d", "e", "f"}})));
}

TEST(OpenTest, RoundtripDimensionUnits) {
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", {{"a", "b", "c"}, {"d", "e", "f"}}},
      {"dtype", "string"},
      {"schema", {{"dimension_units", {{4, "nm"}, {5, "nm"}}}}},
      {"transform",
       {{"input_exclusive_max", {2, 3}}, {"input_inclusive_min", {0, 0}}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(MatchesJson(json_spec)));
}

TEST(OpenTest, InvalidConversion) {
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "array"},
                                    {"array", {"a"}},
                                    {"dtype", "int32"},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Expected integer .*, but received: \"a\""));
}

// Tests that ArrayBackend::spec handles complicated index transforms properly.
TEST(FromArrayTest, Spec) {
  auto context = Context::Default();
  auto store =
      ChainResult(tensorstore::FromArray(
                      context, tensorstore::MakeOffsetArray<std::int32_t>(
                                   {1, 2}, {{1, 2, 3}, {4, 5, 6}})),
                  tensorstore::Dims(1)
                      .IndexArraySlice(tensorstore::MakeArray<Index>({2, 4, 4}))
                      .MoveToBack(),
                  tensorstore::Dims(1).AddNew().ClosedInterval(3, 5))
          .value();
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
      {"transform",
       {{"input_inclusive_min", {1, 3, 0}},
        {"input_exclusive_max", {3, 6, 3}},
        {
            "output",
            {
                {{"offset", -1}, {"input_dimension", 0}},
                {{"input_dimension", 2}},
            },
        }}},
  };
  EXPECT_THAT(store.spec().value().ToJson(),
              ::testing::Optional(MatchesJson(json_spec)));
}

TEST(OpenTest, MissingDataType) {
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "array"},
                                    {"array", {{1, 2, 3}, {4, 5, 6}}},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "dtype must be specified"));
}

TEST(OpenTest, InvalidTransformRank) {
  EXPECT_THAT(tensorstore::Open({
                                    {"driver", "array"},
                                    {"array", {{1, 2, 3}, {4, 5, 6}}},
                                    {"dtype", "int32"},
                                    {"transform",
                                     {
                                         {"input_inclusive_min", {1, 3, 0}},
                                         {"input_exclusive_max", {3, 6, 3}},
                                     }},
                                })
                  .result(),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Error parsing object member \"array\": "
                  "Array rank \\(2\\)\\ does not match expected rank \\(3\\)"));
}

TEST(OpenTest, InvalidRank) {
  EXPECT_THAT(
      tensorstore::Open({
                            {"driver", "array"},
                            {"array", {{1, 2, 3}, {4, 5, 6}}},
                            {"dtype", "int32"},
                            {"rank", 3},
                        })
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"array\": "
                    "Array rank \\(2\\) does not match expected rank \\(3\\)"));
}

// Tests that copying from an array driver to itself does not lead to deadlock.
TEST(CopyTest, SelfCopy) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::FromArray(Context::Default(),
                             tensorstore::MakeArray<int>({1, 2, 3, 4})));
  TENSORSTORE_EXPECT_OK(
      tensorstore::Copy(store | tensorstore::Dims(0).SizedInterval(0, 2),
                        store | tensorstore::Dims(0).SizedInterval(2, 2)));
  EXPECT_EQ(tensorstore::Read(store).result(),
            tensorstore::MakeArray<int>({1, 2, 1, 2}));
}

TEST(ArrayTest, SpecRankPropagation) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, tensorstore::Spec::FromJson({
                                                  {"driver", "array"},
                                                  {"array", {1, 2, 3}},
                                                  {"dtype", "int32"},
                                              }));
  EXPECT_EQ(1, spec.rank());
}

TEST(ArrayDriverHandle, OpenResolveBounds) {
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", {{1, 2, 3}, {4, 5, 6}}},
      {"dtype", "uint32"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  tensorstore::TransactionalOpenOptions options;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto handle,
      tensorstore::internal::OpenDriver(
          std::move(tensorstore::internal_spec::SpecAccess::impl(spec)),
          std::move(options))
          .result());

  // Validates that an identity transform of rank returns the array bounds.
  auto transform_result =
      handle.driver
          ->ResolveBounds(
              {}, tensorstore::IdentityTransform(handle.driver->rank()), {})
          .result();

  EXPECT_THAT(transform_result->input_origin(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(transform_result->input_shape(), ::testing::ElementsAre(2, 3));
}

TEST(ArrayTest, SpecFromArray) {
  auto orig_array = tensorstore::MakeOffsetArray<float>({2}, {1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::SpecFromArray(orig_array));
  EXPECT_EQ(1, spec.rank());
  EXPECT_EQ(tensorstore::dtype_v<float>, spec.dtype());
  EXPECT_THAT(spec.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"driver", "array"},
                  {"array", {1, 2, 3}},
                  {"dtype", "float32"},
                  {"transform",
                   {
                       {"input_inclusive_min", {2}},
                       {"input_exclusive_max", {5}},
                       {"output", {{{"input_dimension", 0}, {"offset", -2}}}},
                   }},
              })));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec).result());
  EXPECT_THAT(tensorstore::Read(store).result(),
              ::testing::Optional(orig_array));
}

TEST(SpecSchemaTest, Basic) {
  TestSpecSchema(
      {
          {"driver", "array"},
          {"array", {{1, 2, 3}, {4, 5, 6}}},
          {"dtype", "float32"},
          {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
      },
      {
          {"rank", 2},
          {"dtype", "float32"},
          {"domain", {{"shape", {2, 3}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(CreateCheckSchemaTest, Basic) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "array"},
          {"array", {{1, 2, 3}, {4, 5, 6}}},
          {"dtype", "float32"},
          {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
      },
      {
          {"rank", 2},
          {"dtype", "float32"},
          {"domain", {{"shape", {2, 3}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(ArrayTest, SpecFromArrayWithDimensionUnits) {
  auto orig_array = tensorstore::MakeOffsetArray<float>({2}, {1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, tensorstore::SpecFromArray(orig_array, {"5nm"}));
  EXPECT_EQ(1, spec.rank());
  EXPECT_EQ(tensorstore::dtype_v<float>, spec.dtype());
  EXPECT_THAT(spec.ToJson(),
              ::testing::Optional(MatchesJson(::nlohmann::json{
                  {"driver", "array"},
                  {"array", {1, 2, 3}},
                  {"dtype", "float32"},
                  {"transform",
                   {
                       {"input_inclusive_min", {2}},
                       {"input_exclusive_max", {5}},
                       {"output", {{{"input_dimension", 0}, {"offset", -2}}}},
                   }},
                  {"schema", {{"dimension_units", {{5, "nm"}}}}},
              })));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec).result());
  EXPECT_THAT(
      store.dimension_units(),
      ::testing::Optional(::testing::ElementsAre(tensorstore::Unit("5nm"))));
}

TEST(ArraySerializationTest, SerializationRoundTest) {
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", 42},
      {"dtype", "uint32"},
      {"transform", {{"input_rank", 0}}},
      {"context", {{"data_copy_concurrency", {{"limit", 1}}}}}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  tensorstore::serialization::TestSerializationRoundTrip(spec);
  tensorstore::TestJsonBinderRoundTrip<tensorstore::Spec>({{spec, json_spec}});
}

}  // namespace open_tests
}  // namespace
