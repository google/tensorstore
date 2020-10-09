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
#include "absl/meta/type_traits.h"
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Context;
using tensorstore::CopyProgress;
using tensorstore::CopyProgressFunction;
using tensorstore::Index;
using tensorstore::MatchesStatus;
using tensorstore::offset_origin;
using tensorstore::ReadProgress;
using tensorstore::ReadProgressFunction;
using tensorstore::ReadWriteMode;
using tensorstore::Status;
using tensorstore::TensorStore;
using tensorstore::WriteProgress;
using tensorstore::WriteProgressFunction;
using tensorstore::zero_origin;

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
  auto future = tensorstore::internal::DriverRead(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver,
      /*target=*/dest_array,
      {/*.progress_function=*/[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }});
  EXPECT_EQ(Status(), GetStatus(future.result()));
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
  auto future = tensorstore::internal::DriverRead(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/transformed_driver,
      /*target_type=*/array.data_type(),
      /*target_layout_order=*/tensorstore::c_order,
      {/*.progress_function=*/[&read_progress](ReadProgress progress) {
        read_progress.push_back(progress);
      }});
  EXPECT_EQ(Status(), GetStatus(future.result()));
  EXPECT_EQ(array, future.value());
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
  EXPECT_THAT(GetStatus(future.result()),
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
  EXPECT_THAT(GetStatus(future.result()),
              MatchesStatus(absl::StatusCode::kOutOfRange,
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 7, 8}}),
            array);
  EXPECT_THAT(write_progress, ::testing::ElementsAre(WriteProgress{2, 2, 0},
                                                     WriteProgress{2, 2, 2}));
}

/// Tests calling Write with an invalid source array that yields an error when
/// MakeNormalizedTransformedArray is called.
TEST(ArrayDriverTest, WriteInvalidSourceTransform) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto [driver, transform, transaction] =
      tensorstore::internal::MakeArrayDriver<offset_origin>(context, array)
          .value();
  std::vector<WriteProgress> write_progress;
  auto write_result = tensorstore::internal::DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/
      tensorstore::TransformedArray(
          tensorstore::MakeOffsetArray<int>({2, 3}, {{7, 8}}),
          tensorstore::IdentityTransform(tensorstore::BoxView({2, 3}, {2, 2}))),
      /*target=*/
      {driver, ChainResult(transform, tensorstore::Dims(0, 1).SizedInterval(
                                          {2, 3}, {1, 2}))
                   .value()},
      {/*.progress_function=*/[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Propagated bounds .* for dimension 0 are "
                            "incompatible with existing bounds .*"));
  EXPECT_EQ(GetStatus(write_result.commit_future.result()),
            GetStatus(write_result.copy_future.result()));
  EXPECT_THAT(write_progress, ::testing::ElementsAre());
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
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(GetStatus(write_result.commit_future.result()),
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
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
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_EQ(GetStatus(write_result.copy_future.result()),
            GetStatus(write_result.commit_future.result()));
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
  EXPECT_EQ(tensorstore::DataTypeOf<int>(), store2.data_type());
  EXPECT_THAT(store2.domain().labels(), ::testing::ElementsAre("", ""));
}

TEST(FromArrayTest, Read) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array).value();
  static_assert(std::is_same<TensorStore<int, 2, ReadWriteMode::read_write>,
                             decltype(store)>::value,
                "");
  std::vector<ReadProgress> read_progress;
  auto dest_array = tensorstore::AllocateArray<int>(array.domain());
  auto future =
      Read(store, dest_array,
           ReadProgressFunction{[&read_progress](ReadProgress progress) {
             read_progress.push_back(progress);
           }});
  EXPECT_EQ(Status(), GetStatus(future.result()));
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
  EXPECT_EQ(Status(), GetStatus(future.result()));
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
  auto future =
      Read(ChainResult(store, tensorstore::AllDims().Label("x", "y")),
           ChainResult(dest_array, tensorstore::AllDims().Label("y", "x")));
  EXPECT_EQ(Status(), GetStatus(future.result()));
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
  EXPECT_EQ(Status(), GetStatus(future.result()));
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
  EXPECT_THAT(GetStatus(future.result()),
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
  EXPECT_THAT(
      GetStatus(future.result()),
      MatchesStatus(absl::StatusCode::kOutOfRange, kOutsideValidRangeRE));
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
  EXPECT_EQ(tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 42, 42}}),
            array);
}

/// Tests calling Write with an invalid source array that yields an error when
/// MakeNormalizedTransformedArray is called.
TEST(FromArrayTest, WriteInvalidSourceTransform) {
  auto array =
      tensorstore::MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}});
  auto context = Context::Default();
  auto store = tensorstore::FromArray(context, array);
  std::vector<WriteProgress> write_progress;
  auto write_result = Write(
      tensorstore::TransformedArray(
          tensorstore::MakeOffsetArray<int>({2, 3}, {{7, 8}}),
          tensorstore::IdentityTransform(tensorstore::BoxView({2, 3}, {2, 2}))),
      ChainResult(store, tensorstore::Dims(0, 1).SizedInterval({2, 3}, {1, 2})),
      WriteProgressFunction{[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Propagated bounds .* for dimension 0 are "
                            "incompatible with existing bounds .*"));
  EXPECT_EQ(GetStatus(write_result.commit_future.result()),
            GetStatus(write_result.copy_future.result()));
  EXPECT_THAT(write_progress, ::testing::ElementsAre());
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
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_THAT(GetStatus(write_result.commit_future.result()),
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
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
  EXPECT_EQ(Status(), GetStatus(write_result.copy_future.result()));
  EXPECT_EQ(Status(), GetStatus(write_result.commit_future.result()));
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
  EXPECT_THAT(GetStatus(write_result.copy_future.result()),
              MatchesStatus(absl::StatusCode::kInvalidArgument, kMismatchRE));
  EXPECT_EQ(GetStatus(write_result.copy_future.result()),
            GetStatus(write_result.commit_future.result()));
  EXPECT_THAT(progress, ::testing::ElementsAre());
}

TEST(FromArrayTest, ReadDataTypeConversion) {
  auto context = Context::Default();
  auto source = tensorstore::MakeArray<std::int32_t>({1, 2, 3});
  auto dest = tensorstore::AllocateArray<std::int64_t>({3});
  EXPECT_EQ(
      Status(),
      GetStatus(tensorstore::Read(tensorstore::FromArray(context, source), dest)
                    .result()));
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
  EXPECT_EQ(Status(),
            GetStatus(tensorstore::Write(source,
                                         tensorstore::FromArray(context, dest))
                          .commit_future.result()));
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
  EXPECT_EQ(Status(),
            GetStatus(tensorstore::Copy(tensorstore::FromArray(context, source),
                                        tensorstore::FromArray(context, dest))
                          .commit_future.result()));
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
      {"dtype", std::string(tensorstore::DataTypeOf<T>().name())},
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
  auto context = Context::Default();
  auto store = tensorstore::Open(context, json_spec).value();
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(json_spec));
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
  auto context = Context::Default();
  auto store = tensorstore::Open(context, json_spec).value();
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(json_spec));
  EXPECT_EQ(
      tensorstore::MakeArray<std::string>({{"a", "b", "c"}, {"d", "e", "f"}}),
      tensorstore::Read(store).value());
}

TEST(OpenTest, InvalidConversion) {
  ::nlohmann::json json_spec{
      {"driver", "array"},
      {"array", {"a"}},
      {"dtype", "int32"},
  };
  auto context = Context::Default();
  EXPECT_THAT(tensorstore::Open(context, json_spec).result(),
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
  EXPECT_THAT(store.spec().value().ToJson(tensorstore::IncludeContext{false}),
              ::testing::Optional(json_spec));
}

TEST(OpenTest, MissingDataType) {
  auto context = Context::Default();
  EXPECT_THAT(tensorstore::Open(context,
                                ::nlohmann::json{
                                    {"driver", "array"},
                                    {"array", {{1, 2, 3}, {4, 5, 6}}},
                                })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Data type must be specified"));
}

TEST(OpenTest, InvalidTransformRank) {
  auto context = Context::Default();
  EXPECT_THAT(tensorstore::Open(context,
                                ::nlohmann::json{
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
  auto context = Context::Default();
  EXPECT_THAT(
      tensorstore::Open(context,
                        ::nlohmann::json{
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
      tensorstore::Copy(
          ChainResult(store, tensorstore::Dims(0).SizedInterval(0, 2)),
          ChainResult(store, tensorstore::Dims(0).SizedInterval(2, 2)))
          .result());
  EXPECT_EQ(tensorstore::Read(store).result(),
            tensorstore::MakeArray<int>({1, 2, 1, 2}));
}

}  // namespace open_tests
}  // namespace
