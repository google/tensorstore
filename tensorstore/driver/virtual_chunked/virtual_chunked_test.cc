// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/virtual_chunked.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::dynamic_rank;
using ::tensorstore::Future;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Promise;
using ::tensorstore::Result;
using ::tensorstore::span;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::internal::ConcurrentQueue;
using ::tensorstore::internal::UniqueNow;
using ::tensorstore::serialization::SerializationRoundTrip;

template <typename... Option>
Result<tensorstore::TensorStore<Index, dynamic_rank,
                                tensorstore::ReadWriteMode::read>>
CoordinatesView(DimensionIndex dim, Option&&... option) {
  return tensorstore::VirtualChunked<Index>(
      tensorstore::NonSerializable{[dim](auto output, auto read_params)
                                       -> Future<TimestampedStorageGeneration> {
        tensorstore::IterateOverIndexRange(
            output.domain(),
            [&](span<const Index> indices) { output(indices) = indices[dim]; });
        return TimestampedStorageGeneration{StorageGeneration::FromString(""),
                                            absl::Now()};
      }},
      std::forward<Option>(option)...);
}

template <typename... Option>
Result<tensorstore::TensorStore<Index, dynamic_rank,
                                tensorstore::ReadWriteMode::read>>
SerializableCoordinatesView(DimensionIndex dim, Option&&... option) {
  return tensorstore::VirtualChunked<Index>(
      tensorstore::serialization::BindFront(
          [](DimensionIndex dim, auto output,
             auto read_params) -> Future<TimestampedStorageGeneration> {
            tensorstore::IterateOverIndexRange(output.domain(),
                                               [&](span<const Index> indices) {
                                                 output(indices) = indices[dim];
                                               });
            return TimestampedStorageGeneration{
                StorageGeneration::FromString(""), absl::Now()};
          },
          dim),
      std::forward<Option>(option)...);
}

using RequestLayout =
    ::tensorstore::StridedLayout<dynamic_rank, ::tensorstore::offset_origin>;

template <typename... Option>
Result<tensorstore::TensorStore<void, dynamic_rank,
                                tensorstore::ReadWriteMode::read>>
LoggingView(std::vector<RequestLayout>& requests, Option&&... option) {
  auto mutex = std::make_shared<absl::Mutex>();
  return tensorstore::VirtualChunked(
      tensorstore::NonSerializable{
          [mutex, &requests](auto output, auto read_params)
              -> Future<TimestampedStorageGeneration> {
            tensorstore::InitializeArray(output);
            absl::MutexLock lock(mutex.get());
            requests.emplace_back(output.layout());
            return TimestampedStorageGeneration{
                StorageGeneration::FromString(""), absl::Now()};
          }},
      std::forward<Option>(option)...);
}

template <typename Element, DimensionIndex Rank, typename Parameters>
struct Request {
  tensorstore::Array<Element, Rank, tensorstore::offset_origin> array;
  Parameters params;
  Promise<TimestampedStorageGeneration> promise;
};

template <typename Element, DimensionIndex Rank, typename Parameters>
auto EnqueueRequestHandler(
    ConcurrentQueue<Request<Element, Rank, Parameters>>& queue) {
  return tensorstore::NonSerializable{
      [&queue](
          tensorstore::Array<Element, Rank, tensorstore::offset_origin> array,
          Parameters params) -> Future<TimestampedStorageGeneration> {
        auto [promise, future] = tensorstore::PromiseFuturePair<
            TimestampedStorageGeneration>::Make();
        queue.push({std::move(array), std::move(params), std::move(promise)});
        return future;
      }};
}

template <typename Element, DimensionIndex Rank>
using ReadRequest =
    Request<Element, Rank, tensorstore::virtual_chunked::ReadParameters>;

template <typename Element, DimensionIndex Rank>
using WriteRequest =
    Request<const Element, Rank, tensorstore::virtual_chunked::WriteParameters>;

template <typename Element, DimensionIndex Rank, typename... Option>
Result<
    tensorstore::TensorStore<Element, Rank, tensorstore::ReadWriteMode::read>>
MockView(ConcurrentQueue<ReadRequest<Element, Rank>>& queue,
         Option&&... option) {
  return tensorstore::VirtualChunked<Element, Rank>(
      EnqueueRequestHandler(queue), std::forward<Option>(option)...);
}

template <typename Element, DimensionIndex Rank, typename... Option>
Result<tensorstore::TensorStore<Element, Rank,
                                tensorstore::ReadWriteMode::read_write>>
MockView(ConcurrentQueue<ReadRequest<Element, Rank>>& read_queue,
         ConcurrentQueue<WriteRequest<Element, Rank>>& write_queue,
         Option&&... option) {
  return tensorstore::VirtualChunked<Element, Rank>(
      EnqueueRequestHandler(read_queue), EnqueueRequestHandler(write_queue),
      std::forward<Option>(option)...);
}

template <typename Element, DimensionIndex Rank, typename... Option>
Result<
    tensorstore::TensorStore<Element, Rank, tensorstore::ReadWriteMode::write>>
MockView(ConcurrentQueue<WriteRequest<Element, Rank>>& write_queue,
         Option&&... option) {
  return tensorstore::VirtualChunkedWriteOnly<Element, Rank>(
      EnqueueRequestHandler(write_queue), std::forward<Option>(option)...);
}

TEST(VirtualChunkedTest, Coordinates) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords0, CoordinatesView(0, tensorstore::Schema::Shape({2, 3})));
  EXPECT_THAT(tensorstore::Read(coords0).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 0, 0}, {1, 1, 1}})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords1, CoordinatesView(1, tensorstore::Schema::Shape({2, 3})));
  EXPECT_THAT(tensorstore::Read(coords1).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 1, 2}, {0, 1, 2}})));
}

TEST(VirtualChunkedTest, CoordinatesUnbounded) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords0, CoordinatesView(0, tensorstore::RankConstraint{2}));
  EXPECT_THAT(
      tensorstore::Read<tensorstore::zero_origin>(
          coords0 | tensorstore::Dims(0, 1).SizedInterval({1000, 2}, {2, 3}))
          .result(),
      ::testing::Optional(tensorstore::MakeArray<Index>(
          {{1000, 1000, 1000}, {1001, 1001, 1001}})));
}

TEST(VirtualChunkedTest, CoordinatesInnerOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords0,
      CoordinatesView(0, tensorstore::Schema::Shape({2, 3}),
                      tensorstore::ChunkLayout::InnerOrder({1, 0})));
  EXPECT_THAT(tensorstore::Read(coords0).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 0, 0}, {1, 1, 1}})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords1,
      CoordinatesView(1, tensorstore::Schema::Shape({2, 3}),
                      tensorstore::ChunkLayout::InnerOrder({1, 0})));
  EXPECT_THAT(tensorstore::Read(coords1).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 1, 2}, {0, 1, 2}})));
}

TEST(VirtualChunkedTest, SerializableCoordinatesInnerOrder) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords0_orig, SerializableCoordinatesView(
                             0, tensorstore::Schema::Shape({2, 3}),
                             tensorstore::ChunkLayout::InnerOrder({1, 0})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto coords0,
                                   SerializationRoundTrip(coords0_orig));

  EXPECT_THAT(tensorstore::Read(coords0).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 0, 0}, {1, 1, 1}})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto coords1_orig, SerializableCoordinatesView(
                             1, tensorstore::Schema::Shape({2, 3}),
                             tensorstore::ChunkLayout::InnerOrder({1, 0})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto coords1,
                                   SerializationRoundTrip(coords1_orig));

  EXPECT_THAT(tensorstore::Read(coords1).result(),
              ::testing::Optional(
                  tensorstore::MakeArray<Index>({{0, 1, 2}, {0, 1, 2}})));
}

TEST(VirtualChunkedTest, ReadChunkShape) {
  std::vector<RequestLayout> requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto view, LoggingView(requests, tensorstore::dtype_v<bool>,
                             tensorstore::Schema::Shape({2, 3}),
                             tensorstore::ChunkLayout::ReadChunkShape({2, 1})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto chunk_layout, view.chunk_layout());
  EXPECT_THAT(chunk_layout.read_chunk_shape(), ::testing::ElementsAre(2, 1));
  TENSORSTORE_ASSERT_OK(tensorstore::Read(view));
  EXPECT_THAT(requests, ::testing::UnorderedElementsAre(
                            // origin, shape, byte_strides
                            RequestLayout({0, 0}, {2, 1}, {1, 1}),
                            RequestLayout({0, 1}, {2, 1}, {1, 1}),
                            RequestLayout({0, 2}, {2, 1}, {1, 1})));
}

TEST(VirtualChunkedTest, InnerOrder) {
  std::vector<RequestLayout> requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto view,
      LoggingView(requests, tensorstore::dtype_v<bool>,
                  tensorstore::Schema::Shape({3, 4, 5}),
                  tensorstore::ChunkLayout::InnerOrder({2, 0, 1}),
                  tensorstore::ChunkLayout::ReadChunkShape({2, 3, 4})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto chunk_layout, view.chunk_layout());
  EXPECT_THAT(chunk_layout.read_chunk_shape(), ::testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(chunk_layout.inner_order(), ::testing::ElementsAre(2, 0, 1));
  TENSORSTORE_ASSERT_OK(tensorstore::Read(view));
  EXPECT_THAT(requests, ::testing::UnorderedElementsAreArray({
                            // origin, shape, byte_strides
                            RequestLayout({0, 0, 0}, {2, 3, 4}, {3, 1, 6}),
                            RequestLayout({2, 0, 0}, {1, 3, 4}, {3, 1, 6}),
                            RequestLayout({0, 3, 0}, {2, 1, 4}, {3, 1, 6}),
                            RequestLayout({2, 3, 0}, {1, 1, 4}, {3, 1, 6}),
                            RequestLayout({0, 0, 4}, {2, 3, 1}, {3, 1, 6}),
                            RequestLayout({2, 0, 4}, {1, 3, 1}, {3, 1, 6}),
                            RequestLayout({0, 3, 4}, {2, 1, 1}, {3, 1, 6}),
                            RequestLayout({2, 3, 4}, {1, 1, 1}, {3, 1, 6}),
                        }));
}

// Tests that the read_function is not called to validate a cached chunk that
// has a timestamp in the past, in the case that `RecheckCachedData{false}` is
// specified.
TEST(VirtualChunkedTest, NoRecheckCache) {
  ConcurrentQueue<ReadRequest<int, 0>> requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, tensorstore::Context::FromJson(
                        {{"cache_pool", {{"total_bytes_limit", 10000000}}}}));
  auto mock_view = MockView<int, 0>(
      requests, tensorstore::RecheckCachedData{false}, context);
  auto read_future = tensorstore::Read(mock_view);
  {
    auto request = requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_not_equal());
    request.array() = 42;
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("abc"), absl::Now()));
  }
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
  read_future = tensorstore::Read(mock_view);
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
}

// Tests that the read_function is called to validate a cached chunk that has a
// timestamp in the past, in the case of `RecheckCachedData{true}` (which is the
// default).
TEST(VirtualChunkedTest, RecheckCache) {
  ConcurrentQueue<ReadRequest<int, 0>> requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, tensorstore::Context::FromJson(
                        {{"cache_pool", {{"total_bytes_limit", 10000000}}}}));
  auto mock_view = MockView<int, 0>(requests, context);
  auto read_future = tensorstore::Read(mock_view);
  {
    auto request = requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_not_equal());
    request.array() = 42;
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("abc"), absl::Now()));
  }
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
  // Ensure previous timestamp is no longer current.
  UniqueNow();
  read_future = tensorstore::Read(mock_view);
  {
    auto request = requests.pop();
    EXPECT_EQ(StorageGeneration::FromString("abc"),
              request.params.if_not_equal());
    request.array() = 43;
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::Unknown(), absl::Now()));
  }
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
}

// Tests that the read_function is not called to validate a cached chunk that
// has a timestamp of `absl::InfiniteFuture()`, even if
// `RecheckCachedData{false}` is not specified.
TEST(VirtualChunkedTest, RecheckCacheImmutable) {
  ConcurrentQueue<ReadRequest<int, 0>> requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, tensorstore::Context::FromJson(
                        {{"cache_pool", {{"total_bytes_limit", 10000000}}}}));
  auto mock_view =
      MockView<int, 0>(requests, tensorstore::RecheckCachedData{true}, context);
  auto read_future = tensorstore::Read(mock_view);
  {
    auto request = requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_not_equal());
    request.array() = 42;
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString(""), absl::InfiniteFuture()));
  }
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
  // Ensure previous timestamp is no longer current.
  UniqueNow();
  read_future = tensorstore::Read(mock_view);
  EXPECT_THAT(read_future.result(),
              ::testing::Optional(tensorstore::MakeScalarArray<int>(42)));
}

TEST(VirtualChunkedTest, ReadWrite) {
  ConcurrentQueue<ReadRequest<int, 1>> read_requests;
  ConcurrentQueue<WriteRequest<int, 1>> write_requests;
  auto mock_view = MockView<int, 1>(read_requests, write_requests,
                                    tensorstore::Schema::Shape({2}));
  auto write_future =
      tensorstore::Write(tensorstore::MakeScalarArray<int>(42),
                         mock_view | tensorstore::Dims(0).IndexSlice(0));
  write_future.Force();
  {
    auto request = read_requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_not_equal());
    request.array(0) = 1;
    request.array(1) = 2;
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("gen1"), absl::Now()));
  }
  {
    auto request = write_requests.pop();
    EXPECT_EQ(StorageGeneration::FromString("gen1"), request.params.if_equal());
    EXPECT_EQ(tensorstore::MakeArray<int>({42, 2}), request.array);
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString("gen2"), absl::Now()));
  }
  TENSORSTORE_ASSERT_OK(write_future);
}

TEST(VirtualChunkedTest, ReadWriteWrite) {
  ConcurrentQueue<ReadRequest<int, 1>> read_requests;
  ConcurrentQueue<WriteRequest<int, 1>> write_requests;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context, tensorstore::Context::FromJson(
                        {{"cache_pool", {{"total_bytes_limit", 1000000}}}}));
  auto mock_view = MockView<int, 1>(read_requests, write_requests, context,
                                    tensorstore::Schema::Shape({2}));
  {
    auto write_future =
        tensorstore::Write(tensorstore::MakeScalarArray<int>(42),
                           mock_view | tensorstore::Dims(0).IndexSlice(0));
    write_future.Force();
    {
      auto request = read_requests.pop();
      EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_not_equal());
      request.array(0) = 1;
      request.array(1) = 2;
      request.promise.SetResult(TimestampedStorageGeneration(
          StorageGeneration::FromString(""), absl::InfiniteFuture()));
    }
    {
      auto request = write_requests.pop();
      EXPECT_EQ(StorageGeneration::FromString(""), request.params.if_equal());
      EXPECT_EQ(tensorstore::MakeArray<int>({42, 2}), request.array);
      request.promise.SetResult(TimestampedStorageGeneration(
          StorageGeneration::FromString(""), absl::InfiniteFuture()));
    }
    TENSORSTORE_ASSERT_OK(write_future);
  }

  {
    auto write_future =
        tensorstore::Write(tensorstore::MakeScalarArray<int>(50),
                           mock_view | tensorstore::Dims(0).IndexSlice(1));
    write_future.Force();
    {
      auto request = write_requests.pop();
      EXPECT_EQ(StorageGeneration::FromString(""), request.params.if_equal());
      EXPECT_EQ(tensorstore::MakeArray<int>({42, 50}), request.array);
      request.promise.SetResult(TimestampedStorageGeneration(
          StorageGeneration::FromString(""), absl::InfiniteFuture()));
    }
    TENSORSTORE_ASSERT_OK(write_future);
  }
}

// Tests a chunk-aligned write to a write-only virtual chunked view.
TEST(VirtualChunkedTest, Write) {
  ConcurrentQueue<WriteRequest<int, 1>> write_requests;

  auto mock_view =
      MockView<int, 1>(write_requests, tensorstore::Schema::Shape({6}),
                       tensorstore::ChunkLayout::ChunkShape({4}));
  // Write to [0, 4), which is chunk aligned.
  {
    auto write_future = tensorstore::Write(
        tensorstore::MakeScalarArray<int>(42),
        mock_view | tensorstore::Dims(0).SizedInterval(0, 4));
    write_future.Force();
    {
      auto request = write_requests.pop();
      EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_equal());
      EXPECT_EQ(tensorstore::MakeArray<int>({42, 42, 42, 42}), request.array);
      request.promise.SetResult(TimestampedStorageGeneration(
          StorageGeneration::FromString(""), absl::Now()));
    }
    TENSORSTORE_ASSERT_OK(write_future);
  }
  // Write to [4, 6), which is also chunk-aligned after clipping the `[4, 8)`
  // chunk to the full domain `[0, 6)`.
  {
    auto write_future = tensorstore::Write(
        tensorstore::MakeScalarArray<int>(42),
        mock_view | tensorstore::Dims(0).SizedInterval(4, 2));
    write_future.Force();
    {
      auto request = write_requests.pop();
      EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_equal());
      EXPECT_EQ(tensorstore::MakeOffsetArray<int>({4}, {42, 42}),
                request.array);
      request.promise.SetResult(TimestampedStorageGeneration(
          StorageGeneration::FromString(""), absl::Now()));
    }
    TENSORSTORE_ASSERT_OK(write_future);
  }
}

// Tests a write of all-zero (the fill value).
TEST(VirtualChunkedTest, WriteFillValue) {
  ConcurrentQueue<WriteRequest<int, 0>> write_requests;

  auto mock_view = MockView<int, 0>(write_requests);
  auto write_future =
      tensorstore::Write(tensorstore::MakeScalarArray<int>(0), mock_view);
  write_future.Force();
  {
    auto request = write_requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_equal());
    EXPECT_EQ(tensorstore::MakeScalarArray<int>(0), request.array);
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString(""), absl::Now()));
  }
  TENSORSTORE_ASSERT_OK(write_future);
}

TEST(VirtualChunkedTest, WriteOnlyError) {
  ConcurrentQueue<WriteRequest<int, 1>> write_requests;

  auto mock_view =
      MockView<int, 1>(write_requests, tensorstore::Schema::Shape({2}));
  EXPECT_THAT(
      tensorstore::Write(tensorstore::MakeScalarArray<int>(42),
                         mock_view | tensorstore::Dims(0).IndexSlice(0))
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Write-only virtual chunked view requires chunk-aligned writes"));
}

TEST(VirtualChunkedTest, AtomicSingleChunk) {
  tensorstore::Transaction transaction(tensorstore::atomic_isolated);

  ConcurrentQueue<WriteRequest<int, 1>> write_requests;
  auto mock_view =
      MockView<int, 1>(write_requests, tensorstore::Schema::Shape({6}),
                       tensorstore::ChunkLayout::ChunkShape({4}), transaction);
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      tensorstore::MakeScalarArray<int>(42),
      mock_view | tensorstore::Dims(0).HalfOpenInterval(0, 4)));

  auto future = transaction.CommitAsync();

  {
    auto request = write_requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_equal());
    EXPECT_EQ(tensorstore::MakeArray<int>({42, 42, 42, 42}), request.array);
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString(""), absl::Now()));
  }

  TENSORSTORE_ASSERT_OK(future);
}

TEST(VirtualChunkedTest, AtomicMultipleChunks) {
  tensorstore::Transaction transaction(tensorstore::atomic_isolated);

  ConcurrentQueue<WriteRequest<int, 1>> write_requests;
  auto mock_view =
      MockView<int, 1>(write_requests, tensorstore::Schema::Shape({6}),
                       tensorstore::ChunkLayout::ChunkShape({4}), transaction);
  EXPECT_THAT(
      tensorstore::Write(tensorstore::MakeScalarArray<int>(42), mock_view)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot write to virtual chunk .* and write to virtual "
                    "chunk .* as single atomic transaction"));
}

TEST(VirtualChunkedTest, NonAtomicSingleChunk) {
  tensorstore::Transaction transaction(tensorstore::isolated);

  ConcurrentQueue<WriteRequest<int, 1>> write_requests;
  auto mock_view =
      MockView<int, 1>(write_requests, tensorstore::Schema::Shape({6}),
                       tensorstore::ChunkLayout::ChunkShape({4}), transaction);
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeScalarArray<int>(42), mock_view));

  auto future = transaction.CommitAsync();

  for (int i = 0; i < 2; ++i) {
    auto request = write_requests.pop();
    EXPECT_EQ(StorageGeneration::Unknown(), request.params.if_equal());
    request.promise.SetResult(TimestampedStorageGeneration(
        StorageGeneration::FromString(""), absl::Now()));
  }

  TENSORSTORE_ASSERT_OK(future);
}

}  // namespace
