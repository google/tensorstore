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

#include "tensorstore/internal/chunk_cache.h"

#include <functional>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/internal/thread_pool.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/result_util.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::ArrayView;
using tensorstore::DimensionIndex;
using tensorstore::Executor;
using tensorstore::Future;
using tensorstore::Index;
using tensorstore::IndexTransform;
using tensorstore::MakeArray;
using tensorstore::MakeCopy;
using tensorstore::MatchesStatus;
using tensorstore::ReadProgressFunction;
using tensorstore::SharedArray;
using tensorstore::span;
using tensorstore::StalenessBound;
using tensorstore::Status;
using tensorstore::StorageGeneration;
using tensorstore::TimestampedStorageGeneration;
using tensorstore::TransformedSharedArrayView;
using tensorstore::WriteFutures;
using tensorstore::WriteProgressFunction;
using tensorstore::internal::CachePool;
using tensorstore::internal::CachePtr;
using tensorstore::internal::ChunkCache;
using tensorstore::internal::ChunkCacheDriver;
using tensorstore::internal::ChunkGridSpecification;
using tensorstore::internal::ConcurrentQueue;
using tensorstore::internal::Driver;
using tensorstore::internal::ElementCopyFunction;
using tensorstore::internal::SimpleElementwiseFunction;
using testing::ElementsAre;

/// Represents a pending read request for a TestCache object.
struct ReadRequest {
  ChunkCache::ReadReceiver receiver;
  StorageGeneration generation;
  absl::Time staleness_bound;

  ChunkCache::Entry* entry() const { return receiver.entry(); }
};

/// Represents a pending writeback request for a TestCache object.
struct WritebackRequest {
  ChunkCache::WritebackReceiver receiver;
  StorageGeneration generation;

  ChunkCache::Entry* entry() const { return receiver.entry(); }
};

/// Contains the queue of pending chunk read and writeback requests for a
/// TestCache object.
struct RequestLog {
  ConcurrentQueue<ReadRequest> reads;
  ConcurrentQueue<WritebackRequest> writebacks;
};

StorageGeneration GetStorageGenerationFromNumber(
    std::uint64_t generation_number) {
  StorageGeneration generation;
  generation.value.resize(sizeof(std::uint64_t));
  std::memcpy(&generation.value[0], &generation_number, sizeof(std::uint64_t));
  return generation;
}

template <typename T>
ElementCopyFunction GetCopyFunction() {
  const auto copy_func ABSL_ATTRIBUTE_UNUSED =
      [](const T* source, T* dest, Status* status) { *dest = *source; };
  return SimpleElementwiseFunction<decltype(copy_func), const T, T>();
}

bool GenerationMatches(std::uint64_t generation_number,
                       const StorageGeneration& condition) {
  if (StorageGeneration::IsUnknown(condition)) return true;
  if (generation_number == 0 && StorageGeneration::IsNoValue(condition))
    return true;
  return GetStorageGenerationFromNumber(generation_number) == condition;
}

/// Data store used for testing that stores the component arrays for each chunk
/// directly in a `flat_hash_map`.
struct MockDataStore {
 public:
  struct StoredChunk {
    std::uint64_t generation = 0;
    std::vector<SharedArray<const void>> data;
  };

  absl::flat_hash_map<std::string, StoredChunk> chunks;

  bool HasChunk(span<const Index> indices) {
    std::string key;
    key.resize(sizeof(Index) * indices.size());
    std::memcpy(&key[0], indices.data(), key.size());
    return chunks.find(key) != chunks.end();
  }

  StoredChunk& GetChunk(span<const Index> indices) {
    std::string key;
    key.resize(sizeof(Index) * indices.size());
    std::memcpy(&key[0], indices.data(), key.size());
    return chunks[key];
  }

  /// Look up a chunk in the `chunks` map specified in the given read request.
  /// If the chunk is found, complete the request successfully (if
  /// `req.generation` does not match) or with `StorageGeneration::Unknown()`
  /// (if `req.generation` does match).  Otherwise, complete the request with
  /// `StorageGeneration::NoValue()`.
  void HandleRead(ReadRequest req) {
    absl::Time read_time = absl::Now();
    auto it = chunks.find(req.entry()->key());
    if (it == chunks.end()) {
      if (StorageGeneration::IsNoValue(req.generation)) {
        req.receiver.NotifyDone(
            ChunkCache::ReadReceiver::ComponentsWithGeneration{
                {}, {StorageGeneration::Unknown(), read_time}});
        return;
      }
      req.receiver.NotifyDone(
          ChunkCache::ReadReceiver::ComponentsWithGeneration{
              {}, {StorageGeneration::NoValue(), read_time}});
      return;
    }
    const auto& stored_chunk = it->second;
    if (!StorageGeneration::IsUnknown(req.generation) &&
        GenerationMatches(stored_chunk.generation, req.generation)) {
      req.receiver.NotifyDone(
          ChunkCache::ReadReceiver::ComponentsWithGeneration{
              {}, {StorageGeneration::Unknown(), read_time}});
      return;
    }
    std::vector<ArrayView<const void>> data(stored_chunk.data.begin(),
                                            stored_chunk.data.end());
    req.receiver.NotifyDone(ChunkCache::ReadReceiver::ComponentsWithGeneration{
        data,
        {GetStorageGenerationFromNumber(stored_chunk.generation), read_time}});
  }

  /// Starts a writeback (using WritebackSnapshot) and returns a function that
  /// completes the writeback (using the `chunks` map) when called.
  std::function<void()> HandleWriteback(WritebackRequest req) {
    ChunkCache::WritebackSnapshot snapshot(req.receiver);
    const bool equals_fill_value = snapshot.equals_fill_value();
    std::vector<SharedArray<const void>> data;
    for (const auto& array : snapshot.component_arrays()) {
      data.push_back(MakeCopy(array));
    }
    auto req_time = absl::Now();
    return [data, req, this, req_time, equals_fill_value] {
      auto& stored_chunk = chunks[req.entry()->key()];
      if (GenerationMatches(stored_chunk.generation, req.generation)) {
        stored_chunk.data = data;
        ++stored_chunk.generation;
        StorageGeneration storage_generation;
        if (equals_fill_value) {
          chunks.erase(req.entry()->key());
          storage_generation = StorageGeneration::NoValue();
        } else {
          storage_generation =
              GetStorageGenerationFromNumber(stored_chunk.generation);
        }
        req.receiver.NotifyDone({std::in_place, storage_generation, req_time});
      } else {
        req.receiver.NotifyDone(
            {std::in_place, StorageGeneration::Unknown(), req_time});
      }
    };
  }
};

class TestCache : public ChunkCache {
 public:
  TestCache(ChunkGridSpecification grid, RequestLog* log)
      : ChunkCache(std::move(grid)), log_(log) {}

  void DoRead(ReadOptions options, ReadReceiver receiver) override {
    log_->reads.push(ReadRequest{std::move(receiver),
                                 std::move(options.existing_generation),
                                 options.staleness_bound});
  }
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override {
    log_->writebacks.push(WritebackRequest{
        std::move(receiver), std::move(existing_generation.generation)});
  }

 private:
  RequestLog* log_;
};

TEST(ChunkGridSpecificationTest, Basic) {
  ChunkGridSpecification grid({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}});
  EXPECT_EQ(1, grid.components[0].fill_value.rank());
  EXPECT_EQ(1, grid.components[0].chunked_to_cell_dimensions.size());
  EXPECT_EQ(1, grid.chunk_shape.size());
}

class TestDriver : public ChunkCacheDriver {
 public:
  using ChunkCacheDriver::ChunkCacheDriver;

  /// Not actually used.
  Executor data_copy_executor() override {
    return tensorstore::InlineExecutor{};
  }
};

// Test fixture for tests to be run with both an anonymous and named cache.  The
// behavior of `ChunkCache` itself does not depend on whether the cache is
// named, but this ensures that the reference counting of the base `Cache`
// object works correctly.
class ChunkCacheTest : public ::testing::TestWithParam<const char*> {
 protected:
  /// Thread pool used by the ChunkCacheDriver to perform read/write operations.
  Executor thread_pool = tensorstore::internal::DetachedThreadPool(1);

  /// Limits used by `InitCache` for creating `pool`.  Tests that override the
  /// default limits must do so before calling `InitCache`.
  CachePool::Limits cache_limits{10000000, 5000000};

  /// Cache pool created by `InitCache` to contain the `TestCache`.
  CachePool::StrongPtr pool;

  /// Contains the queues of pending read/writeback requests made by the
  /// ChunkCache implementation.
  RequestLog log;

  /// Mock data store used to handle pending read/writeback requests, except
  /// when errors need to be explicitly injected, or the test does not require
  /// that the read/writeback request be completed.
  MockDataStore store;

  /// Instance of `TestCache` created by `InitCache`.  Read and writeback
  /// requests are not handled automatically, but are added to `log`.
  CachePtr<ChunkCache> cache;

  /// Cache key used by `InitCache` to obtain `cache`.
  std::string cache_key = GetParam();

  void InitCache(ChunkGridSpecification grid) {
    pool = CachePool::Make(cache_limits);
    cache = pool->GetCache<TestCache>(
        cache_key, [&] { return absl::make_unique<TestCache>(grid, &log); });
  }

  /// Creates a ChunkCacheDriver used by the `Read` and `Write` methods defined
  /// below.
  Driver::Ptr GetDriver(std::size_t component_index,
                        StalenessBound data_staleness = {}) {
    // Create a new driver each time rather than caching it since it reduces
    // complexity and the cost of the additional memory allocation is not
    // important for the tests.
    return Driver::Ptr(new TestDriver(cache, component_index, data_staleness));
  }

  Future<void> Read(std::size_t component_index,
                    IndexTransform<> source_transform,
                    TransformedSharedArrayView<void> dest,
                    StalenessBound data_staleness,
                    ReadProgressFunction read_progress_function = {}) {
    return tensorstore::internal::DriverRead(
        thread_pool,
        {GetDriver(component_index, data_staleness),
         std::move(source_transform)},
        std::move(dest), {std::move(read_progress_function)});
  }

  Future<void> Write(std::size_t component_index,
                     IndexTransform<> dest_transform,
                     TransformedSharedArrayView<const void> source,
                     WriteProgressFunction write_progress_function = {}) {
    auto write_futures = tensorstore::internal::DriverWrite(
        thread_pool, std::move(source),
        {GetDriver(component_index), std::move(dest_transform)},
        {std::move(write_progress_function)});
    // Wait until the copying is complete, and any errors due to copying have
    // propagated to `commit_future` before this function returns.
    write_futures.copy_future.Wait();
    return write_futures.commit_future;
  }
};

INSTANTIATE_TEST_SUITE_P(WithoutCacheKey, ChunkCacheTest,
                         ::testing::Values(""));
INSTANTIATE_TEST_SUITE_P(WithCacheKey, ChunkCacheTest, ::testing::Values("k"));

// Tests reading of chunks not present in the data store.
TEST_P(ChunkCacheTest, ReadSingleComponentOneDimensionalFill) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  // Test that chunks that aren't present in store get filled using the fill
  // value.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    // Index:         0 1 2 3 4 5 ...
    // Fill value:    1 2 1 2 1 2 ...
    // Read region:        [     ]
    EXPECT_EQ(read_array, tensorstore::MakeArray({2, 1, 2}));
  }

  // Test reading cached chunks.  The staleness bound of `absl::InfinitePast()`
  // is always satisfied by an existing read response.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    EXPECT_TRUE(read_future.result());
    // Verify that same response as before is received (fill value).
    EXPECT_EQ(read_array, tensorstore::MakeArray({2, 1, 2}));
  }

  // Test re-reading chunks.  The staleness bound of `absl::InfiniteFuture()` is
  // never satisfied by an existing read response.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfiniteFuture());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    // Verify that result matches fill value after the new read response.
    EXPECT_EQ(read_array, tensorstore::MakeArray({2, 1, 2}));
  }
}

// Tests cancelling a read request.
TEST_P(ChunkCacheTest, CancelRead) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    // Read request is cancelled when `read_future` is destroyed.  At the
    // moment, however, there is no mechanism for propagating cancellation to
    // the implementation of `DoRead` in order to cancel already in-progress
    // reads; however, pending read requests are cancelled.
  }
}

// Special-purpose FlowReceiver used by the `CancelWrite` test defined below,
// passed to `ChunkCache::Write`.
struct CancelWriteReceiver {
  friend void set_starting(CancelWriteReceiver& receiver,
                           tensorstore::AnyCancelReceiver cancel) {
    receiver.cancel = std::move(cancel);
  }
  friend void set_value(CancelWriteReceiver& receiver,
                        tensorstore::internal::WriteChunk chunk,
                        tensorstore::IndexTransform<> cell_transform) {
    EXPECT_FALSE(receiver.set_value_called);
    receiver.set_value_called = true;
    EXPECT_EQ(tensorstore::IndexTransformBuilder<>(1, 1)
                  .input_origin({0})
                  .input_shape({1})
                  .output_single_input_dimension(0, 3, 1, 0)
                  .Finalize()
                  .value(),
              chunk.transform);
    EXPECT_EQ(tensorstore::IndexTransformBuilder<>(1, 1)
                  .input_origin({0})
                  .input_shape({1})
                  .output_single_input_dimension(0, 0, 1, 0)
                  .Finalize()
                  .value(),
              cell_transform);
    receiver.cancel();
  }
  friend void set_done(CancelWriteReceiver& receiver) {}
  friend void set_error(CancelWriteReceiver& receiver, Status status) {}
  friend void set_stopping(CancelWriteReceiver& receiver) {
    receiver.cancel = nullptr;
  }

  bool set_value_called = false;
  tensorstore::AnyCancelReceiver cancel;
};

// Tests cancelling a write request.
TEST_P(ChunkCacheTest, CancelWrite) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  CancelWriteReceiver receiver;
  cache->Write(0,
               ChainResult(tensorstore::IdentityTransform(1),
                           tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                   .value(),
               std::ref(receiver));
  EXPECT_TRUE(receiver.set_value_called);
}

// Tests that the implementation of `ChunkCacheDriver::data_type` returns the
// data type of the associated component array.
TEST_P(ChunkCacheTest, DriverDataType) {
  InitCache(ChunkGridSpecification({
      ChunkGridSpecification::Component{
          SharedArray<const void>(MakeArray<int>({1, 2}))},
      ChunkGridSpecification::Component{
          SharedArray<const void>(MakeArray<float>({{1, 2}, {3, 4}})), {1}},
  }));

  EXPECT_EQ(tensorstore::DataTypeOf<int>(), GetDriver(0)->data_type());

  EXPECT_EQ(tensorstore::DataTypeOf<float>(), GetDriver(1)->data_type());
}

// Tests reading of existing data.
TEST_P(ChunkCacheTest, ReadSingleComponentOneDimensionalExisting) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  // Initialize chunk 1 in the data store.
  {
    auto& chunk = store.GetChunk(span<const Index>({1}));
    chunk.generation = 1;
    chunk.data.resize(1);
    chunk.data[0] = MakeArray<int>({5, 6});
  }

  // Test reading from an existing chunk (read request partially overlaps
  // existing data).
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    // Read chunk 1, position [1], and chunk 2, positions [0] and [1].
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    EXPECT_EQ(read_array, tensorstore::MakeArray({6, 1, 2}));
  }

  // Initialize chunk 2 in the data store.  The cache does not yet reflect this
  // chunk.
  {
    auto& chunk = store.GetChunk(span<const Index>({2}));
    chunk.generation = 1;
    chunk.data.resize(1);
    chunk.data[0] = MakeArray<int>({7, 8});
  }

  // Test reading cached chunks (verifies that the previously cached cells were
  // not evicted).
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    // Read is satisfied without issuing any new read requests.
    EXPECT_TRUE(read_future.result());
    // Read result corresponds to:
    // [0]: chunk 1, position [1]
    // [1]: chunk 2, position [0] (fill value)
    // [2]: chunk 2, position [1] (fill value)
    EXPECT_EQ(read_array, tensorstore::MakeArray({6, 1, 2}));
  }

  // Test re-reading chunks by using a staleness bound of
  // `absl::InfiniteFuture()`, which is never satisfied by an existing read
  // response.  This causes the cache to re-read chunk 2, which has been
  // updated.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfiniteFuture());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(GetStorageGenerationFromNumber(1), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    EXPECT_EQ(read_array, tensorstore::MakeArray({6, 7, 8}));
  }
}

// Test reading the fill value from a two-dimensional chunk cache.
TEST_P(ChunkCacheTest, TwoDimensional) {
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({{1, 2, 3}, {4, 5, 6}})),
      // Transpose the grid dimensions relative to the cell dimensions to test
      // that grid vs. cell indices are correctly handled.
      {1, 0}}}));
  auto read_array = tensorstore::AllocateArray<int>({6, 5});
  auto read_future =
      Read(0,
           ChainResult(
               tensorstore::IdentityTransform(2),
               // Read box starting at {1, 5} of shape {6, 5}.
               tensorstore::Dims(0, 1).TranslateSizedInterval({1, 5}, {6, 5}))
               .value(),
           read_array, absl::InfinitePast());
  for (auto cell_indices : std::vector<std::vector<Index>>{{1, 0},
                                                           {1, 1},
                                                           {1, 2},
                                                           {1, 3},
                                                           {2, 0},
                                                           {2, 1},
                                                           {2, 2},
                                                           {2, 3},
                                                           {3, 0},
                                                           {3, 1},
                                                           {3, 2},
                                                           {3, 3}}) {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(),
                ::testing::ElementsAreArray(cell_indices));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(std::move(r));
  }
  ASSERT_EQ(Status(), GetStatus(read_future.result()));
  EXPECT_EQ(read_array, MakeArray<int>({{6, 4, 5, 6, 4},
                                        {3, 1, 2, 3, 1},
                                        {6, 4, 5, 6, 4},
                                        {3, 1, 2, 3, 1},
                                        {6, 4, 5, 6, 4},
                                        {3, 1, 2, 3, 1}}));
}

// Tests that an invalid transformed array as the read destination leads to an
// error.
TEST_P(ChunkCacheTest, ReadInvalidTransformedArray) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  // Create an invalid transformed array: the array has domain [1,3] but the
  // transform has an output range of [0,2].
  auto read_array = tensorstore::TransformedArray(
      tensorstore::AllocateArray<int>(tensorstore::BoxView<1>({1}, {3})),
      tensorstore::IdentityTransform(tensorstore::BoxView<1>({3})));
  auto read_future =
      Read(0,
           ChainResult(tensorstore::IdentityTransform(1),
                       tensorstore::Dims(0).TranslateSizedInterval(3, 3))
               .value(),
           read_array, absl::InfinitePast());
  EXPECT_THAT(GetStatus(read_future.result()),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST_P(ChunkCacheTest, ReadRequestErrorBasic) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  // Test a read request failing (e.g. because the request to the underlying
  // storage system failed).
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      r.receiver.NotifyDone(absl::UnknownError("Test read error"));
    }
    EXPECT_EQ(absl::UnknownError("Test read error"),
              GetStatus(read_future.result()));
  }

  // Test that the error is cached.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfinitePast());
    EXPECT_EQ(absl::UnknownError("Test read error"),
              GetStatus(read_future.result()));
  }

  // Test that the request is repeated if we require a later timestamp.
  {
    auto read_array = tensorstore::AllocateArray<int>({3});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(3, 3))
                 .value(),
             read_array, absl::InfiniteFuture());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
      store.HandleRead(std::move(r));
    }
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    EXPECT_EQ(read_array, tensorstore::MakeArray({2, 1, 2}));
  }
}

TEST_P(ChunkCacheTest, WriteSingleComponentOneDimensional) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  // Write chunk 1: [1]=3
  // Write chunk 2: [0]=4, [1]=5
  // Write chunk 3: [0]=6
  auto write_future =
      Write(0,
            ChainResult(tensorstore::IdentityTransform(1),
                        tensorstore::Dims(0).TranslateSizedInterval(3, 4))
                .value(),
            MakeArray<int>({3, 4, 5, 6}));

  // Test that reading fully overwritten chunk 2 does not issue a read request.
  {
    auto read_array = tensorstore::AllocateArray<int>({2});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(4, 2))
                 .value(),
             read_array, absl::InfinitePast());
    ASSERT_TRUE(read_future.result());
    EXPECT_EQ(MakeArray<int>({4, 5}), read_array);
  }

  // Test that reading partially overwritten chunk 3 issues a read request.
  {
    auto read_array = tensorstore::AllocateArray<int>({2});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(6, 2))
                 .value(),
             read_array, absl::InfinitePast());
    {
      auto r = log.reads.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(3));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleRead(std::move(r));
    }
    EXPECT_TRUE(read_future.result());
    EXPECT_EQ(read_array, tensorstore::MakeArray({6, 2}));
  }

  // Test that writeback issues a read request for chunk 1, and writeback
  // requests for chunks 2 and 3.
  write_future.Force();
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(2));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  {
    auto& chunk = store.GetChunk(span<const Index>({2}));
    EXPECT_EQ(1, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray<int>({4, 5})));
  }
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(3));
    EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  {
    auto& chunk = store.GetChunk(span<const Index>({3}));
    EXPECT_EQ(1, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray<int>({6, 2})));
  }
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  {
    auto& chunk = store.GetChunk(span<const Index>({1}));
    EXPECT_EQ(1, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray<int>({1, 3})));
  }
  EXPECT_TRUE(write_future.result());
}

// Tests that overwriting a non-present chunk with the fill value results in the
// chunk remaining deleted.
TEST_P(ChunkCacheTest, OverwriteMissingWithFillValue) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Overwrite chunk 1: [0]=1, [1]=2 (matches fill value)
  auto write_future =
      Write(0,
            ChainResult(tensorstore::IdentityTransform(1),
                        tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                .value(),
            MakeArray<int>({1, 2}));
  // Initially the representation is not normalized.
  EXPECT_NE(nullptr, cell_entry->components[0].data);
  write_future.Force();
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    auto complete_writeback = store.HandleWriteback(r);
    // Test that after starting the writeback, the representation has been
    // normalized such that no data is stored.  The normalization is done by
    // `WritebackSnapshot`.
    EXPECT_EQ(nullptr, cell_entry->components[0].data);
    complete_writeback();
  }
  EXPECT_FALSE(store.HasChunk(span<const Index>({1})));
  EXPECT_TRUE(write_future.result());
  // Verify that `cell_entry` still doesn't store data.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
}

// Tests that overwriting an existing chunk with the fill value results in the
// chunk being deleted.
TEST_P(ChunkCacheTest, OverwriteExistingWithFillValue) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Sanity check that entry initially contains no data.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
  // Write initial value to chunk 1: [0]=3, [1]=4
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                  .value(),
              MakeArray<int>({3, 4}));
    write_future.Force();
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleWriteback(r)();
    }
    EXPECT_EQ(Status(), GetStatus(write_future.result()));
    EXPECT_TRUE(store.HasChunk(span<const Index>({1})));
  }
  // Verify that `cell_entry` contains data.
  EXPECT_NE(nullptr, cell_entry->components[0].data);

  // Overwrite chunk 1 with fill value: [0]=1, [1]=2
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                  .value(),
              MakeArray<int>({1, 2}));
    write_future.Force();
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      // Because the chunk was fully overwritten, the prior StorageGeneration is
      // ignored.
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleWriteback(r)();
    }
    EXPECT_FALSE(store.HasChunk(span<const Index>({1})));
    EXPECT_TRUE(write_future.result());
  }
  // Test that after writeback, the representation has been normalized such that
  // no data is stored.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
}

// Tests that deleting a chunk that was previously written results in the chunk
// being deleted.
TEST_P(ChunkCacheTest, DeleteAfterNormalWriteback) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Write initial value to chunk 1: [0]=3, [1]=4
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                  .value(),
              MakeArray<int>({3, 4}));
    write_future.Force();
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleWriteback(r)();
    }
    EXPECT_EQ(Status(), GetStatus(write_future.result()));
    EXPECT_TRUE(store.HasChunk(span<const Index>({1})));
  }
  // Verify that `cell_entry` contains data.
  EXPECT_NE(nullptr, cell_entry->components[0].data);

  // Perform delete.
  auto write_future = cell_entry->Delete();
  write_future.Force();
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleWriteback(r)();
  }
  EXPECT_FALSE(store.HasChunk(span<const Index>({1})));
  EXPECT_TRUE(write_future.result());
  // Test that after writeback, the representation has been normalized such that
  // no data is stored.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
}

TEST_P(ChunkCacheTest, PartialWriteAfterPendingDelete) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Write initial value to chunk 1: [0]=3, [1]=4
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                  .value(),
              MakeArray<int>({3, 4}));
    write_future.Force();
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleWriteback(r)();
    }
    EXPECT_EQ(Status(), GetStatus(write_future.result()));
    EXPECT_TRUE(store.HasChunk(span<const Index>({1})));
  }
  // Verify that `cell_entry` contains data.
  EXPECT_NE(nullptr, cell_entry->components[0].data);

  auto delete_future = cell_entry->Delete();
  // Test that no data is stored.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);

  // Issue partial write: chunk 1, position [0]=3
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 1))
                  .value(),
              MakeArray<int>({3}));
    write_future.Force();
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      // Because the chunk was fully overwritten, the prior StorageGeneration is
      // ignored.
      EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
      store.HandleWriteback(r)();
    }
    EXPECT_EQ(Status(), GetStatus(write_future.result()));
    EXPECT_EQ(Status(), GetStatus(delete_future.result()));

    ASSERT_TRUE(store.HasChunk(span<const Index>({1})));
    auto& chunk = store.GetChunk(span<const Index>({1}));
    EXPECT_EQ(2, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray({3, 2})));
  }

  // Verify that `cell_entry` contains data.
  ASSERT_NE(nullptr, cell_entry->components[0].data);
  EXPECT_EQ(3, static_cast<int*>(cell_entry->components[0].data.get())[0]);

  // Read back value from cache.
  {
    auto read_array = tensorstore::AllocateArray<int>({2});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                 .value(),
             read_array, absl::InfinitePast());
    EXPECT_EQ(Status(), GetStatus(read_future.result()));
    // Read result corresponds to:
    // [0]: chunk 1, position [0]
    // [1]: chunk 1, position [1] (fill value)
    EXPECT_EQ(MakeArray<int>({3, 2}), read_array);
  }
}

// Tests that a partial write works correctly after a delete that has been
// written back.
TEST_P(ChunkCacheTest, PartialWriteAfterWrittenBackDelete) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Cell initially has unknown data because no reads have been performed.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
  auto write_future = cell_entry->Delete();
  // Cell has known data equal to fill value after the Delete (indicated by
  // nullptr).
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
  write_future.Force();
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleWriteback(r)();
  }
  EXPECT_FALSE(store.HasChunk(span<const Index>({1})));
  EXPECT_TRUE(write_future.result());
  // Data should still not be present after writeback.
  EXPECT_EQ(nullptr, cell_entry->components[0].data);

  // Issue partial write: chunk 1, position [0]=3
  {
    auto write_future =
        Write(0,
              ChainResult(tensorstore::IdentityTransform(1),
                          tensorstore::Dims(0).TranslateSizedInterval(2, 1))
                  .value(),
              MakeArray<int>({3}));
    ASSERT_NE(nullptr, cell_entry->components[0].data);
    EXPECT_EQ(3, static_cast<int*>(cell_entry->components[0].data.get())[0]);
    EXPECT_FALSE(cell_entry->components[0].valid_outside_write_mask);

    write_future.Force();
    // Forcing writeback does not affect `valid_outside_write_mask`.
    EXPECT_FALSE(cell_entry->components[0].valid_outside_write_mask);
    {
      auto r = log.writebacks.pop();
      EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
      EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
      auto complete_writeback = store.HandleWriteback(r);
      // WritebackSnapshot fills in the unmasked data values with the fill
      // value.
      EXPECT_TRUE(cell_entry->components[0].valid_outside_write_mask);
      ASSERT_NE(nullptr, cell_entry->components[0].data);
      EXPECT_EQ(3, static_cast<int*>(cell_entry->components[0].data.get())[0]);
      EXPECT_EQ(2, static_cast<int*>(cell_entry->components[0].data.get())[1]);
      complete_writeback();
    }
    EXPECT_EQ(Status(), GetStatus(write_future.result()));
    ASSERT_TRUE(store.HasChunk(span<const Index>({1})));
    auto& chunk = store.GetChunk(span<const Index>({1}));
    EXPECT_EQ(1, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray({3, 2})));
  }

  // Read back value from cache.
  {
    auto read_array = tensorstore::AllocateArray<int>({2});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                 .value(),
             read_array, absl::InfinitePast());
    EXPECT_EQ(Status(), GetStatus(read_future.result()));
    // Read result corresponds to:
    // [0]: chunk 1, position [0]
    // [1]: chunk 1, position [1] (fill value)
    EXPECT_EQ(MakeArray<int>({3, 2}), read_array);
  }
}

// Tests reading a chunk while it has a pending delete.
TEST_P(ChunkCacheTest, ReadAfterPendingDelete) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  // Perform delete.
  auto write_future = cell_entry->Delete();
  EXPECT_EQ(nullptr, cell_entry->components[0].data);

  // Read back value from cache.
  {
    auto read_array = tensorstore::AllocateArray<int>({2});
    auto read_future =
        Read(0,
             ChainResult(tensorstore::IdentityTransform(1),
                         tensorstore::Dims(0).TranslateSizedInterval(2, 2))
                 .value(),
             read_array, absl::InfinitePast());
    EXPECT_EQ(Status(), GetStatus(read_future.result()));
    // Read result corresponds to:
    // [0]: chunk 1, position [0] (fill value)
    // [1]: chunk 1, position [1] (fill value)
    EXPECT_EQ(MakeArray<int>({1, 2}), read_array);
  }
}

// Tests calling Delete on a chunk while a read is pending.
TEST_P(ChunkCacheTest, DeleteWithPendingRead) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));
  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  auto read_array = tensorstore::AllocateArray<int>({2});
  // Read chunk 1, position [0] and [1]
  auto read_future =
      Read(0,
           ChainResult(tensorstore::IdentityTransform(1),
                       tensorstore::Dims(0).TranslateSizedInterval(2, 2))
               .value(),
           read_array, absl::InfinitePast());
  // Wait for the read request to be received.
  auto r = log.reads.pop();

  // Perform delete.  This fully overwrites chunk 1 and makes `read_future`
  // ready even before the read request completes.
  auto write_future = cell_entry->Delete();
  EXPECT_EQ(nullptr, cell_entry->components[0].data);

  EXPECT_EQ(Status(), GetStatus(read_future.result()));
  // Verify that read result matches fill value.
  EXPECT_EQ(MakeArray<int>({1, 2}), read_array);
  EXPECT_EQ(nullptr, cell_entry->components[0].data);

  // Complete the read request even though it has been cancelled.
  EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
  EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
  store.HandleRead(r);
  EXPECT_EQ(nullptr, cell_entry->components[0].data);
}

TEST_P(ChunkCacheTest, WriteToMaskedArrayError) {
  // Dimension 0 is chunked with a size of 2.
  // Dimension 1 has a size of 2 and is not chunked.
  // Fill value is: {{1,2}, {3,4}}
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({{1, 2}, {3, 4}})), {0}}}));

  auto cell_entry = cache->GetEntryForCell(span<const Index>({1}));
  auto write_future =
      Write(0,
            ChainResult(tensorstore::IdentityTransform(2),
                        // Specify out-of-bounds index of 2 for dimension 1.
                        tensorstore::Dims(1)
                            .IndexArraySlice(MakeArray<Index>({2, 2}))
                            .MoveToBack(),
                        // Select single index of dimension 0.
                        tensorstore::Dims(0).IndexSlice(2))
                .value(),
            MakeArray<int>({5, 6}));
  EXPECT_THAT(write_future.result(),
              MatchesStatus(absl::StatusCode::kOutOfRange));
  EXPECT_TRUE(cell_entry->components[0].data);
  EXPECT_EQ(0, cell_entry->components[0].write_mask.num_masked_elements);

  // Verify read of same chunk after failed write returns fill value.
  auto read_array = tensorstore::AllocateArray<int>({2, 2});
  auto read_future =
      Read(0,
           ChainResult(
               tensorstore::IdentityTransform(2),
               tensorstore::Dims(0, 1).TranslateSizedInterval({2, 0}, {2, 2}))
               .value(),
           read_array, absl::InfinitePast());
  // Handle the read request.
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }
  EXPECT_EQ(Status(), GetStatus(read_future.result()));
  EXPECT_EQ(MakeArray<int>({{1, 2}, {3, 4}}), read_array);
}

// Tests writeback where the store was modified after the read and before the
// writeback.
TEST_P(ChunkCacheTest, WriteGenerationMismatch) {
  // Dimension 0 is chunked with a size of 2.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2}))}}));

  auto write_future =
      Write(0,
            ChainResult(tensorstore::IdentityTransform(1),
                        tensorstore::Dims(0).TranslateSizedInterval(3, 1))
                .value(),
            MakeArray<int>({3}));

  write_future.Force();
  // Initialize chunk 1 in the store.
  {
    auto& chunk = store.GetChunk(span<const Index>({1}));
    chunk.generation = 1;
    chunk.data.resize(1);
    chunk.data[0] = MakeArray({5, 6});
  }
  // Handle the read request for chunk 1, which copies the generation 1 data
  // into the cache.
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }
  // Modify chunk 1 in the store.
  {
    auto& chunk = store.GetChunk(span<const Index>({1}));
    chunk.generation = 2;
    chunk.data.resize(1);
    chunk.data[0] = MakeArray({7, 8});
  }
  // Handle the writeback request for chunk 1, which fails due to a generation
  // mismatch.
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(GetStorageGenerationFromNumber(1), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  // Handle the re-issued read request for chunk 1.
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }
  // Handle the re-issued writeback request for chunk 1.
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(1));
    EXPECT_EQ(GetStorageGenerationFromNumber(2), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  EXPECT_TRUE(write_future.result());
  {
    auto& chunk = store.GetChunk(span<const Index>({1}));
    EXPECT_EQ(3, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray({7, 3})));
  }
}

TEST_P(ChunkCacheTest, ModifyDuringWriteback) {
  // Dimension 0 is chunked with a size of 4.
  InitCache(ChunkGridSpecification({ChunkGridSpecification::Component{
      SharedArray<const void>(MakeArray<int>({1, 2, 3, 4}))}}));

  // Partial write to chunk 0: [1]=5, [3]=6
  auto write_future =
      Write(0,
            ChainResult(
                tensorstore::IdentityTransform(1),
                tensorstore::Dims(0).IndexArraySlice(MakeArray<Index>({1, 3})))
                .value(),
            MakeArray<int>({5, 6}));

  write_future.Force();
  // Handle the read request for chunk 0 (the chunk was only partially
  // overwritten).
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(0));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }

  Future<const void> write_future2;
  // Handle the writeback request for chunk 1.
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(0));
    EXPECT_EQ(StorageGeneration::NoValue(), r.generation);
    auto complete_writeback = store.HandleWriteback(std::move(r));
    // While the writeback is in progress, write to chunk 0 again: [2]=7
    write_future2 = Write(
        0,
        ChainResult(tensorstore::IdentityTransform(1),
                    tensorstore::Dims(0).IndexArraySlice(MakeArray<Index>({2})))
            .value(),
        MakeArray<int>({7}));
    complete_writeback();
  }

  EXPECT_TRUE(write_future.result());
  {
    // Verify that the writeback didn't include the second write.
    auto& chunk = store.GetChunk(span<const Index>({0}));
    EXPECT_EQ(1, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray({1, 5, 3, 6})));
    chunk.data[0] = MakeArray({10, 11, 12, 13});
    chunk.generation = 2;
  }
  write_future2.Force();
  // Handle the writeback request for chunk 1 (will fail due to generation
  // mismatch).
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(0));
    EXPECT_EQ(GetStorageGenerationFromNumber(1), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  // Handle the re-issued read request due to the generation mismatch.
  {
    auto r = log.reads.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(0));
    EXPECT_EQ(StorageGeneration::Unknown(), r.generation);
    store.HandleRead(r);
  }
  // Handle the re-issued writeback request for chunk 1.
  {
    auto r = log.writebacks.pop();
    EXPECT_THAT(r.entry()->cell_indices(), ElementsAre(0));
    EXPECT_EQ(GetStorageGenerationFromNumber(2), r.generation);
    store.HandleWriteback(std::move(r))();
  }
  EXPECT_TRUE(write_future2.result());
  // Verify that the writeback only modified the single element touched by the
  // second write.
  {
    auto& chunk = store.GetChunk(span<const Index>({0}));
    EXPECT_EQ(3, chunk.generation);
    EXPECT_THAT(chunk.data, ElementsAre(MakeArray({10, 11, 7, 13})));
  }
}

}  // namespace
