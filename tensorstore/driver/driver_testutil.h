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

#ifndef TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
#define TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_

#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/spec_request_options.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal {

struct TestTensorStoreDriverSpecRoundtripOptions {
  std::string test_name;
  ::nlohmann::json full_spec;
  ::nlohmann::json create_spec{::nlohmann::json::value_t::discarded};
  ::nlohmann::json minimal_spec;
  ContextToJsonOptions to_json_options = IncludeDefaults{false};
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated, tensorstore::atomic_isolated};
  bool check_not_found_before_create = true;
  bool check_not_found_before_commit = true;
  bool check_transactional_open_before_commit = true;
  bool write_value_to_create = false;
};

/// Tests that a TensorStore can be successfully created from `full_spec`, that
/// its full and minimal specs are `full_spec` and `minimal_spec`, respectively,
/// and that a TensorStore can then be successfully opened from `minimal_spec`,
/// and that the resultant TensorStore also has full and minimal specs of
/// `full_spec` and `minimal_spec`, respectively.
void RegisterTensorStoreDriverSpecRoundtripTest(
    TestTensorStoreDriverSpecRoundtripOptions options);

/// Tests that applying `options` to `orig_spec` (via `Spec::Convert`) results
/// in `expected_converted_spec`.
void TestTensorStoreDriverSpecConvert(::nlohmann::json orig_spec,
                                      const SpecRequestOptions& options,
                                      ::nlohmann::json expected_converted_spec);

struct TensorStoreDriverBasicFunctionalityTestOptions {
  std::string test_name;
  ::nlohmann::json create_spec;
  IndexDomain<> expected_domain;
  SharedOffsetArray<const void> initial_value;
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated};

  /// Optional.  Specifies a function to use to compare arrays (e.g. to perform
  /// approximate comparison to accommodate lossy compression).
  ///
  /// The `compare_arrays` function should call a GTest assertion to record any
  /// mismatch.  If not specified, the default function simply calls:
  /// `EXPECT_EQ(expected, actual);`.
  std::function<void(OffsetArrayView<const void> expected,
                     OffsetArrayView<const void> actual)>
      compare_arrays;

  bool check_not_found_before_commit = true;
};

/// Tests that `create_spec` creates a TensorStore with the specified domain and
/// data type, and that `Read`, `Write`, and `ResolveBounds` functions work.
void RegisterTensorStoreDriverBasicFunctionalityTest(
    TensorStoreDriverBasicFunctionalityTestOptions options);

struct TestTensorStoreDriverResizeOptions {
  std::string test_name;
  /// Specifies the initial bounds to use.
  Box<> initial_bounds;
  /// Returns a TensorStore spec for the specified bounds.
  std::function<::nlohmann::json(BoxView<> bounds)> get_create_spec;
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated};
};

/// Tests metadata-only resize functionality.
void RegisterTensorStoreDriverResizeTest(
    TestTensorStoreDriverResizeOptions options);

/// Returns the data from the individual chunks obtained from reading a
/// TensorStore.
///
/// This can be useful for testing.
Future<std::vector<std::pair<SharedOffsetArray<void>, IndexTransform<>>>>
ReadAsIndividualChunks(TensorStore<> store);

/// Returns the individual chunks obtained from reading a TensorStore.
///
/// This can be useful for testing.
Future<std::vector<std::pair<ReadChunk, IndexTransform<>>>> CollectReadChunks(
    TensorStore<> store);

/// Mock TensorStore driver that records Read/Write requests in a queue.
class MockDriver : public Driver {
 public:
  using Ptr = PtrT<MockDriver>;

  explicit MockDriver(DataType data_type, DimensionIndex rank,
                      Executor data_copy_executor = InlineExecutor{})
      : data_type_(data_type),
        rank_(rank),
        executor_(std::move(data_copy_executor)) {}

  struct ReadRequest {
    internal::OpenTransactionPtr transaction;
    IndexTransform<> transform;
    ReadChunkReceiver receiver;
  };

  struct WriteRequest {
    internal::OpenTransactionPtr transaction;
    IndexTransform<> transform;
    WriteChunkReceiver receiver;
  };

  DataType data_type() override { return data_type_; }
  DimensionIndex rank() override { return rank_; }

  void Read(internal::OpenTransactionPtr transaction,
            IndexTransform<> transform, ReadChunkReceiver receiver) override;

  void Write(internal::OpenTransactionPtr transaction,
             IndexTransform<> transform, WriteChunkReceiver receiver) override;

  Executor data_copy_executor() override { return executor_; }

  TensorStore<> Wrap(IndexTransform<> transform = {});

  DataType data_type_;
  DimensionIndex rank_;
  Executor executor_;

  ConcurrentQueue<ReadRequest> read_requests;
  ConcurrentQueue<WriteRequest> write_requests;
};

/// Returns a `ReadChunk` that simply reads from the specified array.
ReadChunk MakeArrayBackedReadChunk(
    NormalizedTransformedArray<Shared<const void>> data);
ReadChunk MakeArrayBackedReadChunk(
    SharedOffsetArray<const void, dynamic_rank, view> data);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
