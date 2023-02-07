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

#include "absl/random/bit_gen_ref.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {

struct TestTensorStoreDriverSpecRoundtripOptions {
  std::string test_name;
  ::nlohmann::json full_spec;
  ::nlohmann::json create_spec = ::nlohmann::json::value_t::discarded;
  ::nlohmann::json minimal_spec;
  SpecRequestOptions spec_request_options;
  JsonSerializationOptions to_json_options;
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
void TestTensorStoreDriverSpecConvertImpl(
    ::nlohmann::json orig_spec, ::nlohmann::json expected_converted_spec,
    SpecConvertOptions&& options);

template <typename... Option>
std::enable_if_t<IsCompatibleOptionSequence<SpecConvertOptions, Option...>,
                 void>
TestTensorStoreDriverSpecConvert(::nlohmann::json orig_spec,
                                 ::nlohmann::json expected_converted_spec,
                                 Option&&... option) {
  SpecConvertOptions options;
  if (absl::Status status;
      !((status = options.Set(std::forward<Option>(option))).ok() && ...)) {
    TENSORSTORE_ASSERT_OK(status);
  }
  TestTensorStoreDriverSpecConvertImpl(std::move(orig_spec),
                                       std::move(expected_converted_spec),
                                       std::move(options));
}

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
  template <typename... Args>
  static ReadWritePtr<MockDriver> Make(ReadWriteMode read_write_mode,
                                       Args&&... args) {
    return MakeReadWritePtr<MockDriver>(read_write_mode,
                                        std::forward<Args>(args)...);
  }

  explicit MockDriver(DataType dtype, DimensionIndex rank,
                      Executor data_copy_executor = InlineExecutor{})
      : dtype_(dtype), rank_(rank), executor_(std::move(data_copy_executor)) {}

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

  DataType dtype() override { return dtype_; }
  DimensionIndex rank() override { return rank_; }

  void Read(internal::OpenTransactionPtr transaction,
            IndexTransform<> transform, ReadChunkReceiver receiver) override;

  void Write(internal::OpenTransactionPtr transaction,
             IndexTransform<> transform, WriteChunkReceiver receiver) override;

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override;

  Executor data_copy_executor() override { return executor_; }

  TensorStore<> Wrap(IndexTransform<> transform = {});

  DataType dtype_;
  DimensionIndex rank_;
  Executor executor_;

  ConcurrentQueue<ReadRequest> read_requests;
  ConcurrentQueue<WriteRequest> write_requests;
};

/// Returns a `ReadChunk` that simply reads from the specified array.
ReadChunk MakeArrayBackedReadChunk(TransformedArray<Shared<const void>> data);

/// DriverRandomOperationTester implements random/fuzz based testing for a
/// driver.
class DriverRandomOperationTester {
 public:
  DriverRandomOperationTester(
      absl::BitGenRef gen,
      TensorStoreDriverBasicFunctionalityTestOptions options);

  void TestBasicFunctionality(TransactionMode transaction_mode,
                              size_t num_iterations);

  void TestConcurrentWrites(TransactionMode transaction_mode,
                            size_t num_iterations);

  void TestMultiTransactionWrite(TransactionMode mode, size_t num_transactions,
                                 size_t num_iterations, bool use_random_values);

  absl::BitGenRef gen;  // Not owned.
  TensorStoreDriverBasicFunctionalityTestOptions options;
  bool log = true;
};

// Options for `TestDriverWriteReadChunks`.
struct TestDriverWriteReadChunksOptions {
  // Context spec to use when opening `tensorstore_spec`.
  tensorstore::Context::Spec context_spec;
  // TensorStore spec to use to open or create the store.
  tensorstore::Spec tensorstore_spec;

  // Strategy to use for choosing a sequence of chunks.
  enum Strategy {
    // Partition the domain into a regular grid with a cell shape of
    // `chunk_shape`, and select chunks in lexicographic (row major) order.
    kSequential,
    // Choose rectangular regions with a shape of `chunk_shape` and randomly
    // sampled start positions (not aligned to a grid).
    kRandom,
  };

  // Strategy to use for choosing which chunks to read and write.
  Strategy strategy = kRandom;

  // Specifies the chunk shape.  Mutually exclusive with `chunk_bytes`.
  std::optional<std::vector<Index>> chunk_shape;

  // Specifies the (approximate) number of bytes per chunk.  A chunk shape is
  // chosen automatically based on this constraint.  Mutually exclusive with
  // `chunk_shape`.
  std::optional<size_t> chunk_bytes;

  // Number of bytes to read.  If negative, specifies a multiple of the total
  // number of bytes within the full domain of the TensorStore.
  int64_t total_read_bytes = -1;

  // Number of bytes to write.  If negative, specifies a multiple of the total
  // number of bytes within the full domain of the TensorStore.
  int64_t total_write_bytes = -2;

  // Number of times to repeat the reads.
  int64_t repeat_reads = 1;

  // Number of times to repeat the writes.
  int64_t repeat_writes = 1;
};

// Tests concurrently reading and/or writing multiple chunks.
absl::Status TestDriverWriteReadChunks(
    absl::BitGenRef gen, const TestDriverWriteReadChunksOptions& options);

// Tests concurrently reading or writing multiple chunks.
//
// Args:
//   gen: Random source.
//   ts: TensorStore on which to operate.
//   chunk_shape: Chunk shape to use.
//   total_bytes: (Approximate) total number of bytes to read/write.
//   strategy: Strategy for selecting chunks to read or write.
//   read: If `true`, perform reads.  If `false`, perform fwrites.
absl::Status TestDriverReadOrWriteChunks(
    absl::BitGenRef gen, tensorstore::TensorStore<> ts,
    span<const Index> chunk_shape, int64_t total_bytes,
    TestDriverWriteReadChunksOptions::Strategy strategy, bool read);

void TestTensorStoreCreateWithSchemaImpl(::nlohmann::json json_spec,
                                         const Schema& schema);

/// Creates a TensorStore using `json_spec` and the specified schema options,
/// and verifies that its schema matches the specified schema.
template <typename... Option>
std::enable_if_t<IsCompatibleOptionSequence<Schema, Option...>, void>
TestTensorStoreCreateWithSchema(::nlohmann::json json_spec,
                                Option&&... option) {
  Schema schema;
  if (absl::Status status; !((status = schema.Set(option)).ok() && ...)) {
    TENSORSTORE_ASSERT_OK(status);
  }
  TestTensorStoreCreateWithSchemaImpl(std::move(json_spec), schema);
}

void TestTensorStoreCreateCheckSchemaImpl(::nlohmann::json json_spec,
                                          const Schema& schema);

/// Creates a TensorStore using `json_spec` and verifies that its schema matches
/// the specified schema.
template <typename... Option>
std::enable_if_t<IsCompatibleOptionSequence<Schema, Option...>, void>
TestTensorStoreCreateCheckSchema(::nlohmann::json json_spec,
                                 Option&&... option) {
  Schema schema;
  if (absl::Status status; !((status = schema.Set(option)).ok() && ...)) {
    TENSORSTORE_ASSERT_OK(status);
  }
  TestTensorStoreCreateCheckSchemaImpl(std::move(json_spec), schema);
}

void TestTensorStoreCreateCheckSchema(::nlohmann::json json_spec,
                                      ::nlohmann::json json_schema);

void TestSpecSchemaImpl(::nlohmann::json json_spec, const Schema& schema);

/// Tests that the schema obtained from `json_spec` is equal to the specified
/// schema.
template <typename... Option>
std::enable_if_t<IsCompatibleOptionSequence<Schema, Option...>, void>
TestSpecSchema(::nlohmann::json json_spec, Option&&... option) {
  Schema schema;
  if (absl::Status status; !((status = schema.Set(option)).ok() && ...)) {
    TENSORSTORE_ASSERT_OK(status);
  }
  TestSpecSchemaImpl(std::move(json_spec), schema);
}

void TestSpecSchema(::nlohmann::json json_spec, ::nlohmann::json json_schema);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
