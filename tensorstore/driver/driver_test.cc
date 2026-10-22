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

#include "tensorstore/driver/driver.h"

#include <stddef.h>

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/write.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AnyFlowReceiver;
using ::tensorstore::StatusIs;
using ::tensorstore::WriteProgress;
using ::tensorstore::internal::DriverWrite;
using ::tensorstore::internal::DriverWriteOptions;
using ::testing::HasSubstr;

class ChunkErrorDriver : public tensorstore::internal::Driver {
 public:
  tensorstore::DataType dtype() override { return tensorstore::dtype_v<int>; }
  tensorstore::DimensionIndex rank() override { return 0; }
  void Read(ReadRequest request, ReadChunkReceiver receiver) override {}
  void Write(WriteRequest request, WriteChunkReceiver receiver) override {
    tensorstore::execution::set_starting(receiver, [] {});
    tensorstore::execution::set_error(receiver,
                                      absl::UnknownError("Chunk error"));
    tensorstore::execution::set_stopping(receiver);
  }
  void GarbageCollectionVisit(
      tensorstore::garbage_collection::GarbageCollectionVisitor& visitor)
      const final {
    // No-op
  }
  tensorstore::Executor data_copy_executor() override {
    return tensorstore::InlineExecutor{};
  }
};

TEST(WriteTest, ChunkError) {
  auto driver = tensorstore::internal::MakeReadWritePtr<ChunkErrorDriver>(
      tensorstore::ReadWriteMode::read_write);
  std::vector<WriteProgress> write_progress;
  DriverWriteOptions options;
  options.progress_function = tensorstore::WriteProgressFunction{
      [&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }};

  auto write_result = DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/tensorstore::MakeScalarArray(3),
      /*target=*/
      {/*.driver=*/driver,
       /*.transform=*/tensorstore::IdentityTransform(0)},
      std::move(options));
  EXPECT_THAT(write_result.copy_future.result(),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Chunk error")));
  EXPECT_EQ(write_result.copy_future.status(),
            write_result.commit_future.status());
}

class LifecycleWriteDriver : public tensorstore::internal::Driver {
 public:
  LifecycleWriteDriver()
      : backing_(tensorstore::AllocateArray<int>({4}, tensorstore::c_order,
                                                 tensorstore::value_init)) {}
  tensorstore::DataType dtype() override { return tensorstore::dtype_v<int>; }
  tensorstore::DimensionIndex rank() override { return 1; }
  void Read(ReadRequest request, ReadChunkReceiver receiver) override {
    struct ReadChunkImpl {
      tensorstore::SharedArray<int> backing;
      absl::Status operator()(
          tensorstore::internal::LockCollection& lock_collection) {
        return absl::OkStatus();
      }
      tensorstore::Result<tensorstore::internal::NDIterable::Ptr> operator()(
          tensorstore::internal::ReadChunk::BeginRead,
          tensorstore::IndexTransform<> chunk_transform,
          tensorstore::internal::Arena* arena) {
        return tensorstore::internal::GetTransformedArrayNDIterable(
            backing, chunk_transform, arena);
      }
    };
    tensorstore::execution::set_starting(receiver, [] {});
    auto cell_transform =
        tensorstore::IdentityTransform(request.transform.input_domain());
    tensorstore::execution::set_value(
        receiver,
        tensorstore::internal::ReadChunk{ReadChunkImpl{backing_},
                                         std::move(request.transform)},
        std::move(cell_transform));
    tensorstore::execution::set_done(receiver);
    tensorstore::execution::set_stopping(receiver);
  }

  void Write(WriteRequest request, WriteChunkReceiver receiver) override {
    struct WriteChunkImpl {
      LifecycleWriteDriver* self;
      absl::Status operator()(
          tensorstore::internal::LockCollection& lock_collection) {
        return absl::OkStatus();
      }
      tensorstore::Result<tensorstore::internal::NDIterable::Ptr> operator()(
          tensorstore::internal::WriteChunk::BeginWrite,
          tensorstore::IndexTransform<> chunk_transform,
          tensorstore::internal::Arena* arena) {
        ++self->begin_write_count;
        return tensorstore::internal::GetTransformedArrayNDIterable(
            self->backing_, chunk_transform, arena);
      }
      tensorstore::internal::WriteChunk::EndWriteResult operator()(
          tensorstore::internal::WriteChunk::EndWrite,
          tensorstore::IndexTransformView<> chunk_transform, bool success,
          tensorstore::internal::Arena* arena) {
        ++self->end_write_count;
        return {};
      }
      bool operator()(
          tensorstore::internal::WriteChunk::WriteArray,
          tensorstore::IndexTransformView<> chunk_transform,
          tensorstore::internal::WriteChunk::GetWriteSourceArrayFunction
              get_source_array,
          tensorstore::internal::Arena* arena,
          tensorstore::internal::WriteChunk::EndWriteResult& end_write_result) {
        return false;
      }
    };
    tensorstore::execution::set_starting(receiver, [] {});
    auto cell_transform =
        tensorstore::IdentityTransform(request.transform.input_domain());
    tensorstore::execution::set_value(
        receiver,
        tensorstore::internal::WriteChunk{WriteChunkImpl{this},
                                          std::move(request.transform)},
        std::move(cell_transform));
    tensorstore::execution::set_done(receiver);
    tensorstore::execution::set_stopping(receiver);
  }

  void GarbageCollectionVisit(
      tensorstore::garbage_collection::GarbageCollectionVisitor& visitor)
      const final {}
  tensorstore::Executor data_copy_executor() override {
    return tensorstore::InlineExecutor{};
  }

  tensorstore::SharedArray<int> backing_;
  int begin_write_count = 0;
  int end_write_count = 0;
};

TEST(WriteTest, SourceNDIterableErrorDoesNotLeaveUnmatchedBeginWrite) {
  auto driver = tensorstore::internal::MakeReadWritePtr<LifecycleWriteDriver>(
      tensorstore::ReadWriteMode::read_write);
  auto source_array = tensorstore::MakeArray<int>({10, 20, 30, 40});
  auto oob_indices = tensorstore::MakeArray<tensorstore::Index>({0, 1, 99, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bad_source,
      tensorstore::TransformedSharedArray<const void>(source_array) |
          tensorstore::Dims(0).IndexArraySlice(oob_indices));

  auto write_result = DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/bad_source,
      /*target=*/
      {/*.driver=*/driver,
       /*.transform=*/tensorstore::IdentityTransform(
           tensorstore::BoxView<1>({4}))},
      /*options=*/{});
  EXPECT_THAT(write_result.copy_future.result(),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_EQ(driver->begin_write_count, driver->end_write_count);
  EXPECT_EQ(driver->begin_write_count, 0);
}

}  // namespace
