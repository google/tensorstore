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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/write.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::AnyFlowReceiver;
using ::tensorstore::IndexTransform;
using ::tensorstore::MatchesStatus;
using ::tensorstore::WriteProgress;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::WriteChunk;

class ChunkErrorDriver : public tensorstore::internal::Driver {
 public:
  tensorstore::DataType dtype() override { return tensorstore::dtype_v<int>; }
  tensorstore::DimensionIndex rank() override { return 0; }
  void Read(tensorstore::internal::OpenTransactionPtr transaction,
            IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override {}
  void Write(tensorstore::internal::OpenTransactionPtr transaction,
             IndexTransform<> transform,
             AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>
                 receiver) override {
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
  auto write_result = tensorstore::internal::DriverWrite(
      /*executor=*/tensorstore::InlineExecutor{},
      /*source=*/tensorstore::MakeScalarArray(3),
      /*target=*/
      {/*.driver=*/driver,
       /*.transform=*/tensorstore::IdentityTransform(0)},
      /*options=*/
      {/*.progress_function=*/[&write_progress](WriteProgress progress) {
        write_progress.push_back(progress);
      }});
  EXPECT_THAT(write_result.copy_future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Chunk error"));
  EXPECT_EQ(write_result.copy_future.status(),
            write_result.commit_future.status());
}

}  // namespace
