// Copyright 2026 The TensorStore Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/internal/testing/hardening.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::StatusIs;

TEST(TensorStoreTest, ReadInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(
      tensorstore::Read(store).result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "TensorStore is not valid"));
}

TEST(TensorStoreTest, WriteInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(
      tensorstore::Write(tensorstore::MakeArray<int>({{1}}), store).result(),
      StatusIs(absl::StatusCode::kInvalidArgument, "TensorStore is not valid"));
}

TEST(TensorStoreTest, SpecInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.spec(), StatusIs(absl::StatusCode::kInvalidArgument,
                                     "TensorStore is not valid"));
}

TEST(TensorStoreTest, SchemaInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.schema(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       "TensorStore is not valid"));
}

TEST(TensorStoreTest, ChunkLayoutInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.chunk_layout(), StatusIs(absl::StatusCode::kInvalidArgument,
                                             "TensorStore is not valid"));
}

TEST(TensorStoreTest, CodecInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.codec(), StatusIs(absl::StatusCode::kInvalidArgument,
                                      "TensorStore is not valid"));
}

TEST(TensorStoreTest, DimensionUnitsInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(
      store.dimension_units(),
      StatusIs(absl::StatusCode::kInvalidArgument, "TensorStore is not valid"));
}

TEST(TensorStoreTest, CopyInvalidSource) {
  tensorstore::TensorStore<int, 2> store;
  auto valid_store = tensorstore::FromArray(tensorstore::MakeArray<int>({{1}}));
  EXPECT_THAT(tensorstore::Copy(store, valid_store).commit_future.result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Source TensorStore is not valid"));
}

TEST(TensorStoreTest, CopyInvalidTarget) {
  tensorstore::TensorStore<int, 2> store;
  auto valid_store = tensorstore::FromArray(tensorstore::MakeArray<int>({{1}}));
  EXPECT_THAT(tensorstore::Copy(valid_store, store).commit_future.result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Target TensorStore is not valid"));
}

TEST(TensorStoreTest, ToUrlInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.ToUrl(), StatusIs(absl::StatusCode::kInvalidArgument,
                                      "TensorStore is not valid"));
}

TEST(TensorStoreTest, FillValueInvalid) {
  tensorstore::TensorStore<int, 2> store;
  EXPECT_THAT(store.fill_value(), StatusIs(absl::StatusCode::kInvalidArgument,
                                           "TensorStore is not valid"));
}

TEST(TensorStoreTest, Hardening) {
  tensorstore::TensorStore<int, 2> store;
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(store.dtype(), "");
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(store.rank(), "");
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(store.domain(), "");
}

}  // namespace
