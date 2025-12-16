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

#include "tensorstore/kvstore/test_util/copy_ops.h"

#include <gmock/gmock.h>
#include "absl/strings/cord.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util/internal.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {

void TestKeyValueStoreCopyRange(const KvStore& store) {
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "w/a", absl::Cord("w_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/a", absl::Cord("value_a")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "x/b", absl::Cord("value_b")));
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "z/a", absl::Cord("z_a")));
  TENSORSTORE_ASSERT_OK(kvstore::ExperimentalCopyRange(
      store.WithPathSuffix("x/"), store.WithPathSuffix("y/")));
  EXPECT_THAT(GetMap(store), IsOkAndHolds(::testing::ElementsAreArray({
                                 ::testing::Pair("w/a", absl::Cord("w_a")),
                                 ::testing::Pair("x/a", absl::Cord("value_a")),
                                 ::testing::Pair("x/b", absl::Cord("value_b")),
                                 ::testing::Pair("y/a", absl::Cord("value_a")),
                                 ::testing::Pair("y/b", absl::Cord("value_b")),
                                 ::testing::Pair("z/a", absl::Cord("z_a")),
                             })));
}

}  // namespace internal
}  // namespace tensorstore
