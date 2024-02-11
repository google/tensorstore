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

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/status_testutil.h"

/// WARNING: This can modify live data!
///
/// This is a test-only binary which runs standard tests against an arbitrary
/// kvstore spec.
///
/// WARNING: This can modify live data!

/* Examples

bazel run //tensorstore/kvstore:live_kvstore_test -- \
    --kvstore_spec='"file:///tmp/tensorstore_kvstore_test"'

bazel run //tensorstore/kvstore:live_kvstore_test -- \
    --kvstore_spec='{"driver":"ocdbt","base":"file:///tmp/tensorstore_kvstore_test"}'

*/

tensorstore::kvstore::Spec DefaultKvStoreSpec() {
  return tensorstore::kvstore::Spec::FromJson({{"driver", "memory"}}).value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          DefaultKvStoreSpec(),
          "KvStore spec for reading data.  See examples at the start of the "
          "source file.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.");

namespace {

using ::tensorstore::Context;

class LiveKvStoreTest : public ::testing::Test {
 public:
  Context GetContext();
  tensorstore::kvstore::Spec GetSpec();
};

Context LiveKvStoreTest::GetContext() {
  static Context* context =
      new Context(absl::GetFlag(FLAGS_context_spec).value);
  return *context;
}

tensorstore::kvstore::Spec LiveKvStoreTest::GetSpec() {
  auto kvstore_spec = absl::GetFlag(FLAGS_kvstore_spec).value;
  if (!kvstore_spec.path.empty() && kvstore_spec.path.back() != '/') {
    kvstore_spec.AppendSuffix("/");
  }
  return kvstore_spec;
}

TEST_F(LiveKvStoreTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open(GetSpec(), GetContext()).result());

  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(LiveKvStoreTest, DeleteRange) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open(GetSpec(), GetContext()).result());
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

void DumpAllMetrics() {
  std::vector<std::string> lines;
  for (const auto& metric :
       tensorstore::internal_metrics::GetMetricRegistry().CollectWithPrefix(
           "")) {
    tensorstore::internal_metrics::FormatCollectedMetric(
        metric, [&lines](bool has_value, std::string line) {
          if (has_value) lines.emplace_back(std::move(line));
        });
  }

  // `lines` is unordered, which isn't great for benchmark comparison.
  std::sort(std::begin(lines), std::end(lines));
  std::cout << std::endl;
  for (const auto& l : lines) {
    std::cout << l << std::endl;
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  int test_result = RUN_ALL_TESTS();
  DumpAllMetrics();
  return test_result;
}
