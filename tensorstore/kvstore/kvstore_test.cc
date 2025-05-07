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

#include "tensorstore/kvstore/kvstore.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::MatchesStatus;

TEST(KeyValueStoreTest, OpenInvalidTensorStore) {
  auto context = tensorstore::Context::Default();
  EXPECT_THAT(
      kvstore::Open({{"driver", "json"}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"driver\": "
                    "\"json\" is a TensorStore driver, not a KvStore driver"));
}

TEST(KeyValueStoreTest, OpenInvalid) {
  auto context = tensorstore::Context::Default();
  EXPECT_THAT(kvstore::Open({{"driver", "invalid"}}, context).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"driver\": "
                            "\"invalid\" is not a registered KvStore driver"));
}

TEST(KeyValueStoreTest, OpenInconsistentTransactions) {
  auto txn1 = tensorstore::Transaction(tensorstore::isolated);
  auto txn2 = tensorstore::Transaction(tensorstore::isolated);
  EXPECT_THAT(kvstore::Open({{"driver", "memory"}}, txn1, txn2).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Inconsistent transactions specified"));
}

TEST(KeyValueStoreTest, EmptyUrl) {
  EXPECT_THAT(kvstore::Spec::FromJson(""),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "URL must be non-empty"));
}

TEST(KeyValueStoreTest, TensorStoreUrl) {
  EXPECT_THAT(
      kvstore::Spec::FromJson("json:"),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"json\" is a kvstore-based TensorStore URL scheme: "
                    "unsupported URL scheme \"json\" in \"json:\""));
}

}  // namespace
