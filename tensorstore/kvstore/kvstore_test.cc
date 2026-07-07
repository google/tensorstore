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
#include "tensorstore/internal/testing/hardening.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::StatusIs;
using ::testing::HasSubstr;

TEST(KeyValueStoreTest, OpenInvalidTensorStore) {
  auto context = tensorstore::Context::Default();
  EXPECT_THAT(
      kvstore::Open({{"driver", "json"}}, context).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Error parsing object member \"driver\": "
                    "\"json\" is a TensorStore driver, not a KvStore driver")));
}

TEST(KeyValueStoreTest, OpenInvalid) {
  auto context = tensorstore::Context::Default();
  EXPECT_THAT(
      kvstore::Open({{"driver", "invalid"}}, context).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Error parsing object member \"driver\": "
                         "\"invalid\" is not a registered KvStore driver")));
}

TEST(KeyValueStoreTest, OpenInconsistentTransactions) {
  auto txn1 = tensorstore::Transaction(tensorstore::isolated);
  auto txn2 = tensorstore::Transaction(tensorstore::isolated);
  EXPECT_THAT(kvstore::Open({{"driver", "memory"}}, txn1, txn2).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Inconsistent transactions specified")));
}

TEST(KeyValueStoreTest, EmptyUrl) {
  EXPECT_THAT(kvstore::Spec::FromJson(""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("URL must be non-empty")));
}

TEST(KeyValueStoreTest, TensorStoreUrl) {
  EXPECT_THAT(
      kvstore::Spec::FromJson("json:"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unsupported URL scheme \"json\" in \"json:\"")));
}

TEST(KeyValueStoreTest, Invalid) {
  kvstore::Spec spec;
  EXPECT_FALSE(spec.valid());
  EXPECT_THAT(
      spec.BindContext(tensorstore::Context::Default()),
      StatusIs(absl::StatusCode::kInvalidArgument, "Invalid kvstore spec"));
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(spec.UnbindContext(), "");
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(spec.StripContext(), "");
  TENSORSTORE_EXPECT_DEATH_IF_HARDENED(spec.context_binding_state(), "");

  kvstore::KvStore store;
  EXPECT_FALSE(store.valid());
  EXPECT_THAT(store.spec(),
              StatusIs(absl::StatusCode::kInvalidArgument, "Invalid kvstore"));
  EXPECT_THAT(store.base(),
              StatusIs(absl::StatusCode::kInvalidArgument, "Invalid kvstore"));
}

}  // namespace
