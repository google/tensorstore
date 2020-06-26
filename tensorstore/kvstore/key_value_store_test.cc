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

#include "tensorstore/kvstore/key_value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::KeyValueStore;
using tensorstore::MatchesStatus;
using tensorstore::StrCat;

TEST(KeyValueStoreTest, OpenInvalid) {
  auto context = tensorstore::Context::Default();
  EXPECT_THAT(KeyValueStore::Open(context, {{"driver", "invalid"}}).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"driver\": "
                            "\"invalid\" is not registered"));
}

}  // namespace
