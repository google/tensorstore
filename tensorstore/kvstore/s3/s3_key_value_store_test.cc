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

#include <stddef.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::StorageGeneration;
using ::tensorstore::StrCat;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::SetDefaultHttpTransport;

static constexpr char kUriScheme[] = "s3";
static constexpr char kDriver[] = "s3";

Context DefaultTestContext() {
  // Opens the s3 driver with small exponential backoff values.
  return Context{Context::Spec::FromJson({{"s3_request_retries",
                                           {{"max_retries", 3},
                                            {"initial_delay", "1ms"},
                                            {"max_delay", "10ms"}}}})
                     .value()};
}

TEST(S3KeyValueStoreTest, Basic) {
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", kDriver}, {"bucket", "abcdefgh"}}, context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(spec.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(
                  MatchesJson({{"driver", kDriver}, {"bucket", "abcdefgh"},
                               {"endpoint", ""}, {"profile", "default",},
                               {"requester_pays", false}})));

  // TODO(sjperkins):
  // Reintroduce when Read, Write, List and Delete are implemented
  // tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
}

TEST(S3KeyValueStoreTest, BadBucketNames) {
  auto context = DefaultTestContext();
  for (auto bucket :
       {"a", "_abc", "abc_", "ABC", "a..b", "a.-.b"}) {
    EXPECT_FALSE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}}, context)
            .result())
        << "bucket: " << bucket;
  }
  for (auto bucket : {"abc", "abc.1-2-3.abc",
        "a."
        "0123456789123456789012345678912345678901234567891234567890"
        "1234567891234567890123456789123456789012345678912345678901"
        "23456789123456789.B"}) {
    EXPECT_TRUE(
        kvstore::Open({{"driver", kDriver}, {"bucket", bucket}}, context)
            .result())
        << "bucket: " << bucket;
  }
}

};
