// Copyright 2023 The TensorStore Authors
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

#include <memory>
#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generation_testutil.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::MatchesStatus;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesTimestampedStorageGeneration;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::SetDefaultHttpTransport;

namespace {

Context DefaultTestContext() {
  // Opens the s3 driver with small exponential backoff values.
  return Context{Context::Spec::FromJson({{"s3_request_retries",
                                           {{"max_retries", 1},
                                            {"initial_delay", "1ms"},
                                            {"max_delay", "2ms"}}}})
                     .value()};
}

TEST(S3KeyValueStoreTest, BadBucketNames) {
  auto context = DefaultTestContext();
  for (auto bucket : {"a", "_abc", "abc_", "a..b", "a.-.b"}) {
    EXPECT_FALSE(kvstore::Open({{"driver", "s3"},
                                {"bucket", bucket},
                                {"endpoint", "https://i.dont.exist"}},
                               context)
                     .result())
        << "bucket: " << bucket;
  }
  for (auto bucket :
       {"abc", "abc.1-2-3.abc",
        "a."
        "0123456789123456789012345678912345678901234567891234567890"
        "1234567891234567890123456789123456789012345678912345678901"
        "23456789123456789.B"}) {
    EXPECT_TRUE(kvstore::Open({{"driver", "s3"},
                               {"bucket", bucket},
                               {"endpoint", "https://i.dont.exist"},
                               {"aws_region", "us-east-1"}},
                              context)
                    .result())
        << "bucket: " << bucket;
  }
}

TEST(S3KeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_write_read = false;
  options.check_data_persists = false;
  options.check_data_after_serialization = false;
  options.full_spec = {{"driver", "s3"}, {"bucket", "mybucket"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(S3KeyValueStoreTest, InvalidSpec) {
  auto context = DefaultTestContext();

  // Test with extra key.
  EXPECT_THAT(kvstore::Open(
                  {{"driver", "s3"}, {"bucket", "my-bucket"}, {"extra", "key"}},
                  context)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with missing `"bucket"` key.
  EXPECT_THAT(kvstore::Open({{"driver", "s3"}}, context).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"bucket"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "s3"}, {"bucket", "a"}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Mock-based tests for s3.
// TODO: Add a more sophisticated s3 mock transport.
class MyMockTransport : public HttpTransport {
 public:
  MyMockTransport(
      const absl::flat_hash_map<std::string, HttpResponse>& url_to_response)
      : url_to_response_(url_to_response) {}

  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override {
    ABSL_LOG(INFO) << request;
    auto it = url_to_response_.find(
        tensorstore::StrCat(request.method, " ", request.url));
    if (it != url_to_response_.end()) {
      return it->second;
    }
    return HttpResponse{404, absl::Cord(), {}};
  }

  const absl::flat_hash_map<std::string, HttpResponse>& url_to_response_;
};

struct DefaultHttpTransportSetter {
  DefaultHttpTransportSetter(std::shared_ptr<HttpTransport> transport) {
    SetDefaultHttpTransport(transport);
  }
  ~DefaultHttpTransportSetter() { SetDefaultHttpTransport(nullptr); }
};

// TODO: Add tests for various responses
TEST(S3KeyValueStoreTest, SimpleMock) {
  // Mocks for s3
  absl::flat_hash_map<std::string, HttpResponse> url_to_response{
      // initial HEAD request responds with an x-amz-bucket-region header.
      {"HEAD https://my-bucket.s3.amazonaws.com",
       HttpResponse{200, absl::Cord(), {{"x-amz-bucket-region", "us-east-1"}}}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/key_read",
       HttpResponse{200,
                    absl::Cord("abcd"),
                    {{"etag", "900150983cd24fb0d6963f7d28e17f72"}}}},

      {"PUT https://my-bucket.s3.us-east-1.amazonaws.com/key_write",
       HttpResponse{
           200, absl::Cord(), {{"etag", "900150983cd24fb0d6963f7d28e17f72"}}}},

      // DELETE 404 => absl::OkStatus()
  };

  auto mock_transport = std::make_shared<MyMockTransport>(url_to_response);
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "s3"}, {"bucket", "my-bucket"}}, context)
          .result());

  auto read_result = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result,
              MatchesKvsReadResult(absl::Cord("abcd"),
                                   StorageGeneration::FromString(
                                       "900150983cd24fb0d6963f7d28e17f72")));

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz")).result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "900150983cd24fb0d6963f7d28e17f72")));

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "key_delete"));
}

// TODO: Add mocking to satisfy kvstore testing methods, such as:
// tensorstore::internal::TestKeyValueStoreReadOps
// tensorstore::internal::TestKeyValueReadWriteOps

}  // namespace
