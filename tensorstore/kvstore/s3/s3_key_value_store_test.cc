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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/http/default_transport.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/mock_http_transport.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;
using ::tensorstore::MatchesStatus;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesListEntry;
using ::tensorstore::internal::MatchesTimestampedStorageGeneration;
using ::tensorstore::internal_http::DefaultMockHttpTransport;
using ::tensorstore::internal_http::HeaderMap;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::SetDefaultHttpTransport;
using ::testing::Contains;
using ::testing::Pair;

namespace {

Context DefaultTestContext() {
  // Opens the s3 driver with small exponential backoff values.
  return Context{
      Context::Spec::FromJson({
                                  {"s3_request_retries",
                                   {{"max_retries", 2},
                                    {"initial_delay", "1ms"},
                                    {"max_delay", "2ms"}}},
                                  {"aws_credentials", {{"type", "anonymous"}}},
                              })
          .value()};
}

struct DefaultHttpTransportSetter {
  DefaultHttpTransportSetter(std::shared_ptr<HttpTransport> transport) {
    old_transport = SetDefaultHttpTransport(transport);
  }
  ~DefaultHttpTransportSetter() { SetDefaultHttpTransport(old_transport); }
  std::shared_ptr<HttpTransport> old_transport;
};

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
  options.url = "s3://mybucket/";
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(S3KeyValueStoreTest, SpecRoundtripCustomEndpoint) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_write_read = false;
  options.check_data_persists = false;
  options.check_data_after_serialization = false;
  options.full_spec = {{"driver", "s3"},
                       {"bucket", "mybucket"},
                       {"endpoint", "https://myendpoint"}};
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

// TODO: Add tests for various responses
TEST(S3KeyValueStoreTest, SimpleMock_VirtualHost) {
  // Mocks for s3
  auto mock_transport = std::make_shared<
      DefaultMockHttpTransport>(DefaultMockHttpTransport::Responses{
      // initial HEAD request responds with an x-amz-bucket-region header.
      {"HEAD https://my-bucket.s3.amazonaws.com",
       HttpResponse{200, absl::Cord(),
                    HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/key_read1",
       HttpResponse{
           200, absl::Cord("abcd"),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
                     {"x-amz-checksum-sha256",
                      "iNQmb9TmM40TuEX88olXnSCciXgjuSF9o+Fhk28DFYk="}}}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/key_read2",
       HttpResponse{200, absl::Cord("abcd"),
                    HeaderMap{
                        {"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
                        {"x-amz-checksum-sha256",
                         "iNQmb9TmM40TuEX88olXnSCciXgjuSF9o+Fhk28DFYk="},
                        {"x-amz-checksum-type", "FULL_OBJECT"},
                    }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/key_read3",
       HttpResponse{
           200, absl::Cord("abcd"),
           HeaderMap{
               {"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
               {"x-amz-checksum-sha256",
                "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},  // mismatch
               {"x-amz-checksum-type", "COMPOSITE"},  // not FULL_OBJECT
           }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/empty_read",
       HttpResponse{200, absl::Cord(),
                    HeaderMap{
                        {"etag", "\"900150983cd24fb0d6963f7d28e17f73\""},
                        {"x-amz-checksum-sha256",
                         "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},
                    }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/sha_mismatch",
       HttpResponse{200, absl::Cord("xyz"),
                    HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f73\""},
                              {"x-amz-checksum-sha256",
                               "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},
                              {"x-amz-checksum-type", "FULL_OBJECT"}}}},

      {"PUT https://my-bucket.s3.us-east-1.amazonaws.com/tmp:1/key_write",
       HttpResponse{
           200, absl::Cord(),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

      // DELETE 404 => absl::OkStatus()
  });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {{"driver", "s3"}, {"bucket", "my-bucket"}, {"path", "tmp:1/"}},
          context)
          .result());

  EXPECT_THAT(
      kvstore::Read(store, "key_read1").result(),
      MatchesKvsReadResult(absl::Cord("abcd"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(
      kvstore::Read(store, "key_read2").result(),
      MatchesKvsReadResult(absl::Cord("abcd"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(
      kvstore::Read(store, "key_read3").result(),
      MatchesKvsReadResult(absl::Cord("abcd"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(kvstore::Read(store, "empty_read").result(),
              MatchesKvsReadResult(
                  absl::Cord(), StorageGeneration::FromString(
                                    "\"900150983cd24fb0d6963f7d28e17f73\"")));

  EXPECT_THAT(kvstore::Read(store, "sha_mismatch").result(),
              MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz")).result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "key_delete"));

  int host_header_validated = 0;
  for (const auto& request : mock_transport->requests()) {
    if (absl::StartsWith(request.url,
                         "https://my-bucket.s3.us-east-1.amazonaws.com/")) {
      host_header_validated++;
      EXPECT_THAT(
          request.headers,
          Contains(Pair("host", "my-bucket.s3.us-east-1.amazonaws.com")));
    }
    if (request.method == "GET") {
      EXPECT_THAT(request.headers,
                  Contains(Pair("x-amz-checksum-mode", "ENABLED")));
    }
    if (request.method == "PUT" &&
        absl::StrContains(request.url, "key_write")) {
      EXPECT_THAT(request.headers,
                  Contains(Pair("x-amz-content-sha256",
                                "3608bca1e44ea6c4d268eb6db02260269892c0b42b"
                                "86bbf1e77a6fa16c3c9282")));
    }
  }
  EXPECT_THAT(host_header_validated, testing::Ge(2));
}

TEST(S3KeyValueStoreTest, SimpleMock_NoVirtualHost) {
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://s3.amazonaws.com/my.bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"GET https://s3.us-east-1.amazonaws.com/my.bucket/key_read",
           HttpResponse{
               200, absl::Cord("abcd"),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          {"PUT https://s3.us-east-1.amazonaws.com/my.bucket/key_write",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          // DELETE 404 => absl::OkStatus()
      });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open({{"driver", "s3"},
                                                  {"bucket", "my.bucket"},
                                                  {"aws_region", "us-east-1"}},
                                                 context)
                                       .result());

  auto read_result = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result, MatchesKvsReadResult(
                               absl::Cord("abcd"),
                               StorageGeneration::FromString(
                                   "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz")).result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "key_delete"));

  int host_header_validated = 0;
  for (const auto& request : mock_transport->requests()) {
    if (absl::StartsWith(request.url, "https://s3.us-east-1.amazonaws.com/")) {
      host_header_validated++;
      EXPECT_THAT(request.headers,
                  Contains(Pair("host", "s3.us-east-1.amazonaws.com")));
    }
  }
  EXPECT_THAT(host_header_validated, testing::Ge(2));
}

TEST(S3KeyValueStoreTest, MockWrite_Conflict) {
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://s3.amazonaws.com/my.bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"PUT https://s3.us-east-1.amazonaws.com/my.bucket/key_write",
           HttpResponse{409, absl::Cord(), {}}},

          {"PUT https://s3.us-east-1.amazonaws.com/my.bucket/key_write",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          // DELETE 404 => absl::OkStatus()
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open({{"driver", "s3"},
                                                  {"bucket", "my.bucket"},
                                                  {"aws_region", "us-east-1"}},
                                                 context)
                                       .result());

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz"),
                             {StorageGeneration::FromString(
                                 "\"11111111111111111111111111111111\"")})
                  .result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "key_delete"));

  int host_header_validated = 0;
  for (const auto& request : mock_transport->requests()) {
    if (absl::StartsWith(request.url, "https://s3.us-east-1.amazonaws.com/")) {
      host_header_validated++;
      EXPECT_THAT(request.headers, testing::Contains(Pair(
                                       "host", "s3.us-east-1.amazonaws.com")));
    }
  }
  EXPECT_THAT(host_header_validated, testing::Ge(2));
}

// TODO: Add tests for various responses
TEST(S3KeyValueStoreTest, SimpleMock_Endpoint) {
  // Mocks for s3
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://localhost:1234/base/my-bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"GET https://localhost:1234/base/my-bucket/tmp:1/key_read",
           HttpResponse{
               200, absl::Cord("abcd"),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          {"PUT https://localhost:1234/base/my-bucket/tmp:1/key_write",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          // DELETE 404 => absl::OkStatus()
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "s3"},
                                 {"bucket", "my-bucket"},
                                 {"endpoint", "https://localhost:1234/base"},
                                 {"aws_credentials", {{"type", "anonymous"}}},
                                 {"path", "tmp:1/"}},
                                context)
                      .result());

  auto read_result = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result, MatchesKvsReadResult(
                               absl::Cord("abcd"),
                               StorageGeneration::FromString(
                                   "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz")).result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "key_delete"));

  int host_header_validated = 0;
  for (const auto& request : mock_transport->requests()) {
    if (absl::StartsWith(request.url, "https://localhost:1234/")) {
      host_header_validated++;
      EXPECT_THAT(request.headers,
                  testing::Contains(Pair("host", "localhost:1234")));
    }
  }
  EXPECT_THAT(host_header_validated, testing::Ge(2));
}

TEST(S3KeyValueStoreTest, SimpleMock_List) {
  const auto kListResultA =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"                            //
      "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"  //
      "<Name>bucket</Name>"                                                   //
      "<Prefix></Prefix>"                                                     //
      "<KeyCount>3</KeyCount>"                                                //
      "<MaxKeys>1000</MaxKeys>"                                               //
      "<IsTruncated>true</IsTruncated>"                                       //
      "<NextContinuationToken>CONTINUE</NextContinuationToken>"               //
      "<Contents><Key>a</Key>"                                                //
      "<LastModified>2023-09-06T17:53:27.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>b</Key>"                                                //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>b/a</Key>"                                              //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "</ListBucketResult>";

  const auto kListResultB =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"                            //
      "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"  //
      "<Name>bucket</Name>"                                                   //
      "<Prefix></Prefix>"                                                     //
      "<KeyCount>2</KeyCount>"                                                //
      "<MaxKeys>1000</MaxKeys>"                                               //
      "<IsTruncated>false</IsTruncated>"                                      //
      "<Contents><Key>b/b</Key>"                                              //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>c</Key>"                                                //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "</ListBucketResult>";

  // Mocks for s3
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://my-bucket.s3.amazonaws.com",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"GET https://my-bucket.s3.us-east-1.amazonaws.com/?list-type=2",
           HttpResponse{200, absl::Cord(kListResultA), {}}},

          {"GET "
           "https://my-bucket.s3.us-east-1.amazonaws.com/"
           "?continuation-token=CONTINUE&list-type=2",
           HttpResponse{200, absl::Cord(kListResultB), {}}},
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "s3"}, {"bucket", "my-bucket"}}, context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto list_result,
                                   kvstore::ListFuture(store, {}).result());
  EXPECT_THAT(list_result, ::testing::ElementsAre(
                               MatchesListEntry("a"), MatchesListEntry("b"),
                               MatchesListEntry("b/a"), MatchesListEntry("b/b"),
                               MatchesListEntry("c")));
}

TEST(S3KeyValueStoreTest, SimpleMock_ListPrefix) {
  const auto kListResult =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"                            //
      "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"  //
      "<Name>bucket</Name>"                                                   //
      "<Prefix>b</Prefix>"                                                    //
      "<KeyCount>4</KeyCount>"                                                //
      "<MaxKeys>1000</MaxKeys>"                                               //
      "<IsTruncated>false</IsTruncated>"                                      //
      "<Contents><Key>b</Key>"                                                //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>b/a</Key>"                                              //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>b/b</Key>"                                              //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "<Contents><Key>c</Key>"                                                //
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"                 //
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"             //
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"        //
      "</ListBucketResult>";

  // Mocks for s3
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://my-bucket.s3.amazonaws.com",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"GET "
           "https://my-bucket.s3.us-east-1.amazonaws.com/"
           "?list-type=2&prefix=b",
           HttpResponse{
               200, absl::Cord(kListResult),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
      });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "s3"}, {"bucket", "my-bucket"}}, context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto list_result,
      kvstore::ListFuture(store, {::tensorstore::KeyRange::Prefix("b")})
          .result());
  EXPECT_THAT(list_result, ::testing::ElementsAre(MatchesListEntry("b"),
                                                  MatchesListEntry("b/a"),
                                                  MatchesListEntry("b/b")));
}

// TODO: Add tests for various responses
TEST(S3KeyValueStoreTest, SimpleMock_RetryTimesOut) {
  absl::Cord retry(R"(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>ThrottledException</Code>
  <Message>Endless retry</Message>
  <Resource>/my-bucket/tmp:1/key_read</Resource>
  <RequestId>4442587FB7D0A2F9</RequestId>
</Error>
)");

  // Mocks for s3
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://localhost:1234/base/my-bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          // 400 => retry
          {"GET https://localhost:1234/base/my-bucket/tmp:1/key_read",
           HttpResponse{400, retry, {}}},
          {"GET https://localhost:1234/base/my-bucket/tmp:1/key_read",
           HttpResponse{400, retry, {}}},
          {"GET https://localhost:1234/base/my-bucket/tmp:1/key_read",
           HttpResponse{400, retry, {}}},
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "s3"},
                                 {"bucket", "my-bucket"},
                                 {"endpoint", "https://localhost:1234/base"},
                                 {"path", "tmp:1/"}},
                                context)
                      .result());

  auto read_result = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result, StatusIs(absl::StatusCode::kAborted));
}

TEST(S3KeyValueStoreTest, SimpleMock_RetryResolveEhr) {
  // Mocks for s3
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request fails.
          {"HEAD https://localhost:1234/base/my-bucket",
           absl::InternalError("Mock error")},
          // Subsequent HEAD responds with an x-amz-bucket-region header.
          {"HEAD https://localhost:1234/base/my-bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"GET https://localhost:1234/base/my-bucket/tmp:1/key_read",
           HttpResponse{
               200, absl::Cord("abcd"),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  // Opens the s3 driver with small exponential backoff values.
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "s3"},
                                 {"bucket", "my-bucket"},
                                 {"endpoint", "https://localhost:1234/base"},
                                 {"path", "tmp:1/"}},
                                context)
                      .result());

  auto read_result1 = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result1, StatusIs(absl::StatusCode::kInternal));

  auto read_result = kvstore::Read(store, "key_read").result();
  EXPECT_THAT(read_result, MatchesKvsReadResult(
                               absl::Cord("abcd"),
                               StorageGeneration::FromString(
                                   "\"900150983cd24fb0d6963f7d28e17f72\"")));
}

// TODO: Add mocking to satisfy kvstore testing methods, such as:
// tensorstore::internal::TestKeyValueStoreReadOps
// tensorstore::internal::TestKeyValueReadWriteOps

}  // namespace
