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
using ::testing::StrEq;

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
                       {"endpoint", "https://myendpoint"},
                       {"unsigned_payload", true}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(S3KeyValueStoreTest, InvalidSpec) {
  auto context = DefaultTestContext();

  // Test with extra key.
  EXPECT_THAT(kvstore::Open(
                  {{"driver", "s3"}, {"bucket", "my-bucket"}, {"extra", "key"}},
                  context)
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Test with missing `"bucket"` key.
  EXPECT_THAT(kvstore::Open({{"driver", "s3"}}, context).result(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"bucket"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "s3"}, {"bucket", "a"}}, context).result(),
      StatusIs(absl::StatusCode::kInvalidArgument));
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

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/key_read1",
       HttpResponse{
           200, absl::Cord("abcd"),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
                     {"x-amz-checksum-sha256",
                      "iNQmb9TmM40TuEX88olXnSCciXgjuSF9o+Fhk28DFYk="}}}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/key_read2",
       HttpResponse{200, absl::Cord("abcd"),
                    HeaderMap{
                        {"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
                        {"x-amz-checksum-sha256",
                         "iNQmb9TmM40TuEX88olXnSCciXgjuSF9o+Fhk28DFYk="},
                        {"x-amz-checksum-type", "FULL_OBJECT"},
                    }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/key_read3",
       HttpResponse{
           200, absl::Cord("abcd"),
           HeaderMap{
               {"etag", "\"900150983cd24fb0d6963f7d28e17f72\""},
               {"x-amz-checksum-sha256",
                "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},  // mismatch
               {"x-amz-checksum-type", "COMPOSITE"},  // not FULL_OBJECT
           }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/empty_read",
       HttpResponse{200, absl::Cord(),
                    HeaderMap{
                        {"etag", "\"900150983cd24fb0d6963f7d28e17f73\""},
                        {"x-amz-checksum-sha256",
                         "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},
                    }}},

      {"GET https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/sha_mismatch",
       HttpResponse{200, absl::Cord("xyz"),
                    HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f73\""},
                              {"x-amz-checksum-sha256",
                               "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="},
                              {"x-amz-checksum-type", "FULL_OBJECT"}}}},

      {"PUT https://my-bucket.s3.us-east-1.amazonaws.com/tmp%3A1/key_write",
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
              StatusIs(absl::StatusCode::kDataLoss));

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
                  Contains(Pair("x-amz-content-sha256", "UNSIGNED-PAYLOAD")));
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
                                                  {"aws_region", "us-east-1"},
                                                  {"unsigned_payload", false}},
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

TEST(S3KeyValueStoreTest, AutoUnsignedPayloadFallbackAndHttp) {
  static constexpr char kXyzSha256[] =
      "3608bca1e44ea6c4d268eb6db02260269892c0b42b86bbf1e77a6fa16c3c9282";
  static constexpr char kRejectedXml[] =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<Error><Code>InvalidArgument</Code>"
      "<Message>UNSIGNED-PAYLOAD not supported</Message></Error>";

  auto mock_transport = std::make_shared<
      DefaultMockHttpTransport>(DefaultMockHttpTransport::Responses{
      // First PUT over https:// uses UNSIGNED-PAYLOAD and is rejected (400).
      {"PUT https://localhost:1234/my-bucket/key1",
       HttpResponse{400, absl::Cord(kRejectedXml), {}}},
      // Fallback retry of key1 uses SHA-256 and succeeds.
      {"PUT https://localhost:1234/my-bucket/key1",
       HttpResponse{
           200, absl::Cord(),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
      // Subsequent PUT on the same store immediately uses SHA-256.
      {"PUT https://localhost:1234/my-bucket/key2",
       HttpResponse{
           200, absl::Cord(),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
      // PUT over http:// with default (unset) unsigned_payload uses SHA-256.
      {"PUT http://localhost:1234/my-bucket/key3",
       HttpResponse{
           200, absl::Cord(),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
      // PUT over http:// with explicit unsigned_payload=true uses
      // UNSIGNED-PAYLOAD.
      {"PUT http://localhost:1234/my-bucket/key4",
       HttpResponse{
           200, absl::Cord(),
           HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
  });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto https_store, kvstore::Open({{"driver", "s3"},
                                       {"bucket", "my-bucket"},
                                       {"endpoint", "https://localhost:1234"},
                                       {"aws_region", "us-east-1"}},
                                      context)
                            .result());
  TENSORSTORE_EXPECT_OK(kvstore::Write(https_store, "key1", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(https_store, "key2", absl::Cord("xyz")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto http_store, kvstore::Open({{"driver", "s3"},
                                      {"bucket", "my-bucket"},
                                      {"endpoint", "http://localhost:1234"},
                                      {"aws_region", "us-east-1"}},
                                     context)
                           .result());
  TENSORSTORE_EXPECT_OK(kvstore::Write(http_store, "key3", absl::Cord("xyz")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto http_unsigned_store,
      kvstore::Open({{"driver", "s3"},
                     {"bucket", "my-bucket"},
                     {"endpoint", "http://localhost:1234"},
                     {"aws_region", "us-east-1"},
                     {"unsigned_payload", true}},
                    context)
          .result());
  TENSORSTORE_EXPECT_OK(
      kvstore::Write(http_unsigned_store, "key4", absl::Cord("xyz")));

  ASSERT_EQ(mock_transport->requests().size(), 5);
  EXPECT_THAT(mock_transport->requests()[0].headers,
              Contains(Pair("x-amz-content-sha256", "UNSIGNED-PAYLOAD")));
  EXPECT_THAT(mock_transport->requests()[1].headers,
              Contains(Pair("x-amz-content-sha256", kXyzSha256)));
  EXPECT_THAT(mock_transport->requests()[2].headers,
              Contains(Pair("x-amz-content-sha256", kXyzSha256)));
  EXPECT_THAT(mock_transport->requests()[3].headers,
              Contains(Pair("x-amz-content-sha256", kXyzSha256)));
  EXPECT_THAT(mock_transport->requests()[4].headers,
              Contains(Pair("x-amz-content-sha256", "UNSIGNED-PAYLOAD")));
}

TEST(S3KeyValueStoreTest, ConditionalWriteBehaviorAndOverride) {
  static constexpr char kEtag1[] = "\"11111111111111111111111111111111\"";
  static constexpr char kEtag2[] = "\"900150983cd24fb0d6963f7d28e17f72\"";
  const auto kGen1 = StorageGeneration::FromString(kEtag1);
  const auto kGen2 = StorageGeneration::FromString(kEtag2);

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // 1. Default conditional write on custom endpoint
          // (ConditionalWriteMode::kDefault) uses preflight HEAD for version
          // checks, and also adds conditional write headers on PUT.
          {"HEAD https://localhost:1234/my-bucket/key1",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag1}}}},
          {"PUT https://localhost:1234/my-bucket/key1",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag2}}}},

          // 2. Explicit use_conditional_write=true sends conditional PUT
          // directly without preceding HEAD.
          {"PUT https://localhost:1234/my-bucket/key2",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag2}}}},

          // 3. Explicit use_conditional_write=false uses preflight HEAD +
          // unconditional PUT (no conditional headers on PUT).
          {"HEAD https://localhost:1234/my-bucket/key3",
           HttpResponse{404, absl::Cord(), {}}},
          {"PUT https://localhost:1234/my-bucket/key3",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag2}}}},

          // 4. Endpoint known to support conditional writes (AWS S3,
          // ConditionalWriteMode::kEnabled) sends conditional PUT directly.
          {"PUT https://my-bucket.s3.us-east-1.amazonaws.com/key4",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag2}}}},
      });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto default_store, kvstore::Open({{"driver", "s3"},
                                         {"bucket", "my-bucket"},
                                         {"endpoint", "https://localhost:1234"},
                                         {"aws_region", "us-east-1"}},
                                        context)
                              .result());
  EXPECT_THAT(kvstore::Write(default_store, "key1", absl::Cord("xyz"), {kGen1})
                  .result(),
              MatchesTimestampedStorageGeneration(kGen2));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto enabled_store, kvstore::Open({{"driver", "s3"},
                                         {"bucket", "my-bucket"},
                                         {"endpoint", "https://localhost:1234"},
                                         {"aws_region", "us-east-1"},
                                         {"use_conditional_write", true}},
                                        context)
                              .result());
  EXPECT_THAT(kvstore::Write(enabled_store, "key2", absl::Cord("xyz"),
                             {StorageGeneration::NoValue()})
                  .result(),
              MatchesTimestampedStorageGeneration(kGen2));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto disabled_store,
      kvstore::Open({{"driver", "s3"},
                     {"bucket", "my-bucket"},
                     {"endpoint", "https://localhost:1234"},
                     {"aws_region", "us-east-1"},
                     {"use_conditional_write", false}},
                    context)
          .result());
  EXPECT_THAT(kvstore::Write(disabled_store, "key3", absl::Cord("xyz"),
                             {StorageGeneration::NoValue()})
                  .result(),
              MatchesTimestampedStorageGeneration(kGen2));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto aws_store,
                                   kvstore::Open({{"driver", "s3"},
                                                  {"bucket", "my-bucket"},
                                                  {"aws_region", "us-east-1"}},
                                                 context)
                                       .result());
  EXPECT_THAT(
      kvstore::Write(aws_store, "key4", absl::Cord("xyz"), {kGen1}).result(),
      MatchesTimestampedStorageGeneration(kGen2));

  ASSERT_EQ(mock_transport->requests().size(), 6);
  // Request 0 & 1: default uses preflight HEAD then conditional PUT with
  // if-match.
  EXPECT_EQ(mock_transport->requests()[0].method, "HEAD");
  EXPECT_THAT(mock_transport->requests()[0].headers,
              Contains(Pair("if-match", kEtag1)));
  EXPECT_EQ(mock_transport->requests()[1].method, "PUT");
  EXPECT_THAT(mock_transport->requests()[1].headers,
              Contains(Pair("if-match", kEtag1)));

  // Request 2: explicit use_conditional_write=true sends conditional PUT
  // directly.
  EXPECT_EQ(mock_transport->requests()[2].method, "PUT");
  EXPECT_THAT(mock_transport->requests()[2].headers,
              Contains(Pair("if-none-match", "*")));

  // Request 3 & 4: explicit use_conditional_write=false uses HEAD +
  // unconditional PUT.
  EXPECT_EQ(mock_transport->requests()[3].method, "HEAD");
  EXPECT_EQ(mock_transport->requests()[4].method, "PUT");
  EXPECT_THAT(mock_transport->requests()[4].headers,
              testing::Not(Contains(Pair("if-match", testing::_))));
  EXPECT_THAT(mock_transport->requests()[4].headers,
              testing::Not(Contains(Pair("if-none-match", testing::_))));

  // Request 5: AWS S3 endpoint sends conditional PUT directly.
  EXPECT_EQ(mock_transport->requests()[5].method, "PUT");
  EXPECT_THAT(mock_transport->requests()[5].headers,
              Contains(Pair("if-match", kEtag1)));
}

TEST(S3KeyValueStoreTest, ChecksumOptions) {
  static constexpr char kEtag[] = "\"900150983cd24fb0d6963f7d28e17f72\"";
  // CRC32C of "abcd" (0x92c80a31) in base64 is "ksgKMQ==".
  static constexpr char kAbcdCrc32c[] = "ksgKMQ==";
  // CRC32C of "xyz" (0x25236885) in base64 is "JSNohQ==".
  static constexpr char kXyzCrc32c[] = "JSNohQ==";

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // 1. Default full GET validates x-amz-checksum-crc32c.
          {"GET https://localhost:1234/my-bucket/read_ok",
           HttpResponse{200, absl::Cord("abcd"),
                        HeaderMap{{"etag", kEtag},
                                  {"x-amz-checksum-crc32c", kAbcdCrc32c}}}},
          // 2. Default full GET detects x-amz-checksum-crc32c mismatch.
          {"GET https://localhost:1234/my-bucket/read_bad_crc",
           HttpResponse{200, absl::Cord("abcd"),
                        HeaderMap{{"etag", kEtag},
                                  {"x-amz-checksum-crc32c", kXyzCrc32c}}}},
          // 3. Partial range GET ignores full-object x-amz-checksum-crc32c
          // returned by servers like GCS XML API.
          {"GET https://localhost:1234/my-bucket/read_range",
           HttpResponse{206, absl::Cord("bc"),
                        HeaderMap{{"etag", kEtag},
                                  {"content-range", "bytes 1-2/4"},
                                  {"x-amz-checksum-crc32c", kAbcdCrc32c}}}},
          // 4. Default PUT includes x-amz-checksum-crc32c.
          {"PUT https://localhost:1234/my-bucket/write_crc",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag}}}},
          // 5. checksum="none" skips x-amz-checksum-mode on GET, ignores bad
          // checksum header on GET, and omits x-amz-checksum-* on PUT.
          {"GET https://localhost:1234/my-bucket/read_none",
           HttpResponse{200, absl::Cord("abcd"),
                        HeaderMap{{"etag", kEtag},
                                  {"x-amz-checksum-crc32c", kXyzCrc32c}}}},
          {"PUT https://localhost:1234/my-bucket/write_none",
           HttpResponse{200, absl::Cord(), HeaderMap{{"etag", kEtag}}}},
      });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto default_store, kvstore::Open({{"driver", "s3"},
                                         {"bucket", "my-bucket"},
                                         {"endpoint", "https://localhost:1234"},
                                         {"aws_region", "us-east-1"}},
                                        context)
                              .result());
  EXPECT_THAT(kvstore::Read(default_store, "read_ok").result(),
              MatchesKvsReadResult(absl::Cord("abcd"),
                                   StorageGeneration::FromString(kEtag)));
  EXPECT_THAT(kvstore::Read(default_store, "read_bad_crc").result(),
              StatusIs(absl::StatusCode::kDataLoss));

  kvstore::ReadOptions range_options;
  range_options.byte_range = tensorstore::OptionalByteRangeRequest(1, 3);
  EXPECT_THAT(
      kvstore::Read(default_store, "read_range", range_options).result(),
      MatchesKvsReadResult(absl::Cord("bc"),
                           StorageGeneration::FromString(kEtag)));

  TENSORSTORE_EXPECT_OK(
      kvstore::Write(default_store, "write_crc", absl::Cord("xyz")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto none_store, kvstore::Open({{"driver", "s3"},
                                      {"bucket", "my-bucket"},
                                      {"endpoint", "https://localhost:1234"},
                                      {"aws_region", "us-east-1"},
                                      {"checksum", "none"}},
                                     context)
                           .result());
  EXPECT_THAT(kvstore::Read(none_store, "read_none").result(),
              MatchesKvsReadResult(absl::Cord("abcd"),
                                   StorageGeneration::FromString(kEtag)));
  TENSORSTORE_EXPECT_OK(
      kvstore::Write(none_store, "write_none", absl::Cord("xyz")));

  ASSERT_EQ(mock_transport->requests().size(), 6);
  EXPECT_THAT(mock_transport->requests()[0].headers,
              Contains(Pair("x-amz-checksum-mode", "ENABLED")));
  EXPECT_THAT(mock_transport->requests()[2].headers,
              testing::Not(Contains(Pair("x-amz-checksum-mode", testing::_))));
  EXPECT_THAT(mock_transport->requests()[3].headers,
              Contains(Pair("x-amz-checksum-crc32c", kXyzCrc32c)));
  EXPECT_THAT(mock_transport->requests()[4].headers,
              testing::Not(Contains(Pair("x-amz-checksum-mode", testing::_))));
  EXPECT_THAT(
      mock_transport->requests()[5].headers,
      testing::Not(Contains(Pair("x-amz-checksum-crc32c", testing::_))));
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

          {"GET https://localhost:1234/base/my-bucket/tmp%3A1/key_read",
           HttpResponse{
               200, absl::Cord("abcd"),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          {"PUT https://localhost:1234/base/my-bucket/tmp%3A1/key_write",
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

TEST(S3KeyValueStoreTest, ListPreservesWhitespaceAndEntities) {
  const auto kListResult =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
      "<Name>bucket</Name>"
      "<Prefix></Prefix>"
      "<KeyCount>2</KeyCount>"
      "<MaxKeys>1000</MaxKeys>"
      "<IsTruncated>false</IsTruncated>"
      "<Contents><Key>   </Key>"
      "<LastModified>2023-09-06T17:53:27.000Z</LastModified>"
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"
      "<Contents><Key>a&apos;b &amp;apos;c</Key>"
      "<LastModified>2023-09-06T17:53:28.000Z</LastModified>"
      "<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>"
      "<Size>0</Size><StorageClass>STANDARD</StorageClass></Contents>"
      "</ListBucketResult>";

  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          {"HEAD https://my-bucket.s3.amazonaws.com",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},
          {"GET https://my-bucket.s3.us-east-1.amazonaws.com/?list-type=2",
           HttpResponse{200, absl::Cord(kListResult), {}}},
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "s3"}, {"bucket", "my-bucket"}}, context)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto list_result,
                                   kvstore::ListFuture(store, {}).result());
  EXPECT_THAT(list_result,
              ::testing::ElementsAre(MatchesListEntry("   "),
                                     MatchesListEntry("a'b &apos;c")));
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
          {"GET https://localhost:1234/base/my-bucket/tmp%3A1/key_read",
           HttpResponse{400, retry, {}}},
          {"GET https://localhost:1234/base/my-bucket/tmp%3A1/key_read",
           HttpResponse{400, retry, {}}},
          {"GET https://localhost:1234/base/my-bucket/tmp%3A1/key_read",
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

          {"GET https://localhost:1234/base/my-bucket/tmp%3A1/key_read",
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

TEST(S3KeyValueStoreTest, RetryOccupiesAdmissionQueueSlot) {
  // The retry path intentionally holds its admission queue slot across retry
  // backoff (matching the HTTP and GCS drivers). Otherwise, queued requests
  // would immediately take the slot while the failing request is backing off,
  // leading to a request storm against an overloaded server.
  auto mock_transport =
      std::make_shared<DefaultMockHttpTransport>(/*queue_requests=*/true);
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "s3"},
                                 {"bucket", "my-bucket"},
                                 {"endpoint", "https://localhost:1234/base"},
                                 {"aws_region", "us-east-1"},
                                 {"context",
                                  {{"aws_credentials", {{"type", "anonymous"}}},
                                   {"s3_request_concurrency", {{"limit", 1}}},
                                   {"s3_request_retries",
                                    {{"max_retries", 2},
                                     {"initial_delay", "10ms"},
                                     {"max_delay", "50ms"}}}}}})
                      .result());

  auto read_future1 = kvstore::Read(store, "abc");
  auto req1 = mock_transport->queued_requests().pop();
  ASSERT_THAT(req1.request.url,
              StrEq("https://localhost:1234/base/my-bucket/abc"));

  // Issue a second read while req1 is still in-flight; because limit == 1,
  // read_future2 is queued in the admission queue until read_future1 finishes.
  auto read_future2 = kvstore::Read(store, "def");
  EXPECT_TRUE(mock_transport->queued_requests().empty());

  // Respond with 503 to req1; read_future1 enters ScheduleAt retry backoff
  // while continuing to occupy the single admission queue slot, so
  // read_future1's retry is issued before read_future2 is admitted.
  req1.set_result(HttpResponse{503, absl::Cord()});

  auto req1_retry = mock_transport->queued_requests().pop();
  EXPECT_THAT(req1_retry.request.url,
              StrEq("https://localhost:1234/base/my-bucket/abc"));
  req1_retry.set_result(HttpResponse{
      200, absl::Cord("value_abc"),
      HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}});
  EXPECT_THAT(
      read_future1.result(),
      MatchesKvsReadResult(absl::Cord("value_abc"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f72\"")));

  auto req2 = mock_transport->queued_requests().pop();
  EXPECT_THAT(req2.request.url,
              StrEq("https://localhost:1234/base/my-bucket/def"));
  req2.set_result(HttpResponse{
      200, absl::Cord("value_def"),
      HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f73\""}}});
  EXPECT_THAT(
      read_future2.result(),
      MatchesKvsReadResult(absl::Cord("value_def"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f73\"")));
}

// TODO: Add mocking to satisfy kvstore testing methods, such as:
// tensorstore::internal::TestKeyValueStoreReadOps
// tensorstore::internal::TestKeyValueReadWriteOps

TEST(S3KeyValueStoreTest, EncodesReservedCharactersInObjectKey) {
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          {"HEAD https://s3.amazonaws.com/my.bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},
          {"GET "
           "https://s3.us-east-1.amazonaws.com/my.bucket/"
           "dir%3A1/file%2520%3Facl%23frag",
           HttpResponse{
               200, absl::Cord("abcd"),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
          {"PUT "
           "https://s3.us-east-1.amazonaws.com/my.bucket/"
           "dir%3A1/file%2520%3Facl%23frag",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
          {"DELETE "
           "https://s3.us-east-1.amazonaws.com/my.bucket/"
           "dir%3A1/file%2520%3Facl%23frag",
           HttpResponse{204, absl::Cord(), {}}},
      });
  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open({{"driver", "s3"},
                                                  {"bucket", "my.bucket"},
                                                  {"aws_region", "us-east-1"}},
                                                 context)
                                       .result());

  EXPECT_THAT(
      kvstore::Read(store, "dir:1/file%20?acl#frag").result(),
      MatchesKvsReadResult(absl::Cord("abcd"),
                           StorageGeneration::FromString(
                               "\"900150983cd24fb0d6963f7d28e17f72\"")));
  EXPECT_THAT(kvstore::Write(store, "dir:1/file%20?acl#frag", absl::Cord("xyz"))
                  .result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));
  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, "dir:1/file%20?acl#frag"));

  ASSERT_EQ(mock_transport->requests().size(), 3);
  for (const auto& request : mock_transport->requests()) {
    EXPECT_EQ(request.url,
              "https://s3.us-east-1.amazonaws.com/my.bucket/"
              "dir%3A1/file%2520%3Facl%23frag");
  }
}

TEST(S3KeyValueStoreTest, RetryHeadOn429) {
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // Endpoint resolution HEAD request gets 429 first, then succeeds.
          {"HEAD https://localhost:1234/base/my-bucket",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          // Conditional Write HEAD ("Peek") gets 429 first, then 200, then PUT.
          {"HEAD https://localhost:1234/base/my-bucket/key_write",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_write",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"11111111111111111111111111111111\""}}}},
          {"PUT https://localhost:1234/base/my-bucket/key_write",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},

          // Conditional Write HEAD ("Peek") exhausting retries does not PUT.
          {"HEAD https://localhost:1234/base/my-bucket/key_write_fail",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_write_fail",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_write_fail",
           HttpResponse{429, absl::Cord(), {}}},

          // Conditional Delete HEAD ("Peek") gets 429 first, then 200, then
          // DELETE.
          {"HEAD https://localhost:1234/base/my-bucket/key_delete",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_delete",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"etag", "\"900150983cd24fb0d6963f7d28e17f72\""}}}},
          {"DELETE https://localhost:1234/base/my-bucket/key_delete",
           HttpResponse{204, absl::Cord(), {}}},

          // Conditional Delete HEAD ("Peek") exhausting retries does not
          // DELETE.
          {"HEAD https://localhost:1234/base/my-bucket/key_delete_fail",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_delete_fail",
           HttpResponse{429, absl::Cord(), {}}},
          {"HEAD https://localhost:1234/base/my-bucket/key_delete_fail",
           HttpResponse{429, absl::Cord(), {}}},
      });

  DefaultHttpTransportSetter mock_transport_setter{mock_transport};
  auto context = DefaultTestContext();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "s3"},
                                 {"bucket", "my-bucket"},
                                 {"endpoint", "https://localhost:1234/base"},
                                 {"use_conditional_write", false}},
                                context)
                      .result());

  EXPECT_THAT(kvstore::Write(store, "key_write", absl::Cord("xyz"),
                             {StorageGeneration::FromString(
                                 "\"11111111111111111111111111111111\"")})
                  .result(),
              MatchesTimestampedStorageGeneration(StorageGeneration::FromString(
                  "\"900150983cd24fb0d6963f7d28e17f72\"")));

  EXPECT_THAT(kvstore::Write(store, "key_write_fail", absl::Cord("xyz"),
                             {StorageGeneration::FromString(
                                 "\"11111111111111111111111111111111\"")})
                  .result(),
              StatusIs(absl::StatusCode::kAborted));

  EXPECT_THAT(
      kvstore::Delete(store, "key_delete",
                      {StorageGeneration::FromString(
                          "\"900150983cd24fb0d6963f7d28e17f72\"")})
          .result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::NoValue()));

  EXPECT_THAT(kvstore::Delete(store, "key_delete_fail",
                              {StorageGeneration::FromString(
                                  "\"900150983cd24fb0d6963f7d28e17f72\"")})
                  .result(),
              StatusIs(absl::StatusCode::kAborted));

  for (const auto& request : mock_transport->requests()) {
    EXPECT_FALSE(request.method == "PUT" &&
                 absl::StrContains(request.url, "key_write_fail"));
    EXPECT_FALSE(request.method == "DELETE" &&
                 absl::StrContains(request.url, "key_delete_fail"));
  }
}

}  // namespace
