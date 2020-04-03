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

#include "tensorstore/kvstore/gcs/object_metadata.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::internal_storage_gcs::ParseObjectMetadata;

namespace {

const char kObjectMetadata[] = R"""({
      "acl": [{
        "kind": "storage#objectAccessControl",
        "id": "acl-id-0",
        "selfLink": "https://www.googleapis.com/kvstore/v1/b/foo-bar/o/baz/acl/user-qux",
        "bucket": "foo-bar",
        "object": "foo",
        "generation": 12345,
        "entity": "user-qux",
        "role": "OWNER",
        "email": "qux@example.com",
        "entityId": "user-qux-id-123",
        "domain": "example.com",
        "projectTeam": {
          "projectNumber": "4567",
          "team": "owners"
        },
        "etag": "AYX="
      }, {
        "kind": "storage#objectAccessControl",
        "id": "acl-id-1",
        "selfLink": "https://www.googleapis.com/kvstore/v1/b/foo-bar/o/baz/acl/user-quux",
        "bucket": "foo-bar",
        "object": "foo",
        "generation": 12345,
        "entity": "user-quux",
        "role": "READER",
        "email": "qux@example.com",
        "entityId": "user-quux-id-123",
        "domain": "example.com",
        "projectTeam": {
          "projectNumber": "4567",
          "team": "viewers"
        },
        "etag": "AYX="
      }
      ],
      "bucket": "foo-bar",
      "cacheControl": "no-cache",
      "componentCount": 7,
      "contentDisposition": "a-disposition",
      "contentEncoding": "an-encoding",
      "contentLanguage": "a-language",
      "contentType": "application/octet-stream",
      "crc32c": "deadbeef",
      "customerEncryption": {
        "encryptionAlgorithm": "some-algo",
        "keySha256": "abc123"
      },
      "etag": "XYZ=",
      "eventBasedHold": true,
      "generation": "12345",
      "id": "foo-bar/baz/12345",
      "kind": "storage#object",
      "kmsKeyName": "/foo/bar/baz/key",
      "md5Hash": "deaderBeef=",
      "mediaLink": "https://www.googleapis.com/kvstore/v1/b/foo-bar/o/baz?generation=12345&alt=media",
      "metadata": {
        "foo": "bar",
        "baz": "qux"
      },
      "metageneration": "4",
      "name": "baz",
      "owner": {
        "entity": "user-qux",
        "entityId": "user-qux-id-123"
      },
      "retentionExpirationTime": "2019-01-01T00:00:00Z",
      "selfLink": "https://www.googleapis.com/kvstore/v1/b/foo-bar/o/baz",
      "size": 102400,
      "storageClass": "STANDARD",
      "temporaryHold": true,
      "timeCreated": "2018-05-19T19:31:14Z",
      "timeDeleted": "2018-05-19T19:32:24Z",
      "timeStorageClassUpdated": "2018-05-19T19:31:34Z",
      "updated": "2018-05-19T19:31:24Z"
})""";

absl::Time AsTime(const std::string& time) {
  absl::Time result;
  if (absl::ParseTime(absl::RFC3339_full, time, &result, nullptr)) {
    return result;
  }
  return absl::InfinitePast();
}

TEST(ParseObjectMetadata, Basic) {
  EXPECT_FALSE(ParseObjectMetadata("").ok());

  auto result = ParseObjectMetadata(kObjectMetadata);
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("foo-bar", result->bucket);
  EXPECT_EQ("no-cache", result->cache_control);
  EXPECT_EQ("a-disposition", result->content_disposition);
  EXPECT_EQ("an-encoding", result->content_encoding);
  EXPECT_EQ("a-language", result->content_language);
  EXPECT_EQ("application/octet-stream", result->content_type);
  EXPECT_EQ("deadbeef", result->crc32c);
  EXPECT_EQ("XYZ=", result->etag);
  EXPECT_EQ("foo-bar/baz/12345", result->id);
  EXPECT_EQ("storage#object", result->kind);
  EXPECT_EQ("/foo/bar/baz/key", result->kms_key_name);
  EXPECT_EQ("deaderBeef=", result->md5_hash);
  EXPECT_EQ(
      "https://www.googleapis.com/kvstore/v1/b/foo-bar/o/"
      "baz?generation=12345&alt=media",
      result->media_link);
  EXPECT_EQ("baz", result->name);
  EXPECT_EQ("https://www.googleapis.com/kvstore/v1/b/foo-bar/o/baz",
            result->self_link);
  EXPECT_EQ("STANDARD", result->storage_class);

  EXPECT_EQ(102400u, result->size);
  EXPECT_EQ(7, result->component_count);
  EXPECT_EQ(12345, result->generation);
  EXPECT_EQ(4, result->metageneration);

  EXPECT_TRUE(result->event_based_hold);
  EXPECT_TRUE(result->temporary_hold);

  EXPECT_EQ(AsTime("2018-12-31T16:00:00-08:00"),
            result->retention_expiration_time);

  EXPECT_EQ(AsTime("2018-05-19T12:31:14-07:00"), result->time_created);
  EXPECT_EQ(AsTime("2018-05-19T12:31:24-07:00"), result->updated);
  EXPECT_EQ(AsTime("2018-05-19T12:32:24-07:00"), result->time_deleted);
  EXPECT_EQ(AsTime("2018-05-19T12:31:34-07:00"),
            result->time_storage_class_updated);
}

const char kObjectMetadata2[] = R"""({
   "name": "fafb_v14/fafb_v14_clahe/128_128_160/0-64_1408-1472_896-960",
   "kind": "storage#object",
   "id": "neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe/128_128_160/0-64_1408-1472_896-960/1540426531840872",
   "bucket": "neuroglancer-fafb-data",
   "generation": "1540426531840872",
   "contentType": "image/jpeg",
   "timeCreated": "2018-10-25T00:15:31.840Z",
   "updated": "2018-10-25T00:15:31.840Z",
   "timeStorageClassUpdated": "2018-10-25T00:15:31.840Z",
   "size": "3404"
})""";

TEST(ParseObjectMetadata, Example2) {
  EXPECT_FALSE(ParseObjectMetadata("").ok());

  auto result = ParseObjectMetadata(kObjectMetadata2);
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ("fafb_v14/fafb_v14_clahe/128_128_160/0-64_1408-1472_896-960",
            result->name);
  EXPECT_EQ("storage#object", result->kind);
  EXPECT_EQ(
      "neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe/128_128_160/"
      "0-64_1408-1472_896-960/1540426531840872",
      result->id);
  EXPECT_EQ("neuroglancer-fafb-data", result->bucket);
  EXPECT_EQ("image/jpeg", result->content_type);

  EXPECT_EQ(3404u, result->size);
  EXPECT_EQ(1540426531840872, result->generation);

  EXPECT_EQ(AsTime("2018-10-24T17:15:31.84-07:00"), result->time_created);
  EXPECT_EQ(AsTime("2018-10-24T17:15:31.84-07:00"), result->updated);
  EXPECT_EQ(AsTime("2018-10-24T17:15:31.84-07:00"),
            result->time_storage_class_updated);

  EXPECT_EQ(0, result->metageneration);
  EXPECT_FALSE(result->event_based_hold);
  EXPECT_FALSE(result->temporary_hold);
  EXPECT_EQ(absl::InfinitePast(), result->retention_expiration_time);
}

}  // namespace
