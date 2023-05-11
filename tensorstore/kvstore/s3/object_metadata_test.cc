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

#include "tensorstore/kvstore/s3/object_metadata.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::internal_storage_s3::ParseObjectMetadata;


const char kObjectMetadata[] = R"""({
  "x-amz-checksum-crc32c": "deadbeef",
  "x-amz-version-id": "version_12345",
  "x-amz-delete-marker": false,
  "Content-Length": "102400",
  "Last-Modified": "2018-05-19T19:31:14Z"
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

  EXPECT_EQ("deadbeef", result->crc32c);
  EXPECT_EQ(102400u, result->size);
  EXPECT_EQ("version_12345", result->version_id);

  EXPECT_EQ(AsTime("2018-05-19T12:31:14-07:00"), result->time_modified);
}

}  // namespace
