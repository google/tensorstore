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

#include "tensorstore/internal/uri/path.h"

#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/testing/on_windows.h"
#include "tensorstore/internal/uri/parse.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOkAndHolds;
using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_testing::OnWindows;
using ::tensorstore::internal_uri::FileUriToOsPath;
using ::tensorstore::internal_uri::OsPathToFileUri;
using ::tensorstore::internal_uri::ParseGenericUri;

namespace {

TEST(OsPathToUriPathTest, Basic) {
  EXPECT_THAT(OsPathToFileUri(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("foo"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("foo/"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OsPathToFileUri("/"), IsOkAndHolds("file:///"));
  EXPECT_THAT(OsPathToFileUri("/foo"), IsOkAndHolds("file:///foo"));
  EXPECT_THAT(OsPathToFileUri("/foo/"), IsOkAndHolds("file:///foo/"));

  EXPECT_THAT(OsPathToFileUri("c:/tmp"),
              OnWindows(IsOkAndHolds("file:///c:/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(OsPathToFileUri("c:/tmp/"),
              OnWindows(IsOkAndHolds("file:///c:/tmp/"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
  EXPECT_THAT(OsPathToFileUri("c:\\tmp\\foo"),
              OnWindows(IsOkAndHolds("file:///c:/tmp/foo"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));

  EXPECT_THAT(OsPathToFileUri("//server/share/tmp"),
              IsOkAndHolds(OnWindows("file://server/share/tmp",
                                     "file:///server/share/tmp")));

  EXPECT_THAT(OsPathToFileUri("\\\\server\\share\\tmp"),
              OnWindows(IsOkAndHolds("file://server/share/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
}

TEST(UriPathToOsPathTest, Basic) {
  auto ToPath = [](std::string_view uri) {
    return FileUriToOsPath(ParseGenericUri(uri));
  };

  EXPECT_THAT(ToPath(""), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ToPath("foo"), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ToPath("foo/"), StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(ToPath("file:///"), IsOkAndHolds("/"));
  EXPECT_THAT(ToPath("file:///foo"), IsOkAndHolds("/foo"));
  EXPECT_THAT(ToPath("file:///foo/"), IsOkAndHolds("/foo/"));
  EXPECT_THAT(ToPath("file:///c:/tmp"),
              IsOkAndHolds(OnWindows("c:/tmp", "/c:/tmp")));
  EXPECT_THAT(ToPath("file:///c:/tmp/"),
              IsOkAndHolds(OnWindows("c:/tmp/", "/c:/tmp/")));

  EXPECT_THAT(ToPath("file://server/share/tmp"),
              OnWindows(IsOkAndHolds("//server/share/tmp"),
                        StatusIs(absl::StatusCode::kInvalidArgument)));
}
}  // namespace
