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

#include "tensorstore/util/status_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::StatusBuilder;
using ::testing::HasSubstr;

TEST(StatusBuilder, OkStatus) {
  absl::Status status = StatusBuilder(absl::OkStatus()).Format("hello");
  EXPECT_THAT(status, IsOk());
}

TEST(StatusBuilder, OkStatusCode) {
  absl::Status status = StatusBuilder(absl::StatusCode::kOk).Format("hello");
  EXPECT_THAT(status, IsOk());
}

TEST(StatusBuilder, Format) {
  // Defaults to prepending the message.
  absl::Status status = StatusBuilder(absl::NotFoundError("initial"))
                            .Format("formatted")
                            .Format(" %d times", 2);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("formatted 2 times: initial")));
}

TEST(StatusBuilder, SetPrepend) {
  absl::Status status = StatusBuilder(absl::NotFoundError("initial"))
                            .Format("formatted")
                            .SetPrepend();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("formatted: initial")));
}

TEST(StatusBuilder, SetAppend) {
  absl::Status status = StatusBuilder(absl::NotFoundError("initial"))
                            .Format("formatted")
                            .SetAppend();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("initial: formatted")));
}

TEST(StatusBuilder, CodeConstructor) {
  absl::Status status =
      StatusBuilder(absl::StatusCode::kNotFound).Format("formatted");
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("formatted")));
}

TEST(StatusBuilder, SetCode) {
  absl::Status status = StatusBuilder(absl::NotFoundError("initial"))
                            .SetCode(absl::StatusCode::kInternal)
                            .SetAppend()
                            .Format("formatted");
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("initial: formatted")));
}

TEST(StatusBuilder, With) {
  StatusBuilder(absl::NotFoundError("initial"))
      .Format("%s", "format")
      .With([](absl::Status status) {
        EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound,
                                     HasSubstr("format: initial")));
      });
}

TEST(StatusBuilder, AddStatusPayload) {
  absl::Status status = StatusBuilder(absl::NotFoundError("initial"))
                            .AddStatusPayload("a", absl::Cord("a1"))
                            .AddStatusPayload("a", absl::Cord("a2"))
                            .AddStatusPayload("a", absl::Cord("a3"));
  EXPECT_THAT(status.GetPayload("a"), testing::Optional(absl::Cord("a1")));
  EXPECT_THAT(status.GetPayload("a[1]"), testing::Optional(absl::Cord("a2")));
  EXPECT_THAT(status.GetPayload("a[2]"), testing::Optional(absl::Cord("a3")));
}

TEST(StatusBuilder, IsOk) {
  EXPECT_THAT(StatusBuilder(absl::OkStatus()), IsOk());
}

}  // namespace
