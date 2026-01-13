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

#include "tensorstore/util/status.h"

#include <functional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::MaybeAnnotateStatus;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::InvokeForStatus;
using ::testing::HasSubstr;

TEST(StatusTest, StrCat) {
  const absl::Status s = absl::UnknownError("Message");
  EXPECT_THAT(s.ToString(), testing::HasSubstr("UNKNOWN: Message"));
  EXPECT_THAT(tensorstore::StrCat(s), testing::HasSubstr("UNKNOWN: Message"));
}

TEST(StatusTest, MaybeAnnotateStatus) {
  EXPECT_THAT(MaybeAnnotateStatus(absl::OkStatus(), "Annotated"), IsOk());

  EXPECT_THAT(MaybeAnnotateStatus(absl::OkStatus(), "Annotated",
                                  tensorstore::SourceLocation::current()),
              IsOk());

  auto bar_status = absl::UnknownError("Bar");
  bar_status.SetPayload("a", absl::Cord("b"));
  auto status = MaybeAnnotateStatus(bar_status, "Annotated");
  EXPECT_TRUE(status.GetPayload("a").has_value());

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnknown,
                               HasSubstr("Annotated: Bar")));
  EXPECT_THAT(tensorstore::StrCat(status), testing::HasSubstr("a='b'"));
}

TEST(StatusTest, InvokeForStatus) {
  int count = 0;

  auto a = [&](int i) { count += i; };
  EXPECT_THAT(InvokeForStatus(a, 1), IsOk());
  EXPECT_EQ(1, count);

  auto b = [&](int i, absl::Status s) {
    count += i;
    return s;
  };
  EXPECT_THAT(InvokeForStatus(b, 2, absl::OkStatus()), IsOk());
  EXPECT_EQ(3, count);

  EXPECT_THAT(InvokeForStatus(b, 4, absl::UnknownError("A")),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("A")));

  EXPECT_EQ(7, count);

  auto c = [](int& i, int j) { i += j; };
  EXPECT_THAT(InvokeForStatus(std::move(c), std::ref(count), 8), IsOk());
  EXPECT_EQ(15, count);
}

TEST(StatusTest, ReturnIfError) {
  const auto Helper = [](absl::Status s) -> absl::Status {
    TENSORSTORE_RETURN_IF_ERROR(s);
    return absl::UnknownError("No error");
  };

  EXPECT_THAT(Helper(absl::Status()),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("No error")));

  EXPECT_THAT(Helper(absl::UnknownError("Got error")),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Got error")));
}

TEST(StatusTest, ReturnIfErrorAnnotate) {
  const auto Helper = [](absl::Status s) -> absl::Status {
    TENSORSTORE_RETURN_IF_ERROR(s, MaybeAnnotateStatus(_, "Annotated"));
    return absl::UnknownError("No error");
  };
  EXPECT_THAT(Helper(absl::Status()),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("No error")));
  EXPECT_THAT(
      Helper(absl::UnknownError("Got error")),
      StatusIs(absl::StatusCode::kUnknown, HasSubstr("Annotated: Got error")));
}

}  // namespace
