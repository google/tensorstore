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

#include <system_error>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::MaybeAnnotateStatus;
using ::tensorstore::internal::InvokeForStatus;
using ::tensorstore::internal::MaybeAnnotateStatusImpl;
using ::tensorstore::internal::MaybeConvertStatusTo;

TEST(StatusTest, StrCat) {
  const absl::Status s = absl::UnknownError("Message");
  EXPECT_THAT(s.ToString(), testing::HasSubstr("UNKNOWN: Message"));
  EXPECT_THAT(tensorstore::StrCat(s), testing::HasSubstr("UNKNOWN: Message"));
}

TEST(StatusTest, MaybeAnnotateStatusImpl) {
  // Just change the code.
  EXPECT_EQ(
      MaybeAnnotateStatusImpl(absl::UnknownError("Boo"), {},
                              absl::StatusCode::kInternal, TENSORSTORE_LOC),
      absl::InternalError("Boo"));

  // Just change the message.
  EXPECT_EQ(MaybeAnnotateStatusImpl(absl::UnknownError("Boo"), "Annotated", {},
                                    TENSORSTORE_LOC),
            absl::UnknownError("Annotated: Boo"));

  // Change both code and message
  EXPECT_EQ(
      MaybeAnnotateStatusImpl(absl::UnknownError("Boo"), "Annotated",
                              absl::StatusCode::kInternal, TENSORSTORE_LOC),
      absl::InternalError("Annotated: Boo"));
}

TEST(StatusTest, MaybeAnnotateStatus) {
  EXPECT_EQ(absl::OkStatus(),  //
            MaybeAnnotateStatus(absl::OkStatus(), "Annotated"));

  EXPECT_EQ(
      absl::OkStatus(),  //
      MaybeAnnotateStatus(absl::OkStatus(), "Annotated", TENSORSTORE_LOC));

  auto bar_status = absl::UnknownError("Bar");
  bar_status.SetPayload("a", absl::Cord("b"));
  auto status = MaybeAnnotateStatus(bar_status, "Annotated");
  EXPECT_TRUE(status.GetPayload("a").has_value());

  // EXEPCT_EQ also verifies status.payloads.
  auto expected = absl::UnknownError("Annotated: Bar");
  expected.SetPayload("a", absl::Cord("b"));
  EXPECT_EQ(expected, status);
}

TEST(StatusTest, MaybeConvertStatusTo) {
  EXPECT_EQ(absl::OkStatus(),  //
            MaybeConvertStatusTo(absl::OkStatus(),
                                 absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(absl::InternalError("Boo"),  //
            MaybeConvertStatusTo(absl::UnknownError("Boo"),
                                 absl::StatusCode::kInternal));
}

TEST(StatusTest, InvokeForStatus) {
  int count = 0;

  auto a = [&](int i) { count += i; };
  EXPECT_EQ(absl::OkStatus(), InvokeForStatus(a, 1));
  EXPECT_EQ(1, count);

  auto b = [&](int i, absl::Status s) {
    count += i;
    return s;
  };
  EXPECT_EQ(absl::OkStatus(), InvokeForStatus(b, 2, absl::OkStatus()));
  EXPECT_EQ(3, count);

  EXPECT_EQ(absl::UnknownError("A"),
            InvokeForStatus(b, 4, absl::UnknownError("A")));
  EXPECT_EQ(7, count);

  auto c = [](int& i, int j) { i += j; };
  EXPECT_EQ(absl::OkStatus(),
            InvokeForStatus(std::move(c), std::ref(count), 8));
  EXPECT_EQ(15, count);
}

TEST(StatusTest, ReturnIfError) {
  const auto Helper = [](absl::Status s) {
    TENSORSTORE_RETURN_IF_ERROR(s);
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(absl::Status()));
  EXPECT_EQ(absl::UnknownError("Got error"),
            Helper(absl::UnknownError("Got error")));
}

TEST(StatusTest, ReturnIfErrorAnnotate) {
  const auto Helper = [](absl::Status s) {
    TENSORSTORE_RETURN_IF_ERROR(s, MaybeAnnotateStatus(_, "Annotated"));
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(absl::Status()));
  EXPECT_EQ(absl::UnknownError("Annotated: Got error"),
            Helper(absl::UnknownError("Got error")));
}

}  // namespace
