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

#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::GetFirstErrorStatus;
using tensorstore::InvokeForStatus;
using tensorstore::MaybeAnnotateStatus;
using tensorstore::Status;

TEST(StatusTest, StrCat) {
  const Status s = absl::UnknownError("Message");
  EXPECT_EQ("UNKNOWN: Message", s.ToString());
  EXPECT_EQ("UNKNOWN: Message", tensorstore::StrCat(s));
}

TEST(StatusTest, GetFirstErrorStatus) {
  EXPECT_EQ(absl::UnknownError("A"),
            GetFirstErrorStatus(absl::OkStatus(), absl::UnknownError("A"),
                                absl::UnknownError("B")));
  EXPECT_EQ(absl::OkStatus(),
            GetFirstErrorStatus(absl::OkStatus(), absl::OkStatus()));
}

TEST(StatusTest, MaybeAnnotateStatus) {
  EXPECT_EQ(absl::OkStatus(),
            MaybeAnnotateStatus(absl::OkStatus(), "Annotated"));
  EXPECT_EQ(absl::UnknownError("Annotated: Bar"),
            MaybeAnnotateStatus(absl::UnknownError("Bar"), "Annotated"));
}

TEST(StatusTest, InvokeForStatus) {
  int count = 0;

  auto a = [&](int i) { count += i; };
  EXPECT_EQ(absl::OkStatus(), InvokeForStatus(a, 1));
  EXPECT_EQ(1, count);

  auto b = [&](int i, Status s) {
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
  const auto Helper = [](Status s) {
    TENSORSTORE_RETURN_IF_ERROR(s);
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(Status()));
  EXPECT_EQ(absl::UnknownError("Got error"),
            Helper(absl::UnknownError("Got error")));
}

TEST(StatusTest, ReturnIfErrorAnnotate) {
  const auto Helper = [](Status s) {
    TENSORSTORE_RETURN_IF_ERROR(s, MaybeAnnotateStatus(_, "Annotated"));
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(Status()));
  EXPECT_EQ(absl::UnknownError("Annotated: Got error"),
            Helper(absl::UnknownError("Got error")));
}

}  // namespace
