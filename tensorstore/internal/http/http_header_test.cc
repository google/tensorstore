// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/http/http_header.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_http::ValidateHttpHeader;

TEST(ValidateHttpHeaderTest, Valid) {
  TENSORSTORE_EXPECT_OK(ValidateHttpHeader("a!#$%&'*+-.^_`|~3X: b\xfe"));
}

TEST(ValidateHttpHeaderTest, Invalid) {
  EXPECT_THAT(ValidateHttpHeader("a"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateHttpHeader("a: \n"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
