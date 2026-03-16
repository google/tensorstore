// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/kvstore/read_result.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::kvstore::ReadResult;

TEST(ReadResultTest, StringifyState) {
  EXPECT_EQ("<unspecified>", absl::StrCat(ReadResult::kUnspecified));
  EXPECT_EQ("<missing>", absl::StrCat(ReadResult::kMissing));
  EXPECT_EQ("<value>", absl::StrCat(ReadResult::kValue));
}

TEST(ReadResultTest, StringifyReadResult) {
  ReadResult r;
  r.state = ReadResult::kMissing;
  EXPECT_EQ("{value=<missing>, stamp={generation=Unknown, time=infinite-past}}",
            absl::StrCat(r));
  r.state = ReadResult::kValue;
  r.value = absl::Cord("abc");
  EXPECT_EQ("{value=\"abc\", stamp={generation=Unknown, time=infinite-past}}",
            absl::StrCat(r));
}

}  // namespace
