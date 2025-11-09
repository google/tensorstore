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

#include "tensorstore/internal/dimension_labels.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::ValidateDimensionLabelsAreUnique;

TEST(ValidateDimensionLabelsAreUniqueTest, Basic) {
  TENSORSTORE_EXPECT_OK(ValidateDimensionLabelsAreUnique(
      std::vector<std::string>{"a", "b", "c"}));
  TENSORSTORE_EXPECT_OK(
      ValidateDimensionLabelsAreUnique(std::vector<std::string>{"", "", ""}));
  TENSORSTORE_EXPECT_OK(ValidateDimensionLabelsAreUnique(
      std::vector<std::string>{"a", "b", "", "d", ""}));
  TENSORSTORE_EXPECT_OK(
      ValidateDimensionLabelsAreUnique(std::vector<std::string>{}));
  EXPECT_THAT(ValidateDimensionLabelsAreUnique(
                  std::vector<std::string>{"a", "b", "c", "a"}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Dimension label.* \"a\" not unique"));
  EXPECT_THAT(ValidateDimensionLabelsAreUnique(
                  std::vector<std::string>{"a", "b", "c", "b"}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Dimension label.* \"b\" not unique"));
}

}  // namespace
