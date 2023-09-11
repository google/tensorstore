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

#include <stddef.h>

#include <algorithm>
#include <string>
#include <string_view>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

namespace {
absl::Status ValidateDimensionLabelsAreUniqueImpl(
    span<std::string_view> sorted_labels) {
  std::sort(sorted_labels.begin(), sorted_labels.end());
  size_t i;
  for (i = 1; i < sorted_labels.size() && sorted_labels[i].empty(); ++i)
    continue;
  std::string error;
  for (; i < sorted_labels.size(); ++i) {
    std::string_view label = sorted_labels[i];
    if (label == sorted_labels[i - 1]) {
      tensorstore::StrAppend(&error, error.empty() ? "" : ", ",
                             QuoteString(label));
    }
  }
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Dimension label(s) ", error, " not unique"));
  }

  return absl::OkStatus();
}
}  // namespace

absl::Status ValidateDimensionLabelsAreUnique(span<const std::string> labels) {
  // TODO(jbms): Consider using a hash set instead.
  absl::FixedArray<std::string_view, kMaxRank> sorted_labels(labels.begin(),
                                                             labels.end());
  return ValidateDimensionLabelsAreUniqueImpl(sorted_labels);
}

absl::Status ValidateDimensionLabelsAreUnique(
    span<const std::string_view> labels) {
  // TODO(jbms): Consider using a hash set instead.
  absl::FixedArray<std::string_view, kMaxRank> sorted_labels(labels.begin(),
                                                             labels.end());
  return ValidateDimensionLabelsAreUniqueImpl(sorted_labels);
}

}  // namespace internal
}  // namespace tensorstore
