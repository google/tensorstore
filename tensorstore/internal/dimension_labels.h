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

#ifndef TENSORSTORE_INTERNAL_DIMENSION_LABELS_H_
#define TENSORSTORE_INTERNAL_DIMENSION_LABELS_H_

#include <string>

#include "absl/status/status.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Validates that non-empty labels are unique.
///
/// \param labels The sequence of labels to validate.
/// \returns `absl::Status()` if valid.
/// \error `absl::StatusCode::kInvalidArgument` if there is a non-unique label.
absl::Status ValidateDimensionLabelsAreUnique(span<const std::string> labels);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DIMENSION_LABELS_H_
