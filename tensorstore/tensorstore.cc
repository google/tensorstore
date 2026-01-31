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

#include "tensorstore/tensorstore.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"

namespace tensorstore {

namespace internal {

absl::Status InvalidModeError(ReadWriteMode mode, ReadWriteMode static_mode) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Run-time mode %s does not match compile-time mode %s",
      absl::FormatStreamed(mode), absl::FormatStreamed(static_mode)));
}

absl::Status ValidateDataTypeAndRank(DataType expected_dtype,
                                     DimensionIndex expected_rank,
                                     DataType actual_dtype,
                                     DimensionIndex actual_rank) {
  if (!tensorstore::RankConstraint::EqualOrUnspecified(expected_rank,
                                                       actual_rank)) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Expected rank of %d but received: %d", expected_rank, actual_rank));
  }
  if (!tensorstore::IsPossiblySameDataType(expected_dtype, actual_dtype)) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Expected data type of %s but received: %s",
                        absl::FormatStreamed(expected_dtype),
                        absl::FormatStreamed(actual_dtype)));
  }
  return absl::OkStatus();
}

}  // namespace internal

namespace internal_tensorstore {
absl::Status ResizeRankError(DimensionIndex rank) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "inclusive_min and exclusive_max must have rank %d", rank));
}
std::string DescribeForCast(DataType dtype, DimensionIndex rank,
                            ReadWriteMode mode) {
  return absl::StrFormat("TensorStore with %s, %s, and mode of %s",
                         StaticCastTraits<DataType>::Describe(dtype),
                         StaticCastTraits<DimensionIndex>::Describe(rank),
                         absl::FormatStreamed(mode));
}

}  // namespace internal_tensorstore
}  // namespace tensorstore
