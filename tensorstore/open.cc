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

#include "tensorstore/open.h"

#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_open {

Status InvalidModeError(ReadWriteMode mode, ReadWriteMode static_mode) {
  return absl::InvalidArgumentError(StrCat("Run-time mode ", mode,
                                           " does not match compile-time mode ",
                                           static_mode));
}

Status ValidateDataTypeAndRank(DataType expected_dtype,
                               DimensionIndex expected_rank,
                               DataType actual_dtype,
                               DimensionIndex actual_rank) {
  if (!tensorstore::IsRankExplicitlyConvertible(expected_rank, actual_rank)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Expected rank of ", expected_rank, " but received: ", actual_rank));
  }
  if (!tensorstore::IsPossiblySameDataType(expected_dtype, actual_dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Expected data type of ", expected_dtype,
                            " but received: ", actual_dtype));
  }
  return absl::OkStatus();
}

}  // namespace internal_open
}  // namespace tensorstore
