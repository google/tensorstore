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

Status ValidateDataTypeAndRank(internal::DriverConstraints expected,
                               internal::DriverConstraints actual) {
  if (!tensorstore::IsRankExplicitlyConvertible(expected.rank, actual.rank)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Expected rank of ", expected.rank, " but received: ", actual.rank));
  }
  if (!tensorstore::IsPossiblySameDataType(expected.data_type,
                                           actual.data_type)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Expected data type of ", expected.data_type,
                            " but received: ", actual.data_type));
  }
  return absl::OkStatus();
}

}  // namespace internal_open
}  // namespace tensorstore
