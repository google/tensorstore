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

#include "tensorstore/data_type_conversion.h"

namespace tensorstore {
namespace internal {

Status ValidateSupportsRead(ReadWriteMode mode) {
  return !(mode & ReadWriteMode::read)
             ? absl::InvalidArgumentError("Source does not support reading.")
             : Status();
}

Status ValidateSupportsWrite(ReadWriteMode mode) {
  return !(mode & ReadWriteMode::write)
             ? absl::InvalidArgumentError(
                   "Destination does not support writing.")
             : Status();
}

Status ValidateSupportsModes(ReadWriteMode mode, ReadWriteMode required_modes) {
  if ((mode & required_modes) != required_modes) {
    if (!!(required_modes & ReadWriteMode::read) &&
        !(mode & ReadWriteMode::read)) {
      return absl::InvalidArgumentError("Read mode not supported");
    }
    if (!!(required_modes & ReadWriteMode::write) &&
        !(mode & ReadWriteMode::write)) {
      return absl::InvalidArgumentError("Write mode not supported");
    }
  }
  return absl::OkStatus();
}

}  // namespace internal

namespace internal_tensorstore {
Status ResizeRankError(DimensionIndex rank) {
  return absl::InvalidArgumentError(
      StrCat("inclusive_min and exclusive_max must have rank ", rank));
}
std::string DescribeForCast(DataType data_type, DimensionIndex rank,
                            ReadWriteMode mode) {
  return StrCat("TensorStore with ",
                StaticCastTraits<DataType>::Describe(data_type), ", ",
                StaticCastTraits<DimensionIndex>::Describe(rank),
                " and mode of ", mode);
}

}  // namespace internal_tensorstore
}  // namespace tensorstore
