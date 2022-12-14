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

#include "tensorstore/rank.h"

#include "tensorstore/util/str_cat.h"

namespace tensorstore {

std::string StaticCastTraits<DimensionIndex>::Describe(DimensionIndex value) {
  if (value == dynamic_rank) return "dynamic rank";
  return tensorstore::StrCat("rank of ", value);
}

absl::Status ValidateRank(DimensionIndex rank) {
  if (!IsValidRank(rank)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Rank ", rank, " is outside valid range [0, ", kMaxRank, "]"));
  }
  return absl::OkStatus();
}

}  // namespace tensorstore
