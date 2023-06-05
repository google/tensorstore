// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"

#include <string>
#include <string_view>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

void Base10LexicographicalGridIndexKeyParser::FormatGridIndex(
    std::string& out, DimensionIndex dim, Index grid_index) const {
  absl::StrAppend(&out, grid_index);
}

bool Base10LexicographicalGridIndexKeyParser::ParseGridIndex(
    std::string_view key, DimensionIndex dim, Index& grid_index) const {
  if (key.empty() || !absl::ascii_isdigit(key.front()) ||
      !absl::ascii_isdigit(key.back()) || !absl::SimpleAtoi(key, &grid_index)) {
    return false;
  }
  return true;
}

Index Base10LexicographicalGridIndexKeyParser::
    MinGridIndexForLexicographicalOrder(DimensionIndex dim,
                                        IndexInterval grid_interval) const {
  return MinValueWithMaxBase10Digits(grid_interval.exclusive_max());
}

Index MinValueWithMaxBase10Digits(Index exclusive_max) {
  if (exclusive_max <= 10) {
    return 0;
  }
  Index min_value = 10;
  while (min_value * 10 < exclusive_max) {
    min_value *= 10;
  }
  return min_value;
}

}  // namespace internal
}  // namespace tensorstore
