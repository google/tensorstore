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
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

std::string Base10LexicographicalGridIndexKeyParser::FormatKey(
    span<const Index> grid_indices) const {
  if (rank == 0) return "0";
  std::string key;
  FormatGridIndexKeyWithDimensionSeparator(
      key, dimension_separator,
      [](std::string& out, DimensionIndex dim, Index grid_index) {
        absl::StrAppend(&out, grid_index);
      },
      rank, grid_indices);
  return key;
}

bool Base10LexicographicalGridIndexKeyParser::ParseKey(
    std::string_view key, span<Index> grid_indices) const {
  return ParseGridIndexKeyWithDimensionSeparator(
      dimension_separator,
      [](std::string_view part, DimensionIndex dim, Index& grid_index) {
        if (part.empty() || !absl::ascii_isdigit(part.front()) ||
            !absl::ascii_isdigit(part.back()) ||
            !absl::SimpleAtoi(part, &grid_index)) {
          return false;
        }
        return true;
      },
      key, grid_indices);
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
