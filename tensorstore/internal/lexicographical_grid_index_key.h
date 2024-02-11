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

#ifndef TENSORSTORE_INTERNAL_LEXICOGRAPHICAL_GRID_INDEX_KEY_H_
#define TENSORSTORE_INTERNAL_LEXICOGRAPHICAL_GRID_INDEX_KEY_H_

#include <stddef.h>

#include <cassert>
#include <string>
#include <string_view>

#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

// Specifies the key format for
// `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys`.
class LexicographicalGridIndexKeyFormatter {
 public:
  // Returns the key or key prefix corresponding to `grid_indices`.
  virtual std::string FormatKey(span<const Index> grid_indices) const = 0;

  // Returns the first grid index for dimension `dim` at which the formatted
  // keys are ordered lexicographically.
  virtual Index MinGridIndexForLexicographicalOrder(
      DimensionIndex dim, IndexInterval grid_interval) const = 0;

  virtual ~LexicographicalGridIndexKeyFormatter() = default;
};

// Specifies the key format for
// `GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys`.
class LexicographicalGridIndexKeyParser
    : public LexicographicalGridIndexKeyFormatter {
 public:
  // Parses a key, inverse of `LexicographicalGridIndexKeyFormatter::FormatKey`.
  virtual bool ParseKey(std::string_view key,
                        span<Index> grid_indices) const = 0;
};

template <typename FormatGridIndex>
void FormatGridIndexKeyWithDimensionSeparator(std::string& out,
                                              char dimension_separator,
                                              FormatGridIndex format_grid_index,
                                              DimensionIndex rank,
                                              span<const Index> grid_indices) {
  assert(grid_indices.size() <= rank);
  for (DimensionIndex i = 0; i < grid_indices.size(); ++i) {
    format_grid_index(out, i, grid_indices[i]);
    if (i + 1 != rank) out += dimension_separator;
  }
}

template <typename ParseGridIndex>
bool ParseGridIndexKeyWithDimensionSeparator(char dimension_separator,
                                             ParseGridIndex parse_grid_index,
                                             std::string_view key,
                                             span<Index> grid_indices) {
  if (key.empty()) return false;
  for (DimensionIndex i = 0; i != grid_indices.size(); ++i) {
    std::string_view part;
    if (i + 1 == grid_indices.size()) {
      part = key;
    } else {
      size_t next = key.find(dimension_separator);
      if (next == std::string_view::npos) return false;
      part = key.substr(0, next);
      key.remove_prefix(part.size() + 1);
    }
    if (!parse_grid_index(part, i, grid_indices[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_LEXICOGRAPHICAL_GRID_INDEX_KEY_H_
