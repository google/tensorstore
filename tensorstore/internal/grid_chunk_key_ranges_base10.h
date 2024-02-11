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

#ifndef TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_BASE10_H_
#define TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_BASE10_H_

#include <string>
#include <string_view>

#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

// Defines the unpadded base-10 grid index format index, for use with
// `GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys` and
// `GetChunkKeyRangesForRegularGridWithSemiLexicographicalKeys`.
class Base10LexicographicalGridIndexKeyParser
    : public LexicographicalGridIndexKeyParser {
 public:
  explicit Base10LexicographicalGridIndexKeyParser(DimensionIndex rank,
                                                   char dimension_separator)
      : rank(rank), dimension_separator(dimension_separator) {}

  std::string FormatKey(span<const Index> grid_indices) const final;

  bool ParseKey(std::string_view key, span<Index> grid_indices) const final;

  Index MinGridIndexForLexicographicalOrder(
      DimensionIndex dim, IndexInterval grid_interval) const final;

  DimensionIndex rank;
  char dimension_separator;
};

// Returns the smallest non-negative value less than `exclusive_max` with the
// same number of base-10 digits as `exclusive_max-1`.
//
// When using the unpadded base-10 index as a key, lexicographical order does
// not correspond to numerical order:
//
//   1
//   12
//   13
//   14
//   2
//   3
//   ...
//
// Consequently, a lexicographical key range, as supported by the kvstore
// interface, can only be used to specify intervals where the inclusive lower
// and upper bounds have the maximum number of base 10 digits of any valid
// index.
Index MinValueWithMaxBase10Digits(Index exclusive_max);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRID_CHUNK_KEY_RANGES_BASE10_H_
