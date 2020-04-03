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

#ifndef TENSORSTORE_UTIL_CONSTANT_BIT_VECTOR_H_
#define TENSORSTORE_UTIL_CONSTANT_BIT_VECTOR_H_

#include <type_traits>

#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/constant_vector.h"

namespace tensorstore {

/// Returns a constant `BitSpan` with unsigned integer block type `Block`
/// containing `value` of compile-time length `Length`.
template <typename Block, bool value, std::ptrdiff_t Length>
constexpr BitSpan<const Block, Length> GetConstantBitVector(
    std::integral_constant<std::ptrdiff_t, Length> = {}) {
  return {GetConstantVector<
              Block, (value ? ~static_cast<Block>(0) : static_cast<Block>(0)),
              BitVectorSizeInBlocks<Block>(Length)>()
              .data(),
          0, Length};
}

/// Returns a constant `BitSpan` with unsigned integer block type `Block`
/// containing `value` of run-time length `length`.
template <typename Block, bool value>
BitSpan<const Block> GetConstantBitVector(std::ptrdiff_t length) {
  return {GetConstantVector<Block, (value ? ~static_cast<Block>(0)
                                          : static_cast<Block>(0))>(
              BitVectorSizeInBlocks<Block>(length))
              .data(),
          0, length};
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_CONSTANT_BIT_VECTOR_H_
