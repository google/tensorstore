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

#include "tensorstore/util/bit_vec_impl.h"

#include <algorithm>
#include <cstring>

#include "absl/base/macros.h"

namespace tensorstore {

namespace internal_bitvec {

void BitVecStorage<dynamic_extent>::resize(std::ptrdiff_t new_size,
                                           bool value) {
  ABSL_ASSERT(new_size >= 0);
  const std::ptrdiff_t existing_size = size_;
  const std::ptrdiff_t existing_num_blocks = num_blocks();
  const std::ptrdiff_t new_num_blocks = BitVectorSizeInBlocks<Block>(new_size);
  constexpr std::ptrdiff_t block_offset_mask = 8 * sizeof(Block) - 1;
  if (new_num_blocks != existing_num_blocks) {
    BitVecStorage temp(new_size);
    std::memcpy(temp.data(), data(),
                sizeof(Block) * std::min(existing_num_blocks, new_num_blocks));
    if (new_num_blocks > existing_num_blocks) {
      // Fix new block.
      std::memset(temp.data() + existing_num_blocks, value ? 0xff : 0,
                  sizeof(Block) * (new_num_blocks - existing_num_blocks));
    }
    temp.swap(*this);
  } else {
    size_ = new_size;
  }
  if (new_size > existing_size && (existing_size & block_offset_mask)) {
    // Fix high bits of last existing block.
    const Block last_block_exclude_mask =
        (~static_cast<Block>(0)) << (existing_size & block_offset_mask);
    Block& last_block = data()[existing_num_blocks - 1];
    last_block &= ~last_block_exclude_mask;
    if (value) last_block |= last_block_exclude_mask;
  }
}

}  // namespace internal_bitvec
}  // namespace tensorstore
