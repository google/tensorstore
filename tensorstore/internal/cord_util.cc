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

#include "tensorstore/internal/cord_util.h"

namespace tensorstore {
namespace internal {

void CopyCordToSpan(absl::Cord::CharIterator& char_it, span<char> output) {
  while (!output.empty()) {
    auto chunk = absl::Cord::ChunkRemaining(char_it);
    size_t n = std::min(chunk.size(), static_cast<size_t>(output.size()));
    std::memcpy(output.data(), chunk.data(), n);
    absl::Cord::Advance(&char_it, n);
    output = {output.data() + n, output.size() - static_cast<ptrdiff_t>(n)};
  }
}

void CopyCordToSpan(const absl::Cord& cord, span<char> output) {
  assert(output.size() <= cord.size());
  auto char_it = cord.char_begin();
  CopyCordToSpan(char_it, output);
}

}  // namespace internal
}  // namespace tensorstore
