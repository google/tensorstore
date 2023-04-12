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

#include "tensorstore/internal/ascii_utils.h"

namespace tensorstore {
namespace internal {

/// Percent encodes any characters in `src` that are not in `unreserved`.
void PercentEncodeReserved(std::string_view src, std::string& dest,
                           AsciiSet unreserved) {
  size_t num_escaped = 0;
  for (char c : src) {
    if (!unreserved.Test(c)) ++num_escaped;
  }
  if (num_escaped == 0) {
    dest = src;
    return;
  }
  dest.clear();
  dest.reserve(src.size() + 2 * num_escaped);
  for (char c : src) {
    if (unreserved.Test(c)) {
      dest += c;
    } else {
      dest += '%';
      dest += IntToHexDigit(static_cast<unsigned char>(c) / 16);
      dest += IntToHexDigit(static_cast<unsigned char>(c) % 16);
    }
  }
}

} // namespace tensorstore
} // namespace internal
