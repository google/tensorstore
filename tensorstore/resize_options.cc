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

#include "tensorstore/resize_options.h"

#include <ostream>

#include "absl/base/macros.h"

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, ResolveBoundsMode mode) {
  constexpr const char* kModeNames[] = {
      "fix_resizable_bounds",
  };
  const char* sep = "";
  constexpr const char* kSep = "|";
  for (std::size_t i = 0; i < ABSL_ARRAYSIZE(kModeNames); ++i) {
    if (static_cast<int>(mode) & (1 << i)) {
      os << sep << kModeNames[i];
      sep = kSep;
    }
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, ResizeMode mode) {
  constexpr const char* kModeNames[] = {
      "resize_metadata_only",
      "resize_tied_bounds",
      "expand_only",
      "shrink_only",
  };
  const char* sep = "";
  constexpr const char* kSep = "|";
  for (std::size_t i = 0; i < ABSL_ARRAYSIZE(kModeNames); ++i) {
    if (static_cast<int>(mode) & (1 << i)) {
      os << sep << kModeNames[i];
      sep = kSep;
    }
  }
  return os;
}

}  // namespace tensorstore
