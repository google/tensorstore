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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_FIND_H_
#define TENSORSTORE_INTERNAL_RIEGELI_FIND_H_

#include <string_view>

#include "riegeli/bytes/reader.h"

namespace tensorstore {
namespace internal {

/// StartsWith()
///
/// Determines whether the Reader starts with the passed string `data`.
/// Similar to absl::Cord::StartsWith.
bool StartsWith(riegeli::Reader &reader, std::string_view needle);

/// FindFirst()
///
/// Seeks for the first occurence of data string starting from the current
/// pos. Implementation note: This implements a naive approach, not
/// Knuth-Morris-Pratt, and is intended to be used for relatively short
/// strings.
bool FindFirst(riegeli::Reader &reader, std::string_view needle);

/// FindLast()
///
/// Seeks for the last occurence of data string starting from reader.
bool FindLast(riegeli::Reader &reader, std::string_view needle);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_FIND_H_
