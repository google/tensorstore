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

#ifndef TENSORSTORE_INTERNAL_CORD_UTIL_H_
#define TENSORSTORE_INTERNAL_CORD_UTIL_H_

#include "absl/strings/cord.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Copies the `output.size()` bytes starting at `char_it`.
///
/// Advances `char_it` by `output.size()` bytes.
///
/// \param char_it The start iterator, must be at least `output.size()` bytes
///     before the end.
void CopyCordToSpan(absl::Cord::CharIterator& char_it, span<char> output);

void CopyCordToSpan(const absl::Cord& cord, span<char> output);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CORD_UTIL_H_
