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

#ifndef TENSORSTORE_INTERNAL_UTF8_H_
#define TENSORSTORE_INTERNAL_UTF8_H_

#include "absl/strings/string_view.h"

/// UTF-8 validation utilities.

namespace tensorstore {
namespace internal {

/// Validates that `code_units` is a valid UTF-8 sequence.
///
/// Surrogate code points, overlong 2, 3, and 4 byte sequences, and 4 byte
/// sequences outside the Unicode range are not considered valid.
///
/// \param code_units The sequence to validate.
/// \returns `true` if the sequence is valid, `false` otherwise.
bool IsValidUtf8(absl::string_view code_units);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_UTF8_H_
