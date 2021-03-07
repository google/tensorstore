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

#ifndef TENSORSTORE_KVSTORE_FILE_UTIL_H_
#define TENSORSTORE_KVSTORE_FILE_UTIL_H_

#include <string_view>

#include "tensorstore/kvstore/key_range.h"

namespace tensorstore {
namespace internal_file_util {

/// A key is valid if its consists of one or more '/'-separated non-empty valid
/// path components, where each valid path component does not contain '\0', and
/// is not equal to "." or "..".
bool IsKeyValid(std::string_view key, std::string_view lock_suffix);

/// Returns the longest directory prefix of a key range.
std::string_view LongestDirectoryPrefix(const KeyRange& range);

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_UTIL_H_
