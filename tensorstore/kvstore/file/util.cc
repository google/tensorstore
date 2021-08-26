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

#include "tensorstore/kvstore/file/util.h"

#include <stddef.h>

#include <string>
#include <string_view>

#include "absl/strings/match.h"
#include "tensorstore/kvstore/key_range.h"

namespace tensorstore {
namespace internal_file_util {

/// A key is valid if its consists of one or more '/'-separated non-empty valid
/// path components, where each valid path component does not contain '\0', and
/// is not equal to "." or "..", and does not end in lock_suffix.
bool IsKeyValid(std::string_view key, std::string_view lock_suffix) {
  if (key.find('\0') != std::string_view::npos) return false;
  // Do not allow `key` to end with '/'.
  if (key.empty()) return false;
  if (key.back() == '/') return false;
  while (true) {
    std::size_t next_delimiter = key.find('/');
    std::string_view component = next_delimiter == std::string_view::npos
                                     ? key
                                     : key.substr(0, next_delimiter);
    if (component == ".") return false;
    if (component == "..") return false;
    if (!lock_suffix.empty() && component.size() >= lock_suffix.size() &&
        absl::EndsWith(component, lock_suffix)) {
      return false;
    }
    if (next_delimiter == std::string_view::npos) return true;
    key.remove_prefix(next_delimiter + 1);
  }
}

std::string_view LongestDirectoryPrefix(const KeyRange& range) {
  std::string_view prefix = tensorstore::LongestPrefix(range);
  const size_t i = prefix.rfind('/');
  if (i == std::string_view::npos) return {};
  return prefix.substr(0, i);
}

}  // namespace internal_file_util
}  // namespace tensorstore
