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

#include <string_view>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "tensorstore/kvstore/key_range.h"

namespace tensorstore {
namespace internal_file_util {

/// A key is valid if its consists of one or more '/'-separated non-empty valid
/// path components, where each valid path component does not contain '\0', and
/// is not equal to "." or "..", and does not end in lock_suffix.
bool IsKeyValid(std::string_view key, std::string_view lock_suffix) {
  if (absl::StrContains(key, '\0')) return false;
  if (key.empty()) return false;
  // Do not allow `key` to end with '/'.
  if (key.back() == '/' || key.back() == '\\') {
    return false;
  }
  // Remove leading / which leads to an empty path component.
  if (key.front() == '/' || key.front() == '\\') {
    key = key.substr(1);
  }
  for (std::string_view component :
       absl::StrSplit(key, absl::ByAnyChar("/\\"))) {
    if (component.empty()) return false;
    if (component == ".") return false;
    if (component == "..") return false;
    if (!lock_suffix.empty() && component.size() >= lock_suffix.size() &&
        absl::EndsWith(component, lock_suffix)) {
      return false;
    }
  }
  return true;
}

std::string_view LongestDirectoryPrefix(const KeyRange& range) {
  std::string_view prefix = tensorstore::LongestPrefix(range);
  const size_t i = prefix.rfind('/');
  if (i == std::string_view::npos) return {};
  return prefix.substr(0, i);
}

}  // namespace internal_file_util
}  // namespace tensorstore
