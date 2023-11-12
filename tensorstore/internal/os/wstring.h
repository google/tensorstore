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

#ifndef TENSORSTORE_INTERNAL_OS_WSTRING_H_
#define TENSORSTORE_INTERNAL_OS_WSTRING_H_

#ifdef _WIN32

#include <string>
#include <string_view>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal {

/// Converts a UTF-8 string to a windows Multibyte string.
/// TODO: Consider consolidating with kvstore/file/windows_file_util.cc
absl::Status ConvertUTF8ToWindowsWide(std::string_view in, std::wstring& out);

}  // namespace internal
}  // namespace tensorstore

#endif  // _WIN32
#endif  // TENSORSTORE_INTERNAL_OS_WSTRING_H_
