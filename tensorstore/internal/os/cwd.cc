// Copyright 2024 The TensorStore Authors
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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "tensorstore/internal/os/cwd.h"
// Normal include order here

#include <stddef.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>  // IWYU pragma: keep

#include "absl/log/absl_check.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/wstring.h"  // IWYU pragma: keep
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

#ifdef _WIN32
using ::tensorstore::internal::ConvertUTF8ToWindowsWide;
using ::tensorstore::internal::ConvertWindowsWideToUTF8;
#endif

using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {

Result<std::string> GetCwd() {
#ifdef _WIN32
  // Determine required buffer size.
  DWORD size = ::GetCurrentDirectoryW(0, nullptr);

  // size is equal to the required length, INCLUDING the terminating NUL.
  while (true) {
    // size of 0 indicates an error
    if (size == 0) break;

    std::vector<wchar_t> buf(size);

    // If `size` was sufficient, `new_size` is equal to the path length,
    // EXCLUDING the terminating NUL.
    //
    // If `size` was insufficient, `new_size` is equal to the path length,
    // INCLUDING the terminating NUL.
    DWORD new_size = ::GetCurrentDirectoryW(size, buf.data());
    if (new_size != size - 1) {
      // Another thread changed the current working directory between the two
      // calls to `GetCurrentDirectoryW`.

      // It is not valid for `size` to exactly equal `new_size`, since that
      // would simultaneously mean `size` was insufficient but also the
      // correct size.
      ABSL_CHECK_NE(new_size, size);

      if (new_size > size) {
        size = new_size;
        continue;
      }
    }

    std::string utf8_buf;
    TENSORSTORE_RETURN_IF_ERROR(ConvertWindowsWideToUTF8(
        std::wstring_view(buf.data(), new_size), utf8_buf));
    return utf8_buf;
  }
  auto get_error = ::GetLastError();
#else
  std::string buf;
  buf.resize(256);
  while (true) {
    if (::getcwd(buf.data(), buf.size()) != nullptr) {
      buf.resize(strlen(buf.data()));
      return buf;
    }
    if (errno == ERANGE) {
      buf.resize(buf.size() * 2);
      continue;
    }
    break;
  }
  auto get_error = errno;
#endif
  return StatusFromOsError(get_error,
                           "Failed to get current working directory");
}

absl::Status SetCwd(const std::string& path) {
#ifdef _WIN32
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  if (::SetCurrentDirectoryW(wpath.c_str())) {
    return absl::OkStatus();
  }
  auto set_error = ::GetLastError();
#else
  if (::chdir(path.c_str()) == 0) {
    return absl::OkStatus();
  }
  auto set_error = errno;
#endif
  return StatusFromOsError(
      set_error,
      "Failed to set current working directory to: ", QuoteString(path));
}

}  // namespace internal_os
}  // namespace tensorstore
