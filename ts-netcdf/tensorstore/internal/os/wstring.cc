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

#include "tensorstore/internal/os/wstring.h"  // IWYU pragma: keep

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <cassert>
#include <limits>  // IWYU pragma: keep
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/os/error_code.h"  // IWYU pragma: keep

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal {

absl::Status ConvertUTF8ToWindowsWide(std::string_view in, std::wstring& out) {
  if (in.size() > std::numeric_limits<int>::max()) {
    return StatusFromOsError(ERROR_BUFFER_OVERFLOW,
                             "ConvertUTF8ToWindowsWide buffer overflow");
  }
  if (in.empty()) {
    out.clear();
    return absl::OkStatus();
  }
  int n = ::MultiByteToWideChar(
      /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS, in.data(),
      static_cast<int>(in.size()), nullptr, 0);
  if (n <= 0) {
    return StatusFromOsError(::GetLastError(),
                             "ConvertUTF8ToWindowsWide failed");
  }
  out.resize(n);
  int m = ::MultiByteToWideChar(
      /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS, in.data(),
      static_cast<int>(in.size()), out.data(), n);
  if (n <= 0) {
    return StatusFromOsError(::GetLastError(),
                             "ConvertUTF8ToWindowsWide failed");
  }
  return absl::OkStatus();
}

/// Converts a windows Multibyte string to a UTF-8 String
absl::Status ConvertWindowsWideToUTF8(std::wstring_view in, std::string& out) {
  if (in.empty()) {
    out.clear();
    return absl::OkStatus();
  }
  int n = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,           //
                                in.data(), static_cast<int>(in.size()),  //
                                nullptr, 0,
                                /*lpDefaultChar=*/nullptr,
                                /*lpUsedDefaultChar=*/nullptr);
  if (n <= 0) {
    return StatusFromOsError(::GetLastError(),
                             "ConvertWindowsWideToUTF8 failed");
  }
  out.resize(n);
  n = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,           //
                            in.data(), static_cast<int>(in.size()),  //
                            out.data(), n,
                            /*lpDefaultChar=*/nullptr,
                            /*lpUsedDefaultChar=*/nullptr);
  if (n <= 0) {
    return StatusFromOsError(::GetLastError(),
                             "ConvertWindowsWideToUTF8 failed");
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore

#endif
