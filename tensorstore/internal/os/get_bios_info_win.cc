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

#ifndef _WIN32
#error "Use get_bios_info_win.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "tensorstore/internal/os/get_bios_info.h"
//

#include <windows.h>
#include <winreg.h>
#include <wtypes.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

Result<std::string> GetGcpProductName() {
  std::string subkey = "SYSTEM\\HardwareConfig\\Current";

  DWORD size = 0;
  LONG result = 0;

  do {
    result =
        RegGetValueA(HKEY_LOCAL_MACHINE, subkey.c_str(), "SystemProductName",
                     RRF_RT_REG_SZ, nullptr, nullptr, &size);
    if (result != ERROR_SUCCESS) {
      break;
    }

    std::string contents;
    contents.resize(size / sizeof(char));
    result =
        RegGetValueA(HKEY_LOCAL_MACHINE, subkey.c_str(), "SystemProductName",
                     RRF_RT_REG_SZ, nullptr, &contents[0], &size);
    if (result != ERROR_SUCCESS) {
      break;
    }

    DWORD content_length = size / sizeof(char);
    content_length--;  // Exclude NUL written by WIN32
    contents.resize(content_length);
    return contents;
  } while (false);

  return absl::UnknownError(
      absl::StrCat("error querying registry for "
                   "\"HKLM\\",
                   subkey, "\\SystemProductName\" with code ", result));
}

}  // namespace internal_os
}  // namespace tensorstore
