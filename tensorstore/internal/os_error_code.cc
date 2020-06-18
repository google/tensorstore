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

#include "tensorstore/internal/os_error_code.h"

#include <string.h>

#include "absl/strings/str_cat.h"

namespace tensorstore {
namespace internal {

#ifdef _WIN32

absl::StatusCode GetOsErrorStatusCode(OsErrorCode error) {
  switch (error) {
    case ERROR_SUCCESS:
      return absl::StatusCode::kOk;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
    case ERROR_DIR_NOT_EMPTY:
      return absl::StatusCode::kAlreadyExists;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_NOT_ENOUGH_MEMORY:
      return absl::StatusCode::kResourceExhausted;
    default:
      return absl::StatusCode::kFailedPrecondition;
  }
}

std::string GetOsErrorMessage(OsErrorCode error) {
  char buf[4096];
  DWORD size = ::FormatMessageA(
      /*dwFlags=*/FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      /*lpSource=*/nullptr,
      /*dwMessageId=*/error,
      /*dwLanguageId=*/0,
      /*lpBuffer=*/buf,
      /*nSize=*/std::size(buf),
      /*Arguments=*/nullptr);
  return std::string(buf, size);
}

#else

// There are two versions of the ::strerror_r function:
//
// XSI-compliant:
//
//     int strerror_r(int errnum, char* buf, size_t buflen);
//
//   Always writes message to supplied buffer.
//
// GNU-specific:
//
//     char *strerror_r(int errnum, char* buf, size_t buflen);
//
//   Either writes message to supplied buffer, or returns a static string.
//
// The following overloads are used to detect the return type and return the
// appropriate result.

namespace {
// GNU version
[[maybe_unused]] const char* GetStrerrorResult(const char* buf,
                                               const char* result) {
  return result;
}
// XSI-compliant version
[[maybe_unused]] const char* GetStrerrorResult(const char* buf, int result) {
  return buf;
}
}  // namespace

std::string GetOsErrorMessage(OsErrorCode error) {
  char buf[4096];
  buf[0] = 0;
  return GetStrerrorResult(buf, ::strerror_r(error, buf, std::size(buf)));
}

absl::StatusCode GetOsErrorStatusCode(OsErrorCode error) {
  switch (error) {
    case ENOENT:
      return absl::StatusCode::kNotFound;
    case EEXIST:
    case ENOTEMPTY:
      return absl::StatusCode::kAlreadyExists;
    case ENOSPC:
    case ENOMEM:
      return absl::StatusCode::kResourceExhausted;
    case EACCES:
    case EPERM:
      return absl::StatusCode::kPermissionDenied;
    default:
      return absl::StatusCode::kFailedPrecondition;
  }
}
#endif

absl::Status StatusFromOsError(OsErrorCode error_code, absl::string_view a,
                               absl::string_view b, absl::string_view c,
                               absl::string_view d) {
  return absl::Status(GetOsErrorStatusCode(error_code),
                      absl::StrCat(a, b, c, d, " [OS error: ",
                                   GetOsErrorMessage(error_code), "]"));
}

}  // namespace internal
}  // namespace tensorstore
