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

#ifndef _WIN32
#error "Use error_code_posix.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "tensorstore/internal/os/error_code.h"
// Normal include order here

#include <string>

#include "absl/base/optimization.h"
#include "absl/status/status.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal {

absl::StatusCode GetOsErrorStatusCode(OsErrorCode error) {
  switch (error) {
    case ERROR_SUCCESS:
      return absl::StatusCode::kOk;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_BAD_PATHNAME:
    case ERROR_DIRECTORY:
    case ERROR_NO_MORE_FILES:
      return absl::StatusCode::kNotFound;
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_DISK_FULL:
    case ERROR_DISK_TOO_FRAGMENTED:
    case ERROR_OUTOFMEMORY:
      return absl::StatusCode::kResourceExhausted;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_INVALID_NAME:
    case ERROR_DELETE_PENDING:
      return absl::StatusCode::kPermissionDenied;
    case ERROR_BUFFER_OVERFLOW:
    case ERROR_FILENAME_EXCED_RANGE:
      return absl::StatusCode::kInvalidArgument;
    case ERROR_DIR_NOT_EMPTY:
    default:
      return absl::StatusCode::kFailedPrecondition;
  }
}

const char* OsErrorCodeLiteral(OsErrorCode error) {
  switch (error) {
    case ERROR_FILE_EXISTS:
      return "ERROR_FILE_EXISTS ";
    case ERROR_ALREADY_EXISTS:
      return "ERROR_ALREADY_EXISTS ";
    case ERROR_FILE_NOT_FOUND:
      return "ERROR_FILE_NOT_FOUND ";
    case ERROR_PATH_NOT_FOUND:
      return "ERROR_PATH_NOT_FOUND ";
    case ERROR_BAD_PATHNAME:
      return "ERROR_BAD_PATHNAME ";
    case ERROR_DIRECTORY:
      return "ERROR_DIRECTORY ";
    case ERROR_NO_MORE_FILES:
      return "ERROR_NO_MORE_FILES ";
    case ERROR_TOO_MANY_OPEN_FILES:
      return "ERROR_TOO_MANY_OPEN_FILES ";
    case ERROR_NOT_ENOUGH_MEMORY:
      return "ERROR_NOT_ENOUGH_MEMORY ";
    case ERROR_HANDLE_DISK_FULL:
      return "ERROR_HANDLE_DISK_FULL ";
    case ERROR_DISK_FULL:
      return "ERROR_DISK_FULL ";
    case ERROR_DISK_TOO_FRAGMENTED:
      return "ERROR_DISK_TOO_FRAGMENTED ";
    case ERROR_OUTOFMEMORY:
      return "ERROR_OUTOFMEMORY ";
    case ERROR_ACCESS_DENIED:
      return "ERROR_ACCESS_DENIED ";
    case ERROR_SHARING_VIOLATION:
      return "ERROR_SHARING_VIOLATION ";
    case ERROR_INVALID_NAME:
      return "ERROR_INVALID_NAME ";
    case ERROR_DELETE_PENDING:
      return "ERROR_DELETE_PENDING ";
    case ERROR_BUFFER_OVERFLOW:
      return "ERROR_BUFFER_OVERFLOW ";
    case ERROR_FILENAME_EXCED_RANGE:
      return "ERROR_FILENAME_EXCED_RANGE ";
    case ERROR_DIR_NOT_EMPTY:
      return "ERROR_DIR_NOT_EMPTY ";
    default:
      return "";
  }
  ABSL_UNREACHABLE();
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

}  // namespace internal
}  // namespace tensorstore
