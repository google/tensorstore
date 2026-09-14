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

#ifndef TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_
#define TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_

#include <cassert>
#include <cerrno>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status_builder.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal {

/// Representation of error code returned by system APIs.
#ifdef _WIN32
using OsErrorCode = DWORD;
#else
using OsErrorCode = int;
#endif

/// Returns the thread-local error code from the most recent system API call
/// that failed.
inline OsErrorCode GetLastErrorCode() {
#ifdef _WIN32
  return ::GetLastError();
#else
  return errno;
#endif
}

/// Returns the absl::StatusCode for a given OS error code.
#ifdef _WIN32
absl::StatusCode GetOsErrorStatusCode(OsErrorCode error);
#else
inline absl::StatusCode GetOsErrorStatusCode(OsErrorCode error) {
  return absl::ErrnoToStatusCode(error);
}
#endif

/// Returns the OsErrorCode associated with an absl::Status if present.
inline std::optional<OsErrorCode> GetOsErrorCode(const absl::Status& status) {
  auto payload = status.GetPayload("os_error_code");
  if (!payload.has_value()) return std::nullopt;
  OsErrorCode code;
  if (absl::SimpleAtoi(std::string(*payload), &code)) {
    return code;
  }
  return std::nullopt;
}

/// Returns a literal of the os error code.
const char* OsErrorCodeLiteral(OsErrorCode error);

/// Returns the error message associated with a system error code.
std::string GetOsErrorMessage(OsErrorCode error);

// Builds an `absl::Status` from an OS error code and/or message.
struct StatusFromOsError {
  absl::StatusCode status_code;
  OsErrorCode error_code;
  SourceLocation loc;

  StatusFromOsError(OsErrorCode error_code,
                    SourceLocation loc = SourceLocation::current())
      : status_code(GetOsErrorStatusCode(error_code)),
        error_code(error_code),
        loc(loc) {
    assert(status_code != absl::StatusCode::kOk);
  }

  StatusFromOsError(absl::StatusCode status_code, OsErrorCode error_code,
                    SourceLocation loc = SourceLocation::current())
      : status_code(status_code), error_code(error_code), loc(loc) {
    assert(status_code != absl::StatusCode::kOk);
  }

  // Returns an `absl::Status` with a message constructed from the provided
  // format string and arguments.
  template <typename... Args>
  absl::Status Format(const absl::FormatSpec<Args...>& format,
                      const Args&... args) {
    return StatusBuilder(status_code, loc)
        .SetPayload("os_error_code", absl::StrFormat("%d", error_code))
        .Format(format, args...)
        .Format(": %s", GetOsErrorMessage(error_code));
  }

  // Returns an `absl::Status` with a message constructed only from the OS error
  // message.
  absl::Status Default() const {
    return StatusBuilder(status_code, loc)
        .SetPayload("os_error_code", absl::StrFormat("%d", error_code))
        .Format("%s", GetOsErrorMessage(error_code));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_
