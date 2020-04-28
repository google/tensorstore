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

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#endif

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

/// Returns the error message associated with a system error code.
std::string GetOsErrorMessage(OsErrorCode error);

/// Converts a system error code to the corresponding `absl::StatusCode`.
absl::StatusCode GetOsErrorStatusCode(OsErrorCode error);

/// Returns a Status from an OS error. The message is composed by catenation of
/// the provided string parts.
absl::Status StatusFromOsError(OsErrorCode error_code, absl::string_view a = {},
                               absl::string_view b = {},
                               absl::string_view c = {},
                               absl::string_view d = {});

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_
