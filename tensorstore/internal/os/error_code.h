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

#include <cerrno>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"

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

/// Returns the error message associated with a system error code.
std::string GetOsErrorMessage(OsErrorCode error);

/// Returns an `absl::Status` from an OS error. The message is composed by
/// catenation of the provided string parts.
absl::Status StatusFromOsError(
    OsErrorCode error_code, std::string_view a = {}, std::string_view b = {},
    std::string_view c = {}, std::string_view d = {}, std::string_view e = {},
    std::string_view f = {},
    SourceLocation loc = tensorstore::SourceLocation::current());

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_
