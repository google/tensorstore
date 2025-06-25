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
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

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

/// Returns a literal of the os error code.
const char* OsErrorCodeLiteral(OsErrorCode error);

/// Returns the error message associated with a system error code.
std::string GetOsErrorMessage(OsErrorCode error);

/// Returns an `absl::Status` with an OS error. The message is composed by
/// catenation of the provided parts {a .. f}
template <typename A = std::string_view, typename B = std::string_view,
          typename C = std::string_view, typename D = std::string_view,
          typename E = std::string_view, typename F = std::string_view>
absl::Status StatusWithOsError(
    absl::StatusCode status_code, OsErrorCode error_code,  //
    A a = {}, B b = {}, C c = {}, D d = {}, E e = {}, F f = {},
    SourceLocation loc = tensorstore::SourceLocation::current()) {
  absl::Status status(
      status_code,
      tensorstore::StrCat(a, b, c, d, e, f, " [OS error ", error_code, ": ",
                          OsErrorCodeLiteral(error_code),
                          GetOsErrorMessage(error_code), "]"));
  MaybeAddSourceLocation(status, loc);
  return status;
}

/// Returns an `absl::Status` from an OS error. The message is composed by
/// catenation of the provided parts {a .. f}
template <typename A = std::string_view, typename B = std::string_view,
          typename C = std::string_view, typename D = std::string_view,
          typename E = std::string_view, typename F = std::string_view>
absl::Status StatusFromOsError(
    OsErrorCode error_code,  //
    A a = {}, B b = {}, C c = {}, D d = {}, E e = {}, F f = {},
    SourceLocation loc = tensorstore::SourceLocation::current()) {
  return StatusWithOsError(GetOsErrorStatusCode(error_code), error_code,
                           std::move(a), std::move(b), std::move(c),
                           std::move(d), std::move(e), std::move(f), loc);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_ERROR_CODE_H_
