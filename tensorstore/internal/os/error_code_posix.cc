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

#include "tensorstore/internal/os/error_code.h"

#ifdef _WIN32
#error "Use error_code_win.cc instead."
#endif

#include <cerrno>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

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
  return absl::ErrnoToStatusCode(error);
}

absl::Status StatusFromOsError(OsErrorCode error_code, std::string_view a,
                               std::string_view b, std::string_view c,
                               std::string_view d, SourceLocation loc) {
  absl::Status status(
      GetOsErrorStatusCode(error_code),
      tensorstore::StrCat(a, b, c, d, " [OS error ", error_code, ": ",
                          GetOsErrorMessage(error_code), "]"));
  MaybeAddSourceLocation(status, loc);
  return status;
}

}  // namespace internal
}  // namespace tensorstore
