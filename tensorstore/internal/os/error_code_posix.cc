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

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {
namespace {

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

// GNU version
[[maybe_unused]] const char* GetStrerrorResult(const char* buf,
                                               const char* result) {
  return result;
}
// XSI-compliant version
[[maybe_unused]] const char* GetStrerrorResult(const char* buf, int result) {
  return buf;
}

// Returns the string representation of the given error number.
const char* ErrnoToCString(int error_number) {
  switch (error_number) {
    case EINVAL:
      return "EINVAL ";
    case ENAMETOOLONG:
      return "ENAMETOOLONG ";
    case E2BIG:
      return "E2BIG ";
    case EDESTADDRREQ:
      return "EDESTADDRREQ ";
    case EDOM:
      return "EDOM ";
    case EFAULT:
      return "EFAULT ";
    case EILSEQ:
      return "EILSEQ ";
    case ENOPROTOOPT:
      return "ENOPROTOOPT ";
    case ENOTSOCK:
      return "ENOTSOCK ";
    case ENOTTY:
      return "ENOTTY ";
    case EPROTOTYPE:
      return "EPROTOTYPE ";
    case ESPIPE:
      return "ESPIPE ";
    case ETIMEDOUT:
      return "ETIMEDOUT ";
    case ENODEV:
      return "ENODEV ";
    case ENOENT:
      return "ENOENT ";
#ifdef ENOMEDIUM
    case ENOMEDIUM:
      return "ENOMEDIUM ";
#endif
    case ENXIO:
      return "ENXIO ";
    case ESRCH:
      return "ESRCH ";
    case EEXIST:
      return "EEXIST ";
    case EADDRNOTAVAIL:
      return "EADDRNOTAVAIL ";
    case EALREADY:
      return "EALREADY ";
#ifdef ENOTUNIQ
    case ENOTUNIQ:
      return "ENOTUNIQ ";
#endif
    case EPERM:
      return "EPERM ";
    case EACCES:
      return "EACCES ";
#ifdef ENOKEY
    case ENOKEY:
      return "ENOKEY ";
#endif
    case EROFS:
      return "EROFS ";
    case ENOTEMPTY:
      return "ENOTEMPTY ";
    case EISDIR:
      return "EISDIR ";
    case ENOTDIR:
      return "ENOTDIR ";
    case EADDRINUSE:
      return "EADDRINUSE ";
    case EBADF:
      return "EBADF ";
#ifdef EBADFD
    case EBADFD:
      return "EBADFD ";
#endif
    case EBUSY:
      return "EBUSY ";
    case ECHILD:
      return "ECHILD ";
    case EISCONN:
      return "EISCONN ";
#ifdef EISNAM
    case EISNAM:
      return "EISNAM ";
#endif
#ifdef ENOTBLK
    case ENOTBLK:
      return "ENOTBLK ";
#endif
    case ENOTCONN:
      return "ENOTCONN ";
    case EPIPE:
      return "EPIPE ";
#ifdef ESHUTDOWN
    case ESHUTDOWN:
      return "ESHUTDOWN ";
#endif
    case ETXTBSY:
      return "ETXTBSY ";
#ifdef EUNATCH
    case EUNATCH:
      return "EUNATCH ";
#endif
    case ENOSPC:
      return "ENOSPC ";
#ifdef EDQUOT
    case EDQUOT:
      return "EDQUOT ";
#endif
    case EMFILE:
      return "EMFILE ";
    case EMLINK:
      return "EMLINK ";
    case ENFILE:
      return "ENFILE ";
    case ENOBUFS:
      return "ENOBUFS ";
    case ENOMEM:
      return "ENOMEM ";
#ifdef EUSERS
    case EUSERS:
      return "EUSERS ";
#endif
#ifdef ECHRNG
    case ECHRNG:
      return "ECHRNG ";
#endif
    case EFBIG:
      return "EFBIG ";
    case EOVERFLOW:
      return "EOVERFLOW ";
    case ERANGE:
      return "ERANGE ";
#ifdef ENOPKG
    case ENOPKG:
      return "ENOPKG ";
#endif
    case ENOSYS:
      return "ENOSYS ";
    case ENOTSUP:
      return "ENOTSUP ";
    case EAFNOSUPPORT:
      return "EAFNOSUPPORT ";
#ifdef EPFNOSUPPORT
    case EPFNOSUPPORT:
      return "EPFNOSUPPORT ";
#endif
    case EPROTONOSUPPORT:
      return "EPROTONOSUPPORT ";
#ifdef ESOCKTNOSUPPORT
    case ESOCKTNOSUPPORT:
      return "ESOCKTNOSUPPORT ";
#endif
    case EXDEV:
      return "EXDEV ";
    case EAGAIN:
      return "EAGAIN ";
#ifdef ECOMM
    case ECOMM:
      return "ECOMM ";
#endif
    case ECONNREFUSED:
      return "ECONNREFUSED ";
    case ECONNABORTED:
      return "ECONNABORTED ";
    case ECONNRESET:
      return "ECONNRESET ";
    case EINTR:
      return "EINTR ";
#ifdef EHOSTDOWN
    case EHOSTDOWN:
      return "EHOSTDOWN ";
#endif
    case EHOSTUNREACH:
      return "EHOSTUNREACH ";
    case ENETDOWN:
      return "ENETDOWN ";
    case ENETRESET:
      return "ENETRESET ";
    case ENETUNREACH:
      return "ENETUNREACH ";
    case ENOLCK:
      return "ENOLCK ";
    case ENOLINK:
      return "ENOLINK ";
#ifdef ENONET
    case ENONET:
      return "ENONET ";
#endif
    case EDEADLK:
      return "EDEADLK ";
#ifdef ESTALE
    case ESTALE:
      return "ESTALE ";
#endif
    case ECANCELED:
      return "ECANCELED ";
    default:
      return "";
  }
  ABSL_UNREACHABLE();
}

}  // namespace

std::string GetOsErrorMessage(OsErrorCode error) {
  char buf[4096];
  buf[0] = 0;
  return GetStrerrorResult(buf, ::strerror_r(error, buf, std::size(buf)));
}

absl::Status StatusFromOsError(OsErrorCode error_code, std::string_view a,
                               std::string_view b, std::string_view c,
                               std::string_view d, std::string_view e,
                               std::string_view f, SourceLocation loc) {
  absl::Status status(
      absl::ErrnoToStatusCode(error_code),
      tensorstore::StrCat(a, b, c, d, e, f, " [OS error ", error_code, ": ",
                          ErrnoToCString(error_code),
                          GetOsErrorMessage(error_code), "]"));
  MaybeAddSourceLocation(status, loc);
  return status;
}

}  // namespace internal
}  // namespace tensorstore
