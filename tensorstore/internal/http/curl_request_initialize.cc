#include "tensorstore/internal/http/curl_handle.h"

namespace tensorstore {
namespace internal_http {

void InitializeCurlHandle(CURL* handle) {
  // Default implementation does nothing.  A different definition can be
  // substituted at build time if needed.
}

}  // namespace internal_http
}  // namespace tensorstore
