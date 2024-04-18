// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_

#include <stdint.h>

#include <cstddef>

#include "absl/log/absl_check.h"
#include "tensorstore/internal/http/curl_factory.h"
#include "tensorstore/internal/http/curl_wrappers.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal_http {

/// Wrap CurlPtr handles.
class CurlHandle {
 public:
  static CurlHandle Create(CurlHandleFactory& factory);
  static void Cleanup(CurlHandleFactory& factory, CurlHandle h);

  explicit CurlHandle(CurlPtr handle);
  ~CurlHandle();

  // This class holds unique ptrs, disable copying.
  CurlHandle(CurlHandle const&) = delete;
  CurlHandle& operator=(CurlHandle const&) = delete;

  CurlHandle(CurlHandle&&) = default;
  CurlHandle& operator=(CurlHandle&&) = default;

  CURL* get() { return handle_.get(); }

  /// Sets a curl option.
  template <typename T>
  void SetOption(CURLoption option, T&& param,
                 SourceLocation loc = tensorstore::SourceLocation::current()) {
    // All curl_easy_setopt non-ok codes are fatal:
    //   CURLE_BAD_FUNCTION_ARGUMENT
    //   CURLE_OUT_OF_MEMORY
    //   CURLE_UNSUPPORTED_PROTOCOL
    //   CURLE_NOT_BUILT_IN
    auto code = curl_easy_setopt(handle_.get(), option, std::forward<T>(param));
    ABSL_CHECK_EQ(CURLE_OK, code) << loc.file_name() << ":" << loc.line() << " "
                                  << curl_easy_strerror(code);
  }

  void SetOption(CURLoption option, std::nullptr_t,
                 SourceLocation loc = tensorstore::SourceLocation::current()) {
    auto code = curl_easy_setopt(handle_.get(), option, nullptr);
    ABSL_CHECK_EQ(CURLE_OK, code) << loc.file_name() << ":" << loc.line() << " "
                                  << curl_easy_strerror(code);
  }

  /// Gets CURLINFO values.
  template <typename T>
  void GetInfo(CURLINFO info, T* out,
               SourceLocation loc = tensorstore::SourceLocation::current()) {
    // Possible return codes
    //   CURLE_UNKNOWN_OPTION
    //   CURLE_BAD_FUNCTION_ARGUMENT
    auto code = curl_easy_getinfo(handle_.get(), info, out);
    ABSL_CHECK_EQ(CURLE_OK, code) << loc.file_name() << ":" << loc.line() << " "
                                  << curl_easy_strerror(code);
  }

  /// Gets the HTTP response code.  This is not valid until the transfer has
  /// completed successfully.
  int32_t GetResponseCode();

 private:
  CurlPtr handle_;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_
