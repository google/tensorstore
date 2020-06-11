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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include <curl/curl.h>  // IWYU pragma: export
#include "tensorstore/internal/attributes.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_http {

/// Cleanup types for libcurl pointers.
struct CurlPtrCleanup {
  void operator()(CURL*);
};
struct CurlMultiCleanup {
  void operator()(CURLM*);
};
struct CurlSlistCleanup {
  void operator()(curl_slist*);
};

/// CurlPtr holds a CURL* handle and automatically clean it up.
using CurlPtr = std::unique_ptr<CURL, CurlPtrCleanup>;

/// CurlMulti holds a CURLM* handle and automatically clean it up.
using CurlMulti = std::unique_ptr<CURLM, CurlMultiCleanup>;

/// CurlHeaders holds a singly-linked list of headers.
using CurlHeaders = std::unique_ptr<curl_slist, CurlSlistCleanup>;

/// CurlHandleFactory creates and cleans up CURL* (CurlPtr) handles
/// and CURLM* (CurlMulti) handles.
///
/// NOTE: These methods are virtual so that a curl factory can re-use
/// curl handles.
class CurlHandleFactory {
 public:
  virtual ~CurlHandleFactory() = default;

  virtual CurlPtr CreateHandle() = 0;
  virtual void CleanupHandle(CurlPtr&&) = 0;

  virtual CurlMulti CreateMultiHandle() = 0;
  virtual void CleanupMultiHandle(CurlMulti&&) = 0;
};

/// Returns the default CurlHandleFactory.
std::shared_ptr<CurlHandleFactory> GetDefaultCurlHandleFactory();

/// Returns the default GetCurlUserAgentSuffix.
std::string GetCurlUserAgentSuffix();

/// Returns a Status object for a corresponding CURLcode.
Status CurlCodeToStatus(CURLcode code, absl::string_view);

/// Returns a Status object for a corresponding CURLcode.
Status CurlMCodeToStatus(CURLMcode code, absl::string_view);

/// URL-escapes a string.
std::string CurlEscapeString(absl::string_view s);

/// URL-unescapes a string.
std::string CurlUnescapeString(absl::string_view s);

template <typename T>
inline void CurlEasySetopt(CURL* handle, CURLoption option, T value) {
  auto e = curl_easy_setopt(handle, option, value);
  TENSORSTORE_CHECK(e == CURLE_OK);
}

template <typename T>
inline void CurlMultiSetopt(CURLM* handle, CURLMoption option, T value) {
  auto e = curl_multi_setopt(handle, option, value);
  TENSORSTORE_CHECK(e == CURLM_OK);
}

/// Returns the HTTP response code from a curl handle.
int32_t CurlGetResponseCode(CURL* handle);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_
