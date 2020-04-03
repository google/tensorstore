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

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "absl/strings/str_cat.h"
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

template <typename Func>
std::size_t CurlCallbackImpl(void* contents, std::size_t size,
                             std::size_t nmemb, void* userdata) {
  return (*static_cast<Func*>(userdata))(
      absl::string_view(static_cast<char const*>(contents), size * nmemb));
}

/// RAII write callback setter for CURL handles.
///
/// \tparam Func Callable with signature `std::size_t (absl::string_view)`.  The
///     return value is the number of bytes that were consumed.
template <typename Func>
class CurlWriteCallback {
 public:
  CurlWriteCallback(CURL* handle, Func func)
      : handle_(handle), func_(std::move(func)) {
    CurlEasySetopt(handle_, CURLOPT_WRITEDATA, &func_);
    CurlEasySetopt(handle_, CURLOPT_WRITEFUNCTION, &CurlCallbackImpl<Func>);
  }

  ~CurlWriteCallback() {
    CurlEasySetopt(handle_, CURLOPT_WRITEDATA, static_cast<void*>(nullptr));
    CurlEasySetopt(handle_, CURLOPT_WRITEFUNCTION,
                   static_cast<void (*)()>(nullptr));
  }

 private:
  CURL* handle_;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Func func_;
};

template <typename Func>
CurlWriteCallback(CURL* handle, Func func)->CurlWriteCallback<Func>;

/// RAII header callback setter for CURL handles.
///
/// \tparam Func Callable with signature `std::size_t (absl::string_view)`.  The
///     return value is the number of bytes from the string that were consumed.
template <typename Func>
class CurlHeaderCallback {
 public:
  CurlHeaderCallback(CURL* handle, Func func)
      : handle_(handle), func_(std::move(func)) {
    CurlEasySetopt(handle_, CURLOPT_HEADERDATA, &func_);
    CurlEasySetopt(handle_, CURLOPT_HEADERFUNCTION, &CurlCallbackImpl<Func>);
  }

  ~CurlHeaderCallback() {
    CurlEasySetopt(handle_, CURLOPT_HEADERDATA, static_cast<void*>(nullptr));
    CurlEasySetopt(handle_, CURLOPT_HEADERFUNCTION,
                   static_cast<void (*)()>(nullptr));
  }

 private:
  CURL* handle_;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Func func_;
};

template <typename Func>
CurlHeaderCallback(CURL* handle, Func func)->CurlHeaderCallback<Func>;

int32_t CurlGetResponseCode(CURL* handle);
Status CurlEasyPerform(CURL* handle);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_HANDLE_H_
