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

#ifndef TENSORSTORE_INTERNAL_HTTP_CURL_WRAPPERS_H_
#define TENSORSTORE_INTERNAL_HTTP_CURL_WRAPPERS_H_

#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include <curl/curl.h>  // IWYU pragma: export
#include "tensorstore/internal/source_location.h"

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

/// Returns the default GetCurlUserAgentSuffix.
std::string GetCurlUserAgentSuffix();

/// Returns a absl::Status object for a corresponding CURLcode.
absl::Status CurlCodeToStatus(
    CURLcode code, std::string_view detail,
    SourceLocation loc = tensorstore::SourceLocation::current());

/// Returns a absl::Status object for a corresponding CURLcode.
absl::Status CurlMCodeToStatus(
    CURLMcode code, std::string_view,
    SourceLocation loc = tensorstore::SourceLocation::current());

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_CURL_WRAPPERS_H_
