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

#include "tensorstore/internal/http/http_request.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/http/curl_handle.h"  // for CurlEscapeString
#include "tensorstore/internal/logging.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

HttpRequestBuilder::HttpRequestBuilder(std::string base_url)
    : query_parameter_separator_("?") {
  request_.url_ = std::move(base_url);
}

HttpRequest HttpRequestBuilder::BuildRequest() { return std::move(request_); }

HttpRequestBuilder& HttpRequestBuilder::AddUserAgentPrefix(
    absl::string_view prefix) {
  request_.user_agent_ = absl::StrCat(prefix, request_.user_agent_);
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddHeader(std::string header) {
  request_.headers_.emplace_back(std::move(header));
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddQueryParameter(
    absl::string_view key, absl::string_view value) {
  std::string parameter =
      absl::StrCat(query_parameter_separator_, CurlEscapeString(key), "=",
                   CurlEscapeString(value));
  query_parameter_separator_ = "&";
  request_.url_.append(parameter);
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::SetMethod(absl::string_view method) {
  request_.method_.assign(method.data(), method.size());
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::EnableAcceptEncoding() {
  request_.accept_encoding_ = true;
  return *this;
}

std::string GetRangeHeader(OptionalByteRangeRequest byte_range) {
  if (byte_range.exclusive_max) {
    return StrCat("Range: bytes=", byte_range.inclusive_min, "-",
                  *byte_range.exclusive_max - 1);
  } else {
    return StrCat("Range: bytes=", byte_range.inclusive_min, "-");
  }
}

}  // namespace internal_http
}  // namespace tensorstore
