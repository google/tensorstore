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

#include "tensorstore/internal/http/curl_request_builder.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorstore/internal/logging.h"

namespace tensorstore {
namespace internal_http {

CurlRequestBuilder::CurlRequestBuilder(
    std::string base_url, std::shared_ptr<CurlHandleFactory> factory)
    : query_parameter_separator_("?") {
  request_.url_ = std::move(base_url);
  request_.factory_ = std::move(factory);
}

CurlRequest CurlRequestBuilder::BuildRequest() {
  assert(request_.factory_);
  request_.user_agent_ += GetCurlUserAgentSuffix();
  return std::move(request_);
}

CurlRequestBuilder& CurlRequestBuilder::AddUserAgentPrefix(
    absl::string_view prefix) {
  assert(request_.factory_);
  request_.user_agent_ = absl::StrCat(prefix, request_.user_agent_);
  return *this;
}

CurlRequestBuilder& CurlRequestBuilder::AddHeader(const std::string& header) {
  assert(request_.factory_);
  auto new_header = curl_slist_append(request_.headers_.get(), header.c_str());
  (void)request_.headers_.release();
  request_.headers_.reset(new_header);
  return *this;
}

CurlRequestBuilder& CurlRequestBuilder::AddQueryParameter(
    absl::string_view key, absl::string_view value) {
  assert(request_.factory_);
  std::string parameter =
      absl::StrCat(query_parameter_separator_, CurlEscapeString(key), "=",
                   CurlEscapeString(value), "&");
  query_parameter_separator_ = "&";
  request_.url_.append(parameter);
  return *this;
}

CurlRequestBuilder& CurlRequestBuilder::SetMethod(absl::string_view method) {
  assert(request_.factory_);
  request_.method_.assign(method.data(), method.size());
  return *this;
}

CurlRequestBuilder& CurlRequestBuilder::EnableAcceptEncoding() {
  assert(request_.factory_);
  request_.accept_encoding_ = true;
  return *this;
}

std::string GetRangeHeader(OptionalByteRangeRequest byte_range) {
  if (byte_range.exclusive_max) {
    return StrCat("Range: ", byte_range.inclusive_min, "-",
                  *byte_range.exclusive_max - 1);
  } else {
    return StrCat("Range: ", byte_range.inclusive_min, "-");
  }
}

}  // namespace internal_http
}  // namespace tensorstore
