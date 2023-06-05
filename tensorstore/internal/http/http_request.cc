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

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

HttpRequestBuilder::HttpRequestBuilder(std::string_view method,
                                       std::string base_url,
                                       absl::FunctionRef<std::string(std::string_view)> uri_encoder)
    : query_parameter_separator_("?"), uri_encoder_(uri_encoder) {
  request_.method_ = method;
  request_.url_ = std::move(base_url);
}

HttpRequest HttpRequestBuilder::BuildRequest() { return std::move(request_); }

HttpRequestBuilder& HttpRequestBuilder::AddUserAgentPrefix(
    std::string_view prefix) {
  request_.user_agent_ = tensorstore::StrCat(prefix, request_.user_agent_);
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddHeader(std::string header) {
  request_.headers_.emplace_back(std::move(header));
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddQueryParameter(
    std::string_view key, std::string_view value) {
  std::string parameter = tensorstore::StrCat(
      query_parameter_separator_, uri_encoder_(key), "=", uri_encoder_(value));
  query_parameter_separator_ = "&";
  request_.url_.append(parameter);
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::EnableAcceptEncoding() {
  request_.accept_encoding_ = true;
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddRangeHeader(
        OptionalByteRangeRequest byte_range, bool & result) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Range
  if (byte_range.exclusive_max) {
    assert(byte_range.SatisfiesInvariants());

    AddHeader(absl::StrFormat("Range: bytes=%d-%d",
                              byte_range.inclusive_min,
                              *byte_range.exclusive_max - 1));
    result = true;
    return *this;
  }
  if (byte_range.inclusive_min > 0) {
    AddHeader(absl::StrFormat("Range: bytes=%d-", byte_range.inclusive_min));
    result = true;
    return *this;
  }
  result = false;
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddCacheControlMaxAgeHeader(absl::Duration max_age, bool & result) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control
  if (max_age >= absl::InfiniteDuration()) {
    result = false;
    return *this;
  }

  // Since max-age is specified as an integer number of seconds, always
  // round down to ensure our requirement is met.
  auto max_age_seconds = absl::ToInt64Seconds(max_age);
  if (max_age_seconds > 0) {
    AddHeader(absl::StrFormat("cache-control: max-age=%d", max_age_seconds));
  } else {
    AddHeader("cache-control: no-cache");
  }
  result = true;
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::AddStalenessBoundCacheControlHeader(absl::Time staleness_bound, bool & result) {
  // `absl::InfiniteFuture()`  indicates that the result must be current.
  // `absl::InfinitePast()`  disables the cache-control header.
  if (staleness_bound == absl::InfinitePast()) {
    result = false;
    return *this;
  }
  absl::Time now;
  absl::Duration duration = absl::ZeroDuration();
  if (staleness_bound != absl::InfiniteFuture() &&
      (now = absl::Now()) > staleness_bound) {
    // the max age is in the past.
    duration = now - staleness_bound;
  }
  AddCacheControlMaxAgeHeader(duration, result);
  return *this;
}

}  // namespace internal_http
}  // namespace tensorstore
