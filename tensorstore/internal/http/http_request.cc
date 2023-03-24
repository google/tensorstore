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
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

HttpRequestBuilder::HttpRequestBuilder(std::string_view method,
                                       std::string base_url)
    : query_parameter_separator_("?") {
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
      query_parameter_separator_, internal::PercentEncodeUriComponent(key), "=",
      internal::PercentEncodeUriComponent(value));
  query_parameter_separator_ = "&";
  request_.url_.append(parameter);
  return *this;
}

HttpRequestBuilder& HttpRequestBuilder::EnableAcceptEncoding() {
  request_.accept_encoding_ = true;
  return *this;
}

bool AddRangeHeader(HttpRequestBuilder& request_builder,
                    OptionalByteRangeRequest byte_range) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Range
  if (byte_range.exclusive_max) {
    assert(byte_range.SatisfiesInvariants());
    request_builder.AddHeader(absl::StrFormat("Range: bytes=%d-%d",
                                              byte_range.inclusive_min,
                                              *byte_range.exclusive_max - 1));
    return true;
  }
  if (byte_range.inclusive_min > 0) {
    request_builder.AddHeader(
        absl::StrFormat("Range: bytes=%d-", byte_range.inclusive_min));
    return true;
  }
  return false;
}

bool AddCacheControlMaxAgeHeader(HttpRequestBuilder& request_builder,
                                 absl::Duration max_age) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control
  if (max_age >= absl::InfiniteDuration()) {
    return false;
  }

  // Since max-age is specified as an integer number of seconds, always
  // round down to ensure our requirement is met.
  auto max_age_seconds = absl::ToInt64Seconds(max_age);
  if (max_age_seconds > 0) {
    request_builder.AddHeader(
        absl::StrFormat("cache-control: max-age=%d", max_age_seconds));
  } else {
    request_builder.AddHeader("cache-control: no-cache");
  }
  return true;
}

bool AddStalenessBoundCacheControlHeader(HttpRequestBuilder& request_builder,
                                         absl::Time staleness_bound) {
  // `absl::InfiniteFuture()`  indicates that the result must be current.
  // `absl::InfinitePast()`  disables the cache-control header.
  if (staleness_bound == absl::InfinitePast()) {
    return false;
  }
  absl::Time now;
  absl::Duration duration = absl::ZeroDuration();
  if (staleness_bound != absl::InfiniteFuture() &&
      (now = absl::Now()) > staleness_bound) {
    // the max age is in the past.
    duration = now - staleness_bound;
  }
  return AddCacheControlMaxAgeHeader(request_builder, duration);
}

}  // namespace internal_http
}  // namespace tensorstore
