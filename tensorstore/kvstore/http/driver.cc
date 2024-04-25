// Copyright 2022 The TensorStore Authors
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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/concurrency_resource_provider.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generic_coalescing_batch_util.h"
#include "tensorstore/kvstore/http/byte_range_util.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

/// specializations
#include "tensorstore/internal/cache_key/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep

using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

auto& http_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/http/read", "http driver kvstore::Read calls");

auto& http_batch_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/http/batch_read", "http driver reads after batching");

auto& http_bytes_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/http/bytes_read",
    "Bytes read by the http kvstore driver");

ABSL_CONST_INIT internal_log::VerboseFlag http_logging("http_kvstore");

struct HttpRequestConcurrencyResource : public internal::ConcurrencyResource {
  static constexpr char id[] = "http_request_concurrency";
};

/// Specifies a limit on the number of retries.
struct HttpRequestRetries
    : public internal::RetriesResource<HttpRequestRetries> {
  static constexpr char id[] = "http_request_retries";
};

struct HttpRequestConcurrencyResourceTraits
    : public internal::ConcurrencyResourceTraits,
      public internal::ContextResourceTraits<HttpRequestConcurrencyResource> {
  HttpRequestConcurrencyResourceTraits() : ConcurrencyResourceTraits(32) {}
};
const internal::ContextResourceRegistration<
    HttpRequestConcurrencyResourceTraits>
    http_request_concurrency_registration;

const internal::ContextResourceRegistration<HttpRequestRetries>
    http_request_retries_registration;

/// Returns whether the absl::Status is a retriable request.
bool IsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

absl::Status ValidateParsedHttpUrl(const internal::ParsedGenericUri& parsed) {
  if (parsed.scheme != "http" && parsed.scheme != "https") {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Expected scheme of \"http\" or \"https\" but received: ",
        tensorstore::QuoteString(parsed.scheme)));
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  return absl::OkStatus();
}

void SplitParsedHttpUrl(const internal::ParsedGenericUri& parsed,
                        std::string& base_url, std::string& path) {
  size_t end_of_authority = parsed.authority_and_path.find('/');
  std::string_view authority =
      parsed.authority_and_path.substr(0, end_of_authority);
  std::string_view encoded_path =
      (end_of_authority == std::string_view::npos)
          ? "/"
          : parsed.authority_and_path.substr(end_of_authority);
  base_url = tensorstore::StrCat(parsed.scheme, "://", authority,
                                 parsed.query.empty() ? "" : "?", parsed.query);
  path = internal::PercentDecode(encoded_path);
}

struct HttpKeyValueStoreSpecData {
  std::string base_url;
  Context::Resource<HttpRequestConcurrencyResource> request_concurrency;
  Context::Resource<HttpRequestRetries> retries;
  std::vector<std::string> headers;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.base_url, x.request_concurrency, x.retries, x.headers);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member(
          "base_url",
          jb::Projection<&HttpKeyValueStoreSpecData::base_url>(
              jb::Validate([](const auto& options, const std::string* x) {
                return ValidateParsedHttpUrl(internal::ParseGenericUri(*x));
              }))),
      jb::Member("headers",
                 jb::Projection<&HttpKeyValueStoreSpecData::headers>(
                     jb::DefaultInitializedValue(jb::Array(jb::Validate(
                         [](const auto& options, const std::string* x) {
                           return internal_http::ValidateHttpHeader(*x);
                         }))))),
      jb::Member(
          HttpRequestConcurrencyResource::id,
          jb::Projection<&HttpKeyValueStoreSpecData::request_concurrency>()),
      jb::Member(HttpRequestRetries::id,
                 jb::Projection<&HttpKeyValueStoreSpecData::retries>()));

  std::string GetUrl(std::string_view path) const {
    auto parsed = internal::ParseGenericUri(base_url);
    return tensorstore::StrCat(parsed.scheme, "://", parsed.authority_and_path,
                               absl::StartsWith(path, "/") ? "" : "/",
                               internal::PercentEncodeUriPath(path),
                               parsed.query.empty() ? "" : "?", parsed.query);
  }
};

class HttpKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<HttpKeyValueStoreSpec,
                                                    HttpKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "http";
  Future<kvstore::DriverPtr> DoOpen() const override;
  Result<std::string> ToUrl(std::string_view path) const override {
    return data_.GetUrl(path);
  }
  absl::Status NormalizeSpec(std::string& path) override {
    auto parsed = internal::ParseGenericUri(data_.base_url);
    std::string base_url;
    std::string new_path;
    SplitParsedHttpUrl(parsed, base_url, new_path);
    if (path.empty()) {
      path = std::move(new_path);
    } else if (path[0] != '/') {
      internal::AppendPathComponent(new_path, path);
      path = std::move(new_path);
    } else if (new_path != "/") {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Cannot specify absolute path ", tensorstore::QuoteString(path),
          " in conjunction with base URL ",
          tensorstore::QuoteString(data_.base_url),
          " that includes a path component"));
    }
    data_.base_url = std::move(base_url);
    return absl::OkStatus();
  }
};

/// Implements the KeyValueStore interface for HTTP servers.
class HttpKeyValueStore
    : public internal_kvstore::RegisteredDriver<HttpKeyValueStore,
                                                HttpKeyValueStoreSpec> {
 public:
  internal_kvstore_batch::CoalescingOptions GetBatchReadCoalescingOptions()
      const {
    return internal_kvstore_batch::kDefaultRemoteStorageCoalescingOptions;
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;
  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);

  const Executor& executor() const {
    return spec_.request_concurrency->executor;
  }

  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  std::string DescribeKey(std::string_view key) override {
    return spec_.GetUrl(key);
  }

  HttpKeyValueStoreSpecData spec_;

  std::shared_ptr<HttpTransport> transport_;
};

Future<kvstore::DriverPtr> HttpKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<HttpKeyValueStore>();
  driver->spec_ = data_;
  driver->transport_ = internal_http::GetDefaultHttpTransport();
  return driver;
}

/// A ReadTask is a function object used to satisfy a
/// HttpKeyValueStore::Read request.
struct ReadTask {
  IntrusivePtr<HttpKeyValueStore> owner;
  std::string url;
  kvstore::ReadOptions options;

  HttpResponse httpresponse;

  absl::Status DoRead() {
    HttpRequestBuilder request_builder(
        options.byte_range.size() == 0 ? "HEAD" : "GET", url);
    for (const auto& header : owner->spec_.headers) {
      request_builder.AddHeader(header);
    }
    if (options.byte_range.size() != 0) {
      request_builder.MaybeAddRangeHeader(options.byte_range);
    }

    request_builder
        .MaybeAddStalenessBoundCacheControlHeader(options.staleness_bound)
        .EnableAcceptEncoding();

    if (StorageGeneration::IsCleanValidValue(
            options.generation_conditions.if_equal)) {
      request_builder.AddHeader(absl::StrFormat(
          "if-match: \"%s\"", StorageGeneration::DecodeString(
                                  options.generation_conditions.if_equal)));
    }
    if (StorageGeneration::IsCleanValidValue(
            options.generation_conditions.if_not_equal)) {
      request_builder.AddHeader(
          absl::StrFormat("if-none-match: \"%s\"",
                          StorageGeneration::DecodeString(
                              options.generation_conditions.if_not_equal)));
    }

    auto request = request_builder.BuildRequest();

    ABSL_LOG_IF(INFO, http_logging) << "[http] Read: " << request;

    auto response = owner->transport_->IssueRequest(request, {}).result();
    if (!response.ok()) return response.status();
    httpresponse = std::move(*response);
    http_bytes_read.IncrementBy(httpresponse.payload.size());
    ABSL_LOG_IF(INFO, http_logging.Level(1))
        << "[http] Read response: " << httpresponse;

    switch (httpresponse.status_code) {
      // Special status codes handled outside the retry loop.
      case 412:
      case 404:
      case 304:
        return absl::OkStatus();
    }
    return HttpResponseCodeToStatus(httpresponse);
  }

  Result<kvstore::ReadResult> HandleResult(absl::Time start_time) {
    // Parse `Date` header from response to correctly handle cached responses.
    absl::Time response_date;
    if (auto date_it = httpresponse.headers.find("date");
        date_it != httpresponse.headers.end()) {
      if (!absl::ParseTime(internal_http::kHttpTimeFormat, date_it->second,
                           &response_date, /*err=*/nullptr) ||
          response_date == absl::InfiniteFuture() ||
          response_date == absl::InfinitePast()) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Invalid \"date\" response header: ",
                                tensorstore::QuoteString(date_it->second)));
      }
      if (response_date < start_time) {
        if (options.staleness_bound < start_time &&
            response_date < options.staleness_bound) {
          // `response_date` does not satisfy the `staleness_bound`
          // requirement, possibly due to time skew.  Due to the way we
          // compute `max-age` in the request header, in the case of time skew
          // it is correct to just use `staleness_bound` instead.
          start_time = options.staleness_bound;
        } else {
          start_time = response_date;
        }
      }
    }

    switch (httpresponse.status_code) {
      case 204:
      case 404:
        // Object not found.
        return kvstore::ReadResult::Missing(start_time);
      case 412:
        // "Failed precondition": indicates the If-Match condition did
        // not hold.
        return kvstore::ReadResult::Unspecified(TimestampedStorageGeneration{
            StorageGeneration::Unknown(), start_time});
      case 304:
        // "Not modified": indicates that the If-None-Match condition did
        // not hold.
        return kvstore::ReadResult::Unspecified(TimestampedStorageGeneration{
            options.generation_conditions.if_not_equal, start_time});
    }

    absl::Cord value;
    if (options.byte_range.size() != 0) {
      // Currently unused
      ByteRange byte_range;
      int64_t total_size;

      TENSORSTORE_RETURN_IF_ERROR(internal_http::ValidateResponseByteRange(
          httpresponse, options.byte_range, value, byte_range, total_size));
    }

    // Parse `ETag` header from response.
    StorageGeneration generation = StorageGeneration::Invalid();
    {
      auto it = httpresponse.headers.find("etag");
      if (it != httpresponse.headers.end() && it->second.size() > 2 &&
          it->second.front() == '"' && it->second.back() == '"') {
        // Ignore weak etags.
        std::string_view etag(it->second);
        etag.remove_prefix(1);
        etag.remove_suffix(1);
        generation = StorageGeneration::FromString(etag);
      }
    }

    return kvstore::ReadResult::Value(
        std::move(value),
        TimestampedStorageGeneration{std::move(generation), start_time});
  }

  Result<kvstore::ReadResult> operator()() {
    absl::Time start_time;
    absl::Status status;
    const int max_retries = owner->spec_.retries->max_retries;
    int attempt = 0;
    for (; attempt < max_retries; attempt++) {
      start_time = absl::Now();
      status = DoRead();
      if (status.ok() || !IsRetriable(status)) break;

      auto delay = internal::BackoffForAttempt(
          attempt, owner->spec_.retries->initial_delay,
          owner->spec_.retries->max_delay,
          std::min(absl::Seconds(1), owner->spec_.retries->initial_delay));

      ABSL_LOG_IF(INFO, http_logging)
          << "The operation failed and will be automatically retried in "
          << delay << " seconds (attempt " << attempt + 1 << " out of "
          << max_retries << "), caused by: " << status;

      // NOTE: At some point migrate from a sleep-based retry to an operation
      // queue.
      absl::SleepFor(delay);
    }
    if (!status.ok()) {
      // Return AbortedError, so that it doesn't get retried again somewhere
      // at a higher level.
      if (IsRetriable(status)) {
        return MaybeAnnotateStatus(
            std::move(status),
            absl::StrFormat("All %d retry attempts failed", attempt),
            absl::StatusCode::kAborted);
      }
      return status;
    }

    return HandleResult(start_time);
  }
};

Future<kvstore::ReadResult> HttpKeyValueStore::Read(Key key,
                                                    ReadOptions options) {
  http_read.Increment();
  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> HttpKeyValueStore::ReadImpl(Key&& key,
                                                        ReadOptions&& options) {
  http_batch_read.Increment();
  std::string url = spec_.GetUrl(key);
  return MapFuture(executor(), ReadTask{IntrusivePtr<HttpKeyValueStore>(this),
                                        std::move(url), std::move(options)});
}

Result<kvstore::Spec> ParseHttpUrl(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  TENSORSTORE_RETURN_IF_ERROR(ValidateParsedHttpUrl(parsed));
  std::string path;
  auto driver_spec = internal::MakeIntrusivePtr<HttpKeyValueStoreSpec>();
  SplitParsedHttpUrl(parsed, driver_spec->data_.base_url, path);
  driver_spec->data_.request_concurrency =
      Context::Resource<HttpRequestConcurrencyResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<HttpRequestRetries>::DefaultSpec();
  return {std::in_place, std::move(driver_spec), std::move(path)};
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::HttpKeyValueStore)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::HttpKeyValueStoreSpecData)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::HttpKeyValueStoreSpec>
    registration;
const tensorstore::internal_kvstore::UrlSchemeRegistration
    http_url_scheme_registration{"http", tensorstore::ParseHttpUrl};
const tensorstore::internal_kvstore::UrlSchemeRegistration
    https_url_scheme_registration{"https", tensorstore::ParseHttpUrl};
}  // namespace
