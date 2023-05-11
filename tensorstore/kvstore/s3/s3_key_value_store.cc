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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/match.h"

#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/s3/s3_resource.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"


// specializations
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/util/garbage_collection/std_optional.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/std_optional.h"


using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_storage_gcs::IsRetriable;
using ::tensorstore::internal_storage_gcs::RateLimiter;
using ::tensorstore::internal_storage_gcs::RateLimiterNode;
using ::tensorstore::internal_storage_s3::S3ConcurrencyResource;
using ::tensorstore::internal_storage_s3::S3RateLimiterResource;
using ::tensorstore::internal_storage_s3::S3RequestRetries;
using ::tensorstore::internal_storage_s3::S3RequestBuilder;
using ::tensorstore::internal_storage_s3::IsValidBucketName;
using ::tensorstore::internal_storage_s3::IsValidObjectName;
using ::tensorstore::internal_storage_s3::UriEncode;

#ifndef TENSORSTORE_INTERNAL_S3_LOG_REQUESTS
#define TENSORSTORE_INTERNAL_S3_LOG_REQUESTS 1
#endif

#ifndef TENSORSTORE_INTERNAL_S3_LOG_RESPONSES
#define TENSORSTORE_INTERNAL_S3_LOG_RESPONSES 1
#endif


namespace {
static constexpr char kUriScheme[] = "s3";
static constexpr char kDotAmazonAwsDotCom[] = ".amazonaws.com";
}  // namespace

namespace tensorstore {
namespace {
namespace jb = tensorstore::internal_json_binding;

auto& s3_bytes_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/bytes_read",
    "Bytes read by the s3 kvstore driver");

auto& s3_bytes_written = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/bytes_written",
    "Bytes written by the s3 kvstore driver");

auto& s3_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/retries",
    "Count of all retried S3 requests (read/write/delete)");

auto& s3_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/read", "S3 driver kvstore::Read calls");

auto& s3_read_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/s3/read_latency_ms",
        "S3 driver kvstore::Read latency (ms)");

auto& s3_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/write", "S3 driver kvstore::Write calls");

auto& s3_write_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/s3/write_latency_ms",
        "S3 driver kvstore::Write latency (ms)");

auto& s3_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/delete_range",
    "S3 driver kvstore::DeleteRange calls");

auto& s3_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/s3/list", "S3 driver kvstore::List calls");

struct S3KeyValueStoreSpecData {
  std::string bucket;
  bool requester_pays;
  std::string endpoint;
  std::string profile;

  Context::Resource<S3ConcurrencyResource> request_concurrency;
  std::optional<Context::Resource<S3RateLimiterResource>> rate_limiter;
  Context::Resource<S3RequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.bucket, x.request_concurrency, x.rate_limiter,
             x.requester_pays, x.endpoint, x.profile,
             x.retries, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      // Bucket is specified in the `spec` since it identifies the resource
      // being accessed.
      jb::Member("bucket",
                 jb::Projection<&S3KeyValueStoreSpecData::bucket>(jb::Validate(
                     [](const auto& options, const std::string* x) {
                       if (!IsValidBucketName(*x)) {
                         return absl::InvalidArgumentError(tensorstore::StrCat(
                             "Invalid S3 bucket name: ", QuoteString(*x)));
                       }
                       return absl::OkStatus();
                     }))),

      jb::Member("requester_pays",
                 jb::Projection<&S3KeyValueStoreSpecData::requester_pays>(
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                      [](auto* v) { *v = false; })
                  )),
      jb::OptionalMember("endpoint",
                 jb::Projection<&S3KeyValueStoreSpecData::endpoint>()),
      jb::Member("profile",
                 jb::Projection<&S3KeyValueStoreSpecData::profile>(
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                      [](auto* v) { *v = "default"; })
                  )),
      jb::Member(S3ConcurrencyResource::id,
                 jb::Projection<&S3KeyValueStoreSpecData::request_concurrency>()),
      jb::Member(S3RateLimiterResource::id,
                 jb::Projection<&S3KeyValueStoreSpecData::rate_limiter>()),
      jb::Member(S3RequestRetries::id,
                 jb::Projection<&S3KeyValueStoreSpecData::retries>()),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<
                     &S3KeyValueStoreSpecData::data_copy_concurrency>()) /**/
  );
};


std::string GetS3Url(std::string_view bucket, std::string_view path) {
  return tensorstore::StrCat(kUriScheme, "://", bucket, "/", UriEncode(path));
}


class S3KeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<S3KeyValueStoreSpec,
                                                    S3KeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "s3";

  Future<kvstore::DriverPtr> DoOpen() const override;
  Result<std::string> ToUrl(std::string_view path) const override {
    return GetS3Url(data_.bucket, path);
  }
};

class S3KeyValueStore
    : public internal_kvstore::RegisteredDriver<S3KeyValueStore,
                                                S3KeyValueStoreSpec> {
 public:
  const std::string & endpoint() const { return endpoint_; }
  bool is_aws_endpoint() const { return absl::EndsWith(endpoint_, kDotAmazonAwsDotCom); }

  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  const Executor& executor() const {
    return spec_.data_copy_concurrency->executor;
  }

  RateLimiter& read_rate_limiter() {
    if (spec_.rate_limiter.has_value()) {
      return *(spec_.rate_limiter.value()->read_limiter);
    }
    return no_rate_limiter_;
  }
  RateLimiter& write_rate_limiter() {
    if (spec_.rate_limiter.has_value()) {
      return *(spec_.rate_limiter.value()->write_limiter);
    }
    return no_rate_limiter_;
  }

  RateLimiter& admission_queue() { return *spec_.request_concurrency->queue; }


// Apply default backoff/retry logic to the task.
  // Returns whether the task will be retried. On false, max retries have
  // been met or exceeded.  On true, `task->Retry()` will be scheduled to run
  // after a suitable backoff period.
  template <typename Task>
  absl::Status BackoffForAttemptAsync(
      absl::Status status, int attempt, Task* task,
      SourceLocation loc = ::tensorstore::SourceLocation::current()) {
    if (attempt >= spec_.retries->max_retries) {
      return MaybeAnnotateStatus(
          status,
          tensorstore::StrCat("All ", attempt, " retry attempts failed"),
          absl::StatusCode::kAborted, loc);
    }

    // https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff
    s3_retries.Increment();
    auto delay = internal::BackoffForAttempt(
        attempt, spec_.retries->initial_delay, spec_.retries->max_delay,
        /*jitter=*/std::min(absl::Seconds(1), spec_.retries->initial_delay));
        ScheduleAt(absl::Now() + delay,
               WithExecutor(executor(), [task = IntrusivePtr<Task>(task)] {
                 task->Retry();
               }));

    return absl::OkStatus();
  }


  std::shared_ptr<HttpTransport> transport_;
  internal_storage_gcs::NoRateLimiter no_rate_limiter_;
  std::string endpoint_;  // endpoint url
  SpecData spec_;
};


Future<kvstore::DriverPtr> S3KeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<S3KeyValueStore>();
  driver->spec_ = data_;

  if(data_.endpoint.size() == 0) {
    driver->endpoint_ = tensorstore::StrCat("https://", data_.bucket, kDotAmazonAwsDotCom);
  } else {
    auto parsed = internal::ParseGenericUri(data_.endpoint);
    if(parsed.scheme != "http" || parsed.scheme != "https") {
      return absl::InvalidArgumentError(
        tensorstore::StrCat("Endpoint ", data_.endpoint,
                            " has invalid schema ", parsed.scheme,
                            ". Should be http(s)."));
    }
    if(!parsed.query.empty()) {
      return absl::InvalidArgumentError(
        tensorstore::StrCat("Query in endpoint unsupported ", data_.endpoint));
    }
    if(!parsed.fragment.empty()) {
      return absl::InvalidArgumentError(
        tensorstore::StrCat("Fragment in endpoint unsupported ", data_.endpoint));
    }

    driver->endpoint_ = data_.endpoint;
    driver->transport_ = internal_http::GetDefaultHttpTransport();
  }

  // NOTE: Remove temporary logging use of experimental feature.
  if (data_.rate_limiter.has_value()) {
    ABSL_LOG(INFO) << "Using experimental_s3_rate_limiter";
  }
  return driver;
}


Result<kvstore::Spec> ParseS3Url(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == kUriScheme);
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  size_t end_of_bucket = parsed.authority_and_path.find('/');
  std::string_view bucket = parsed.authority_and_path.substr(0, end_of_bucket);
  if (!IsValidBucketName(bucket)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid S3 bucket name: ", QuoteString(bucket)));
  }
  std::string_view path =
      (end_of_bucket == std::string_view::npos)
          ? std::string_view{}
          : parsed.authority_and_path.substr(end_of_bucket + 1);
  auto driver_spec = internal::MakeIntrusivePtr<S3KeyValueStoreSpec>();
  driver_spec->data_.bucket = bucket;
  driver_spec->data_.request_concurrency =
      Context::Resource<S3ConcurrencyResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<S3RequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();

  driver_spec->data_.requester_pays = true;
  driver_spec->data_.profile = "default";
  driver_spec->data_.endpoint = tensorstore::StrCat("https://", bucket, ".s3.amazonaws.com");

  return {std::in_place, std::move(driver_spec), std::string(path)};
}



} // namespace
} // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::S3KeyValueStore)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::S3KeyValueStoreSpec>
    registration;
const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{kUriScheme, tensorstore::ParseS3Url};
}  // namespace

