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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_http/rate_limiter.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/s3/aws_credential_provider.h"
#include "tensorstore/kvstore/s3/s3_metadata.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/s3/s3_resource.h"
#include "tensorstore/kvstore/s3/s3_uri_utils.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/url_registry.h"
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
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_optional.h"  // IWYU pragma: keep

#ifndef TENSORSTORE_INTERNAL_S3_LOG_REQUESTS
#define TENSORSTORE_INTERNAL_S3_LOG_REQUESTS 0
#endif

#ifndef TENSORSTORE_INTERNAL_S3_LOG_RESPONSES
#define TENSORSTORE_INTERNAL_S3_LOG_RESPONSES 0
#endif

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal::SHA256Digester;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_kvstore_gcs_http::RateLimiter;
using ::tensorstore::internal_kvstore_gcs_http::RateLimiterNode;
using ::tensorstore::internal_kvstore_s3::AwsCredentialProvider;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::FindTag;
using ::tensorstore::internal_kvstore_s3::GetAwsCredentialProvider;
using ::tensorstore::internal_kvstore_s3::GetTag;
using ::tensorstore::internal_kvstore_s3::IsValidBucketName;
using ::tensorstore::internal_kvstore_s3::IsValidObjectName;
using ::tensorstore::internal_kvstore_s3::IsValidStorageGeneration;
using ::tensorstore::internal_kvstore_s3::S3ConcurrencyResource;
using ::tensorstore::internal_kvstore_s3::S3RateLimiterResource;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::internal_kvstore_s3::S3RequestRetries;
using ::tensorstore::internal_kvstore_s3::S3UriEncode;
using ::tensorstore::internal_kvstore_s3::S3UriObjectKeyEncode;
using ::tensorstore::internal_kvstore_s3::StorageGenerationFromHeaders;
using ::tensorstore::internal_kvstore_s3::TagAndPosition;
using ::tensorstore::internal_storage_gcs::IsRetriable;
using ::tensorstore::kvstore::Key;
using ::tensorstore::kvstore::ListOptions;

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

/// S3 strings
static constexpr char kUriScheme[] = "s3";
static constexpr char kAmzBucketRegionHeader[] = "x-amz-bucket-region";

/// sha256 hash of an empty string
static constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// An empty etag which should not collide with an actual payload hash
static constexpr char kEmptyEtag[] = "\"\"";

/// Adds the generation header to the provided builder.
bool AddGenerationHeader(S3RequestBuilder* builder, std::string_view header,
                         const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen)) {
    // Unconditional.
    return false;
  }

  // If no generation is provided, we still need to provide an empty etag
  auto etag = StorageGeneration::IsNoValue(gen)
                  ? kEmptyEtag
                  : StorageGeneration::DecodeString(gen);

  builder->AddHeader(absl::StrCat(header, ": ", etag));
  return true;
}

std::string payload_sha256(const absl::Cord& cord = absl::Cord()) {
  SHA256Digester sha256;
  sha256.Write(cord);
  auto digest = sha256.Digest();
  auto digest_sv = std::string_view(reinterpret_cast<const char*>(&digest[0]),
                                    digest.size());

  return absl::BytesToHexString(digest_sv);
}

/// Specifies the AWS profile name.
/// TODO: Allow more complex credential specification, which could be any of:
///  {"profile", "filename"} or { "access_key", "secret_key", "session_token" }
///
struct AwsCredentialsResource
    : public internal::ContextResourceTraits<AwsCredentialsResource> {
  static constexpr char id[] = "aws_credentials";

  struct Spec {
    std::string profile;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.profile);
    };
  };

  struct Resource {
    Spec spec;
    std::shared_ptr<AwsCredentialProvider> credential_provider_;

    Result<std::optional<AwsCredentials>> GetCredentials();
  };

  static Spec Default() { return Spec{}; }

  static constexpr auto JsonBinder() {
    return jb::Object(
        jb::Member("profile", jb::Projection<&Spec::profile>()) /**/
    );
  }

  Result<Resource> Create(
      const Spec& spec,
      internal::ContextResourceCreationContext context) const {
    auto result = GetAwsCredentialProvider(
        spec.profile, internal_http::GetDefaultHttpTransport());
    if (!result.ok() && absl::IsNotFound(result.status())) {
      return Resource{spec, nullptr};
    }
    TENSORSTORE_RETURN_IF_ERROR(result);
    return Resource{spec, std::move(*result)};
  }

  Spec GetSpec(const Resource& resource,
               const internal::ContextSpecBuilder& builder) const {
    return resource.spec;
  }
};

Result<std::optional<AwsCredentials>>
AwsCredentialsResource::Resource::GetCredentials() {
  if (!credential_provider_) return std::nullopt;
  auto credential_result_ = credential_provider_->GetCredentials();
  if (!credential_result_.ok() &&
      absl::IsNotFound(credential_result_.status())) {
    return std::nullopt;
  }
  return credential_result_;
}

const internal::ContextResourceRegistration<AwsCredentialsResource>
    aws_credentials_registration;

struct S3KeyValueStoreSpecData {
  std::string bucket;
  bool requester_pays;
  std::optional<std::string> endpoint;
  std::optional<std::string> host;
  std::string aws_region;

  Context::Resource<AwsCredentialsResource> aws_credentials;
  Context::Resource<S3ConcurrencyResource> request_concurrency;
  std::optional<Context::Resource<S3RateLimiterResource>> rate_limiter;
  Context::Resource<S3RequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.bucket, x.requester_pays, x.endpoint, x.host, x.aws_region,
             x.aws_credentials, x.request_concurrency, x.rate_limiter,
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
                     jb::DefaultValue([](auto* v) { *v = false; }))),
      jb::Member("host", jb::Projection<&S3KeyValueStoreSpecData::host>()),
      jb::Member("endpoint",
                 jb::Projection<&S3KeyValueStoreSpecData::endpoint>()),
      jb::Member("aws_region",
                 jb::Projection<&S3KeyValueStoreSpecData::aws_region>(
                     jb::DefaultValue([](auto* v) { *v = ""; }))),
      jb::Member(AwsCredentialsResource::id,
                 jb::Projection<&S3KeyValueStoreSpecData::aws_credentials>()),
      jb::Member(
          S3ConcurrencyResource::id,
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

struct S3EndpointHostRegion {
  std::string endpoint;
  std::string host_header;
  std::string aws_region;
};

// https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html
// NOTE: If the bucket name contains a '.', then "you can't use
// virtual-host-style addressing over HTTPS, unless you perform your own
// certificate validation".
S3EndpointHostRegion GetS3EndpointHostRegion(std::string_view bucket,
                                             std::string aws_region) {
  std::string host;
  std::string endpoint;
  if (absl::StrContains(bucket, ".")) {
    host = tensorstore::StrCat("s3.", aws_region, ".amazonaws.com");
    endpoint = tensorstore::StrCat("https://", host, "/", bucket);
  } else {
    host = tensorstore::StrCat(bucket, ".s3.", aws_region, ".amazonaws.com");
    endpoint = tensorstore::StrCat("https://", host);
  }
  return S3EndpointHostRegion{std::move(endpoint), std::move(host),
                              std::move(aws_region)};
}

std::string GetS3Url(std::string_view bucket, std::string_view path) {
  return tensorstore::StrCat(kUriScheme, "://", bucket, "/", S3UriEncode(path));
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
  S3KeyValueStore(std::shared_ptr<HttpTransport> transport,
                  S3KeyValueStoreSpecData spec)
      : transport_(std::move(transport)), spec_(std::move(spec)) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  Future<const void> DeleteRange(KeyRange range) override;

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

  Result<std::optional<AwsCredentials>> GetCredentials() {
    return spec_.aws_credentials->GetCredentials();
  }

  // Resolves the region endpoint for the bucket.
  Future<const S3EndpointHostRegion> MaybeResolveRegion();

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

  internal_kvstore_gcs_http::NoRateLimiter no_rate_limiter_;
  std::shared_ptr<HttpTransport> transport_;
  SpecData spec_;

  absl::Mutex mutex_;  // Guards resolve_ehr_ creation.
  Future<const S3EndpointHostRegion> resolve_ehr_;
};

/// A ReadTask is a function object used to satisfy a
/// S3KeyValueStore::Read request.
struct ReadTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ReadTask> {
  IntrusivePtr<S3KeyValueStore> owner;
  std::string encoded_object_name;
  kvstore::ReadOptions options;
  Promise<kvstore::ReadResult> promise;

  std::string read_url_;  // the url to read from
  ReadyFuture<const S3EndpointHostRegion> endpoint_host_region_;

  int attempt_ = 0;
  absl::Time start_time_;

  ReadTask(IntrusivePtr<S3KeyValueStore> owner, std::string encoded_object_name,
           kvstore::ReadOptions options, Promise<kvstore::ReadResult> promise)
      : owner(std::move(owner)),
        encoded_object_name(std::move(encoded_object_name)),
        options(std::move(options)),
        promise(std::move(promise)) {}

  ~ReadTask() { owner->admission_queue().Finish(this); }

  static void Start(void* task) {
    auto* self = reinterpret_cast<ReadTask*>(task);
    self->owner->read_rate_limiter().Finish(self);
    self->owner->admission_queue().Admit(self, &ReadTask::Admit);
  }

  static void Admit(void* task) {
    auto* self = reinterpret_cast<ReadTask*>(task);
    self->owner->executor()(
        [state = IntrusivePtr<ReadTask>(self, internal::adopt_object_ref)] {
          state->Retry();
        });
  }

  void Retry() {
    if (!promise.result_needed()) {
      return;
    }

    AwsCredentials credentials;
    if (auto maybe_credentials = owner->GetCredentials();
        !maybe_credentials.ok()) {
      promise.SetResult(maybe_credentials.status());
      return;
    } else if (maybe_credentials.value().has_value()) {
      credentials = std::move(*maybe_credentials.value());
    }

    auto request_builder = S3RequestBuilder(
        options.byte_range.size() == 0 ? "HEAD" : "GET", read_url_);

    AddGenerationHeader(&request_builder, "if-none-match",
                        options.if_not_equal);
    AddGenerationHeader(&request_builder, "if-match", options.if_equal);

    if (options.byte_range.size() != 0) {
      request_builder.MaybeAddRangeHeader(options.byte_range);
    }

    const auto& ehr = endpoint_host_region_.value();
    start_time_ = absl::Now();
    auto request = request_builder.EnableAcceptEncoding()
                       .MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(ehr.host_header, credentials,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "ReadTask: " << request;
    auto future = owner->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady([self = IntrusivePtr<ReadTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnResponse(response.result());
    });
  }

  void OnResponse(const Result<HttpResponse>& response) {
    if (!promise.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "ReadTask " << *response;

    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) return response.status();
      switch (response.value().status_code) {
        // Special status codes handled outside the retry loop.
        case 412:
        case 404:
        case 304:
          return absl::OkStatus();
      }
      return HttpResponseCodeToStatus(response.value());
    }();

    if (!status.ok() && IsRetriable(status)) {
      status = owner->BackoffForAttemptAsync(status, attempt_++, this);
      if (status.ok()) {
        return;
      }
    }
    if (!status.ok()) {
      promise.SetResult(status);
    } else {
      promise.SetResult(FinishResponse(response.value()));
    }
  }

  Result<kvstore::ReadResult> FinishResponse(const HttpResponse& httpresponse) {
    s3_bytes_read.IncrementBy(httpresponse.payload.size());
    auto latency = absl::Now() - start_time_;
    s3_read_latency_ms.Observe(absl::ToInt64Milliseconds(latency));

    // Parse `Date` header from response to correctly handle cached responses.
    // The GCS servers always send a `date` header.
    kvstore::ReadResult read_result;
    read_result.stamp.time = start_time_;

    switch (httpresponse.status_code) {
      case 204:
      case 404:
        // Object not found.
        read_result.stamp.generation = StorageGeneration::NoValue();
        read_result.state = kvstore::ReadResult::kMissing;
        return read_result;
      case 412:
        // "Failed precondition": indicates the ifGenerationMatch condition
        // did not hold.
        // NOTE: This is returned even when the object does not exist.
        read_result.stamp.generation = StorageGeneration::Unknown();
        return read_result;
      case 304:
        // "Not modified": indicates that the ifGenerationNotMatch condition
        // did not hold.
        read_result.stamp.generation = options.if_not_equal;
        return read_result;
    }

    read_result.state = kvstore::ReadResult::kValue;
    if (options.byte_range.size() != 0) {
      if (httpresponse.status_code != 206) {
        // This may or may not have been a range request; attempt to validate.
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto byte_range,
            options.byte_range.Validate(httpresponse.payload.size()));
        read_result.value =
            internal::GetSubCord(httpresponse.payload, byte_range);
      } else {
        read_result.value = httpresponse.payload;
        // Server should return a parseable content-range header.
        TENSORSTORE_ASSIGN_OR_RETURN(auto content_range_tuple,
                                     ParseContentRangeHeader(httpresponse));

        if (auto request_size = options.byte_range.size();
            (options.byte_range.inclusive_min != -1 &&
             options.byte_range.inclusive_min !=
                 std::get<0>(content_range_tuple)) ||
            (request_size >= 0 && request_size != read_result.value.size())) {
          // Return an error when the response does not start at the requested
          // offset of when the response is smaller than the desired size.
          return absl::OutOfRangeError(
              tensorstore::StrCat("Requested byte range ", options.byte_range,
                                  " was not satisfied by S3 response of size ",
                                  httpresponse.payload.size()));
        }
      }
    }

    TENSORSTORE_ASSIGN_OR_RETURN(
        read_result.stamp.generation,
        StorageGenerationFromHeaders(httpresponse.headers));
    return read_result;
  }
};

Future<kvstore::ReadResult> S3KeyValueStore::Read(Key key,
                                                  ReadOptions options) {
  s3_read.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid S3 object name");
  }
  if (!IsValidStorageGeneration(options.if_equal) ||
      !IsValidStorageGeneration(options.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }

  auto op = PromiseFuturePair<ReadResult>::Make();
  auto state = internal::MakeIntrusivePtr<ReadTask>(
      internal::IntrusivePtr<S3KeyValueStore>(this), S3UriObjectKeyEncode(key),
      std::move(options), std::move(op.promise));
  MaybeResolveRegion().ExecuteWhenReady(
      [state =
           std::move(state)](ReadyFuture<const S3EndpointHostRegion> ready) {
        if (!ready.status().ok()) {
          state->promise.SetResult(ready.status());
          return;
        }
        state->read_url_ = tensorstore::StrCat(ready.value().endpoint, "/",
                                               state->encoded_object_name);
        state->endpoint_host_region_ = std::move(ready);
        intrusive_ptr_increment(state.get());  // adopted by ReadTask::Start.
        state->owner->read_rate_limiter().Admit(state.get(), &ReadTask::Start);
      });
  return std::move(op.future);
}

/// A WriteTask is a function object used to satisfy a
/// S3KeyValueStore::Write request.
struct WriteTask : public RateLimiterNode,
                   public internal::AtomicReferenceCount<WriteTask> {
  IntrusivePtr<S3KeyValueStore> owner;
  std::string encoded_object_name;
  absl::Cord value;
  kvstore::WriteOptions options;
  Promise<TimestampedStorageGeneration> promise;

  std::string upload_url_;
  ReadyFuture<const S3EndpointHostRegion> endpoint_host_region_;

  AwsCredentials credentials_;
  int attempt_ = 0;
  absl::Time start_time_;

  WriteTask(IntrusivePtr<S3KeyValueStore> owner,
            std::string encoded_object_name, absl::Cord value,
            kvstore::WriteOptions options,
            Promise<TimestampedStorageGeneration> promise)
      : owner(std::move(owner)),
        encoded_object_name(std::move(encoded_object_name)),
        value(std::move(value)),
        options(std::move(options)),
        promise(std::move(promise)) {}

  ~WriteTask() { owner->admission_queue().Finish(this); }

  static void Start(void* task) {
    auto* self = reinterpret_cast<WriteTask*>(task);
    self->owner->write_rate_limiter().Finish(self);
    self->owner->admission_queue().Admit(self, &WriteTask::Admit);
  }
  static void Admit(void* task) {
    auto* self = reinterpret_cast<WriteTask*>(task);
    self->owner->executor()(
        [state = IntrusivePtr<WriteTask>(self, internal::adopt_object_ref)] {
          state->Retry();
        });
  }

  /// Writes an object to S3.
  void Retry() {
    if (!promise.result_needed()) {
      return;
    }

    if (auto maybe_credentials = owner->GetCredentials();
        !maybe_credentials.ok()) {
      promise.SetResult(maybe_credentials.status());
      return;
    } else if (maybe_credentials.value().has_value()) {
      credentials_ = std::move(*maybe_credentials.value());
    }

    if (StorageGeneration::IsUnknown(options.if_equal)) {
      DoPut();
      return;
    }

    // S3 doesn't support conditional PUT, so we use a HEAD call
    // to test the if-match condition
    auto builder = S3RequestBuilder("HEAD", upload_url_);
    AddGenerationHeader(&builder, "if-match", options.if_equal);

    auto now = absl::Now();
    const auto& ehr = endpoint_host_region_.value();
    auto request = builder.MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(ehr.host_header, credentials_,
                                     ehr.aws_region, kEmptySha256, now);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "WriteTask (Peek): " << request;

    auto future = owner->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady([self = IntrusivePtr<WriteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnPeekResponse(response.result());
    });
  }

  void OnPeekResponse(const Result<HttpResponse>& response) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "WriteTask (Peek) " << *response;

    if (!response.ok()) {
      promise.SetResult(response.status());
      return;
    }

    TimestampedStorageGeneration r;
    r.time = absl::Now();
    switch (response.value().status_code) {
      case 304:
        // Not modified implies that the generation did not match.
        [[fallthrough]];
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        promise.SetResult(r);
        return;
      case 404:
        if (!StorageGeneration::IsUnknown(options.if_equal) &&
            !StorageGeneration::IsNoValue(options.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          promise.SetResult(r);
          return;
        }
        break;
      default:
        break;
    }

    DoPut();
  }

  void DoPut() {
    // NOTE: This was changed from POST to PUT as a basic POST does not work
    // Some more headers need to be added to allow POST to work:
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-authentication-HTTPPOST.html

    start_time_ = absl::Now();
    auto content_sha256 = payload_sha256(value);

    const auto& ehr = endpoint_host_region_.value();
    auto request =
        S3RequestBuilder("PUT", upload_url_)
            .AddHeader("Content-Type: application/octet-stream")
            .AddHeader(absl::StrCat("Content-Length: ", value.size()))
            .MaybeAddRequesterPayer(owner->spec_.requester_pays)
            .BuildRequest(ehr.host_header, credentials_, ehr.aws_region,
                          content_sha256, start_time_);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "WriteTask: " << request << " size=" << value.size();

    auto future = owner->transport_->IssueRequest(request, value);
    future.ExecuteWhenReady([self = IntrusivePtr<WriteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnResponse(response.result());
    });
  }

  void OnResponse(const Result<HttpResponse>& response) {
    if (!promise.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "WriteTask " << *response;

    absl::Status status = !response.ok()
                              ? response.status()
                              : HttpResponseCodeToStatus(response.value());

    if (!status.ok() && IsRetriable(status)) {
      status = owner->BackoffForAttemptAsync(status, attempt_++, this);
      if (status.ok()) {
        return;
      }
    }
    if (!status.ok()) {
      promise.SetResult(status);
      return;
    }

    promise.SetResult(FinishResponse(response.value()));
  }

  Result<TimestampedStorageGeneration> FinishResponse(
      const HttpResponse& response) {
    TimestampedStorageGeneration r;
    r.time = start_time_;
    switch (response.status_code) {
      case 404:
        if (!StorageGeneration::IsUnknown(options.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          return r;
        }
    }

    auto latency = absl::Now() - start_time_;
    s3_write_latency_ms.Observe(absl::ToInt64Milliseconds(latency));
    s3_bytes_written.IncrementBy(value.size());
    TENSORSTORE_ASSIGN_OR_RETURN(
        r.generation, StorageGenerationFromHeaders(response.headers));
    return r;
  }
};

/// A DeleteTask is a function object used to satisfy a
/// S3KeyValueStore::Delete request.
struct DeleteTask : public RateLimiterNode,
                    public internal::AtomicReferenceCount<DeleteTask> {
  IntrusivePtr<S3KeyValueStore> owner;
  std::string encoded_object_name;
  kvstore::WriteOptions options;
  Promise<TimestampedStorageGeneration> promise;

  std::string delete_url_;
  ReadyFuture<const S3EndpointHostRegion> endpoint_host_region_;

  int attempt_ = 0;
  absl::Time start_time_;
  AwsCredentials credentials_;

  DeleteTask(IntrusivePtr<S3KeyValueStore> owner,
             std::string encoded_object_name, kvstore::WriteOptions options,
             Promise<TimestampedStorageGeneration> promise)
      : owner(std::move(owner)),
        encoded_object_name(std::move(encoded_object_name)),
        options(std::move(options)),
        promise(std::move(promise)) {}

  ~DeleteTask() { owner->admission_queue().Finish(this); }

  static void Start(void* task) {
    auto* self = reinterpret_cast<DeleteTask*>(task);
    self->owner->write_rate_limiter().Finish(self);
    self->owner->admission_queue().Admit(self, &DeleteTask::Admit);
  }

  static void Admit(void* task) {
    auto* self = reinterpret_cast<DeleteTask*>(task);
    self->owner->executor()(
        [state = IntrusivePtr<DeleteTask>(self, internal::adopt_object_ref)] {
          state->Retry();
        });
  }

  /// Removes an object from S3.
  void Retry() {
    if (!promise.result_needed()) {
      return;
    }
    if (!IsValidStorageGeneration(options.if_equal)) {
      promise.SetResult(
          absl::InvalidArgumentError("Malformed StorageGeneration"));
      return;
    }

    if (auto maybe_credentials = owner->GetCredentials();
        !maybe_credentials.ok()) {
      promise.SetResult(maybe_credentials.status());
      return;
    } else if (maybe_credentials.value().has_value()) {
      credentials_ = std::move(*maybe_credentials.value());
    }

    if (StorageGeneration::IsUnknown(options.if_equal)) {
      DoDelete();
      return;
    }

    // S3 doesn't support conditional DELETE,
    // use a HEAD call to test the if-match condition
    auto builder = S3RequestBuilder("HEAD", delete_url_);
    AddGenerationHeader(&builder, "if-match", options.if_equal);

    auto now = absl::Now();
    const auto& ehr = endpoint_host_region_.value();
    auto request = builder.MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(ehr.host_header, credentials_,
                                     ehr.aws_region, kEmptySha256, now);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "DeleteTask (Peek): " << request;

    auto future = owner->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady([self = IntrusivePtr<DeleteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnPeekResponse(response.result());
    });
  }

  void OnPeekResponse(const Result<HttpResponse>& response) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "DeleteTask (Peek) " << *response;

    if (!response.ok()) {
      promise.SetResult(response.status());
      return;
    }

    TimestampedStorageGeneration r;
    r.time = absl::Now();
    switch (response.value().status_code) {
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        promise.SetResult(std::move(r));
        return;
      case 404:
        if (!StorageGeneration::IsUnknown(options.if_equal) &&
            !StorageGeneration::IsNoValue(options.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          promise.SetResult(std::move(r));
          return;
        }
        break;
      default:
        break;
    }

    DoDelete();
  }

  void DoDelete() {
    start_time_ = absl::Now();

    const auto& ehr = endpoint_host_region_.value();
    auto request = S3RequestBuilder("DELETE", delete_url_)
                       .MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(ehr.host_header, credentials_,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "DeleteTask: " << request;

    auto future = owner->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady([self = IntrusivePtr<DeleteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnResponse(response.result());
    });
  }

  void OnResponse(const Result<HttpResponse>& response) {
    if (!promise.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "DeleteTask " << *response;

    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) return response.status();
      switch (response.value().status_code) {
        case 404:
          return absl::OkStatus();
        default:
          break;
      }
      return HttpResponseCodeToStatus(response.value());
    }();

    if (!status.ok() && IsRetriable(status)) {
      status = owner->BackoffForAttemptAsync(status, attempt_++, this);
      if (status.ok()) {
        return;
      }
    }
    if (!status.ok()) {
      promise.SetResult(status);
      return;
    }

    TimestampedStorageGeneration r;
    r.time = start_time_;
    switch (response.value().status_code) {
      case 404:
        // 404 Not Found means aborted when a StorageGeneration was specified.
        if (!StorageGeneration::IsNoValue(options.if_equal) &&
            !StorageGeneration::IsUnknown(options.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          break;
        }
        [[fallthrough]];
      default:
        r.generation = StorageGeneration::NoValue();
        break;
    }
    promise.SetResult(std::move(r));
  }
};

Future<TimestampedStorageGeneration> S3KeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  s3_write.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid S3 object name");
  }
  if (!IsValidStorageGeneration(options.if_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }

  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  if (value) {
    auto state = internal::MakeIntrusivePtr<WriteTask>(
        IntrusivePtr<S3KeyValueStore>(this), S3UriObjectKeyEncode(key),
        std::move(*value), std::move(options), std::move(op.promise));
    MaybeResolveRegion().ExecuteWhenReady(
        [state =
             std::move(state)](ReadyFuture<const S3EndpointHostRegion> ready) {
          if (!ready.status().ok()) {
            state->promise.SetResult(ready.status());
            return;
          }
          state->upload_url_ = tensorstore::StrCat(ready.value().endpoint, "/",
                                                   state->encoded_object_name);
          state->endpoint_host_region_ = std::move(ready);
          intrusive_ptr_increment(state.get());  // adopted by WriteTask::Start.
          state->owner->write_rate_limiter().Admit(state.get(),
                                                   &WriteTask::Start);
        });
  } else {
    auto state = internal::MakeIntrusivePtr<DeleteTask>(
        IntrusivePtr<S3KeyValueStore>(this), S3UriObjectKeyEncode(key),
        std::move(options), std::move(op.promise));

    MaybeResolveRegion().ExecuteWhenReady(
        [state =
             std::move(state)](ReadyFuture<const S3EndpointHostRegion> ready) {
          if (!ready.status().ok()) {
            state->promise.SetResult(ready.status());
            return;
          }
          state->delete_url_ = tensorstore::StrCat(ready.value().endpoint, "/",
                                                   state->encoded_object_name);
          state->endpoint_host_region_ = std::move(ready);

          intrusive_ptr_increment(
              state.get());  // adopted by DeleteTask::Start.
          state->owner->write_rate_limiter().Admit(state.get(),
                                                   &DeleteTask::Start);
        });
  }
  return std::move(op.future);
}

/// ListTask implements the ListImpl execution flow.
struct ListTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<S3KeyValueStore> owner_;
  ListOptions options_;
  AnyFlowReceiver<absl::Status, Key> receiver_;

  std::string resource_;
  ReadyFuture<const S3EndpointHostRegion> endpoint_host_region_;

  std::string continuation_token_;
  absl::Time start_time_;
  int attempt_ = 0;
  bool has_query_parameters_;
  std::atomic<bool> cancelled_{false};

  ListTask(internal::IntrusivePtr<S3KeyValueStore> owner, ListOptions options,
           AnyFlowReceiver<absl::Status, Key> receiver)
      : owner_(std::move(owner)),
        options_(std::move(options)),
        receiver_(std::move(receiver)) {
    execution::set_starting(receiver_, [this] {
      cancelled_.store(true, std::memory_order_relaxed);
    });
  }

  ~ListTask() {
    execution::set_stopping(receiver_);
    owner_->admission_queue().Finish(this);
  }

  inline bool is_cancelled() {
    return cancelled_.load(std::memory_order_relaxed);
  }

  static void Start(void* task) {
    auto* self = reinterpret_cast<ListTask*>(task);
    self->owner_->read_rate_limiter().Finish(self);
    self->owner_->admission_queue().Admit(self, &ListTask::Admit);
  }
  static void Admit(void* task) {
    auto* self = reinterpret_cast<ListTask*>(task);
    self->owner_->executor()(
        [state = IntrusivePtr<ListTask>(self, internal::adopt_object_ref)] {
          state->IssueRequest();
        });
  }

  void Retry() { IssueRequest(); }

  void IssueRequest() {
    if (is_cancelled()) {
      execution::set_done(receiver_);
      return;
    }

    // https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    auto request_builder =
        S3RequestBuilder("GET", resource_).AddQueryParameter("list-type", "2");
    if (auto prefix = LongestPrefix(options_.range); !prefix.empty()) {
      request_builder.AddQueryParameter("prefix", std::string(prefix));
    }
    // NOTE: Consider adding a start-after query parameter, however that
    // would require a predecessor to inclusive_min key.
    if (!continuation_token_.empty()) {
      request_builder.AddQueryParameter("continuation-token",
                                        continuation_token_);
    }

    AwsCredentials credentials;
    if (auto maybe_credentials = owner_->GetCredentials();
        !maybe_credentials.ok()) {
      execution::set_error(receiver_, std::move(maybe_credentials).status());
      return;
    } else if (maybe_credentials.value().has_value()) {
      credentials = std::move(*maybe_credentials.value());
    }

    const auto& ehr = endpoint_host_region_.value();
    start_time_ = absl::Now();

    auto request =
        request_builder.BuildRequest(ehr.host_header, credentials,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_REQUESTS)
        << "List: " << request;

    auto future = owner_->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady(WithExecutor(
        owner_->executor(), [self = IntrusivePtr<ListTask>(this)](
                                ReadyFuture<HttpResponse> response) {
          self->OnResponse(response.result());
        }));
  }

  void OnResponse(const Result<HttpResponse>& response) {
    auto status = OnResponseImpl(response);
    // OkStatus are handled by OnResponseImpl
    if (absl::IsCancelled(status)) {
      execution::set_done(receiver_);
      return;
    }
    if (!status.ok()) {
      execution::set_error(receiver_, std::move(status));
      return;
    }
  }

  absl::Status OnResponseImpl(const Result<HttpResponse>& response) {
    if (is_cancelled()) {
      return absl::CancelledError();
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_RESPONSES && response.ok())
        << "List " << *response;

    absl::Status status =
        response.ok() ? HttpResponseCodeToStatus(*response) : response.status();
    if (!status.ok() && IsRetriable(status)) {
      return owner_->BackoffForAttemptAsync(status, attempt_++, this);
    }

    TagAndPosition tag_and_pos;
    auto cord = response->payload;
    auto payload = cord.Flatten();
    // TODO: Use an xml parser, such as tinyxml2.
    // Then this would could just iterate over the path elements:
    //    /ListBucketResult/KeyCount
    //    /ListBucketResult/NextContinuationToken
    //    /ListBucketResult/Contents/Key
    auto kListBucketOpenTag =
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">";
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto start_pos, FindTag(payload, kListBucketOpenTag, 0, false));
    TENSORSTORE_ASSIGN_OR_RETURN(
        tag_and_pos, GetTag(payload, "<KeyCount>", "</KeyCount>", start_pos));
    size_t keycount = 0;
    if (!absl::SimpleAtoi(tag_and_pos.tag, &keycount)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Malformed KeyCount ", tag_and_pos.tag));
    }

    for (size_t k = 0; k < keycount; ++k) {
      if (is_cancelled()) {
        return absl::CancelledError();
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto contents_pos,
          FindTag(payload, "<Contents>", tag_and_pos.pos, false));
      TENSORSTORE_ASSIGN_OR_RETURN(
          tag_and_pos, GetTag(payload, "<Key>", "</Key>", contents_pos));

      const auto& key = tag_and_pos.tag;
      if (key < options_.range.inclusive_min) continue;
      if (KeyRange::CompareKeyAndExclusiveMax(
              key, options_.range.exclusive_max) >= 0) {
        // Objects are returned sorted in ascending order of the respective
        // key names, so after the current key exceeds exclusive max no
        // additional requests need to be made.
        execution::set_done(receiver_);
        return absl::OkStatus();
      }
      if (key.size() >= options_.strip_prefix_length) {
        execution::set_value(receiver_,
                             key.substr(options_.strip_prefix_length));
      }
    }

    // Successful request, so clear the retry_attempt for the next request.
    attempt_ = 0;
    TENSORSTORE_ASSIGN_OR_RETURN(
        tag_and_pos,
        GetTag(payload, "<IsTruncated>", "</IsTruncated>", start_pos));

    if (tag_and_pos.tag == "true") {
      TENSORSTORE_ASSIGN_OR_RETURN(
          tag_and_pos, GetTag(payload, "<NextContinuationToken>",
                              "</NextContinuationToken>", start_pos));
      continuation_token_ = tag_and_pos.tag;
      IssueRequest();
    } else {
      execution::set_done(receiver_);
    }
    return absl::OkStatus();
  }
};

void S3KeyValueStore::ListImpl(ListOptions options,
                               AnyFlowReceiver<absl::Status, Key> receiver) {
  s3_list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }

  auto state = internal::MakeIntrusivePtr<ListTask>(
      IntrusivePtr<S3KeyValueStore>(this), std::move(options),
      std::move(receiver));

  MaybeResolveRegion().ExecuteWhenReady(
      [state =
           std::move(state)](ReadyFuture<const S3EndpointHostRegion> ready) {
        if (!ready.status().ok()) {
          execution::set_error(state->receiver_, ready.status());
          return;
        }
        state->resource_ = tensorstore::StrCat(ready.value().endpoint, "/");
        state->endpoint_host_region_ = std::move(ready);
        intrusive_ptr_increment(state.get());
        state->owner_->read_rate_limiter().Admit(state.get(), &ListTask::Start);
      });
}

// Receiver used by `DeleteRange` for processing the results from `List`.
struct DeleteRangeListReceiver {
  IntrusivePtr<S3KeyValueStore> owner_;
  Promise<void> promise_;
  FutureCallbackRegistration cancel_registration_;

  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration_ = promise_.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_value(std::string key) {
    assert(!key.empty());
    if (!key.empty()) {
      LinkError(promise_, owner_->Delete(std::move(key)));
    }
  }

  void set_error(absl::Status error) {
    SetDeferredResult(promise_, std::move(error));
    promise_ = Promise<void>();
  }

  void set_done() { promise_ = Promise<void>(); }

  void set_stopping() { cancel_registration_.Unregister(); }
};

Future<const void> S3KeyValueStore::DeleteRange(KeyRange range) {
  s3_delete_range.Increment();
  if (range.empty()) return absl::OkStatus();

  // TODO(jbms): It could make sense to rate limit the list operation, so that
  // we don't get way ahead of the delete operations.  Currently our
  // sender/receiver abstraction does not support back pressure, though.
  auto op = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
  ListOptions list_options;
  list_options.range = std::move(range);
  ListImpl(list_options, DeleteRangeListReceiver{
                             internal::IntrusivePtr<S3KeyValueStore>(this),
                             std::move(op.promise)});
  return std::move(op.future);
}

// Resolves the region endpoint for the bucket.
Future<const S3EndpointHostRegion> S3KeyValueStore::MaybeResolveRegion() {
  absl::MutexLock l(&mutex_);
  if (!resolve_ehr_.null()) return resolve_ehr_;

  // Make global request to get bucket region from response headers,
  // then create region specific endpoint
  auto url = tensorstore::StrCat("https://", spec_.bucket, ".s3.amazonaws.com");
  auto request = internal_http::HttpRequestBuilder("HEAD", url).BuildRequest();
  auto op = PromiseFuturePair<S3EndpointHostRegion>::Link(
      WithExecutor(
          executor(),
          [self = internal::IntrusivePtr<S3KeyValueStore>{this}](
              Promise<S3EndpointHostRegion> promise,
              ReadyFuture<HttpResponse> ready) {
            if (!promise.result_needed()) return;
            auto& headers = ready.value().headers;
            if (auto it = headers.find(kAmzBucketRegionHeader);
                it != headers.end()) {
              const auto& aws_region = it->second;
              // TODO: propagate resolved region back into spec.
              auto host_endpoint =
                  GetS3EndpointHostRegion(self->spec_.bucket, aws_region);
              ABSL_LOG_IF(INFO, (TENSORSTORE_INTERNAL_S3_LOG_REQUESTS ||
                                 TENSORSTORE_INTERNAL_S3_LOG_RESPONSES))
                  << "S3 driver using endpoint [" << host_endpoint.endpoint
                  << "]";
              promise.SetResult(std::move(host_endpoint));
              return;
            }
            promise.SetResult(absl::FailedPreconditionError(tensorstore::StrCat(
                "bucket ", self->spec_.bucket, " does not exist")));
          }),
      transport_->IssueRequest(request, {}));

  resolve_ehr_ = std::move(op.future);
  return resolve_ehr_;
}

Future<kvstore::DriverPtr> S3KeyValueStoreSpec::DoOpen() const {
  // TODO: The transport should support the AWS_CA_BUNDLE environment variable.
  auto driver = internal::MakeIntrusivePtr<S3KeyValueStore>(
      internal_http::GetDefaultHttpTransport(), data_);

  // NOTE: Remove temporary logging use of experimental feature.
  if (data_.rate_limiter.has_value()) {
    ABSL_LOG(INFO) << "Using experimental_s3_rate_limiter";
  }

  if (data_.endpoint.has_value()) {
    auto endpoint = data_.endpoint.value();
    auto parsed = internal::ParseGenericUri(endpoint);
    if (parsed.scheme != "http" && parsed.scheme != "https") {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Endpoint ", endpoint, " has invalid scheme ",
                              parsed.scheme, ". Should be http(s)."));
    }
    if (!parsed.query.empty()) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Query in endpoint unsupported ", endpoint));
    }
    if (!parsed.fragment.empty()) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Fragment in endpoint unsupported ", endpoint));
    }

    std::string host_header;
    if (data_.host.has_value()) {
      host_header = data_.host.value();
    } else {
      auto parsed = internal::ParseGenericUri(endpoint);
      size_t end_of_host = parsed.authority_and_path.find('/');
      host_header = parsed.authority_and_path.substr(0, end_of_host);
    }

    driver->resolve_ehr_ =
        MakeReadyFuture<S3EndpointHostRegion>(S3EndpointHostRegion{
            endpoint, std::move(host_header), data_.aws_region});

  } else if (!data_.aws_region.empty() ||
             internal_kvstore_s3::ClassifyBucketName(data_.bucket) ==
                 internal_kvstore_s3::BucketNameType::kOldUSEast1) {
    if (!data_.aws_region.empty() && data_.aws_region != "us-east-1") {
      // This is an old-style bucket name, so the region must be us-east-1
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Bucket ", QuoteString(data_.bucket),
                              " requires aws_region \"us-east-1\", not ",
                              QuoteString(data_.aws_region)));
    }

    driver->resolve_ehr_ = MakeReadyFuture<S3EndpointHostRegion>(
        GetS3EndpointHostRegion(data_.bucket, data_.aws_region));
  } else if (absl::StrContains(data_.bucket, ".")) {
    // TODO: Rework how 'x-amz-bucket-region' is handled. The technique of using
    // a HEAD request on bucket.s3.amazonaws.com to acquire the aws_region does
    // not work when there is a . in the bucket name; for now require the
    // aws_region to be set.
    //
    // The aws cli issues a request against the aws-global endpoint,
    // using host:s3.amazonaws.com, with a string-to-sign using "us-east-1"
    // zone. The response will be a 301 request with an 'x-amz-bucket-region'
    // header. We might be able to just do a signed HEAD request against an
    // possibly non-existent file... But try this later.
    return absl::InvalidArgumentError(
        tensorstore::StrCat("bucket ", QuoteString(data_.bucket),
                            " requires aws_region to be set."));
  }

  ABSL_LOG_IF(INFO, (TENSORSTORE_INTERNAL_S3_LOG_REQUESTS ||
                     TENSORSTORE_INTERNAL_S3_LOG_RESPONSES) &&
                        !driver->resolve_ehr_.null())
      << "S3 driver using endpoint [" << driver->resolve_ehr_.value().endpoint
      << "]";

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
  std::string path = internal::PercentDecode(
      (end_of_bucket == std::string_view::npos)
          ? std::string_view{}
          : parsed.authority_and_path.substr(end_of_bucket + 1));
  auto driver_spec = internal::MakeIntrusivePtr<S3KeyValueStoreSpec>();
  driver_spec->data_.bucket = bucket;
  driver_spec->data_.requester_pays = false;

  driver_spec->data_.aws_credentials =
      Context::Resource<AwsCredentialsResource>::DefaultSpec();
  driver_spec->data_.request_concurrency =
      Context::Resource<S3ConcurrencyResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<S3RequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();

  return {std::in_place, std::move(driver_spec), std::move(path)};
}

const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::S3KeyValueStoreSpec>
    registration;
const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{kUriScheme, tensorstore::ParseS3Url};

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::S3KeyValueStore)
