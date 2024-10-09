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

#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generic_coalescing_batch_util.h"
#include "tensorstore/kvstore/http/byte_range_util.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/s3/aws_credentials_resource.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/kvstore/s3/s3_endpoint.h"
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
#include "tinyxml2.h"

// specializations
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_optional.h"  // IWYU pragma: keep

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::RateLimiter;
using ::tensorstore::internal::RateLimiterNode;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal::SHA256Digester;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::AwsCredentialsResource;
using ::tensorstore::internal_kvstore_s3::AwsHttpResponseToStatus;
using ::tensorstore::internal_kvstore_s3::GetNodeInt;
using ::tensorstore::internal_kvstore_s3::GetNodeText;
using ::tensorstore::internal_kvstore_s3::IsValidBucketName;
using ::tensorstore::internal_kvstore_s3::IsValidObjectName;
using ::tensorstore::internal_kvstore_s3::IsValidStorageGeneration;
using ::tensorstore::internal_kvstore_s3::S3ConcurrencyResource;
using ::tensorstore::internal_kvstore_s3::S3EndpointRegion;
using ::tensorstore::internal_kvstore_s3::S3RateLimiterResource;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::internal_kvstore_s3::S3RequestRetries;
using ::tensorstore::internal_kvstore_s3::S3UriEncode;
using ::tensorstore::internal_kvstore_s3::StorageGenerationFromHeaders;
using ::tensorstore::kvstore::Key;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListOptions;
using ::tensorstore::kvstore::ListReceiver;

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

struct S3Metrics : public internal_kvstore::CommonMetrics {
  internal_metrics::Counter<int64_t>& retries;
  // no additional members
};

auto s3_metrics = []() -> S3Metrics {
  return {
      TENSORSTORE_KVSTORE_COMMON_METRICS(s3),
      TENSORSTORE_KVSTORE_COUNTER_IMPL(
          s3, retries, "count of all retried requests (read/write/delete)")};
}();

ABSL_CONST_INIT internal_log::VerboseFlag s3_logging("s3");

/// S3 strings
static constexpr char kUriScheme[] = "s3";

/// sha256 hash of an empty string
static constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// An empty etag which should not collide with an actual payload hash
static constexpr char kEmptyEtag[] = "\"\"";

static constexpr size_t kMaxS3PutSize = size_t{5} * 1024 * 1024 * 1024;  // 5GB

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

bool DefaultIsRetryableCode(absl::StatusCode code) {
  return code == absl::StatusCode::kDeadlineExceeded ||
         code == absl::StatusCode::kUnavailable;
}

struct S3KeyValueStoreSpecData {
  std::string bucket;
  bool requester_pays;
  std::optional<std::string> endpoint;
  std::optional<std::string> host_header;
  std::string aws_region;

  Context::Resource<AwsCredentialsResource> aws_credentials;
  Context::Resource<S3ConcurrencyResource> request_concurrency;
  std::optional<Context::Resource<S3RateLimiterResource>> rate_limiter;
  Context::Resource<S3RequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.bucket, x.requester_pays, x.endpoint, x.host_header,
             x.aws_region, x.aws_credentials, x.request_concurrency,
             x.rate_limiter, x.retries, x.data_copy_concurrency);
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
      jb::Member("host_header",
                 jb::Projection<&S3KeyValueStoreSpecData::host_header>()),
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
      : transport_(std::move(transport)),
        spec_(std::move(spec)),
        host_header_(spec_.host_header.value_or(std::string())) {}

  internal_kvstore_batch::CoalescingOptions GetBatchReadCoalescingOptions()
      const {
    internal_kvstore_batch::CoalescingOptions options;
    options.max_extra_read_bytes = 4095;
    options.target_coalesced_size = 128 * 1024 * 1024;
    return options;
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

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
  Future<const S3EndpointRegion> MaybeResolveRegion();

  // Apply default backoff/retry logic to the task.
  // Returns whether the task will be retried. On false, max retries have
  // been met or exceeded.  On true, `task->Retry()` will be scheduled to run
  // after a suitable backoff period.
  template <typename Task>
  absl::Status BackoffForAttemptAsync(
      absl::Status status, int attempt, Task* task,
      SourceLocation loc = ::tensorstore::SourceLocation::current()) {
    assert(task != nullptr);
    auto delay = spec_.retries->BackoffForAttempt(attempt);
    if (!delay) {
      return MaybeAnnotateStatus(std::move(status),
                                 absl::StrFormat("All %d retry attempts failed",
                                                 spec_.retries->max_retries),
                                 absl::StatusCode::kAborted, loc);
    }
    s3_metrics.retries.Increment();
    ScheduleAt(absl::Now() + *delay,
               WithExecutor(executor(), [task = IntrusivePtr<Task>(task)] {
                 task->Retry();
               }));

    return absl::OkStatus();
  }

  internal::NoRateLimiter no_rate_limiter_;
  std::shared_ptr<HttpTransport> transport_;
  S3KeyValueStoreSpecData spec_;
  std::string host_header_;

  absl::Mutex mutex_;  // Guards resolve_ehr_ creation.
  Future<const S3EndpointRegion> resolve_ehr_;
};

/// A ReadTask is a function object used to satisfy a
/// S3KeyValueStore::Read request.
struct ReadTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ReadTask> {
  IntrusivePtr<S3KeyValueStore> owner;
  std::string object_name;
  kvstore::ReadOptions options;
  Promise<kvstore::ReadResult> promise;

  std::string read_url_;  // the url to read from
  ReadyFuture<const S3EndpointRegion> endpoint_region_;

  int attempt_ = 0;
  absl::Time start_time_;

  ReadTask(IntrusivePtr<S3KeyValueStore> owner, std::string object_name,
           kvstore::ReadOptions options, Promise<kvstore::ReadResult> promise)
      : owner(std::move(owner)),
        object_name(std::move(object_name)),
        options(std::move(options)),
        promise(std::move(promise)) {}

  ~ReadTask() { owner->admission_queue().Finish(this); }

  static void Start(RateLimiterNode* task) {
    auto* self = static_cast<ReadTask*>(task);
    self->owner->read_rate_limiter().Finish(self);
    self->owner->admission_queue().Admit(self, &ReadTask::Admit);
  }

  static void Admit(RateLimiterNode* task) {
    auto* self = static_cast<ReadTask*>(task);
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
                        options.generation_conditions.if_not_equal);
    AddGenerationHeader(&request_builder, "if-match",
                        options.generation_conditions.if_equal);

    if (options.byte_range.size() != 0) {
      request_builder.MaybeAddRangeHeader(options.byte_range);
    }

    const auto& ehr = endpoint_region_.value();
    start_time_ = absl::Now();
    auto request = request_builder.EnableAcceptEncoding()
                       .MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(owner->host_header_, credentials,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, s3_logging) << "ReadTask: " << request;
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
    ABSL_LOG_IF(INFO, s3_logging.Level(1) && response.ok())
        << "ReadTask " << *response;

    bool is_retryable = false;
    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) {
        is_retryable = DefaultIsRetryableCode(response.status().code());
        return response.status();
      }
      switch (response.value().status_code) {
        // Special status codes handled outside the retry loop.
        case 412:
        case 404:
        case 304:
          return absl::OkStatus();
      }
      return AwsHttpResponseToStatus(response.value(), is_retryable);
    }();

    if (!status.ok() && is_retryable) {
      status =
          owner->BackoffForAttemptAsync(std::move(status), attempt_++, this);
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
    s3_metrics.bytes_read.IncrementBy(httpresponse.payload.size());
    auto latency = absl::Now() - start_time_;
    s3_metrics.read_latency_ms.Observe(absl::ToInt64Milliseconds(latency));

    switch (httpresponse.status_code) {
      case 204:
      case 404:
        // Object not found.
        return kvstore::ReadResult::Missing(start_time_);
      case 412:
        // "Failed precondition": indicates the ifGenerationMatch condition
        // did not hold.
        // NOTE: This is returned even when the object does not exist.
        return kvstore::ReadResult::Unspecified(TimestampedStorageGeneration{
            StorageGeneration::Unknown(), start_time_});
      case 304:
        // "Not modified": indicates that the ifGenerationNotMatch condition
        // did not hold.
        return kvstore::ReadResult::Unspecified(TimestampedStorageGeneration{
            options.generation_conditions.if_not_equal, start_time_});
    }

    absl::Cord value;
    if (options.byte_range.size() != 0) {
      // Currently unused
      ByteRange byte_range;
      int64_t total_size;

      TENSORSTORE_RETURN_IF_ERROR(internal_http::ValidateResponseByteRange(
          httpresponse, options.byte_range, value, byte_range, total_size));
    }

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto generation, StorageGenerationFromHeaders(httpresponse.headers));

    return kvstore::ReadResult::Value(
        std::move(value),
        TimestampedStorageGeneration{std::move(generation), start_time_});
  }
};

Future<kvstore::ReadResult> S3KeyValueStore::Read(Key key,
                                                  ReadOptions options) {
  s3_metrics.read.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid S3 object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal) ||
      !IsValidStorageGeneration(options.generation_conditions.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> S3KeyValueStore::ReadImpl(Key&& key,
                                                      ReadOptions&& options) {
  s3_metrics.batch_read.Increment();
  auto op = PromiseFuturePair<ReadResult>::Make();
  auto state = internal::MakeIntrusivePtr<ReadTask>(
      internal::IntrusivePtr<S3KeyValueStore>(this), key, std::move(options),
      std::move(op.promise));
  MaybeResolveRegion().ExecuteWhenReady(
      [state = std::move(state)](ReadyFuture<const S3EndpointRegion> ready) {
        if (!ready.status().ok()) {
          state->promise.SetResult(ready.status());
          return;
        }
        state->read_url_ = tensorstore::StrCat(ready.value().endpoint, "/",
                                               state->object_name);
        state->endpoint_region_ = std::move(ready);
        intrusive_ptr_increment(state.get());  // adopted by ReadTask::Start.
        state->owner->read_rate_limiter().Admit(state.get(), &ReadTask::Start);
      });
  return std::move(op.future);
}

// S3 doesn't support conditional PUT, so we use a HEAD request
// to test the if-match condition; unfortunately this still has races.
// Base must provide the following methods:
//   bool IsCancelled()
//   void Fail(absl::Status)
//   void OnHeadResponse(const Result<HttpResponse>& response)
//   void AfterHeadRequest()
//
template <typename Base>
struct ConditionTask : public RateLimiterNode,
                       public internal::AtomicReferenceCount<Base> {
  using Self = ConditionTask<Base>;

  IntrusivePtr<S3KeyValueStore> owner;
  kvstore::WriteOptions options_;
  ReadyFuture<const S3EndpointRegion> endpoint_region_;
  std::string object_url_;

  AwsCredentials credentials_;

  ConditionTask(IntrusivePtr<S3KeyValueStore> owner,
                kvstore::WriteOptions options,
                ReadyFuture<const S3EndpointRegion> endpoint_region,
                std::string object_url)
      : owner(std::move(owner)),
        options_(std::move(options)),
        endpoint_region_(std::move(endpoint_region)),
        object_url_(std::move(object_url)) {}

  static void Start(RateLimiterNode* task) {
    auto* self = static_cast<Base*>(task);
    self->owner->write_rate_limiter().Finish(self);
    self->owner->admission_queue().Admit(self, &Base::Admit);
  }

  static void Admit(RateLimiterNode* task) {
    auto* self = static_cast<Base*>(task);
    self->owner->executor()(
        [state = IntrusivePtr<Base>(self, internal::adopt_object_ref)] {
          state->Retry();
        });
  }

  void Retry() {
    if (static_cast<Base*>(this)->IsCancelled()) {
      return;
    }
    if (auto maybe_credentials = owner->GetCredentials();
        !maybe_credentials.ok()) {
      static_cast<Base*>(this)->Fail(maybe_credentials.status());
      return;
    } else if (maybe_credentials.value().has_value()) {
      credentials_ = std::move(*maybe_credentials.value());
    }

    if (StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
      static_cast<Base*>(this)->AfterHeadRequest();
      return;
    }

    auto builder = S3RequestBuilder("HEAD", object_url_);
    AddGenerationHeader(&builder, "if-match",
                        options_.generation_conditions.if_equal);

    auto now = absl::Now();
    const auto& ehr = endpoint_region_.value();
    auto request = builder.MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(owner->host_header_, credentials_,
                                     ehr.aws_region, kEmptySha256, now);

    ABSL_LOG_IF(INFO, s3_logging) << "Peek: " << request;

    auto future = owner->transport_->IssueRequest(request, {});
    future.ExecuteWhenReady([self = IntrusivePtr<Base>(static_cast<Base*>(
                                 this))](ReadyFuture<HttpResponse> response) {
      ABSL_LOG_IF(INFO, s3_logging.Level(1) && response.result().ok())
          << "Peek (Response): " << response.value();
      if (self->IsCancelled()) return;
      self->OnHeadResponse(response.result());
    });
  }
};

// A WriteTask is a function object used to satisfy S3KeyValueStore::Write.
struct WriteTask : public ConditionTask<WriteTask> {
  using Base = ConditionTask<WriteTask>;

  absl::Cord value_;
  Promise<TimestampedStorageGeneration> promise;

  int attempt_ = 0;
  absl::Time start_time_;

  WriteTask(IntrusivePtr<S3KeyValueStore> o, kvstore::WriteOptions options,
            ReadyFuture<const S3EndpointRegion> endpoint_region,
            std::string object_url, absl::Cord value,
            Promise<TimestampedStorageGeneration> promise)
      : Base(std::move(o), std::move(options), std::move(endpoint_region),
             std::move(object_url)),
        value_(std::move(value)),
        promise(std::move(promise)) {}

  ~WriteTask() { owner->admission_queue().Finish(this); }

  bool IsCancelled() { return !promise.result_needed(); }
  void Fail(absl::Status status) { promise.SetResult(std::move(status)); }

  void OnHeadResponse(const Result<HttpResponse>& response) {
    // TODO: Retry these.
    if (!response.ok()) {
      Fail(response.status());
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
        if (!options_.generation_conditions.MatchesNoValue()) {
          r.generation = StorageGeneration::Unknown();
          promise.SetResult(r);
          return;
        }
        break;
      default:
        break;
    }

    AfterHeadRequest();
  }

  void AfterHeadRequest() {
    // NOTE: This was changed from POST to PUT as a basic POST does not work
    // Some more headers need to be added to allow POST to work:
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-authentication-HTTPPOST.html

    start_time_ = absl::Now();
    auto content_sha256 = payload_sha256(value_);

    const auto& ehr = endpoint_region_.value();
    auto request =
        S3RequestBuilder("PUT", object_url_)
            .AddHeader("Content-Type: application/octet-stream")
            .AddHeader(absl::StrCat("Content-Length: ", value_.size()))
            .MaybeAddRequesterPayer(owner->spec_.requester_pays)
            .BuildRequest(owner->host_header_, credentials_, ehr.aws_region,
                          content_sha256, start_time_);

    ABSL_LOG_IF(INFO, s3_logging)
        << "WriteTask: " << request << " size=" << value_.size();

    auto future = owner->transport_->IssueRequest(
        request, internal_http::IssueRequestOptions(value_));
    future.ExecuteWhenReady([self = IntrusivePtr<WriteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnResponse(response.result());
    });
  }

  void OnResponse(const Result<HttpResponse>& response) {
    if (!promise.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, s3_logging.Level(1) && response.ok())
        << "WriteTask " << *response;

    bool is_retryable = false;
    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) {
        is_retryable = DefaultIsRetryableCode(response.status().code());
        return response.status();
      }
      return AwsHttpResponseToStatus(response.value(), is_retryable);
    }();
    if (!status.ok() && is_retryable) {
      status =
          owner->BackoffForAttemptAsync(std::move(status), attempt_++, this);
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
        if (!StorageGeneration::IsUnknown(
                options_.generation_conditions.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          return r;
        }
    }

    auto latency = absl::Now() - start_time_;
    s3_metrics.write_latency_ms.Observe(absl::ToInt64Milliseconds(latency));
    s3_metrics.bytes_written.IncrementBy(value_.size());
    TENSORSTORE_ASSIGN_OR_RETURN(
        r.generation, StorageGenerationFromHeaders(response.headers));
    return r;
  }
};

/// A DeleteTask is a function object used to satisfy S3KeyValueStore::Delete.
struct DeleteTask : public ConditionTask<DeleteTask> {
  using Base = ConditionTask<DeleteTask>;

  Promise<TimestampedStorageGeneration> promise;

  int attempt_ = 0;
  absl::Time start_time_;

  DeleteTask(IntrusivePtr<S3KeyValueStore> o, kvstore::WriteOptions options,
             ReadyFuture<const S3EndpointRegion> endpoint_region,
             std::string object_url,
             Promise<TimestampedStorageGeneration> promise)
      : Base(std::move(o), std::move(options), std::move(endpoint_region),
             std::move(object_url)),
        promise(std::move(promise)) {}

  ~DeleteTask() { owner->admission_queue().Finish(this); }

  bool IsCancelled() { return !promise.result_needed(); }
  void Fail(absl::Status status) { promise.SetResult(std::move(status)); }

  void OnHeadResponse(const Result<HttpResponse>& response) {
    // TODO: Retry these.
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
        if (!options_.generation_conditions.MatchesNoValue()) {
          r.generation = StorageGeneration::Unknown();
          promise.SetResult(std::move(r));
          return;
        }
        break;
      default:
        break;
    }

    AfterHeadRequest();
  }

  void AfterHeadRequest() {
    start_time_ = absl::Now();

    const auto& ehr = endpoint_region_.value();
    auto request = S3RequestBuilder("DELETE", object_url_)
                       .MaybeAddRequesterPayer(owner->spec_.requester_pays)
                       .BuildRequest(owner->host_header_, credentials_,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, s3_logging) << "DeleteTask: " << request;

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
    ABSL_LOG_IF(INFO, s3_logging.Level(1) && response.ok())
        << "DeleteTask " << *response;

    bool is_retryable = false;
    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) {
        is_retryable = DefaultIsRetryableCode(response.status().code());
        return response.status();
      }
      switch (response.value().status_code) {
        case 404:
          return absl::OkStatus();
        default:
          break;
      }
      return AwsHttpResponseToStatus(response.value(), is_retryable);
    }();
    if (!status.ok() && is_retryable) {
      status =
          owner->BackoffForAttemptAsync(std::move(status), attempt_++, this);
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
        if (!StorageGeneration::IsNoValue(
                options_.generation_conditions.if_equal) &&
            !StorageGeneration::IsUnknown(
                options_.generation_conditions.if_equal)) {
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
  s3_metrics.write.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid S3 object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  if (value && value->size() > kMaxS3PutSize) {
    // TODO: Support multi-part uploads of files larger than 5GB.
    // Generally, aws-cli splits uploads which exceed ~8MB into multiple
    // parts.
    return absl::InvalidArgumentError(absl::StrCat(
        "Object size ", value->size(), " exceeds S3 limit of ", kMaxS3PutSize));
  }

  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  MaybeResolveRegion().ExecuteWhenReady(
      [self = IntrusivePtr<S3KeyValueStore>(this),
       promise = std::move(op.promise), key = std::move(key),
       value = std::move(value), options = std::move(options)](
          ReadyFuture<const S3EndpointRegion> ready) {
        if (!ready.status().ok()) {
          promise.SetResult(ready.status());
          return;
        }
        std::string object_url =
            tensorstore::StrCat(ready.value().endpoint, "/", key);

        if (!value) {
          // Write with a std::nullopt value is a delete.
          auto state = internal::MakeIntrusivePtr<DeleteTask>(
              std::move(self), std::move(options), std::move(ready),
              std::move(object_url), std::move(promise));

          intrusive_ptr_increment(
              state.get());  // adopted by DeleteTask::Admit.
          state->owner->write_rate_limiter().Admit(state.get(),
                                                   &DeleteTask::Start);
          return;
        }

        auto state = internal::MakeIntrusivePtr<WriteTask>(
            std::move(self), std::move(options), std::move(ready),
            std::move(object_url), *std::move(value), std::move(promise));

        intrusive_ptr_increment(state.get());  // adopted by WriteTask::Admit.
        state->owner->write_rate_limiter().Admit(state.get(),
                                                 &WriteTask::Start);
      });

  return std::move(op.future);
}

/// ListTask implements the ListImpl execution flow.
struct ListTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<S3KeyValueStore> owner_;
  ListOptions options_;
  ListReceiver receiver_;

  std::string resource_;
  ReadyFuture<const S3EndpointRegion> endpoint_region_;

  std::string continuation_token_;
  absl::Time start_time_;
  int attempt_ = 0;
  bool has_query_parameters_;
  std::atomic<bool> cancelled_{false};

  ListTask(internal::IntrusivePtr<S3KeyValueStore>&& owner,
           ListOptions&& options, ListReceiver&& receiver)
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

  static void Start(RateLimiterNode* task) {
    auto* self = static_cast<ListTask*>(task);
    self->owner_->read_rate_limiter().Finish(self);
    self->owner_->admission_queue().Admit(self, &ListTask::Admit);
  }
  static void Admit(RateLimiterNode* task) {
    auto* self = static_cast<ListTask*>(task);
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

    const auto& ehr = endpoint_region_.value();
    start_time_ = absl::Now();

    auto request =
        request_builder.BuildRequest(owner_->host_header_, credentials,
                                     ehr.aws_region, kEmptySha256, start_time_);

    ABSL_LOG_IF(INFO, s3_logging) << "List: " << request;

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
    ABSL_LOG_IF(INFO, s3_logging.Level(1) && response.ok())
        << "List " << *response;

    bool is_retryable = false;
    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) {
        is_retryable = DefaultIsRetryableCode(response.status().code());
        return response.status();
      }
      return AwsHttpResponseToStatus(response.value(), is_retryable);
    }();
    if (!status.ok() && is_retryable) {
      return owner_->BackoffForAttemptAsync(std::move(status), attempt_++,
                                            this);
    }

    auto cord = response->payload;
    auto payload = cord.Flatten();

    tinyxml2::XMLDocument xmlDocument;
    if (int xmlcode = xmlDocument.Parse(payload.data(), payload.size());
        xmlcode != tinyxml2::XML_SUCCESS) {
      return absl::InvalidArgumentError(
          absl::StrCat("Malformed List response: ", xmlcode));
    }
    auto* root = xmlDocument.FirstChildElement("ListBucketResult");
    if (root == nullptr) {
      return absl::InvalidArgumentError(
          "Malformed List response: missing <ListBucketResult>");
    }

    // TODO: Visit /ListBucketResult/KeyCount?
    // Visit /ListBucketResult/Contents
    for (auto* contents = root->FirstChildElement("Contents");
         contents != nullptr;
         contents = contents->NextSiblingElement("Contents")) {
      if (is_cancelled()) {
        return absl::CancelledError();
      }

      // Visit  /ListBucketResult/Contents/Key
      auto* key_node = contents->FirstChildElement("Key");
      if (key_node == nullptr) {
        return absl::InvalidArgumentError(
            "Malformed List response: missing <Key> in <Contents>");
      }
      std::string key = GetNodeText(key_node);
      if (key < options_.range.inclusive_min) continue;
      if (KeyRange::CompareKeyAndExclusiveMax(
              key, options_.range.exclusive_max) >= 0) {
        // Objects are returned sorted in ascending order of the respective
        // key names, so after the current key exceeds exclusive max no
        // additional requests need to be made.
        execution::set_done(receiver_);
        return absl::OkStatus();
      }

      // Visit /ListBucketResult/Contents/Size
      int64_t size =
          GetNodeInt(contents->FirstChildElement("Size")).value_or(-1);

      // TODO: Visit /ListBucketResult/Contents/LastModified?
      if (key.size() > options_.strip_prefix_length) {
        execution::set_value(
            receiver_,
            ListEntry{key.substr(options_.strip_prefix_length), size});
      }
    }

    // Successful request, so clear the retry_attempt for the next request.
    // Visit /ListBucketResult/IsTruncated
    // Visit /ListBucketResult/NextContinuationToken
    attempt_ = 0;
    if (GetNodeText(root->FirstChildElement("IsTruncated")) == "true") {
      auto* next_continuation_token =
          root->FirstChildElement("NextContinuationToken");
      if (next_continuation_token == nullptr) {
        return absl::InvalidArgumentError(
            "Malformed List response: missing <NextContinuationToken>");
      }
      continuation_token_ = GetNodeText(next_continuation_token);
      IssueRequest();
    } else {
      execution::set_done(receiver_);
    }
    return absl::OkStatus();
  }
};

void S3KeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  s3_metrics.list.Increment();
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
      [state = std::move(state)](ReadyFuture<const S3EndpointRegion> ready) {
        if (!ready.status().ok()) {
          execution::set_error(state->receiver_, ready.status());
          return;
        }
        state->resource_ = tensorstore::StrCat(ready.value().endpoint, "/");
        state->endpoint_region_ = std::move(ready);
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

  void set_value(ListEntry entry) {
    assert(!entry.key.empty());
    if (!entry.key.empty()) {
      LinkError(promise_, owner_->Delete(std::move(entry.key)));
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
  s3_metrics.delete_range.Increment();
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
Future<const S3EndpointRegion> S3KeyValueStore::MaybeResolveRegion() {
  absl::MutexLock l(&mutex_);
  if (!resolve_ehr_.null()) return resolve_ehr_;

  resolve_ehr_ = internal_kvstore_s3::ResolveEndpointRegion(
      spec_.bucket,
      !spec_.endpoint.has_value() || spec_.endpoint.value().empty()
          ? std::string_view{}
          : std::string_view(spec_.endpoint.value()),
      spec_.host_header.value_or(std::string{}), transport_);
  resolve_ehr_.ExecuteWhenReady([](ReadyFuture<const S3EndpointRegion> ready) {
    if (!ready.status().ok()) {
      ABSL_LOG_IF(INFO, s3_logging)
          << "S3 driver failed to resolve endpoint: " << ready.status();
    } else {
      ABSL_LOG_IF(INFO, s3_logging)
          << "S3 driver using endpoint [" << ready.value() << "]";
    }
  });

  return resolve_ehr_;
}

Future<kvstore::DriverPtr> S3KeyValueStoreSpec::DoOpen() const {
  // TODO: The transport should support the AWS_CA_BUNDLE environment
  // variable.
  auto driver = internal::MakeIntrusivePtr<S3KeyValueStore>(
      internal_http::GetDefaultHttpTransport(), data_);

  // NOTE: Remove temporary logging use of experimental feature.
  if (data_.rate_limiter.has_value()) {
    ABSL_LOG_IF(INFO, s3_logging) << "Using experimental_s3_rate_limiter";
  }

  auto result = internal_kvstore_s3::ValidateEndpoint(
      data_.bucket, data_.aws_region, data_.endpoint.value_or(std::string{}),
      driver->host_header_);
  if (auto* status = std::get_if<absl::Status>(&result);
      status != nullptr && !status->ok()) {
    return std::move(*status);
  }
  if (auto* ehr = std::get_if<S3EndpointRegion>(&result); ehr != nullptr) {
    ABSL_LOG_IF(INFO, s3_logging)
        << "S3 driver using endpoint [" << *ehr << "]";
    driver->resolve_ehr_ = MakeReadyFuture<S3EndpointRegion>(std::move(*ehr));
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
  if (!IsValidBucketName(parsed.authority)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid S3 bucket name: ", QuoteString(parsed.authority)));
  }
  auto decoded_path = parsed.path.empty()
                          ? std::string()
                          : internal::PercentDecode(parsed.path.substr(1));

  auto driver_spec = internal::MakeIntrusivePtr<S3KeyValueStoreSpec>();
  driver_spec->data_.bucket = std::string(parsed.authority);
  driver_spec->data_.requester_pays = false;

  driver_spec->data_.aws_credentials =
      Context::Resource<AwsCredentialsResource>::DefaultSpec();
  driver_spec->data_.request_concurrency =
      Context::Resource<S3ConcurrencyResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<S3RequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();

  return {std::in_place, std::move(driver_spec), std::move(decoded_path)};
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
