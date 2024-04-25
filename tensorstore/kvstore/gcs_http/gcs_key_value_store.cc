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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/google_auth_provider.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/gcs/gcs_resource.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_http/gcs_resource.h"
#include "tensorstore/kvstore/gcs_http/object_metadata.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generic_coalescing_batch_util.h"
#include "tensorstore/kvstore/http/byte_range_util.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
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

/// specializations
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_optional.h"  // IWYU pragma: keep

// GCS reference links are:
//
// https://cloud.google.com/storage/docs/uploads-downloads
// https://cloud.google.com/storage/docs/json_api/v1/
// https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff

ABSL_FLAG(std::optional<std::string>, tensorstore_gcs_http_url, std::nullopt,
          "Url to used for http access to google cloud storage. "
          "Overrides TENSORSTORE_GCS_HTTP_URL.");

ABSL_FLAG(std::optional<std::string>, tensorstore_gcs_http_version,
          std::nullopt,
          "Url to used for http access to google cloud storage. "
          "Overrides TENSORSTORE_GCS_HTTP_VERSION.");

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::NoRateLimiter;
using ::tensorstore::internal::RateLimiter;
using ::tensorstore::internal::RateLimiterNode;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::internal_kvstore_gcs_http::GcsConcurrencyResource;
using ::tensorstore::internal_kvstore_gcs_http::GcsRateLimiterResource;
using ::tensorstore::internal_kvstore_gcs_http::ObjectMetadata;
using ::tensorstore::internal_kvstore_gcs_http::ParseObjectMetadata;
using ::tensorstore::internal_storage_gcs::GcsRequestRetries;
using ::tensorstore::internal_storage_gcs::GcsUserProjectResource;
using ::tensorstore::internal_storage_gcs::IsRetriable;
using ::tensorstore::internal_storage_gcs::IsValidBucketName;
using ::tensorstore::internal_storage_gcs::IsValidObjectName;
using ::tensorstore::internal_storage_gcs::IsValidStorageGeneration;
using ::tensorstore::kvstore::Key;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListOptions;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::SupportedFeatures;

namespace {
static constexpr char kUriScheme[] = "gs";
}  // namespace

namespace tensorstore {
namespace {
namespace jb = tensorstore::internal_json_binding;

auto& gcs_bytes_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/bytes_read",
    "Bytes read by the gcs kvstore driver");

auto& gcs_bytes_written = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/bytes_written",
    "Bytes written by the gcs kvstore driver");

auto& gcs_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/retries",
    "Count of all retried GCS requests (read/write/delete)");

auto& gcs_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/read", "GCS driver kvstore::Read calls");

auto& gcs_batch_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/batch_read", "gcs driver reads after batching");

auto& gcs_read_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/gcs/read_latency_ms",
        "GCS driver kvstore::Read latency (ms)");

auto& gcs_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/write", "GCS driver kvstore::Write calls");

auto& gcs_write_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/gcs/write_latency_ms",
        "GCS driver kvstore::Write latency (ms)");

auto& gcs_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/delete_range",
    "GCS driver kvstore::DeleteRange calls");

auto& gcs_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs/list", "GCS driver kvstore::List calls");

ABSL_CONST_INIT internal_log::VerboseFlag gcs_http_logging("gcs_http");

std::string GetGcsBaseUrl() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_gcs_http_url,
                           "TENSORSTORE_GCS_HTTP_URL")
      .value_or("https://storage.googleapis.com");
}

IssueRequestOptions::HttpVersion GetHttpVersion() {
  using HttpVersion = IssueRequestOptions::HttpVersion;
  static auto http_version = []() -> HttpVersion {
    auto version = GetFlagOrEnvValue(FLAGS_tensorstore_gcs_http_version,
                                     "TENSORSTORE_GCS_HTTP_VERSION");
    if (!version) {
      return HttpVersion::kDefault;
    }
    if (*version == "1" || *version == "1.1") {
      return HttpVersion::kHttp1;
    }
    if (*version == "2" || *version == "2.0") {
      return HttpVersion::kHttp2PriorKnowledge;
    }
    return HttpVersion::kHttp2TLS;
  }();
  return http_version;
}

/// Adds the generation query parameter to the provided url.
bool AddGenerationParam(std::string* url, const bool has_query,
                        std::string_view param_name,
                        const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen)) {
    // Unconditional.
    return false;
  } else {
    // One of two cases applies:
    //
    // 1. `gen` is a `StorageGeneration::FromUint64` generation.  In this
    //    case, the condition is specified as `=N`, where `N` is the decimal
    //    representation of the generation number.
    //
    // 2. `gen` is `StorageGeneration::NoValue()`.  In this case, the
    //    condition is specified as `=0`.
    //
    // In either case, `StorageGeneration::ToUint64` provides the correct
    // result.
    absl::StrAppend(url, (has_query ? "&" : "?"), param_name, "=",
                    StorageGeneration::ToUint64(gen));
    return true;
  }
}

/// Adds the userProject query parameter to the provided url.
bool AddUserProjectParam(std::string* url, const bool has_query,
                         std::string_view encoded_user_project) {
  if (!encoded_user_project.empty()) {
    absl::StrAppend(url, (has_query ? "&" : "?"),
                    "userProject=", encoded_user_project);
    return true;
  }
  return false;
}

/// Composes the resource root uri for the GCS API using the bucket
/// and constants for the host, api-version, etc.
std::string BucketResourceRoot(std::string_view bucket) {
  const char kVersion[] = "v1";
  return tensorstore::StrCat(GetGcsBaseUrl(), "/storage/", kVersion, "/b/",
                             bucket);
}

/// Composes the resource upload root uri for the GCS API using the bucket
/// and constants for the host, api-version, etc.
std::string BucketUploadRoot(std::string_view bucket) {
  const char kVersion[] = "v1";
  return tensorstore::StrCat(GetGcsBaseUrl(), "/upload/storage/", kVersion,
                             "/b/", bucket);
}

struct GcsKeyValueStoreSpecData {
  std::string bucket;

  Context::Resource<GcsConcurrencyResource> request_concurrency;
  std::optional<Context::Resource<GcsRateLimiterResource>> rate_limiter;
  Context::Resource<GcsUserProjectResource> user_project;
  Context::Resource<GcsRequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.bucket, x.request_concurrency, x.rate_limiter, x.user_project,
             x.retries, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      // Bucket is specified in the `spec` since it identifies the resource
      // being accessed.
      jb::Member("bucket",
                 jb::Projection<&GcsKeyValueStoreSpecData::bucket>(jb::Validate(
                     [](const auto& options, const std::string* x) {
                       if (!IsValidBucketName(*x)) {
                         return absl::InvalidArgumentError(tensorstore::StrCat(
                             "Invalid GCS bucket name: ", QuoteString(*x)));
                       }
                       return absl::OkStatus();
                     }))),

      jb::Member(
          GcsConcurrencyResource::id,
          jb::Projection<&GcsKeyValueStoreSpecData::request_concurrency>()),
      jb::Member(GcsRateLimiterResource::id,
                 jb::Projection<&GcsKeyValueStoreSpecData::rate_limiter>()),

      // `user_project` project ID to use for billing is obtained from the
      // `context` since it is not part of the identity of the resource being
      // accessed.
      jb::Member(GcsUserProjectResource::id,
                 jb::Projection<&GcsKeyValueStoreSpecData::user_project>()),
      jb::Member(GcsRequestRetries::id,
                 jb::Projection<&GcsKeyValueStoreSpecData::retries>()),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<
                     &GcsKeyValueStoreSpecData::data_copy_concurrency>()) /**/
  );
};

std::string GetGcsUrl(std::string_view bucket, std::string_view path) {
  return tensorstore::StrCat(kUriScheme, "://", bucket, "/",
                             internal::PercentEncodeUriPath(path));
}

class GcsKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<GcsKeyValueStoreSpec,
                                                    GcsKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "gcs";

  Future<kvstore::DriverPtr> DoOpen() const override;
  Result<std::string> ToUrl(std::string_view path) const override {
    return GetGcsUrl(data_.bucket, path);
  }
};

/// Implements the KeyValueStore interface for storing tensorstore data into a
/// GCS storage bucket.
class GcsKeyValueStore
    : public internal_kvstore::RegisteredDriver<GcsKeyValueStore,
                                                GcsKeyValueStoreSpec> {
 public:
  /// The resource_root is the url used to read data and metadata from the GCS
  /// bucket.
  const std::string& resource_root() const { return resource_root_; }

  /// The upload_root is the url used to upload data to the GCS bucket.
  const std::string& upload_root() const { return upload_root_; }

  /// The userProject field, or empty.
  const std::string& encoded_user_project() const {
    return encoded_user_project_;
  }

  internal_kvstore_batch::CoalescingOptions GetBatchReadCoalescingOptions()
      const {
    return internal_kvstore_batch::kDefaultRemoteStorageCoalescingOptions;
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  Future<const void> DeleteRange(KeyRange range) override;

  /// Returns the Auth header for a GCS request.
  Result<std::optional<std::string>> GetAuthHeader() {
    absl::MutexLock lock(&auth_provider_mutex_);
    if (!auth_provider_) {
      auto result = tensorstore::internal_oauth2::GetSharedGoogleAuthProvider();
      if (!result.ok() && absl::IsNotFound(result.status())) {
        auth_provider_ = nullptr;
      } else {
        TENSORSTORE_RETURN_IF_ERROR(result);
        auth_provider_ = std::move(*result);
      }
    }
    if (!*auth_provider_) return std::nullopt;
    auto auth_header_result = (*auth_provider_)->GetAuthHeader();
    if (!auth_header_result.ok() &&
        absl::IsNotFound(auth_header_result.status())) {
      return std::nullopt;
    }
    return auth_header_result;
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

  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  std::string DescribeKey(std::string_view key) override {
    return GetGcsUrl(spec_.bucket, key);
  }

  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
           SupportedFeatures::kAtomicWriteWithoutOverwrite;
  }

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

    gcs_retries.Increment();
    ScheduleAt(absl::Now() + *delay,
               WithExecutor(executor(), [task = IntrusivePtr<Task>(task)] {
                 task->Retry();
               }));

    return absl::OkStatus();
  }

  SpecData spec_;
  std::string resource_root_;  // bucket resource root.
  std::string upload_root_;    // bucket upload root.
  std::string encoded_user_project_;
  NoRateLimiter no_rate_limiter_;

  std::shared_ptr<HttpTransport> transport_;

  absl::Mutex auth_provider_mutex_;
  // Optional state indicates whether the provider has been obtained.  A
  // nullptr provider is valid and indicates to use anonymous access.
  std::optional<std::shared_ptr<internal_oauth2::AuthProvider>> auth_provider_;
};

Future<kvstore::DriverPtr> GcsKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<GcsKeyValueStore>();
  driver->spec_ = data_;
  driver->resource_root_ = BucketResourceRoot(data_.bucket);
  driver->upload_root_ = BucketUploadRoot(data_.bucket);
  driver->transport_ = internal_http::GetDefaultHttpTransport();

  // NOTE: Remove temporary logging use of experimental feature.
  if (data_.rate_limiter.has_value()) {
    ABSL_LOG(INFO) << "Using experimental_gcs_rate_limiter";
  }
  if (const auto& project_id = data_.user_project->project_id) {
    driver->encoded_user_project_ =
        internal::PercentEncodeUriComponent(*project_id);
  }
  return driver;
}

// GCS does not follow HTTP spec as far as respecting `cache-control` request
// headers.
//
// Instead, if the object metadata does not disallow caching and the bucket is
// public, a cached response may be returned even if `cache-control:
// max-age=0` is specified.
//
// b/243544317
//
// As a workaround, specify a unique query parameter in every request.  That
// ensures the cache is bypassed.
void AddUniqueQueryParameterToDisableCaching(std::string& url) {
  struct RandomState {
    absl::Mutex mutex;
    absl::BitGen gen ABSL_GUARDED_BY(mutex);
  };
  static RandomState random_state;
  uint64_t uuid[2];
  absl::MutexLock lock(&random_state.mutex);
  for (auto& x : uuid) {
    x = absl::Uniform<uint64_t>(random_state.gen);
  }
  tensorstore::StrAppend(&url,
                         "&tensorstore=", absl::Hex(uuid[0], absl::kZeroPad16),
                         absl::Hex(uuid[1], absl::kZeroPad16));
}

////////////////////////////////////////////////////

/// A ReadTask is a function object used to satisfy a
/// GcsKeyValueStore::Read request.
struct ReadTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ReadTask> {
  IntrusivePtr<GcsKeyValueStore> owner;
  std::string resource;
  kvstore::ReadOptions options;
  Promise<kvstore::ReadResult> promise;

  int attempt_ = 0;
  absl::Time start_time_;

  ReadTask(IntrusivePtr<GcsKeyValueStore> owner, std::string resource,
           kvstore::ReadOptions options, Promise<kvstore::ReadResult> promise)
      : owner(std::move(owner)),
        resource(std::move(resource)),
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
    /// Reads contents of a GCS object.
    std::string media_url = tensorstore::StrCat(
        resource, options.byte_range.size() == 0 ? "?alt=json" : "?alt=media");

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&media_url, true, "ifGenerationNotMatch",
                       options.generation_conditions.if_not_equal);
    AddGenerationParam(&media_url, true, "ifGenerationMatch",
                       options.generation_conditions.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&media_url, true, owner->encoded_user_project());

    AddUniqueQueryParameterToDisableCaching(media_url);

    // TODO: Configure timeouts.
    auto maybe_auth_header = owner->GetAuthHeader();
    if (!maybe_auth_header.ok()) {
      promise.SetResult(maybe_auth_header.status());
      return;
    }

    HttpRequestBuilder request_builder("GET", media_url);
    if (maybe_auth_header.value().has_value()) {
      request_builder.AddHeader(*maybe_auth_header.value());
    }
    if (options.byte_range.size() != 0) {
      request_builder.MaybeAddRangeHeader(options.byte_range);
    }

    auto request = request_builder.EnableAcceptEncoding().BuildRequest();
    start_time_ = absl::Now();

    ABSL_LOG_IF(INFO, gcs_http_logging) << "ReadTask: " << request;
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
    ABSL_LOG_IF(INFO, gcs_http_logging.Level(1) && response.ok())
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
    gcs_bytes_read.IncrementBy(httpresponse.payload.size());
    auto latency = absl::Now() - start_time_;
    gcs_read_latency_ms.Observe(absl::ToInt64Milliseconds(latency));

    // Parse `Date` header from response to correctly handle cached responses.
    // The GCS servers always send a `date` header.
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
    ObjectMetadata metadata;
    if (options.byte_range.size() != 0) {
      // Currently unused
      ByteRange byte_range;
      int64_t total_size;

      TENSORSTORE_RETURN_IF_ERROR(internal_http::ValidateResponseByteRange(
          httpresponse, options.byte_range, value, byte_range, total_size));
      // TODO: Avoid parsing the entire metadata & only extract the
      // generation field.
      SetObjectMetadataFromHeaders(httpresponse.headers, &metadata);
    } else {
      // 0-range reads issue metadata requests; parse the metadata for
      // the storage generation.
      absl::Cord cord = httpresponse.payload;
      TENSORSTORE_ASSIGN_OR_RETURN(metadata,
                                   ParseObjectMetadata(cord.Flatten()));
    }

    auto generation = StorageGeneration::FromUint64(metadata.generation);
    return kvstore::ReadResult::Value(
        std::move(value),
        TimestampedStorageGeneration{std::move(generation), start_time_});
  }
};

Future<kvstore::ReadResult> GcsKeyValueStore::Read(Key key,
                                                   ReadOptions options) {
  gcs_read.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid GCS object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal) ||
      !IsValidStorageGeneration(options.generation_conditions.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> GcsKeyValueStore::ReadImpl(Key&& key,
                                                       ReadOptions&& options) {
  gcs_batch_read.Increment();
  auto encoded_object_name = internal::PercentEncodeUriComponent(key);
  std::string resource = tensorstore::internal::JoinPath(resource_root_, "/o/",
                                                         encoded_object_name);

  auto op = PromiseFuturePair<ReadResult>::Make();
  auto state = internal::MakeIntrusivePtr<ReadTask>(
      internal::IntrusivePtr<GcsKeyValueStore>(this), std::move(resource),
      std::move(options), std::move(op.promise));

  intrusive_ptr_increment(state.get());  // adopted by ReadTask::Start.
  read_rate_limiter().Admit(state.get(), &ReadTask::Start);
  return std::move(op.future);
}

/// A WriteTask is a function object used to satisfy a
/// GcsKeyValueStore::Write request.
struct WriteTask : public RateLimiterNode,
                   public internal::AtomicReferenceCount<WriteTask> {
  IntrusivePtr<GcsKeyValueStore> owner;
  std::string encoded_object_name;
  absl::Cord value;
  kvstore::WriteOptions options;
  Promise<TimestampedStorageGeneration> promise;

  int attempt_ = 0;
  absl::Time start_time_;

  WriteTask(IntrusivePtr<GcsKeyValueStore> owner,
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

  /// Writes an object to GCS.
  void Retry() {
    if (!promise.result_needed()) {
      return;
    }
    // We use the SimpleUpload technique.

    std::string upload_url =
        tensorstore::StrCat(owner->upload_root(), "/o", "?uploadType=media",
                            "&name=", encoded_object_name);

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&upload_url, true, "ifGenerationMatch",
                       options.generation_conditions.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&upload_url, true, owner->encoded_user_project());

    auto maybe_auth_header = owner->GetAuthHeader();
    if (!maybe_auth_header.ok()) {
      promise.SetResult(maybe_auth_header.status());
      return;
    }
    HttpRequestBuilder request_builder("POST", upload_url);
    if (maybe_auth_header.value().has_value()) {
      request_builder.AddHeader(*maybe_auth_header.value());
    }
    auto request =
        request_builder.AddHeader("Content-Type: application/octet-stream")
            .AddHeader(tensorstore::StrCat("Content-Length: ", value.size()))
            .BuildRequest();
    start_time_ = absl::Now();

    ABSL_LOG_IF(INFO, gcs_http_logging)
        << "WriteTask: " << request << " size=" << value.size();

    auto future = owner->transport_->IssueRequest(
        request, IssueRequestOptions(value).SetHttpVersion(GetHttpVersion()));
    future.ExecuteWhenReady([self = IntrusivePtr<WriteTask>(this)](
                                ReadyFuture<HttpResponse> response) {
      self->OnResponse(response.result());
    });
  }

  void OnResponse(const Result<HttpResponse>& response) {
    if (!promise.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, gcs_http_logging.Level(1) && response.ok())
        << "WriteTask " << *response;

    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) return response.status();
      switch (response.value().status_code) {
        case 304:
          // Not modified implies that the generation did not match.
          [[fallthrough]];
        case 412:
          // Failed precondition implies the generation did not match.
          return absl::OkStatus();
        case 404:
          if (!options.generation_conditions.MatchesNoValue()) {
            return absl::OkStatus();
          }
          break;
        default:
          break;
      }
      return HttpResponseCodeToStatus(response.value());
    }();

    if (!status.ok() && IsRetriable(status)) {
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

  Result<TimestampedStorageGeneration> FinishResponse(
      const HttpResponse& httpresponse) {
    TimestampedStorageGeneration r;
    r.time = start_time_;
    switch (httpresponse.status_code) {
      case 304:
        // Not modified implies that the generation did not match.
        [[fallthrough]];
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        return r;
      case 404:
        if (!StorageGeneration::IsUnknown(
                options.generation_conditions.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          return r;
        }
    }

    auto latency = absl::Now() - start_time_;
    gcs_write_latency_ms.Observe(absl::ToInt64Milliseconds(latency));
    gcs_bytes_written.IncrementBy(value.size());

    // TODO: Avoid parsing the entire metadata & only extract the
    // generation field.
    auto payload = httpresponse.payload;
    auto parsed_object_metadata = ParseObjectMetadata(payload.Flatten());
    TENSORSTORE_RETURN_IF_ERROR(parsed_object_metadata);

    r.generation =
        StorageGeneration::FromUint64(parsed_object_metadata->generation);
    return r;
  }
};

/// A DeleteTask is a function object used to satisfy a
/// GcsKeyValueStore::Delete request.
struct DeleteTask : public RateLimiterNode,
                    public internal::AtomicReferenceCount<DeleteTask> {
  IntrusivePtr<GcsKeyValueStore> owner;
  std::string resource;
  kvstore::WriteOptions options;
  Promise<TimestampedStorageGeneration> promise;

  int attempt_ = 0;
  absl::Time start_time_;

  DeleteTask(IntrusivePtr<GcsKeyValueStore> owner, std::string resource,
             kvstore::WriteOptions options,
             Promise<TimestampedStorageGeneration> promise)
      : owner(std::move(owner)),
        resource(std::move(resource)),
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

  /// Removes an object from GCS.
  void Retry() {
    if (!promise.result_needed()) {
      return;
    }
    std::string delete_url = resource;

    // Add the ifGenerationNotMatch condition.
    bool has_query = AddGenerationParam(&delete_url, false, "ifGenerationMatch",
                                        options.generation_conditions.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&delete_url, has_query, owner->encoded_user_project());

    auto maybe_auth_header = owner->GetAuthHeader();
    if (!maybe_auth_header.ok()) {
      promise.SetResult(maybe_auth_header.status());
      return;
    }
    HttpRequestBuilder request_builder("DELETE", delete_url);
    if (maybe_auth_header.value().has_value()) {
      request_builder.AddHeader(*maybe_auth_header.value());
    }

    auto request = request_builder.BuildRequest();
    start_time_ = absl::Now();

    ABSL_LOG_IF(INFO, gcs_http_logging) << "DeleteTask: " << request;

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
    ABSL_LOG_IF(INFO, gcs_http_logging.Level(1) && response.ok())
        << "DeleteTask " << *response;

    absl::Status status = [&]() -> absl::Status {
      if (!response.ok()) return response.status();
      switch (response.value().status_code) {
        case 412:
          // Failed precondition implies the generation did not match.
          [[fallthrough]];
        case 404:
          return absl::OkStatus();
        default:
          break;
      }
      return HttpResponseCodeToStatus(response.value());
    }();

    if (!status.ok() && IsRetriable(status)) {
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
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        break;
      case 404:
        // 404 Not Found means aborted when a StorageGeneration was specified.
        if (!options.generation_conditions.MatchesNoValue()) {
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

Future<TimestampedStorageGeneration> GcsKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  gcs_write.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid GCS object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }

  std::string encoded_object_name = internal::PercentEncodeUriComponent(key);
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  if (value) {
    auto state = internal::MakeIntrusivePtr<WriteTask>(
        IntrusivePtr<GcsKeyValueStore>(this), std::move(encoded_object_name),
        std::move(*value), std::move(options), std::move(op.promise));

    intrusive_ptr_increment(state.get());  // adopted by WriteTask::Start.
    write_rate_limiter().Admit(state.get(), &WriteTask::Start);
  } else {
    std::string resource = tensorstore::internal::JoinPath(
        resource_root_, "/o/", encoded_object_name);

    auto state = internal::MakeIntrusivePtr<DeleteTask>(
        IntrusivePtr<GcsKeyValueStore>(this), std::move(resource),
        std::move(options), std::move(op.promise));

    intrusive_ptr_increment(state.get());  // adopted by DeleteTask::Start.
    write_rate_limiter().Admit(state.get(), &DeleteTask::Start);
  }
  return std::move(op.future);
}

// List responds with a Json payload that includes these fields.
struct GcsListResponsePayload {
  std::string next_page_token;        // used to page through list results.
  std::vector<ObjectMetadata> items;  // individual result metadata.
};

constexpr static auto GcsListResponsePayloadBinder = jb::Object(
    jb::Member("nextPageToken",
               jb::Projection(&GcsListResponsePayload::next_page_token,
                              jb::DefaultInitializedValue())),
    jb::Member("items", jb::Projection(&GcsListResponsePayload::items,
                                       jb::DefaultInitializedValue())),
    jb::DiscardExtraMembers);

/// ListTask implements the ListImpl execution flow.
struct ListTask : public RateLimiterNode,
                  public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<GcsKeyValueStore> owner_;
  ListOptions options_;
  ListReceiver receiver_;
  std::string resource_;

  std::string base_list_url_;
  std::string next_page_token_;
  int attempt_ = 0;
  bool has_query_parameters_;
  std::atomic<bool> cancelled_{false};

  ListTask(internal::IntrusivePtr<GcsKeyValueStore>&& owner,
           ListOptions&& options, ListReceiver&& receiver,
           std::string&& resource)
      : owner_(std::move(owner)),
        options_(std::move(options)),
        receiver_(std::move(receiver)),
        resource_(std::move(resource)) {
    // Construct the base LIST url. This will be modified to include the
    // nextPageToken
    base_list_url_ = resource_;
    has_query_parameters_ = AddUserProjectParam(&base_list_url_, false,
                                                owner_->encoded_user_project());
    if (auto& inclusive_min = options_.range.inclusive_min;
        !inclusive_min.empty()) {
      absl::StrAppend(
          &base_list_url_, (has_query_parameters_ ? "&" : "?"),
          "startOffset=", internal::PercentEncodeUriComponent(inclusive_min));
      has_query_parameters_ = true;
    }
    if (auto& exclusive_max = options_.range.exclusive_max;
        !exclusive_max.empty()) {
      absl::StrAppend(
          &base_list_url_, (has_query_parameters_ ? "&" : "?"),
          "endOffset=", internal::PercentEncodeUriComponent(exclusive_max));
      has_query_parameters_ = true;
    }
  }

  ~ListTask() { owner_->admission_queue().Finish(this); }

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
    execution::set_starting(self->receiver_, [self] {
      self->cancelled_.store(true, std::memory_order_relaxed);
    });
    self->owner_->executor()(
        [state = IntrusivePtr<ListTask>(self, internal::adopt_object_ref)] {
          state->IssueRequest();
        });
  }

  void Retry() { IssueRequest(); }

  void IssueRequest() {
    if (is_cancelled()) {
      execution::set_done(receiver_);
      execution::set_stopping(receiver_);
      return;
    }

    std::string list_url = base_list_url_;
    if (!next_page_token_.empty()) {
      absl::StrAppend(&list_url, (has_query_parameters_ ? "&" : "?"),
                      "pageToken=", next_page_token_);
    }

    auto auth_header = owner_->GetAuthHeader();
    if (!auth_header.ok()) {
      execution::set_error(receiver_, std::move(auth_header).status());
      execution::set_stopping(receiver_);
      return;
    }

    HttpRequestBuilder request_builder("GET", list_url);
    if (auth_header->has_value()) {
      request_builder.AddHeader(auth_header->value());
    }

    auto request = request_builder.BuildRequest();
    ABSL_LOG_IF(INFO, gcs_http_logging) << "List: " << request;

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
      execution::set_stopping(receiver_);
      return;
    }
    if (!status.ok()) {
      execution::set_error(receiver_, std::move(status));
      execution::set_stopping(receiver_);
      return;
    }
  }

  absl::Status OnResponseImpl(const Result<HttpResponse>& response) {
    if (is_cancelled()) {
      return absl::CancelledError();
    }
    ABSL_LOG_IF(INFO, gcs_http_logging.Level(1) && response.ok())
        << "List " << *response;

    absl::Status status =
        response.ok() ? HttpResponseCodeToStatus(*response) : response.status();
    if (!status.ok() && IsRetriable(status)) {
      return owner_->BackoffForAttemptAsync(std::move(status), attempt_++,
                                            this);
    }
    auto payload = response->payload;
    auto j = internal::ParseJson(payload.Flatten());
    if (j.is_discarded()) {
      return absl::InternalError(tensorstore::StrCat(
          "Failed to parse response metadata: ", payload.Flatten()));
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto parsed_payload,
        jb::FromJson<GcsListResponsePayload>(j, GcsListResponsePayloadBinder));
    for (auto& metadata : parsed_payload.items) {
      if (is_cancelled()) {
        return absl::CancelledError();
      }
      std::string_view name = metadata.name;
      if (options_.strip_prefix_length) {
        name = name.substr(options_.strip_prefix_length);
      }
      execution::set_value(receiver_,
                           ListEntry{
                               std::string(name),
                               ListEntry::checked_size(metadata.size),
                           });
    }

    // Successful request, so clear the retry_attempt for the next request.
    attempt_ = 0;
    next_page_token_ = std::move(parsed_payload.next_page_token);
    if (!next_page_token_.empty()) {
      IssueRequest();
    } else {
      execution::set_done(receiver_);
      execution::set_stopping(receiver_);
    }
    return absl::OkStatus();
  }
};

void GcsKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  gcs_list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }

  auto state = internal::MakeIntrusivePtr<ListTask>(
      IntrusivePtr<GcsKeyValueStore>(this), std::move(options),
      std::move(receiver),
      /*resource=*/tensorstore::internal::JoinPath(resource_root_, "/o"));

  intrusive_ptr_increment(state.get());  // adopted by ListTask::Start.
  read_rate_limiter().Admit(state.get(), &ListTask::Start);
}

// Receiver used by `DeleteRange` for processing the results from `List`.
struct DeleteRangeListReceiver {
  IntrusivePtr<GcsKeyValueStore> owner_;
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

Future<const void> GcsKeyValueStore::DeleteRange(KeyRange range) {
  gcs_delete_range.Increment();
  if (range.empty()) return absl::OkStatus();

  // TODO(jbms): It could make sense to rate limit the list operation, so that
  // we don't get way ahead of the delete operations.  Currently our
  // sender/receiver abstraction does not support back pressure, though.
  auto op = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
  ListOptions list_options;
  list_options.range = std::move(range);
  ListImpl(list_options, DeleteRangeListReceiver{
                             internal::IntrusivePtr<GcsKeyValueStore>(this),
                             std::move(op.promise)});
  return std::move(op.future);
}

Result<kvstore::Spec> ParseGcsUrl(std::string_view url) {
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
        tensorstore::StrCat("Invalid GCS bucket name: ", QuoteString(bucket)));
  }
  std::string_view encoded_path =
      (end_of_bucket == std::string_view::npos)
          ? std::string_view{}
          : parsed.authority_and_path.substr(end_of_bucket + 1);
  auto driver_spec = internal::MakeIntrusivePtr<GcsKeyValueStoreSpec>();
  driver_spec->data_.bucket = bucket;
  driver_spec->data_.request_concurrency =
      Context::Resource<GcsConcurrencyResource>::DefaultSpec();
  driver_spec->data_.user_project =
      Context::Resource<GcsUserProjectResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<GcsRequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();

  return {std::in_place, std::move(driver_spec),
          internal::PercentDecode(encoded_path)};
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::GcsKeyValueStore)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::GcsKeyValueStoreSpec>
    registration;
const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{kUriScheme, tensorstore::ParseGcsUrl};
}  // namespace
