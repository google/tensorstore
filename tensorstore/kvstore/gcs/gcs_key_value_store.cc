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

#include <atomic>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/concurrency_resource_provider.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/google_auth_provider.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/gcs/object_metadata.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// Best Practices for GCS Storage usage & other GCS reference
// links are:
//
// https://cloud.google.com/storage/docs/downloading-objects
// https://cloud.google.com/storage/docs/uploading-objects
// https://cloud.google.com/storage/docs/best-practices#uploading
// https://cloud.google.com/storage/docs/json_api/v1/

using tensorstore::internal::IntrusivePtr;
using tensorstore::internal_http::HttpRequest;
using tensorstore::internal_http::HttpRequestBuilder;
using tensorstore::internal_http::HttpResponse;
using tensorstore::internal_http::HttpTransport;
using tensorstore::internal_storage_gcs::ObjectMetadata;
using tensorstore::internal_storage_gcs::ParseObjectMetadata;

// Uncomment to log all http requests.
// #define TENSORSTORE_INTERNAL_GCS_LOG_REQUESTS
// Uncomment to log all http responses
// #define TENSORSTORE_INTERNAL_GCS_LOG_RESPONSES

namespace tensorstore {
namespace {
namespace jb = tensorstore::internal_json_binding;

std::string_view GetGcsBaseUrl() {
  static std::string url = []() -> std::string {
    auto maybe_url = internal::GetEnv("TENSORSTORE_GCS_HTTP_URL");
    if (maybe_url) return std::move(*maybe_url);
    return "https://storage.googleapis.com";
  }();
  return url;
}

struct GcsRequestConcurrencyResource : public internal::ConcurrencyResource {
  static constexpr char id[] = "gcs_request_concurrency";
};

/// Optionally specifies a project to which all requests are billed.
///
/// If not specified, requests to normal buckets are billed to the project that
/// owns the bucket, and requests to "requestor pays"-enabled buckets fail.
struct GcsUserProjectResource
    : public internal::ContextResourceTraits<GcsUserProjectResource> {
  static constexpr char id[] = "gcs_user_project";
  struct Spec {
    std::optional<std::string> project_id;
  };
  using Resource = Spec;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    return jb::Object(
        jb::Member("project_id", jb::Projection(&Spec::project_id)));
  }
  static Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) {
    return spec;
  }
  static Spec GetSpec(const Spec& spec,
                      const internal::ContextSpecBuilder& builder) {
    return spec;
  }
};

/// Specifies a limit on the number of retries.
struct GcsRequestRetries : public internal::RetriesResource<GcsRequestRetries> {
  static constexpr char id[] = "gcs_request_retries";
};

struct GcsRequestConcurrencyResourceTraits
    : public internal::ConcurrencyResourceTraits,
      public internal::ContextResourceTraits<GcsRequestConcurrencyResource> {
  GcsRequestConcurrencyResourceTraits() : ConcurrencyResourceTraits(32) {}
};
const internal::ContextResourceRegistration<GcsRequestConcurrencyResourceTraits>
    gcs_request_concurrency_registration;
const internal::ContextResourceRegistration<GcsUserProjectResource>
    gcs_user_project_registration;
const internal::ContextResourceRegistration<GcsRequestRetries>
    gcs_request_retries_registration;

// Returns whether the bucket name is valid.
// https://cloud.google.com/storage/docs/naming#requirements
bool IsValidBucketName(std::string_view bucket) {
  // Buckets containing dots can contain up to 222 characters.
  if (bucket.size() < 3 || bucket.size() > 222) return false;

  // Bucket names must start and end with a number or letter.
  if (!absl::ascii_isdigit(*bucket.begin()) &&
      !absl::ascii_islower(*bucket.begin())) {
    return false;
  }
  if (!absl::ascii_isdigit(*bucket.rbegin()) &&
      !absl::ascii_islower(*bucket.rbegin())) {
    return false;
  }

  for (std::string_view v : absl::StrSplit(bucket, absl::ByChar('.'))) {
    if (v.empty()) return false;
    if (v.size() > 63) return false;
    if (*v.begin() == '-') return false;
    if (*v.rbegin() == '-') return false;

    for (std::string_view::size_type i = 0; i < v.size(); i++) {
      // Bucket names must contain only lowercase letters, numbers,
      // dashes (-), underscores (_), and dots (.).
      // Names containing dots require verification.
      const auto ch = v[i];
      if (ch != '-' && ch != '_' && !absl::ascii_isdigit(ch) &&
          !absl::ascii_islower(ch)) {
        return false;
      }
    }
  }

  // Not validated:
  // Bucket names cannot begin with the "goog" prefix.
  // Bucket names cannot contain "google" or close misspellings, such as
  // "g00gle".
  // NOTE: ip-address-style bucket names are also invalid, but not checked here.
  return true;
}

// Returns whether the object name is a valid GCS object name.
bool IsValidObjectName(std::string_view name) {
  if (name == "." || name == "..") return false;
  if (absl::StartsWith(name, ".well-known/acme-challenge")) return false;
  if (name.find('\r') != std::string_view::npos) return false;
  if (name.find('\n') != std::string_view::npos) return false;
  // TODO: Validate that object is a correct utf-8 string.
  return true;
}

/// Returns an error Status when either the object name or the StorageGeneration
/// are not legal values for the GCS storage backend.
absl::Status ValidateObjectAndStorageGeneration(std::string_view object,
                                                const StorageGeneration& gen) {
  if (!IsValidObjectName(object)) {
    return absl::InvalidArgumentError("Invalid GCS object name");
  }
  if (!StorageGeneration::IsUnknown(gen) &&
      !StorageGeneration::IsNoValue(gen) && !StorageGeneration::IsUint64(gen)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  return absl::OkStatus();
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
    // 1. `gen` is a `StorageGeneration::FromUint64` generation.  In this case,
    //    the condition is specified as `=N`, where `N` is the decimal
    //    representation of the generation number.
    //
    // 2. `gen` is `StorageGeneration::NoValue()`.  In this case, the condition
    //    is specified as `=0`.
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

/// Returns whether the Status is a retriable request.
bool IsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

struct GcsKeyValueStoreSpecData {
  std::string bucket;
  Context::Resource<GcsRequestConcurrencyResource> request_concurrency;
  Context::Resource<GcsUserProjectResource> user_project;
  Context::Resource<GcsRequestRetries> retries;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.bucket, x.request_concurrency, x.user_project, x.retries);
  };

  constexpr static auto default_json_binder = jb::Object(
      // Bucket is specified in the `spec` since it identifies the resource
      // being accessed.
      jb::Member("bucket",
                 jb::Projection<&GcsKeyValueStoreSpecData::bucket>(jb::Validate(
                     [](const auto& options, const std::string* x) {
                       if (!IsValidBucketName(*x)) {
                         return absl::InvalidArgumentError(StrCat(
                             "Invalid GCS bucket name: ", QuoteString(*x)));
                       }
                       return absl::OkStatus();
                     }))),
      jb::Member(
          GcsRequestConcurrencyResource::id,
          jb::Projection<&GcsKeyValueStoreSpecData::request_concurrency>()),
      // `user_project` project ID to use for billing is obtained from the
      // `context` since it is not part of the identity of the resource being
      // accessed.
      jb::Member(GcsUserProjectResource::id,
                 jb::Projection<&GcsKeyValueStoreSpecData::user_project>()),
      jb::Member(GcsRequestRetries::id,
                 jb::Projection<&GcsKeyValueStoreSpecData::retries>()));
};

std::string GetGcsUrl(std::string_view bucket, std::string_view path) {
  return tensorstore::StrCat("gs://", bucket, "/",
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

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  Future<void> DeleteRange(KeyRange range) override;

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
    return (*auth_provider_)->GetAuthHeader();
  }

  const Executor& executor() const {
    return spec_.request_concurrency->executor;
  }

  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  std::string DescribeKey(std::string_view key) override {
    return GetGcsUrl(spec_.bucket, key);
  }

  // Wrap transport to allow our old mocking to work.
  Result<HttpResponse> IssueRequest(const char* description,
                                    const HttpRequest& request,
                                    const absl::Cord& payload) {
    auto result = transport_->IssueRequest(request, payload).result();
#ifdef TENSORSTORE_INTERNAL_GCS_LOG_REQUESTS
    if (result.ok()) {
      TENSORSTORE_LOG(description, " ", result->status_code, " ",
                      request.url());
    } else {
      TENSORSTORE_LOG(description, " ", result.status(), " ", request.url());
    }
#endif
#ifdef TENSORSTORE_INTERNAL_GCS_LOG_RESPONSES
    if (result.ok()) {
      for (auto& [key, value] : result->headers) {
        TENSORSTORE_LOG(description, ": ", key, ": ", value);
      }
      TENSORSTORE_LOG(description, ": Response: ", result->payload);
    }
#endif
    return result;
  }

  // https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff
  absl::Status RetryRequestWithBackoff(std::function<absl::Status()> function) {
    return internal::RetryWithBackoff(
        std::move(function), spec_.retries->max_retries,
        spec_.retries->initial_delay, spec_.retries->max_delay,
        spec_.retries->initial_delay, IsRetriable);
  }

  SpecData spec_;
  std::string resource_root_;  // bucket resource root.
  std::string upload_root_;    // bucket upload root.
  std::string encoded_user_project_;

  std::shared_ptr<HttpTransport> transport_;

  absl::Mutex auth_provider_mutex_;
  // Optional state indicates whether the provider has been obtained.  A nullptr
  // provider is valid and indicates to use anonymous access.
  std::optional<std::shared_ptr<internal_oauth2::AuthProvider>> auth_provider_;
};

Future<kvstore::DriverPtr> GcsKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<GcsKeyValueStore>();
  driver->spec_ = data_;
  driver->resource_root_ = BucketResourceRoot(data_.bucket);
  driver->upload_root_ = BucketUploadRoot(data_.bucket);
  driver->transport_ = internal_http::GetDefaultHttpTransport();
  if (const auto& project_id = data_.user_project->project_id) {
    driver->encoded_user_project_ =
        internal::PercentEncodeUriComponent(*project_id);
  }
  return driver;
}

/// A ReadTask is a function object used to satisfy a
/// GcsKeyValueStore::Read request.
struct ReadTask {
  IntrusivePtr<GcsKeyValueStore> owner;
  std::string resource;
  kvstore::ReadOptions options;

  Result<kvstore::ReadResult> operator()() {
    /// Reads contents of a GCS object.
    std::string media_url = StrCat(resource, "?alt=media");

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&media_url, true, "ifGenerationNotMatch",
                       options.if_not_equal);
    AddGenerationParam(&media_url, true, "ifGenerationMatch", options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&media_url, true, owner->encoded_user_project());
    kvstore::ReadResult read_result;

    // TODO: Configure timeouts.
    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
      TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());
      HttpRequestBuilder request_builder("GET", media_url);
      if (auth_header) request_builder.AddHeader(*auth_header);
      // For requests to buckets that are publicly readable, GCS may return a
      // stale response unless it is prohibited by the cache-control header.
      internal_http::AddStalenessBoundCacheControlHeader(
          request_builder, options.staleness_bound);
      if (options.byte_range.inclusive_min != 0 ||
          options.byte_range.exclusive_max) {
        request_builder.AddHeader(
            internal_http::GetRangeHeader(options.byte_range));
      }
      auto request = request_builder.EnableAcceptEncoding().BuildRequest();
      read_result.stamp.time = absl::Now();
      auto response = owner->IssueRequest("ReadTask", request, {});
      if (!response.ok()) return GetStatus(response);
      httpresponse = std::move(*response);
      switch (httpresponse.status_code) {
        // Special status codes handled outside the retry loop.
        case 412:
        case 404:
        case 304:
          return absl::OkStatus();
      }
      return HttpResponseCodeToStatus(httpresponse);
    });

    TENSORSTORE_RETURN_IF_ERROR(retry_status);

    // Parse `Date` header from response to correctly handle cached responses.
    // The GCS servers always send a `date` header.
    {
      absl::Time response_date;
      auto date_it = httpresponse.headers.find("date");
      if (date_it == httpresponse.headers.end()) {
        return absl::InvalidArgumentError("Missing \"date\" response header");
      }
      if (!absl::ParseTime(internal_http::kHttpTimeFormat, date_it->second,
                           &response_date, /*err=*/nullptr) ||
          response_date == absl::InfiniteFuture() ||
          response_date == absl::InfinitePast()) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Invalid \"date\" response header: ",
                                tensorstore::QuoteString(date_it->second)));
      }
      if (response_date < read_result.stamp.time) {
        if (options.staleness_bound < read_result.stamp.time &&
            response_date < options.staleness_bound) {
          // `response_date` does not satisfy the `staleness_bound` requirement,
          // possibly due to time skew.  Due to the way we compute `max-age` in
          // the request header, in the case of time skew it is correct to just
          // use `staleness_bound` instead.
          read_result.stamp.time = options.staleness_bound;
        } else {
          read_result.stamp.time = response_date;
        }
      }
    }

    switch (httpresponse.status_code) {
      case 204:
      case 404:
        // Object not found.
        read_result.stamp.generation = StorageGeneration::NoValue();
        read_result.state = kvstore::ReadResult::kMissing;
        return read_result;
      case 412:
        // "Failed precondition": indicates the ifGenerationMatch condition did
        // not hold.
        read_result.stamp.generation = StorageGeneration::Unknown();
        return read_result;
      case 304:
        // "Not modified": indicates that the ifGenerationNotMatch condition did
        // not hold.
        read_result.stamp.generation = options.if_not_equal;
        return read_result;
    }

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto byte_range,
        GetHttpResponseByteRange(httpresponse, options.byte_range));
    read_result.state = kvstore::ReadResult::kValue;
    read_result.value = internal::GetSubCord(httpresponse.payload, byte_range);

    // TODO: Avoid parsing the entire metadata & only extract the
    // generation field.
    ObjectMetadata metadata;
    SetObjectMetadataFromHeaders(httpresponse.headers, &metadata);

    read_result.stamp.generation =
        StorageGeneration::FromUint64(metadata.generation);
    return read_result;
  }
};

Future<kvstore::ReadResult> GcsKeyValueStore::Read(Key key,
                                                   ReadOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateObjectAndStorageGeneration(key, options.if_not_equal));

  auto encoded_object_name = internal::PercentEncodeUriComponent(key);
  std::string resource = tensorstore::internal::JoinPath(resource_root_, "/o/",
                                                         encoded_object_name);
  return MapFuture(executor(),
                   ReadTask{IntrusivePtr<GcsKeyValueStore>(this),
                            std::move(resource), std::move(options)});
}

/// A WriteTask is a function object used to satisfy a
/// GcsKeyValueStore::Write request.
struct WriteTask {
  using Value = kvstore::Value;

  IntrusivePtr<GcsKeyValueStore> owner;
  std::string encoded_object_name;
  Value value;
  kvstore::WriteOptions options;

  /// Writes an object to GCS.
  Result<TimestampedStorageGeneration> operator()() {
    // We use the SimpleUpload technique.

    std::string upload_url =
        StrCat(owner->upload_root(), "/o", "?uploadType=media",
               "&name=", encoded_object_name);

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&upload_url, true, "ifGenerationMatch",
                       options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&upload_url, true, owner->encoded_user_project());

    TimestampedStorageGeneration r;

    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
      TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());
      HttpRequestBuilder request_builder("POST", upload_url);
      if (auth_header) request_builder.AddHeader(*auth_header);
      auto request = request_builder  //
                         .AddHeader("Content-Type: application/octet-stream")
                         .AddHeader(StrCat("Content-Length: ", value.size()))
                         .BuildRequest();
      r.time = absl::Now();
      auto response = owner->IssueRequest("WriteTask", request, value);
      if (!response.ok()) return GetStatus(response);
      httpresponse = std::move(*response);
      switch (httpresponse.status_code) {
        case 304:
          // Not modified implies that the generation did not match.
          [[fallthrough]];
        case 412:
          // Failed precondition implies the generation did not match.
          return absl::OkStatus();
        case 404:
          if (!StorageGeneration::IsUnknown(options.if_equal) &&
              !StorageGeneration::IsNoValue(options.if_equal)) {
            return absl::OkStatus();
          }
          break;
        default:
          break;
      }
      return HttpResponseCodeToStatus(httpresponse);
    });

    TENSORSTORE_RETURN_IF_ERROR(retry_status);

    switch (httpresponse.status_code) {
      case 304:
        // Not modified implies that the generation did not match.
        [[fallthrough]];
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        return r;
      case 404:
        if (!StorageGeneration::IsUnknown(options.if_equal)) {
          r.generation = StorageGeneration::Unknown();
          return r;
        }
    }

    // TODO: Avoid parsing the entire metadata & only extract the
    // generation field.
    auto parsed_object_metadata =
        ParseObjectMetadata(httpresponse.payload.Flatten());
    TENSORSTORE_RETURN_IF_ERROR(parsed_object_metadata);

    r.generation =
        StorageGeneration::FromUint64(parsed_object_metadata->generation);

    return r;
  }
};

/// A DeleteTask is a function object used to satisfy a
/// GcsKeyValueStore::Delete request.
struct DeleteTask {
  IntrusivePtr<GcsKeyValueStore> owner;
  std::string resource;
  kvstore::WriteOptions options;

  /// Writes an object to GCS.
  Result<TimestampedStorageGeneration> operator()() {
    std::string delete_url = resource;

    // Add the ifGenerationNotMatch condition.
    bool has_query = AddGenerationParam(&delete_url, false, "ifGenerationMatch",
                                        options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&delete_url, has_query, owner->encoded_user_project());
    TimestampedStorageGeneration r;

    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
      TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());
      HttpRequestBuilder request_builder("DELETE", delete_url);
      if (auth_header) request_builder.AddHeader(*auth_header);
      auto request = request_builder  //
                         .BuildRequest();
      r.time = absl::Now();
      auto response = owner->IssueRequest("DeleteTask", request, {});
      if (!response.ok()) return GetStatus(response);
      httpresponse = std::move(*response);
      switch (httpresponse.status_code) {
        case 412:
          // Failed precondition implies the generation did not match.
          [[fallthrough]];
        case 404:
          return absl::OkStatus();
        default:
          break;
      }
      return HttpResponseCodeToStatus(httpresponse);
    });

    TENSORSTORE_RETURN_IF_ERROR(retry_status);

    switch (httpresponse.status_code) {
      case 412:
        // Failed precondition implies the generation did not match.
        r.generation = StorageGeneration::Unknown();
        break;
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
    return r;
  }
};

Future<TimestampedStorageGeneration> GcsKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateObjectAndStorageGeneration(key, options.if_equal));

  std::string encoded_object_name = internal::PercentEncodeUriComponent(key);
  if (value) {
    return MapFuture(
        executor(), WriteTask{IntrusivePtr<GcsKeyValueStore>(this),
                              std::move(encoded_object_name), std::move(*value),
                              std::move(options)});
  } else {
    std::string resource = tensorstore::internal::JoinPath(
        resource_root_, "/o/", encoded_object_name);
    return MapFuture(executor(),
                     DeleteTask{IntrusivePtr<GcsKeyValueStore>(this),
                                std::move(resource), std::move(options)});
  }
}

std::string BuildListQueryParameters(const KeyRange& range,
                                     std::optional<int> max_results) {
  std::string result;
  if (!range.inclusive_min.empty()) {
    result = StrCat("startOffset=",
                    internal::PercentEncodeUriComponent(range.inclusive_min));
  }
  if (!range.exclusive_max.empty()) {
    absl::StrAppend(&result, (result.empty() ? "" : "&"), "endOffset=",
                    internal::PercentEncodeUriComponent(range.exclusive_max));
  }
  if (max_results.has_value()) {
    absl::StrAppend(&result, (result.empty() ? "" : "&"),
                    "maxResults=", *max_results);
  }
  return result;
}

template <typename Receiver>
struct ListState : public internal::AtomicReferenceCount<ListState<Receiver>> {
  IntrusivePtr<GcsKeyValueStore> owner;
  Executor executor;
  std::string resource;
  std::string query_parameters;

  Receiver receiver;
  std::atomic<bool> cancelled{false};

  inline bool is_cancelled() {
    return cancelled.load(std::memory_order_relaxed);
  }

  // Helpers forward to the receiver.
  inline void set_starting() {
    execution::set_starting(
        receiver, [this] { cancelled.store(true, std::memory_order_relaxed); });
  }
};

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

template <typename Receiver>
struct ListOp {
  using State = ListState<Receiver>;
  IntrusivePtr<State> state;

  inline absl::Status maybe_cancelled() {
    return state->is_cancelled() ? absl::CancelledError("") : absl::OkStatus();
  }

  void operator()() {
    state->set_starting();
    auto status = Run();
    if (!status.ok() && !state->is_cancelled()) {
      if (absl::IsInvalidArgument(status)) {
        status = absl::InternalError(status.message());
      }
      execution::set_error(state->receiver, std::move(status));
      return;
    }

    // Either the request has been cancelled, or it has been completed.
    execution::set_done(state->receiver);
    execution::set_stopping(state->receiver);
  }

  absl::Status Run() {
    // Construct the base LIST url. This will be modified to
    // include the nextPageToken
    std::string base_list_url = state->resource;
    bool url_has_query = AddUserProjectParam(
        &base_list_url, false, state->owner->encoded_user_project());
    if (!state->query_parameters.empty()) {
      absl::StrAppend(&base_list_url, (url_has_query ? "&" : "?"),
                      state->query_parameters);
      url_has_query = true;
    }

    GcsListResponsePayload last_payload;
    while (true) {
      TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());

      std::string list_url = base_list_url;
      if (!last_payload.next_page_token.empty()) {
        absl::StrAppend(&list_url, (url_has_query ? "&" : "?"),
                        "pageToken=", last_payload.next_page_token);
      }

      HttpResponse httpresponse;
      auto retry_status = state->owner->RetryRequestWithBackoff([&] {
        TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());
        TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header,
                                     state->owner->GetAuthHeader());
        TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());
        HttpRequestBuilder request_builder("GET", list_url);
        if (auth_header) request_builder.AddHeader(*auth_header);
        auto request = request_builder.BuildRequest();
        auto response = state->owner->IssueRequest("List", request, {});
        if (!response.ok()) return GetStatus(response);
        httpresponse = std::move(*response);
        return HttpResponseCodeToStatus(httpresponse);
      });

      TENSORSTORE_RETURN_IF_ERROR(retry_status);
      TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());

      auto j = internal::ParseJson(httpresponse.payload.Flatten());
      if (j.is_discarded()) {
        return absl::InternalError(StrCat("Failed to parse response metadata: ",
                                          httpresponse.payload.Flatten()));
      }

      TENSORSTORE_ASSIGN_OR_RETURN(last_payload,
                                   jb::FromJson<GcsListResponsePayload>(
                                       j, GcsListResponsePayloadBinder));

      execution::set_value(state->receiver, std::move(last_payload.items));

      // Are we done yet?
      if (last_payload.next_page_token.empty()) {
        return absl::OkStatus();
      }
    }
  }
};

struct ListReceiver {
  AnyFlowReceiver<absl::Status, kvstore::Key> receiver;
  size_t strip_prefix_length;

  // set_value extracts the name from the object metadata.
  [[maybe_unused]] friend void set_value(ListReceiver& self,
                                         std::vector<ObjectMetadata> v) {
    for (auto& metadata : v) {
      metadata.name.erase(0, self.strip_prefix_length);
      execution::set_value(self.receiver, std::move(metadata.name));
    }
  }
  [[maybe_unused]] friend void set_done(ListReceiver& self) {
    execution::set_done(self.receiver);
  }
  [[maybe_unused]] friend void set_stopping(ListReceiver& self) {
    execution::set_stopping(self.receiver);
  }

  // Other methods just forward to the underlying receiver.
  template <typename CancelReceiver>
  friend void set_starting(ListReceiver& self, CancelReceiver cancel) {
    execution::set_starting(self.receiver, std::move(cancel));
  }
  template <typename E>
  friend void set_error(ListReceiver& self, E e) {
    execution::set_error(self.receiver, std::move(e));
  }
};

void GcsKeyValueStore::ListImpl(ListOptions options,
                                AnyFlowReceiver<absl::Status, Key> receiver) {
  using State = ListState<ListReceiver>;
  auto state = internal::MakeIntrusivePtr<State>();
  state->owner = IntrusivePtr<GcsKeyValueStore>(this);
  state->executor = executor();
  state->resource = tensorstore::internal::JoinPath(resource_root_, "/o");
  state->query_parameters =
      BuildListQueryParameters(options.range, std::nullopt);
  state->receiver.receiver = std::move(receiver);
  state->receiver.strip_prefix_length = options.strip_prefix_length;

  executor()(ListOp<ListReceiver>{std::move(state)});
}

// Receiver used by `DeleteRange` for processing the results from `List`.
struct DeleteRangeListReceiver {
  Promise<void> promise_;
  IntrusivePtr<GcsKeyValueStore> owner;
  FutureCallbackRegistration cancel_registration_;

  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration_ = promise_.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_value(std::string key) {
    LinkError(promise_, owner->Delete(std::move(key)));
  }

  void set_error(absl::Status error) {
    if (internal_future::FutureAccess::rep(promise_).LockResult()) {
      promise_.raw_result() = std::move(error);
    }
    promise_ = Promise<void>();
  }

  void set_done() { promise_ = Promise<void>(); }

  void set_stopping() { cancel_registration_.Unregister(); }
};

Future<void> GcsKeyValueStore::DeleteRange(KeyRange range) {
  // TODO(jbms): It could make sense to rate limit the list operation, so that
  // we don't get way ahead of the delete operations.  Currently our
  // sender/receiver abstraction does not support back pressure, though.
  auto [promise, future] =
      PromiseFuturePair<void>::Make(tensorstore::MakeResult());
  ListOptions list_options;
  list_options.range = std::move(range);
  ListImpl(list_options,
           DeleteRangeListReceiver{std::move(promise),
                                   IntrusivePtr<GcsKeyValueStore>(this)});
  return future;
}

Result<kvstore::Spec> ParseGcsUrl(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == "gs");
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
        StrCat("Invalid GCS bucket name: ", QuoteString(bucket)));
  }
  std::string_view encoded_path =
      (end_of_bucket == std::string_view::npos)
          ? std::string_view{}
          : parsed.authority_and_path.substr(end_of_bucket + 1);
  auto driver_spec = internal::MakeIntrusivePtr<GcsKeyValueStoreSpec>();
  driver_spec->data_.bucket = bucket;
  driver_spec->data_.request_concurrency =
      Context::Resource<GcsRequestConcurrencyResource>::DefaultSpec();
  driver_spec->data_.user_project =
      Context::Resource<GcsUserProjectResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<GcsRequestRetries>::DefaultSpec();
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
    url_scheme_registration{"gs", tensorstore::ParseGcsUrl};
}  // namespace
