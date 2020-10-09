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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/concurrency_resource_provider.h"
#include "tensorstore/internal/http/curl_handle.h"
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
#include "tensorstore/internal/retry.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/gcs/object_metadata.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/util/execution.h"
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

using tensorstore::internal_http::HttpRequest;
using tensorstore::internal_http::HttpRequestBuilder;
using tensorstore::internal_http::HttpResponse;
using tensorstore::internal_http::HttpTransport;
using tensorstore::internal_storage_gcs::ObjectMetadata;
using tensorstore::internal_storage_gcs::ParseObjectMetadata;

namespace tensorstore {
namespace {
namespace jb = tensorstore::internal::json_binding;

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
  static Result<Spec> Create(const Spec& spec,
                             internal::ContextResourceCreationContext context) {
    return spec;
  }
  static Spec GetSpec(const Spec& spec,
                      const internal::ContextSpecBuilder& builder) {
    return spec;
  }
};

/// Specifies a limit on the number of retries.
struct GcsRequestRetries
    : public internal::ContextResourceTraits<GcsRequestRetries> {
  static constexpr char id[] = "gcs_request_retries";
  struct Spec {
    int64_t max_retries = 32;
  };
  using Resource = Spec;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    return jb::Object(
        jb::Member("max_retries",
                   jb::Projection(&Spec::max_retries,
                                  jb::DefaultValue([](auto* v) { *v = 32; },
                                                   jb::Integer<int64_t>(1)))));
  }
  static Result<Spec> Create(const Spec& spec,
                             internal::ContextResourceCreationContext context) {
    return spec;
  }
  static Spec GetSpec(const Spec& spec,
                      const internal::ContextSpecBuilder& builder) {
    return spec;
  }
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
bool IsValidBucketName(absl::string_view bucket) {
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

  for (absl::string_view v : absl::StrSplit(bucket, absl::ByChar('.'))) {
    if (v.empty()) return false;
    if (v.size() > 63) return false;
    if (*v.begin() == '-') return false;
    if (*v.rbegin() == '-') return false;

    for (absl::string_view::size_type i = 0; i < v.size(); i++) {
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
bool IsValidObjectName(absl::string_view name) {
  if (name == "." || name == "..") return false;
  if (absl::StartsWith(name, ".well-known/acme-challenge")) return false;
  if (name.find('\r') != absl::string_view::npos) return false;
  if (name.find('\n') != absl::string_view::npos) return false;
  // TODO: Validate that object is a correct utf-8 string.
  return true;
}

/// Returns an error Status when either the object name or the StorageGeneration
/// are not legal values for the GCS storage backend.
Status ValidateObjectAndStorageGeneration(absl::string_view object,
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
                        absl::string_view param_name,
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
                         absl::string_view encoded_user_project) {
  if (!encoded_user_project.empty()) {
    absl::StrAppend(url, (has_query ? "&" : "?"),
                    "userProject=", encoded_user_project);
    return true;
  }
  return false;
}

/// Composes the resource root uri for the GCS API using the bucket
/// and constants for the host, api-version, etc.
std::string BucketResourceRoot(absl::string_view bucket) {
  const char kHostname[] = "www.googleapis.com";
  const char kVersion[] = "v1";
  return tensorstore::internal::JoinPath("https://", kHostname, "/storage/",
                                         kVersion, "/b/", bucket);
}

/// Composes the resource upload root uri for the GCS API using the bucket
/// and constants for the host, api-version, etc.
std::string BucketUploadRoot(absl::string_view bucket) {
  const char kHostname[] = "www.googleapis.com";
  const char kVersion[] = "v1";
  return tensorstore::internal::JoinPath(
      "https://", kHostname, "/upload/storage/", kVersion, "/b/", bucket);
}

/// Returns whether the Status is a retriable request.
bool IsRetriable(const tensorstore::Status& status) {
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

/// Implements the KeyValueStore interface for storing tensorstore data into a
/// GCS storage bucket.
class GcsKeyValueStore
    : public internal::RegisteredKeyValueStore<GcsKeyValueStore> {
 public:
  static constexpr char id[] = "gcs";

  template <template <typename T> class MaybeBound>
  struct SpecT {
    std::string bucket;
    MaybeBound<Context::ResourceSpec<GcsRequestConcurrencyResource>>
        request_concurrency;
    MaybeBound<Context::ResourceSpec<GcsUserProjectResource>> user_project;
    MaybeBound<Context::ResourceSpec<GcsRequestRetries>> retries;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(x.bucket, x.request_concurrency, x.user_project, x.retries);
    };
  };

  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  constexpr static auto json_binder = jb::Object(
      // Bucket is specified in the `spec` since it identifies the resource
      // being accessed.
      jb::Member("bucket",
                 jb::Projection(
                     &SpecData::bucket, jb::Validate([](const auto& options,
                                                        const std::string* x) {
                       if (!IsValidBucketName(*x)) {
                         return Status(absl::StatusCode::kInvalidArgument,
                                       StrCat("Invalid GCS bucket name: ",
                                              QuoteString(*x)));
                       }
                       return absl::OkStatus();
                     }))),
      jb::Member(GcsRequestConcurrencyResource::id,
                 jb::Projection(&SpecData::request_concurrency)),
      // `user_project` project ID to use for billing is obtained from the
      // `context` since it is not part of the identity of the resource being
      // accessed.
      jb::Member(GcsUserProjectResource::id,
                 jb::Projection(&SpecData::user_project)),
      jb::Member(GcsRequestRetries::id, jb::Projection(&SpecData::retries)));

  static void EncodeCacheKey(std::string* out, const BoundSpecData& spec) {
    internal::EncodeCacheKey(out, spec.bucket, spec.request_concurrency,
                             spec.user_project->project_id,
                             spec.retries->max_retries);
  }

  static Status ConvertSpec(SpecData* spec,
                            KeyValueStore::SpecRequestOptions options) {
    return absl::OkStatus();
  }

  using KeyValueStore::Key;
  using KeyValueStore::ReadResult;
  using Ptr = KeyValueStore::PtrT<GcsKeyValueStore>;

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

  void ListImpl(const ListOptions& options,
                AnyFlowReceiver<Status, Key> receiver) override;

  /// Returns the Auth header for a GCS request.
  Result<std::string> GetAuthHeader() {
    absl::MutexLock lock(&auth_provider_mutex_);
    if (!auth_provider_) {
      auto result =
          tensorstore::internal_oauth2::GetGoogleAuthProvider(transport_);
      TENSORSTORE_RETURN_IF_ERROR(result);
      auth_provider_ = std::move(*result);
    }
    return auth_provider_->GetAuthHeader();
  }

  const Executor& executor() const {
    return spec_.request_concurrency->executor;
  }

  static void Open(internal::KeyValueStoreOpenState<GcsKeyValueStore> state) {
    auto& d = state.driver();
    d.spec_ = state.spec();
    d.resource_root_ = BucketResourceRoot(d.spec_.bucket);
    d.upload_root_ = BucketUploadRoot(d.spec_.bucket);
    d.transport_ = internal_http::GetDefaultHttpTransport();
    if (const auto& user_project = d.spec_.user_project->project_id) {
      d.encoded_user_project_ =
          tensorstore::internal_http::CurlEscapeString(*user_project);
    }
  }

  Status GetBoundSpecData(BoundSpecData* spec) const {
    *spec = spec_;
    return absl::OkStatus();
  }

  std::string DescribeKey(absl::string_view key) override {
    return tensorstore::QuoteString(
        tensorstore::StrCat("gs://", spec_.bucket, "/", key));
  }

  // Wrap transport to allow our old mocking to work.
  Result<HttpResponse> IssueRequest(const char* description,
                                    const HttpRequest& request,
                                    const absl::Cord& payload) {
    auto result = transport_->IssueRequest(request, payload).result();
#if 0
  // If we want to log the URL & the response code, uncomment this.
  TENSORSTORE_LOG(description, " ", response.status_code, " ", request.url());
#endif
#if 0
  // If we want to log the entire request, uncomment this.
  TENSORSTORE_LOG(description, " ",
                  DumpRequestResponse(request, {}, response, {}));
#endif
    return result;
  }

  absl::Status RetryRequestWithBackoff(std::function<Status()> function) {
    return internal::RetryWithBackoff(
        std::move(function), spec_.retries->max_retries,
        absl::Milliseconds(100), absl::Seconds(5), IsRetriable);
  }

  BoundSpecData spec_;
  std::string resource_root_;  // bucket resource root.
  std::string upload_root_;    // bucket upload root.
  std::string encoded_user_project_;

  std::shared_ptr<HttpTransport> transport_;

  absl::Mutex auth_provider_mutex_;
  std::unique_ptr<internal_oauth2::AuthProvider> auth_provider_;
};

/// A ReadTask is a function object used to satisfy a
/// GcsKeyValueStore::Read request.
struct ReadTask {
  GcsKeyValueStore::Ptr owner;
  std::string resource;
  KeyValueStore::ReadOptions options;

  Result<KeyValueStore::ReadResult> operator()() {
    TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());

    /// Reads contents of a GCS object.
    std::string media_url = StrCat(resource, "?alt=media");

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&media_url, true, "ifGenerationNotMatch",
                       options.if_not_equal);
    AddGenerationParam(&media_url, true, "ifGenerationMatch", options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&media_url, true, owner->encoded_user_project());

    HttpRequestBuilder request_builder(media_url);
    if (options.byte_range.inclusive_min != 0 ||
        options.byte_range.exclusive_max) {
      request_builder.AddHeader(
          internal_http::GetRangeHeader(options.byte_range));
    }
    auto request = request_builder.EnableAcceptEncoding()
                       .AddHeader(auth_header)
                       .BuildRequest();
    KeyValueStore::ReadResult read_result;

    // TODO: Configure timeouts.
    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
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
    switch (httpresponse.status_code) {
      case 204:
      case 404:
        // Object not found.
        read_result.stamp.generation = StorageGeneration::NoValue();
        read_result.state = KeyValueStore::ReadResult::kMissing;
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
    read_result.state = KeyValueStore::ReadResult::kValue;
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

Future<KeyValueStore::ReadResult> GcsKeyValueStore::Read(Key key,
                                                         ReadOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateObjectAndStorageGeneration(key, options.if_not_equal));

  auto encoded_object_name = tensorstore::internal_http::CurlEscapeString(key);
  std::string resource = tensorstore::internal::JoinPath(resource_root_, "/o/",
                                                         encoded_object_name);
  return MapFuture(executor(),
                   ReadTask{GcsKeyValueStore::Ptr(this), std::move(resource),
                            std::move(options)});
}

/// A WriteTask is a function object used to satisfy a
/// GcsKeyValueStore::Write request.
struct WriteTask {
  using Value = KeyValueStore::Value;

  GcsKeyValueStore::Ptr owner;
  std::string encoded_object_name;
  Value value;
  KeyValueStore::WriteOptions options;

  /// Writes an object to GCS.
  Result<TimestampedStorageGeneration> operator()() {
    // We use the SimpleUpload technique.
    TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());

    std::string upload_url =
        StrCat(owner->upload_root(), "/o", "?uploadType=media",
               "&name=", encoded_object_name);

    // Add the ifGenerationNotMatch condition.
    AddGenerationParam(&upload_url, true, "ifGenerationMatch",
                       options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&upload_url, true, owner->encoded_user_project());

    auto request = HttpRequestBuilder(upload_url)
                       .AddHeader(auth_header)
                       .AddHeader("Content-Type: application/octet-stream")
                       .AddHeader(StrCat("Content-Length: ", value.size()))
                       .BuildRequest();

    TimestampedStorageGeneration r;

    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
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
          if (!StorageGeneration::IsUnknown(options.if_equal)) {
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
  GcsKeyValueStore::Ptr owner;
  std::string resource;
  KeyValueStore::WriteOptions options;

  /// Writes an object to GCS.
  Result<TimestampedStorageGeneration> operator()() {
    TENSORSTORE_ASSIGN_OR_RETURN(auto auth_header, owner->GetAuthHeader());

    std::string delete_url = resource;

    // Add the ifGenerationNotMatch condition.
    bool has_query = AddGenerationParam(&delete_url, false, "ifGenerationMatch",
                                        options.if_equal);

    // Assume that if the user_project field is set, that we want to provide
    // it on the uri for a requestor pays bucket.
    AddUserProjectParam(&delete_url, has_query, owner->encoded_user_project());

    auto request = HttpRequestBuilder(delete_url)
                       .AddHeader(auth_header)
                       .SetMethod("DELETE")
                       .BuildRequest();

    TimestampedStorageGeneration r;

    HttpResponse httpresponse;
    auto retry_status = owner->RetryRequestWithBackoff([&] {
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

  std::string encoded_object_name =
      tensorstore::internal_http::CurlEscapeString(key);
  if (value) {
    return MapFuture(
        executor(),
        WriteTask{GcsKeyValueStore::Ptr(this), std::move(encoded_object_name),
                  std::move(*value), std::move(options)});
  } else {
    std::string resource = tensorstore::internal::JoinPath(
        resource_root_, "/o/", encoded_object_name);
    return MapFuture(
        executor(), DeleteTask{GcsKeyValueStore::Ptr(this), std::move(resource),
                               std::move(options)});
  }
}

std::string BuildListQueryParameters(const KeyRange& range,
                                     absl::optional<int> max_results) {
  std::string result;
  if (!range.inclusive_min.empty()) {
    result = StrCat(
        "startOffset=",
        tensorstore::internal_http::CurlEscapeString(range.inclusive_min));
  }
  if (!range.exclusive_max.empty()) {
    absl::StrAppend(
        &result, (result.empty() ? "" : "&"), "endOffset=",
        tensorstore::internal_http::CurlEscapeString(range.exclusive_max));
  }
  if (max_results.has_value()) {
    absl::StrAppend(&result, (result.empty() ? "" : "&"),
                    "maxResults=", *max_results);
  }
  return result;
}

template <typename Receiver>
struct ListState : public internal::AtomicReferenceCount<ListState<Receiver>> {
  GcsKeyValueStore::Ptr owner;
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

template <typename Receiver>
struct ListOp {
  using State = ListState<Receiver>;
  internal::IntrusivePtr<State> state;

  inline Status maybe_cancelled() {
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

  Status Run() {
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

    // The nextPageToken is used to page through the list results.
    std::string nextPageToken;

    while (true) {
      TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());

      TENSORSTORE_ASSIGN_OR_RETURN(auto header, state->owner->GetAuthHeader());
      TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());

      std::string list_url = base_list_url;
      if (!nextPageToken.empty()) {
        absl::StrAppend(&list_url, (url_has_query ? "&" : "?"),
                        "pageToken=", nextPageToken);
      }

      auto request =
          HttpRequestBuilder(list_url).AddHeader(header).BuildRequest();

      HttpResponse httpresponse;
      auto retry_status = state->owner->RetryRequestWithBackoff([&] {
        TENSORSTORE_RETURN_IF_ERROR(maybe_cancelled());
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

      nextPageToken.clear();
      TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
          j, "nextPageToken", [&nextPageToken](const ::nlohmann::json& value) {
            return internal::JsonRequireValueAs(value, &nextPageToken);
          }));

      std::vector<ObjectMetadata> items;
      TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
          j, "items", [&items](const ::nlohmann::json& value) {
            return internal::JsonParseArray(
                value,  //
                [&items](std::ptrdiff_t i) {
                  items.reserve(i);
                  return absl::OkStatus();
                },
                [&items](const ::nlohmann::json& item, std::ptrdiff_t) {
                  ObjectMetadata metadata;
                  SetObjectMetadata(item, &metadata);
                  items.emplace_back(std::move(metadata));
                  return absl::OkStatus();
                });
          }));

      execution::set_value(state->receiver, std::move(items));

      // Are we done yet?
      if (nextPageToken.empty()) {
        return absl::OkStatus();
      }
    }
  }
};

struct ListReceiver {
  AnyFlowReceiver<Status, KeyValueStore::Key> receiver;

  // set_value extracts the name from the object metadata.
  friend void set_value(ListReceiver& self, std::vector<ObjectMetadata> v) {
    for (auto& metadata : v) {
      execution::set_value(self.receiver, std::move(metadata.name));
    }
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
  friend void set_done(ListReceiver& self) {
    execution::set_done(self.receiver);
  }
  friend void set_stopping(ListReceiver& self) {
    execution::set_stopping(self.receiver);
  }
};

void GcsKeyValueStore::ListImpl(const ListOptions& options,
                                AnyFlowReceiver<Status, Key> receiver) {
  using State = ListState<ListReceiver>;
  internal::IntrusivePtr<State> state(new State);
  state->owner = GcsKeyValueStore::Ptr{this};
  state->executor = executor();
  state->resource = tensorstore::internal::JoinPath(resource_root_, "/o");
  state->query_parameters =
      BuildListQueryParameters(options.range, absl::nullopt);
  state->receiver.receiver = std::move(receiver);

  executor()(ListOp<ListReceiver>{state});
}

const internal::KeyValueStoreDriverRegistration<GcsKeyValueStore> registration;

}  // namespace
}  // namespace tensorstore
