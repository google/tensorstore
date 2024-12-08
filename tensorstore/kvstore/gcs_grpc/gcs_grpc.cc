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

// NOTE: This is an experimental driver which uses the gRPC protocol
// for Google GCS. Also note, this feature is not GA and accounts need to
// be enrolled in early access for support; contact your account manager.
//
// https://googleapis.dev/cpp/google-cloud-storage/latest/storage-grpc.html

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/gcs/gcs_resource.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_grpc/get_credentials.h"
#include "tensorstore/kvstore/gcs_grpc/state.h"
#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"
#include "tensorstore/kvstore/gcs_grpc/use_directpath.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generic_coalescing_batch_util.h"
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

// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/context_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/enum.h"  // IWYU pragma: keep
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"  // IWYU pragma: keep

// protos
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::GrpcStatusToAbslStatus;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_gcs_grpc::GetCredentialsForEndpoint;
using ::tensorstore::internal_gcs_grpc::GetSharedStorageStubPool;
using ::tensorstore::internal_gcs_grpc::StorageStubPool;
using ::tensorstore::internal_storage_gcs::GcsUserProjectResource;
using ::tensorstore::internal_storage_gcs::IsRetriable;
using ::tensorstore::internal_storage_gcs::IsValidBucketName;
using ::tensorstore::internal_storage_gcs::IsValidObjectName;
using ::tensorstore::internal_storage_gcs::IsValidStorageGeneration;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::SupportedFeatures;

using ::google::storage::v2::DeleteObjectRequest;
using ::google::storage::v2::ListObjectsRequest;
using ::google::storage::v2::ListObjectsResponse;
using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::ReadObjectResponse;
using ::google::storage::v2::ServiceConstants;
using ::google::storage::v2::WriteObjectRequest;
using ::google::storage::v2::WriteObjectResponse;
using ::google::storage::v2::Storage;

namespace {
static constexpr char kUriScheme[] = "gcs_grpc";
static constexpr size_t kMaxWriteBytes =
    ServiceConstants::MAX_WRITE_CHUNK_BYTES;
}  // namespace

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

struct GcsMetrics : public internal_kvstore::CommonMetrics {
  internal_metrics::Counter<int64_t>& retries;
  // no additional members
};

auto gcs_grpc_metrics = []() -> GcsMetrics {
  return {TENSORSTORE_KVSTORE_COMMON_METRICS(gcs_grpc),
          TENSORSTORE_KVSTORE_COUNTER_IMPL(
              gcs_grpc, retries,
              "Ccunt of all retried requests (read/write/delete)")};
}();

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

struct GcsGrpcKeyValueStoreSpecData {
  std::string bucket;
  std::string endpoint;
  uint32_t num_channels = 0;
  absl::Duration timeout = absl::ZeroDuration();
  absl::Duration wait_for_connection = absl::ZeroDuration();
  Context::Resource<GcsUserProjectResource> user_project;
  Context::Resource<internal_storage_gcs::GcsRequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.bucket, x.endpoint, x.num_channels, x.timeout,
             x.wait_for_connection, x.user_project, x.retries,
             x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member(
          "bucket",
          jb::Projection<&GcsGrpcKeyValueStoreSpecData::bucket>(
              jb::Validate([](const auto& options, const std::string* x) {
                if (!IsValidBucketName(*x)) {
                  return absl::InvalidArgumentError(tensorstore::StrCat(
                      "Invalid GCS bucket name: ", QuoteString(*x)));
                }
                return absl::OkStatus();
              }))),
      jb::Member("endpoint",
                 jb::Projection<&GcsGrpcKeyValueStoreSpecData::endpoint>(
                     jb::DefaultInitializedValue())),
      jb::Member("num_channels",
                 jb::Projection<&GcsGrpcKeyValueStoreSpecData::num_channels>(
                     jb::DefaultInitializedValue())),
      jb::Member("timeout",
                 jb::Projection<&GcsGrpcKeyValueStoreSpecData::timeout>(
                     jb::DefaultValue<jb::kNeverIncludeDefaults>(
                         [](auto* x) { *x = absl::ZeroDuration(); }))),
      jb::Member(
          "wait_for_connection",
          jb::Projection<&GcsGrpcKeyValueStoreSpecData::wait_for_connection>(
              jb::DefaultValue<jb::kNeverIncludeDefaults>(
                  [](auto* x) { *x = absl::ZeroDuration(); }))),
      jb::Member(GcsUserProjectResource::id,
                 jb::Projection<&GcsGrpcKeyValueStoreSpecData::user_project>()),
      jb::Member(internal_storage_gcs::GcsRequestRetries::id,
                 jb::Projection<&GcsGrpcKeyValueStoreSpecData::retries>()),
      jb::Member(
          DataCopyConcurrencyResource::id,
          jb::Projection<
              &GcsGrpcKeyValueStoreSpecData::data_copy_concurrency>()), /**/
      jb::DiscardExtraMembers);
};

class GcsGrpcKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          GcsGrpcKeyValueStoreSpec, GcsGrpcKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "gcs_grpc";
  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status NormalizeSpec(std::string& path) override {
    if (!path.empty() && !IsValidObjectName(path)) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Invalid GCS path: ", QuoteString(path)));
    }
    return absl::OkStatus();
  }

  Result<std::string> ToUrl(std::string_view path) const override {
    if (!data_.endpoint.empty()) {
      return absl::UnimplementedError(
          "URL representation does not support test endpoints");
    }
    return tensorstore::StrCat(kUriScheme, "://", data_.bucket, "/",
                               internal::PercentEncodeUriPath(path));
  }
};

/// Defines the "gcs_grpc" KeyValueStore driver.
class GcsGrpcKeyValueStore
    : public internal_kvstore::RegisteredDriver<GcsGrpcKeyValueStore,
                                                GcsGrpcKeyValueStoreSpec> {
 public:
  internal_kvstore_batch::CoalescingOptions GetBatchReadCoalescingOptions()
      const {
    return internal_kvstore_batch::kDefaultRemoteStorageCoalescingOptions;
  }

  /// Key value store operations.
  Future<ReadResult> Read(Key key, ReadOptions options) override;
  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  Future<const void> DeleteRange(KeyRange range) override;

  /// Obtains a `SpecData` representation from an open `Driver`.
  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
           SupportedFeatures::kAtomicWriteWithoutOverwrite;
  }

  const Executor& executor() const {
    return spec_.data_copy_concurrency->executor;
  }

  // Bucket names are in the form: 'projects/{project-id}/buckets/{bucket-id}'
  std::string bucket_name() { return bucket_; }

  std::shared_ptr<Storage::StubInterface> get_stub() {
    return storage_stub_pool_->get_next_stub();
  }

  std::unique_ptr<grpc::ClientContext> AllocateContext() {
    auto context = std::make_unique<grpc::ClientContext>();

    // For a requestor-pays bucket we need to set x-goog-user-project.
    if (spec_.user_project->project_id &&
        !spec_.user_project->project_id->empty()) {
      context->AddMetadata("x-goog-user-project",
                           *spec_.user_project->project_id);
    }

    // gRPC requests need to have routing parameters added.
    context->AddMetadata("x-goog-request-params",
                         absl::StrFormat("bucket=%s", bucket_name()));

    // NOTE: Evaluate this a bit more?
    // context.set_wait_for_ready(false);
    if (spec_.timeout > absl::ZeroDuration() &&
        spec_.timeout < absl::InfiniteDuration()) {
      context->set_deadline(absl::ToChronoTime(absl::Now() + spec_.timeout));
    }

    if (call_credentials_fn_) {
      // The gRPC credentials model includes per-channel credentials,
      // `ChannelCredentials`, and per-request credentials,
      // `CallCredentials`. Instead of using a composite credentials object,
      // set the CallCredentials on each request which allows using a shared
      // channel pool.
      context->set_credentials(call_credentials_fn_());
    }
    return context;
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
    gcs_grpc_metrics.retries.Increment();
    ScheduleAt(absl::Now() + *delay,
               WithExecutor(executor(), [task = internal::IntrusivePtr<Task>(
                                             task)] { task->Retry(); }));

    return absl::OkStatus();
  }

  SpecData spec_;
  std::string bucket_;
  std::shared_ptr<StorageStubPool> storage_stub_pool_;
  std::function<std::shared_ptr<grpc::CallCredentials>()> call_credentials_fn_;
};

////////////////////////////////////////////////////

// Implements GcsGrpcKeyValueStore::Read
// rpc ReadObject(ReadObjectRequest) returns (stream ReadObjectResponse) {}
struct ReadTask : public internal::AtomicReferenceCount<ReadTask>,
                  public grpc::ClientReadReactor<ReadObjectResponse> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  internal_gcs_grpc::ReadState state_;
  Promise<kvstore::ReadResult> promise_;

  // working state.
  ReadObjectRequest request_;
  ReadObjectResponse response_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  ReadTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
           kvstore::ReadOptions options, Promise<kvstore::ReadResult> promise)
      : driver_(std::move(driver)),
        state_(std::move(options)),
        promise_(std::move(promise)) {}

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  void Start(const std::string& object_name) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging) << "ReadTask " << object_name;

    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->TryCancel(); });

    request_.set_bucket(driver_->bucket_name());
    request_.set_object(object_name);
    state_.SetupRequest(request_);

    Retry();
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }

    // Retry always starts from a "clean" request, so clear the value.
    state_.ResetWorkingState();

    {
      absl::MutexLock lock(&mutex_);
      assert(context_ == nullptr);
      context_ = driver_->AllocateContext();
      auto stub = driver_->get_stub();

      // Start a call.
      intrusive_ptr_increment(this);  // adopted in OnDone.
      stub->async()->ReadObject(context_.get(), &request_, this);
    }

    StartRead(&response_);
    StartCall();
  }

  void OnReadDone(bool ok) override {
    if (!ok) return;
    if (!promise_.result_needed()) {
      TryCancel();
      return;
    }
    if (response_.has_checksummed_data()) {
      gcs_grpc_metrics.bytes_read.IncrementBy(
          response_.checksummed_data().content().size());
    }
    if (auto status = state_.HandleResponse(response_); !status.ok()) {
      promise_.SetResult(status);
      TryCancel();
      return;
    }

    // Issue next request, if necessary.
    StartRead(&response_);
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<ReadTask> self(this, internal::adopt_object_ref);
    driver_->executor()(
        [self = std::move(self), status = GrpcStatusToAbslStatus(s)]() {
          self->ReadFinished(std::move(status));
        });
  }

  void ReadFinished(absl::Status status) {
    // Streaming read complete.
    if (!promise_.result_needed()) {
      return;
    }
    {
      absl::MutexLock lock(&mutex_);
      context_ = nullptr;
    }

    auto latency = state_.GetLatency();
    gcs_grpc_metrics.read_latency_ms.Observe(
        absl::ToInt64Milliseconds(latency));

    if (!status.ok() && attempt_ == 0 &&
        status.code() == absl::StatusCode::kUnauthenticated) {
      // Allow a single unauthenticated error.
      attempt_++;
      Retry();
      return;
    }
    if (!status.ok() && IsRetriable(status)) {
      status =
          driver_->BackoffForAttemptAsync(std::move(status), attempt_++, this);
      if (status.ok()) {
        return;
      }
    }

    promise_.SetResult(state_.HandleFinalStatus(status));
  }
};

// Implements GcsGrpcKeyValueStore::Write
// rpc ReadObject(ReadObjectRequest) returns (stream ReadObjectResponse) {}
// rpc StartResumableWrite(StartResumableWriteRequest) returns
// (StartResumableWriteResponse) {}
struct WriteTask : public internal::AtomicReferenceCount<WriteTask>,
                   public grpc::ClientWriteReactor<WriteObjectRequest> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  internal_gcs_grpc::WriteState state_;
  Promise<TimestampedStorageGeneration> promise_;
  std::string object_name_;

  // working state.
  WriteObjectRequest request_;
  WriteObjectResponse response_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  WriteTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
            kvstore::WriteOptions options, absl::Cord data,
            Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)),
        state_(std::move(options), std::move(data)),
        promise_(std::move(promise)) {}

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  void Start(std::string object_name) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging) << "WriteTask " << object_name;

    object_name_ = std::move(object_name);
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<WriteTask>(
                                       this)] { self->TryCancel(); });
    Retry();
  }

  // Retry/Continue a call.
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }

    // Retry always starts from a "clean" request, so clear the value.
    request_.Clear();
    state_.ResetWorkingState();

    // First request, make sure that the spec is setup correctly.
    auto& resource = *request_.mutable_write_object_spec()->mutable_resource();
    resource.set_bucket(driver_->bucket_name());
    resource.set_name(object_name_);

    {
      absl::MutexLock lock(&mutex_);
      assert(context_ == nullptr);
      context_ = driver_->AllocateContext();
      auto stub = driver_->get_stub();
      // Initiate the write.
      intrusive_ptr_increment(this);
      stub->async()->WriteObject(context_.get(), &response_, this);
    }

    state_.UpdateRequestForNextWrite(request_);

    auto options = grpc::WriteOptions();
    if (request_.finish_write()) {
      options.set_last_message();
    }
    StartWrite(&request_, options);
    StartCall();
  }

  void OnWriteDone(bool ok) override {
    // Not streaming any additional data bits.
    if (!ok) return;
    if (request_.finish_write()) return;

    state_.UpdateRequestForNextWrite(request_);

    auto options = grpc::WriteOptions();
    if (request_.finish_write()) {
      options.set_last_message();
    }
    StartWrite(&request_, options);
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<WriteTask> self(this, internal::adopt_object_ref);
    driver_->executor()(
        [self = std::move(self), status = GrpcStatusToAbslStatus(s)] {
          self->WriteFinished(std::move(status));
        });
  }

  void WriteFinished(absl::Status status) {
    if (!promise_.result_needed()) {
      return;
    }

    auto latency = state_.GetLatency();
    gcs_grpc_metrics.write_latency_ms.Observe(
        absl::ToInt64Milliseconds(latency));
    {
      absl::MutexLock lock(&mutex_);
      context_ = nullptr;
    }

    if (!status.ok() && attempt_ == 0 &&
        status.code() == absl::StatusCode::kUnauthenticated) {
      // Allow a single unauthenticated error.
      attempt_++;
      Retry();
      return;
    }

    if (!status.ok() && IsRetriable(status)) {
      status =
          driver_->BackoffForAttemptAsync(std::move(status), attempt_++, this);
      if (status.ok()) {
        return;
      }
    }

    promise_.SetResult(state_.HandleFinalStatus(status, response_));
  }
};

// Implements GcsGrpcKeyValueStore::Delete
// rpc DeleteObject(DeleteObjectRequest) returns (google.protobuf.Empty) {}
struct DeleteTask : public internal::AtomicReferenceCount<DeleteTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::WriteOptions options_;
  Promise<TimestampedStorageGeneration> promise_;

  // Working state
  absl::Time start_time_;
  DeleteObjectRequest request_;
  ::google::protobuf::Empty response_;
  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  void Start(const std::string& object_name) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging) << "DeleteTask " << object_name;

    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<DeleteTask>(
                                       this)] { self->TryCancel(); });

    request_.set_bucket(driver_->bucket_name());
    request_.set_object(object_name);
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      auto gen =
          StorageGeneration::ToUint64(options_.generation_conditions.if_equal);
      request_.set_if_generation_match(gen);
    }
    Retry();
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }

    start_time_ = absl::Now();
    {
      absl::MutexLock lock(&mutex_);
      assert(context_ == nullptr);
      context_ = driver_->AllocateContext();
      auto stub = driver_->get_stub();

      intrusive_ptr_increment(this);  // Adopted by OnDone
      stub->async()->DeleteObject(
          context_.get(), &request_, &response_,
          WithExecutor(driver_->executor(), [this](::grpc::Status s) {
            internal::IntrusivePtr<DeleteTask> self(this,
                                                    internal::adopt_object_ref);
            self->DeleteFinished(GrpcStatusToAbslStatus(s));
          }));
    }
  }

  void DeleteFinished(absl::Status status) {
    if (!promise_.result_needed()) {
      return;
    }

    {
      absl::MutexLock lock(&mutex_);
      context_ = nullptr;
    }

    if (!status.ok() && attempt_ == 0 &&
        status.code() == absl::StatusCode::kUnauthenticated) {
      // Allow a single unauthenticated error.
      attempt_++;
      Retry();
      return;
    }
    if (!status.ok() && IsRetriable(status)) {
      status =
          driver_->BackoffForAttemptAsync(std::move(status), attempt_++, this);
      if (status.ok()) {
        return;
      }
    }

    TimestampedStorageGeneration r;
    r.time = start_time_;
    r.generation = StorageGeneration::NoValue();
    if (absl::IsFailedPrecondition(status)) {
      // precondition did not match.
      r.generation = StorageGeneration::Unknown();
    } else if (absl::IsNotFound(status)) {
      // object missing; that's probably ok.
      if (!options_.generation_conditions.MatchesNoValue()) {
        r.generation = StorageGeneration::Unknown();
      }
    } else if (!status.ok()) {
      promise_.SetResult(std::move(status));
      return;
    }
    promise_.SetResult(std::move(r));
  }
};

// Implements GcsGrpcKeyValueStore::List
// rpc ListObjects(ListObjectsRequest) returns (ListObjectsResponse) {}
struct ListTask : public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::ListOptions options_;
  ListReceiver receiver_;

  // working state.
  std::shared_ptr<Storage::StubInterface> stub_;
  ListObjectsRequest request;
  ListObjectsResponse response;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  bool cancelled_ ABSL_GUARDED_BY(mutex_) = false;

  ListTask(internal::IntrusivePtr<GcsGrpcKeyValueStore>&& driver,
           kvstore::ListOptions&& options, ListReceiver&& receiver)
      : driver_(std::move(driver)),
        options_(std::move(options)),
        receiver_(std::move(receiver)) {
    // Start a call.
    execution::set_starting(receiver_, [this] { TryCancel(); });
  }

  ~ListTask() {
    {
      absl::MutexLock l(&mutex_);
      context_ = nullptr;
    }
    driver_ = {};
    execution::set_stopping(receiver_);
  }

  bool is_cancelled() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock l(&mutex_);
    return cancelled_;
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock l(&mutex_);
    if (!cancelled_) {
      cancelled_ = true;
      if (context_) context_->TryCancel();
    }
  }

  void Start() {
    ABSL_LOG_IF(INFO, gcs_grpc_logging) << "ListTask " << options_.range;

    request.set_lexicographic_start(options_.range.inclusive_min);
    request.set_lexicographic_end(options_.range.exclusive_max);
    request.set_parent(driver_->bucket_name());
    request.set_page_size(1000);  // maximum.

    Retry();
  }

  // Retry/Continue a call.
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (is_cancelled()) {
      execution::set_done(receiver_);
      return;
    }

    {
      absl::MutexLock lock(&mutex_);
      context_ = driver_->AllocateContext();
      stub_ = driver_->get_stub();

      intrusive_ptr_increment(this);
      stub_->async()->ListObjects(
          context_.get(), &request, &response,
          WithExecutor(driver_->executor(), [this](::grpc::Status s) {
            internal::IntrusivePtr<ListTask> self(this,
                                                  internal::adopt_object_ref);
            self->ListFinished(GrpcStatusToAbslStatus(s));
          }));
    }
  }

  void ListFinished(absl::Status status) {
    if (is_cancelled()) {
      execution::set_done(receiver_);
      return;
    }

    if (!status.ok() && IsRetriable(status)) {
      status =
          driver_->BackoffForAttemptAsync(std::move(status), attempt_++, this);
      if (status.ok()) {
        return;
      }
    }

    if (!status.ok()) {
      execution::set_error(receiver_, std::move(status));
      return;
    }

    bool done = false;
    for (const auto& o : response.objects()) {
      if (is_cancelled()) {
        done = true;
        break;
      }
      std::string_view name = o.name();
      if (!Contains(options_.range, name)) {
        if (KeyRange::CompareKeyAndExclusiveMax(
                name, options_.range.exclusive_max) >= 0) {
          done = true;
          break;
        }
        continue;
      }
      if (options_.strip_prefix_length) {
        name = name.substr(options_.strip_prefix_length);
      }
      execution::set_value(receiver_, ListEntry{
                                          std::string(name),
                                          ListEntry::checked_size(o.size()),
                                      });
    }
    if (!done && !response.next_page_token().empty()) {
      // If there is a continuation token, issue the next request.
      request.set_page_token(response.next_page_token());
      response.Clear();
      attempt_ = 0;
      Retry();
      return;
    }

    execution::set_done(receiver_);
  }
};

// Receiver used by `DeleteRange` for processing the results from `List`.
struct DeleteRangeListReceiver {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  Promise<void> promise_;
  FutureCallbackRegistration cancel_registration_;

  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration_ = promise_.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_value(ListEntry entry) {
    assert(!entry.key.empty());
    if (!entry.key.empty()) {
      LinkError(promise_, driver_->Delete(std::move(entry.key)));
    }
  }

  void set_error(absl::Status error) {
    SetDeferredResult(promise_, std::move(error));
    promise_ = Promise<void>();
  }

  void set_done() { promise_ = Promise<void>(); }

  void set_stopping() {
    cancel_registration_.Unregister();
    driver_ = {};
  }
};

////////////////////////////////////////////////////

/// Key value store operations.
Future<kvstore::ReadResult> GcsGrpcKeyValueStore::Read(Key key,
                                                       ReadOptions options) {
  gcs_grpc_metrics.read.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid blob object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal) ||
      !IsValidStorageGeneration(options.generation_conditions.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> GcsGrpcKeyValueStore::ReadImpl(
    Key&& key, ReadOptions&& options) {
  gcs_grpc_metrics.batch_read.Increment();
  auto op = PromiseFuturePair<ReadResult>::Make();

  auto task = internal::MakeIntrusivePtr<ReadTask>(
      internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), std::move(options),
      std::move(op.promise));
  task->Start(key);
  return std::move(op.future);
}

Future<TimestampedStorageGeneration> GcsGrpcKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  gcs_grpc_metrics.write.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid blob object name");
  }
  if (!IsValidStorageGeneration(options.generation_conditions.if_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }

  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();
  if (!value) {
    auto task = internal::MakeIntrusivePtr<DeleteTask>();
    task->driver_ = internal::IntrusivePtr<GcsGrpcKeyValueStore>(this);
    task->options_ = std::move(options);
    task->promise_ = std::move(op.promise);
    task->Start(key);
  } else {
    auto task = internal::MakeIntrusivePtr<WriteTask>(
        internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), std::move(options),
        *std::move(value), std::move(op.promise));
    task->Start(key);
  }
  return std::move(op.future);
}

void GcsGrpcKeyValueStore::ListImpl(ListOptions options,
                                    ListReceiver receiver) {
  gcs_grpc_metrics.list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }

  auto task = internal::MakeIntrusivePtr<ListTask>(
      internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), std::move(options),
      std::move(receiver));
  task->Start();
}

Future<const void> GcsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  gcs_grpc_metrics.delete_range.Increment();
  if (range.empty()) return absl::OkStatus();

  // TODO(jbms): It could make sense to rate limit the list operation, so that
  // we don't get way ahead of the delete operations.  Currently the
  // sender/receiver abstraction does not support back pressure, though.
  auto op = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
  ListOptions list_options;
  list_options.range = std::move(range);
  ListImpl(list_options, DeleteRangeListReceiver{
                             internal::IntrusivePtr<GcsGrpcKeyValueStore>(this),
                             std::move(op.promise)});
  return std::move(op.future);
}

Future<kvstore::DriverPtr> GcsGrpcKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<GcsGrpcKeyValueStore>();
  driver->spec_ = data_;
  driver->bucket_ = absl::StrFormat("projects/_/buckets/%s", data_.bucket);

  // Use direct path endpoint by default when running on a GCP machine.
  // https://github.com/googleapis/google-cloud-cpp/google/cloud/storage/internal/grpc/default_options.cc
  std::string endpoint = data_.endpoint;
  if (endpoint.empty()) {
    if (internal_gcs_grpc::UseDirectPathGcsEndpointByDefault()) {
      endpoint = "google-c2p:///storage.googleapis.com";
    } else {
      endpoint = "dns:///storage.googleapis.com";
    }
  }

  auto channel_credentials =
      GetCredentialsForEndpoint(endpoint, driver->call_credentials_fn_);
  driver->storage_stub_pool_ = GetSharedStorageStubPool(
      endpoint, data_.num_channels, std::move(channel_credentials),
      driver->spec_.wait_for_connection);

  return driver;
}

Result<kvstore::Spec> ParseGcsGrpcUrl(std::string_view url) {
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
        "Invalid GCS bucket name: ", QuoteString(parsed.authority)));
  }
  auto decoded_path = parsed.path.empty()
                          ? std::string()
                          : internal::PercentDecode(parsed.path.substr(1));

  auto driver_spec = internal::MakeIntrusivePtr<GcsGrpcKeyValueStoreSpec>();
  driver_spec->data_.bucket = std::string(parsed.authority);
  driver_spec->data_.user_project =
      Context::Resource<GcsUserProjectResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<internal_storage_gcs::GcsRequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();
  return {std::in_place, std::move(driver_spec), std::move(decoded_path)};
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::GcsGrpcKeyValueStore)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::GcsGrpcKeyValueStoreSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{kUriScheme, tensorstore::ParseGcsGrpcUrl};

}  // namespace
