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

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/crc/crc32c.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
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
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/gcs/gcs_resource.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_grpc/get_credentials.h"
#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

/// specializations
#include "tensorstore/internal/cache_key/absl_time.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/serialization/absl_time.h"
#include "tensorstore/serialization/fwd.h"

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

// To enable debug checks, specify:
// bazel build --//tensorstore/kvstore/gcs_grpc:debug
#ifndef TENSORSTORE_GCS_GRPC_DEBUG
#define TENSORSTORE_GCS_GRPC_DEBUG 0
#endif

namespace {
static constexpr char kUriScheme[] = "gcs_grpc";
}

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

auto& gcs_grpc_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs_grpc/read", "GCS driver kvstore::Read calls");

auto& gcs_grpc_read_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/gcs_grpc/read_latency_ms",
        "GCS driver kvstore::Read latency (ms)");

auto& gcs_grpc_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs_grpc/write", "GCS driver kvstore::Write calls");

auto& gcs_grpc_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs_grpc/delete_range",
    "GCS driver kvstore::DeleteRange calls");

auto& gcs_grpc_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs_grpc/list", "GCS driver kvstore::List calls");

auto& gcs_grpc_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/gcs_grpc/retries",
    "Count of all retried GCS requests (read/write/delete)");

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

  Result<std::string> ToUrl(std::string_view path) const override {
    if (!data_.endpoint.empty()) {
      return absl::UnimplementedError(
          "URL representation does not support test endpoints");
    }
    return tensorstore::StrCat(kUriScheme, "://", data_.bucket, "/",
                               internal::PercentEncodeUriPath(path));
  }
};

/// Defines the "grpc_kvstore" KeyValueStore driver.
class GcsGrpcKeyValueStore
    : public internal_kvstore::RegisteredDriver<GcsGrpcKeyValueStore,
                                                GcsGrpcKeyValueStoreSpec> {
 public:
  /// Key value store operations.
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

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

  void SetDefaultContextOptions(grpc::ClientContext& context) {
    // NOTE: Evaluate this a bit more?
    // context.set_wait_for_ready(false);
    if (spec_.timeout > absl::ZeroDuration() &&
        spec_.timeout < absl::InfiniteDuration()) {
      context.set_deadline(absl::ToChronoTime(absl::Now() + spec_.timeout));
    }

    // For a requestor-pays bucket we need to set x-goog-user-project.
    if (spec_.user_project->project_id &&
        !spec_.user_project->project_id->empty()) {
      context.AddMetadata("x-goog-user-project",
                          *spec_.user_project->project_id);
    }

    // gRPC requests need to have routing parameters added.
    context.AddMetadata("x-goog-request-params",
                        absl::StrFormat("bucket=%s", bucket_name()));

    if (call_credentials_fn_) {
      // The gRPC credentials model includes per-channel credentials,
      // `ChannelCredentials`, and per-request credentials,
      // `CallCredentials`. Instead of using a composite credentials object,
      // set the CallCredentials on each request which allows using a shared
      // channel pool.
      context.set_credentials(call_credentials_fn_());
    }
  }

  // Apply default backoff/retry logic to the task.
  // Returns whether the task will be retried. On false, max retries have
  // been met or exceeded.  On true, `task->Retry()` will be scheduled to run
  // after a suitable backoff period.
  template <typename Task>
  bool BackoffForAttemptAsync(int attempt, Task* task) {
    assert(task != nullptr);
    if (attempt >= spec_.retries->max_retries) return false;
    // https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff
    gcs_grpc_retries.Increment();
    auto delay = internal::BackoffForAttempt(
        attempt, spec_.retries->initial_delay, spec_.retries->max_delay,
        /*jitter=*/std::min(absl::Seconds(1), spec_.retries->initial_delay));
    ScheduleAt(absl::Now() + delay,
               WithExecutor(executor(), [task = internal::IntrusivePtr<Task>(
                                             task)] { task->Retry(); }));
    return true;
  }

  SpecData spec_;
  std::string bucket_;
  std::shared_ptr<StorageStubPool> storage_stub_pool_;
  std::function<std::shared_ptr<grpc::CallCredentials>()> call_credentials_fn_;
};

////////////////////////////////////////////////////

/// Abseil has a convenient crc32_t type, but it doesn't handle cord.
[[maybe_unused]] absl::crc32c_t ComputeCrc32c(const absl::Cord& cord) {
  absl::crc32c_t crc{0};
  for (auto chunk : cord.Chunks()) {
    crc = absl::ExtendCrc32c(crc, chunk);
  }
  return crc;
}

[[maybe_unused]] absl::crc32c_t ComputeCrc32c(std::string_view data) {
  return absl::ComputeCrc32c(data);
}

// Set google::storage::v2::ChecksummedData::content, which could be either an
// absl::Cord or a std::string, depending on options. This function is a
// template to avoid instantiation with the incorrect type.
template <typename T>
void SetContentImpl(std::true_type, T& checksummed_data, absl::Cord subcord) {
  checksummed_data.set_content(std::move(subcord));
}

template <typename T>
void SetContentImpl(std::false_type, T& checksummed_data, absl::Cord subcord) {
  auto* content_ptr = checksummed_data.mutable_content();
  absl::CopyCordToString(subcord, content_ptr);
  assert(content_ptr->size() == subcord.size());
}

// Implements GcsGrpcKeyValueStore::Read
// rpc ReadObject(ReadObjectRequest) returns (stream ReadObjectResponse) {}
struct ReadTask : public internal::AtomicReferenceCount<ReadTask>,
                  public grpc::ClientReadReactor<ReadObjectResponse> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::ReadOptions options_;
  Promise<kvstore::ReadResult> promise_;

  // working state.
  Storage::StubInterface* stub_ = nullptr;
  ReadObjectRequest request_;
  ReadObjectResponse response_;
  std::optional<absl::crc32c_t> crc32c_;
  kvstore::ReadResult read_result_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  void TryCancel() {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  grpc::ClientContext* AllocateContext() {
    absl::MutexLock lock(&mutex_);
    context_ = std::make_unique<grpc::ClientContext>();
    driver_->SetDefaultContextOptions(*context_);
    return context_.get();
  }

  void Start(const std::string& object_name) {
    stub_ = driver_->get_stub().get();
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->TryCancel(); });

    request_.set_bucket(driver_->bucket_name());
    request_.set_object(object_name);

    if (!StorageGeneration::IsUnknown(options_.if_equal)) {
      uint64_t gen = StorageGeneration::IsNoValue(options_.if_equal)
                         ? 0
                         : StorageGeneration::ToUint64(options_.if_equal);
      request_.set_if_generation_match(gen);
    }
    if (!StorageGeneration::IsUnknown(options_.if_not_equal)) {
      uint64_t gen = StorageGeneration::IsNoValue(options_.if_not_equal)
                         ? 0
                         : StorageGeneration::ToUint64(options_.if_not_equal);
      request_.set_if_generation_not_match(gen);
    }
    if (options_.byte_range.inclusive_min != 0) {
      request_.set_read_offset(options_.byte_range.inclusive_min);
    }
    if (options_.byte_range.exclusive_max != -1) {
      auto target_size = options_.byte_range.size();
      assert(target_size >= 0);
      // read_limit == 0 reads the entire object; instead just read a single
      // byte.
      request_.set_read_limit(target_size == 0 ? 1 : target_size);
    }

    Retry();
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    read_result_.stamp.time = absl::Now();
    read_result_.stamp.generation = StorageGeneration::Unknown();

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "Read: " << this << " " << ConciseDebugString(request_);

    grpc::ClientContext* context = AllocateContext();

    // Start a call.
    intrusive_ptr_increment(this);  // adopted in OnDone.
    stub_->async()->ReadObject(context, &request_, this);
    StartRead(&response_);
    StartCall();
  }

  void OnReadDone(bool ok) override {
    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "ReadDone: " << this << " " << ok << " "
        << (ok ? ConciseDebugString(response_) : std::string());
    if (!ok) return;
    if (!promise_.result_needed()) {
      TryCancel();
      return;
    }
    if (response_.has_metadata()) {
      read_result_.stamp.generation =
          StorageGeneration::FromUint64(response_.metadata().generation());
    }
    if (response_.has_object_checksums() &&
        response_.object_checksums().crc32c() != 0 &&
        options_.byte_range.inclusive_min == 0 &&
        !options_.byte_range.exclusive_max) {
      // Do not validate byte-range requests.
      crc32c_ = absl::crc32c_t(response_.object_checksums().crc32c());
    }
    if (response_.has_content_range()) {
      // The content-range request indicates the expected data. If it does not
      // satisfy the byte range request, cancel the read with an error. Allow
      // the returned size to exceed the requested size.
      auto returned_size =
          response_.content_range().end() - response_.content_range().start();
      if (auto size = options_.byte_range.size();
          (size > 0 && size != returned_size) ||
          (options_.byte_range.inclusive_min >= 0 &&
           response_.content_range().start() !=
               options_.byte_range.inclusive_min)) {
        promise_.SetResult(absl::OutOfRangeError(
            tensorstore::StrCat("Requested byte range ", options_.byte_range,
                                " was not satisfied by GCS object with size ",
                                response_.content_range().complete_length())));
        TryCancel();
        return;
      }
    }
    if (response_.has_checksummed_data()) {
      if (response_.checksummed_data().has_crc32c() &&
          ComputeCrc32c(response_.checksummed_data().content()) !=
              absl::crc32c_t(response_.checksummed_data().crc32c())) {
        promise_.SetResult(absl::DataLossError(
            "Object fragment crc32c does not match expected crc32c"));
        TryCancel();
        return;
      }

      read_result_.value.Append(response_.checksummed_data().content());
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
    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "ReadFinished: " << this << " " << status;

    if (!status.ok() && IsRetriable(status) && read_result_.value.empty()) {
      if (driver_->BackoffForAttemptAsync(attempt_++, this)) {
        return;
      }
      status = absl::AbortedError(
          tensorstore::StrCat("All retry attempts failed: ", status));
    }

    auto latency = absl::Now() - read_result_.stamp.time;
    gcs_grpc_read_latency_ms.Observe(absl::ToInt64Milliseconds(latency));

    if (!status.ok()) {
      if (absl::IsFailedPrecondition(status) || absl::IsAborted(status)) {
        // Failed precondition is set when either the if_generation_match or
        // the if_generation_not_match fails.
        read_result_.value.Clear();
        if (!StorageGeneration::IsUnknown(options_.if_equal)) {
          read_result_.stamp.generation = StorageGeneration::Unknown();
        } else {
          read_result_.stamp.generation = options_.if_not_equal;
        }
        promise_.SetResult(std::move(read_result_));
      } else if (absl::IsNotFound(status)) {
        read_result_.stamp.generation = StorageGeneration::NoValue();
        read_result_.state = kvstore::ReadResult::kMissing;
        read_result_.value.Clear();
        promise_.SetResult(std::move(read_result_));
      } else {
        promise_.SetResult(std::move(status));
      }
      return;
    }
    if (StorageGeneration::IsUnknown(read_result_.stamp.generation)) {
      // Bad metadata was returned by BlobService; this is unexpected, and
      // usually indicates a bug in our testing.
      promise_.SetResult(
          absl::InternalError("Object missing a valid generation"));
      return;
    }
    if (options_.byte_range.size() == 0) {
      read_result_.value.Clear();
    } else if (crc32c_.has_value() &&
               ComputeCrc32c(read_result_.value) != *crc32c_) {
      promise_.SetResult(
          absl::DataLossError("Object crc32c does not match expected crc32c"));
      return;
    }
    read_result_.state = kvstore::ReadResult::kValue;
    promise_.SetResult(std::move(read_result_));
  }
};

// Implements GcsGrpcKeyValueStore::Write
// rpc ReadObject(ReadObjectRequest) returns (stream ReadObjectResponse) {}
// rpc StartResumableWrite(StartResumableWriteRequest) returns
// (StartResumableWriteResponse) {}
struct WriteTask : public internal::AtomicReferenceCount<WriteTask>,
                   public grpc::ClientWriteReactor<WriteObjectRequest> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::WriteOptions options_;
  Promise<TimestampedStorageGeneration> promise_;

  // working state.
  absl::Cord value_;
  absl::crc32c_t crc32c_{0};
  size_t write_offset_ = 0;
  Storage::StubInterface* stub_ = nullptr;
  WriteObjectRequest request_;
  WriteObjectResponse response_;
  TimestampedStorageGeneration write_result_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  void TryCancel() {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  grpc::ClientContext* AllocateContext() {
    absl::MutexLock lock(&mutex_);
    context_ = std::make_unique<grpc::ClientContext>();
    driver_->SetDefaultContextOptions(*context_);
    return context_.get();
  }

  // TODO(laramiel): We could write these chunks in parallel.
  void Start(const std::string& object_name, absl::Cord value) {
    value_ = std::move(value);
    stub_ = driver_->get_stub().get();
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<WriteTask>(
                                       this)] { self->TryCancel(); });

    auto& resource = *request_.mutable_write_object_spec()->mutable_resource();
    resource.set_bucket(driver_->bucket_name());
    resource.set_name(object_name);
    request_.mutable_write_object_spec()->set_object_size(value_.size());
    if (!StorageGeneration::IsUnknown(options_.if_equal)) {
      auto gen = StorageGeneration::ToUint64(options_.if_equal);
      request_.mutable_write_object_spec()->set_if_generation_match(gen);
    }

    AddChunkData();
    Retry();
  }

  // Retry/Continue a call.
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }

    /// Set this as the last write time if this is the initial chunk.
    if (request_.write_offset() == 0) {
      write_result_.time = absl::Now();
    }

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "Write: " << this << " " << ConciseDebugString(request_);

    grpc::ClientContext* context = AllocateContext();

    // Initiate the write.
    intrusive_ptr_increment(this);
    stub_->async()->WriteObject(context, &response_, this);
    if (request_.finish_write()) {
      StartWriteLast(&request_, grpc::WriteOptions());
    } else {
      StartWrite(&request_, grpc::WriteOptions());
    }
    StartCall();
  }

  void AddChunkData() {
    request_.set_write_offset(write_offset_);
    size_t next_write_offset = std::min(
        write_offset_ + ServiceConstants::MAX_WRITE_CHUNK_BYTES, value_.size());

    auto& checksummed_data = *request_.mutable_checksummed_data();
    SetContentImpl(
        std::is_same<absl::Cord,
                     internal::remove_cvref_t<
                         decltype(checksummed_data.content())>>::type{},
        checksummed_data,
        value_.Subcord(write_offset_, next_write_offset - write_offset_));

    write_offset_ = next_write_offset;

    auto chunk_crc32c = ComputeCrc32c(checksummed_data.content());
    checksummed_data.set_crc32c(static_cast<uint32_t>(chunk_crc32c));
    crc32c_ = absl::ConcatCrc32c(crc32c_, chunk_crc32c,
                                 checksummed_data.content().size());

    if (write_offset_ == value_.size()) {
      /// This is the last request.
      request_.mutable_object_checksums()->set_crc32c(
          static_cast<uint32_t>(crc32c_));
      request_.set_finish_write(true);
    }
  }

  void OnWriteDone(bool ok) override {
    // Not streaming any additional data bits.
    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "WriteDone: " << this << " " << ok;

    if (!ok) return;
    if (request_.finish_write()) return;
    request_.clear_write_object_spec();

    AddChunkData();

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "Write: " << this << " " << ConciseDebugString(request_);

    if (request_.finish_write()) {
      StartWriteLast(&request_, grpc::WriteOptions());
    } else {
      StartWrite(&request_, grpc::WriteOptions());
    }
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
    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "WriteFinished: " << this << " " << status << " "
        << ConciseDebugString(response_);

    if (!status.ok() && IsRetriable(status)) {
      if (driver_->BackoffForAttemptAsync(attempt_++, this)) {
        return;
      }
      status = absl::AbortedError(
          tensorstore::StrCat("All retry attempts failed: ", status));
    }

    if (response_.has_resource()) {
      write_result_.generation =
          StorageGeneration::FromUint64(response_.resource().generation());
    }
    if (absl::IsFailedPrecondition(status) || absl::IsAlreadyExists(status)) {
      /// if_equal condition did not match.
      write_result_.generation = StorageGeneration::Unknown();
      promise_.SetResult(std::move(write_result_));
    } else if (absl::IsNotFound(status) &&
               !StorageGeneration::IsUnknown(options_.if_equal)) {
      /// precondition did not match.
      write_result_.generation = StorageGeneration::Unknown();
      promise_.SetResult(std::move(write_result_));
    } else if (!status.ok()) {
      promise_.SetResult(status);
    } else {
      promise_.SetResult(std::move(write_result_));
    }
  }
};

// Implements GcsGrpcKeyValueStore::Delete
// rpc DeleteObject(DeleteObjectRequest) returns (google.protobuf.Empty) {}
struct DeleteTask : public internal::AtomicReferenceCount<DeleteTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::WriteOptions options_;
  Promise<TimestampedStorageGeneration> promise_;

  // Working state
  Storage::StubInterface* stub_ = nullptr;
  absl::Time start_time_;
  DeleteObjectRequest request_;
  ::google::protobuf::Empty response_;
  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  void TryCancel() {
    absl::MutexLock lock(&mutex_);
    if (context_) context_->TryCancel();
  }

  grpc::ClientContext* AllocateContext() {
    absl::MutexLock lock(&mutex_);
    context_ = std::make_unique<grpc::ClientContext>();
    driver_->SetDefaultContextOptions(*context_);
    return context_.get();
  }

  void Start(const std::string& object_name) {
    stub_ = driver_->get_stub().get();
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<DeleteTask>(
                                       this)] { self->TryCancel(); });

    request_.set_bucket(driver_->bucket_name());
    request_.set_object(object_name);
    if (!StorageGeneration::IsUnknown(options_.if_equal)) {
      auto gen = StorageGeneration::ToUint64(options_.if_equal);
      request_.set_if_generation_match(gen);
    }
    Retry();
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    grpc::ClientContext* context = AllocateContext();

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "Delete: " << this << " " << ConciseDebugString(request_);

    start_time_ = absl::Now();
    intrusive_ptr_increment(this);  // Adopted by OnDone
    stub_->async()->DeleteObject(
        context, &request_, &response_,
        WithExecutor(driver_->executor(), [this](::grpc::Status s) {
          internal::IntrusivePtr<DeleteTask> self(this,
                                                  internal::adopt_object_ref);
          self->DeleteFinished(GrpcStatusToAbslStatus(s));
        }));
  }

  void DeleteFinished(absl::Status status) {
    if (!promise_.result_needed()) {
      return;
    }

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "DeleteFinished: " << this << " " << status;

    if (!status.ok() && IsRetriable(status)) {
      if (driver_->BackoffForAttemptAsync(attempt_++, this)) {
        return;
      }
      status = absl::AbortedError(
          tensorstore::StrCat("All retry attempts failed: ", status));
    }

    TimestampedStorageGeneration r;
    r.time = start_time_;
    r.generation = StorageGeneration::NoValue();
    if (absl::IsFailedPrecondition(status)) {
      /// precondition did not match.
      r.generation = StorageGeneration::Unknown();
    } else if (absl::IsNotFound(status)) {
      /// object missing; that's probably ok.
      if (!StorageGeneration::IsNoValue(options_.if_equal) &&
          !StorageGeneration::IsUnknown(options_.if_equal)) {
        r.generation = StorageGeneration::Unknown();
      }
    } else if (!status.ok()) {
      ABSL_LOG(INFO) << status;
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
  AnyFlowReceiver<absl::Status, kvstore::Key> receiver_;
  kvstore::ListOptions options_;

  // working state.
  Storage::StubInterface* stub_ = nullptr;
  ListObjectsRequest request;
  ListObjectsResponse response;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  bool cancelled_ ABSL_GUARDED_BY(mutex_) = false;

  ListTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
           AnyFlowReceiver<absl::Status, kvstore::Key> receiver,
           kvstore::ListOptions options)
      : driver_(std::move(driver)),
        receiver_(std::move(receiver)),
        options_(std::move(options)) {
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

  bool is_cancelled() {
    absl::MutexLock l(&mutex_);
    return cancelled_;
  }

  void TryCancel() {
    absl::MutexLock l(&mutex_);
    if (!cancelled_) {
      cancelled_ = true;
      if (context_) context_->TryCancel();
    }
  }

  grpc::ClientContext* AllocateContext() {
    absl::MutexLock lock(&mutex_);
    context_ = std::make_unique<grpc::ClientContext>();
    driver_->SetDefaultContextOptions(*context_);
    return context_.get();
  }

  void Start() {
    stub_ = driver_->get_stub().get();
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

    grpc::ClientContext* context = AllocateContext();

    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "List: " << this << " " << ConciseDebugString(request);

    intrusive_ptr_increment(this);
    stub_->async()->ListObjects(
        context, &request, &response,
        WithExecutor(driver_->executor(), [this](::grpc::Status s) {
          internal::IntrusivePtr<ListTask> self(this,
                                                internal::adopt_object_ref);
          self->ListFinished(GrpcStatusToAbslStatus(s));
        }));
  }

  void ListFinished(absl::Status status) {
    if (is_cancelled()) {
      execution::set_done(receiver_);
      return;
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_GCS_GRPC_DEBUG)
        << "ListResponse: " << this << " " << status << " :"
        << ConciseDebugString(response);

    if (!status.ok() && IsRetriable(status)) {
      if (driver_->BackoffForAttemptAsync(attempt_++, this)) {
        return;
      }
      status = absl::AbortedError(
          tensorstore::StrCat("All retry attempts failed: ", status));
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
      execution::set_value(receiver_, std::string(name));
    }
    if (!done && !response.next_page_token().empty()) {
      /// If there is a continuation token, issue the next request.
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

  void set_value(std::string key) {
    assert(!key.empty());
    if (!key.empty()) {
      LinkError(promise_, driver_->Delete(std::move(key)));
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
  gcs_grpc_read.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid blob object name");
  }
  if (!IsValidStorageGeneration(options.if_equal) ||
      !IsValidStorageGeneration(options.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed StorageGeneration");
  }
  auto op = PromiseFuturePair<ReadResult>::Make();

  auto task = internal::MakeIntrusivePtr<ReadTask>();
  task->driver_ = internal::IntrusivePtr<GcsGrpcKeyValueStore>(this);
  task->options_ = std::move(options);
  task->promise_ = std::move(op.promise);
  task->Start(key);
  return std::move(op.future);
}

Future<TimestampedStorageGeneration> GcsGrpcKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  gcs_grpc_write.Increment();
  if (!IsValidObjectName(key)) {
    return absl::InvalidArgumentError("Invalid blob object name");
  }
  if (!IsValidStorageGeneration(options.if_equal)) {
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
    auto task = internal::MakeIntrusivePtr<WriteTask>();
    task->driver_ = internal::IntrusivePtr<GcsGrpcKeyValueStore>(this);
    task->options_ = std::move(options);
    task->promise_ = std::move(op.promise);
    task->Start(key, std::move(*value));
  }
  return std::move(op.future);
}

void GcsGrpcKeyValueStore::ListImpl(
    ListOptions options, AnyFlowReceiver<absl::Status, Key> receiver) {
  gcs_grpc_list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }

  auto task = internal::MakeIntrusivePtr<ListTask>(
      internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), std::move(receiver),
      std::move(options));
  task->Start();
}

Future<const void> GcsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  gcs_grpc_delete_range.Increment();
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

  std::string endpoint = data_.endpoint;
  if (endpoint.empty()) {
    endpoint = "dns:///storage.googleapis.com";
  }

  // TODO(laramiel): Introduce a common mechanism for gRPC credentials, to the extent
  // possible.
  auto channel_credentials =
      GetCredentialsForEndpoint(endpoint, driver->call_credentials_fn_);
  driver->storage_stub_pool_ = GetSharedStorageStubPool(
      endpoint, data_.num_channels, std::move(channel_credentials));

  // Maybe wait for the channels to be connected.
  if (driver->spec_.wait_for_connection > absl::ZeroDuration()) {
    driver->storage_stub_pool_->WaitForConnected(
        driver->spec_.wait_for_connection);
  }
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

  std::pair<std::string_view, std::string_view> bucket_and_path =
      absl::StrSplit(parsed.authority_and_path, absl::MaxSplits('/', 1));
  if (!IsValidBucketName(bucket_and_path.first)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid bucket name: ", QuoteString(bucket_and_path.first)));
  }

  auto driver_spec = internal::MakeIntrusivePtr<GcsGrpcKeyValueStoreSpec>();
  driver_spec->data_.bucket = bucket_and_path.first;
  driver_spec->data_.user_project =
      Context::Resource<GcsUserProjectResource>::DefaultSpec();
  driver_spec->data_.retries =
      Context::Resource<internal_storage_gcs::GcsRequestRetries>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<DataCopyConcurrencyResource>::DefaultSpec();

  return {std::in_place, std::move(driver_spec),
          internal::PercentDecode(bucket_and_path.second)};
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
