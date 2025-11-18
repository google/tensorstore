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

#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/gcs/exp_credentials_resource.h"
#include "tensorstore/kvstore/gcs/exp_credentials_spec.h"
#include "tensorstore/kvstore/gcs/gcs_resource.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_grpc/default_endpoint.h"
#include "tensorstore/kvstore/gcs_grpc/default_strategy.h"
#include "tensorstore/kvstore/gcs_grpc/op_delete.h"
#include "tensorstore/kvstore/gcs_grpc/op_list.h"
#include "tensorstore/kvstore/gcs_grpc/op_read.h"
#include "tensorstore/kvstore/gcs_grpc/op_write.h"
#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"
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
#include "google/storage/v2/storage.pb.h"

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::ScheduleAt;
using ::tensorstore::internal_gcs_grpc::GetSharedStorageStubPool;
using ::tensorstore::internal_gcs_grpc::StorageStubPool;
using ::tensorstore::internal_storage_gcs::ExperimentalGcsGrpcCredentials;
using ::tensorstore::internal_storage_gcs::GcsUserProjectResource;
using ::tensorstore::internal_storage_gcs::IsValidBucketName;
using ::tensorstore::internal_storage_gcs::IsValidObjectName;
using ::tensorstore::internal_storage_gcs::IsValidStorageGeneration;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::SupportedFeatures;


namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

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

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    GcsGrpcKeyValueStoreSpecData,
    jb::Object(
        jb::Member(
            "bucket",
            jb::Projection<&GcsGrpcKeyValueStoreSpecData::bucket>(
                jb::Validate([](const auto& options, const std::string* x) {
                  if (!internal_storage_gcs::IsValidBucketName(*x)) {
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
                       jb::DefaultValue<jb::kNeverIncludeDefaults>([](auto* x) {
                         *x = absl::ZeroDuration();
                       }))),
        jb::Member(
            "wait_for_connection",
            jb::Projection<&GcsGrpcKeyValueStoreSpecData::wait_for_connection>(
                jb::DefaultValue<jb::kNeverIncludeDefaults>([](auto* x) {
                  *x = absl::ZeroDuration();
                }))),
        jb::Member(
            GcsUserProjectResource::id,
            jb::Projection<&GcsGrpcKeyValueStoreSpecData::user_project>()),
        jb::Member(internal_storage_gcs::GcsRequestRetries::id,
                   jb::Projection<&GcsGrpcKeyValueStoreSpecData::retries>()),
        jb::Member(DataCopyConcurrencyResource::id,
                   jb::Projection<
                       &GcsGrpcKeyValueStoreSpecData::data_copy_concurrency>()),
        jb::Member(
            ExperimentalGcsGrpcCredentials::id,
            jb::Projection<&GcsGrpcKeyValueStoreSpecData::credentials>()), /**/
        jb::DiscardExtraMembers));

/// Obtains a `SpecData` representation from an open `Driver`.
absl::Status GcsGrpcKeyValueStore::GetBoundSpecData(
    GcsGrpcKeyValueStoreSpecData& spec) const {
  spec = spec_;
  return absl::OkStatus();
}

Future<std::shared_ptr<grpc::ClientContext>>
GcsGrpcKeyValueStore::AllocateContext() {
  auto context = std::make_shared<grpc::ClientContext>();

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

  if (!auth_strategy_->RequiresConfigureContext()) {
    return std::move(context);
  }
  return auth_strategy_->ConfigureContext(std::move(context));
}

// Apply default backoff/retry logic to the task.
// Returns whether the task will be retried. On false, max retries have
// been met or exceeded.  On true, `task->Retry()` will be scheduled to run
// after a suitable backoff period.
absl::Status GcsGrpcKeyValueStore::BackoffForAttemptAsync(
    absl::Status status, int attempt, absl::AnyInvocable<void() &&> task,
    SourceLocation loc) {
  auto delay = spec_.retries->BackoffForAttempt(attempt);
  if (!delay) {
    return MaybeAnnotateStatus(std::move(status),
                               absl::StrFormat("All %d retry attempts failed",
                                               spec_.retries->max_retries),
                               absl::StatusCode::kAborted, loc);
  }
  gcs_grpc_metrics.retries.Increment();
  ScheduleAt(absl::Now() + *delay,
             WithExecutor(executor(), [task = std::move(task)]() mutable {
               std::move(task)();
             }));
  return absl::OkStatus();
}

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
  return internal_gcs_grpc::InitiateRead(
      internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), gcs_grpc_metrics, key,
      std::move(options));
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

  if (!value) {
    return internal_gcs_grpc::InitiateDelete(
        internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), std::move(key),
        std::move(options));
  } else {
    return internal_gcs_grpc::InitiateWrite(
        internal::IntrusivePtr<GcsGrpcKeyValueStore>(this), gcs_grpc_metrics,
        std::move(key), std::move(*value), std::move(options));
  }
}

void GcsGrpcKeyValueStore::ListImpl(ListOptions options,
                                    ListReceiver receiver) {
  gcs_grpc_metrics.list.Increment();
  InitiateList(internal::IntrusivePtr<GcsGrpcKeyValueStore>(this),
               std::move(options), std::move(receiver));
}

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

Future<const void> GcsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  gcs_grpc_metrics.delete_range.Increment();
  if (range.empty()) return absl::OkStatus();

  // TODO(jbms): It could make sense to rate limit the list operation, so that
  // we don't get way ahead of the delete operations.  Currently the
  // sender/receiver abstraction does not support back pressure, though.
  auto op = PromiseFuturePair<void>::Make(tensorstore::MakeResult());

  ListOptions list_options;
  list_options.range = std::move(range);
  InitiateList(internal::IntrusivePtr<GcsGrpcKeyValueStore>(this),
               std::move(list_options),
               DeleteRangeListReceiver{
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
    endpoint = internal_gcs_grpc::GetDefaultGcsGrpcEndpoint();
    ABSL_LOG_IF_FIRST_N(INFO, gcs_grpc_logging, 1)
        << "Using gcs_grpc default endpoint: " << endpoint;
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      driver->auth_strategy_,
      internal_storage_gcs::MakeGrpcAuthenticationStrategy(*data_.credentials,
                                                           {}));
  if (!driver->auth_strategy_) {
    // Use default authentication strategy.
    driver->auth_strategy_ =
        internal_gcs_grpc::CreateDefaultGrpcAuthenticationStrategy(endpoint);
  }

  driver->storage_stub_pool_ = GetSharedStorageStubPool(
      endpoint, data_.num_channels, driver->auth_strategy_,
      driver->spec_.wait_for_connection);

  return driver;
}

// Registers the driver.
namespace {

Result<kvstore::Spec> ParseGcsGrpcUrl(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  TENSORSTORE_RETURN_IF_ERROR(internal::EnsureSchemaWithAuthorityDelimiter(
      parsed, GcsGrpcKeyValueStoreSpec::id));
  TENSORSTORE_RETURN_IF_ERROR(internal::EnsureNoQueryOrFragment(parsed));
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
  driver_spec->data_.credentials =
      Context::Resource<ExperimentalGcsGrpcCredentials>::DefaultSpec();
  return {std::in_place, std::move(driver_spec), std::move(decoded_path)};
}

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{GcsGrpcKeyValueStoreSpec::id, ParseGcsGrpcUrl};

const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_gcs_grpc::GcsGrpcKeyValueStoreSpec>
    registration;

}  // namespace
}  // namespace internal_gcs_grpc
}  // namespace tensorstore
