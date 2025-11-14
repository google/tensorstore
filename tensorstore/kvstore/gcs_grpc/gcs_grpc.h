// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_GCS_GRPC_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_GCS_GRPC_H_

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/gcs/exp_credentials_resource.h"
#include "tensorstore/kvstore/gcs/gcs_resource.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"  // IWYU pragma: keep

// protos
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"

namespace tensorstore {
namespace internal_gcs_grpc {

using GcsUserProjectResource = internal_storage_gcs::GcsUserProjectResource;
using GcsRequestRetries = internal_storage_gcs::GcsRequestRetries;
using ExperimentalGcsGrpcCredentials =
    internal_storage_gcs::ExperimentalGcsGrpcCredentials;
using DataCopyConcurrencyResource = internal::DataCopyConcurrencyResource;

class GcsGrpcKeyValueStoreSpec;

struct GcsGrpcKeyValueStoreSpecData {
  std::string bucket;
  std::string endpoint;
  uint32_t num_channels = 0;
  absl::Duration timeout = absl::ZeroDuration();
  absl::Duration wait_for_connection = absl::ZeroDuration();
  Context::Resource<GcsUserProjectResource> user_project;
  Context::Resource<GcsRequestRetries> retries;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;
  Context::Resource<ExperimentalGcsGrpcCredentials> credentials;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.bucket, x.endpoint, x.num_channels, x.timeout,
             x.wait_for_connection, x.user_project, x.retries,
             x.data_copy_concurrency, x.credentials);
  };

  TENSORSTORE_INTERNAL_DECLARE_JSON_BINDER_IMPL(
      JsonBinderImpl, GcsGrpcKeyValueStoreSpecData,
      internal_kvstore::DriverFromJsonOptions, JsonSerializationOptions,
      ::nlohmann::json::object_t);

  static inline constexpr JsonBinderImpl default_json_binder = {};
};

class GcsGrpcKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          GcsGrpcKeyValueStoreSpec, GcsGrpcKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "gcs_grpc";
  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status NormalizeSpec(std::string& path) override {
    if (!path.empty() && !internal_storage_gcs::IsValidObjectName(path)) {
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
    return tensorstore::StrCat(id, "://", data_.bucket, "/",
                               internal::PercentEncodeKvStoreUriPath(path));
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
  Future<kvstore::ReadResult> Read(kvstore::Key key,
                                   kvstore::ReadOptions options) override;
  Future<kvstore::ReadResult> ReadImpl(kvstore::Key&& key,
                                       kvstore::ReadOptions&& options);

  Future<TimestampedStorageGeneration> Write(
      kvstore::Key key, std::optional<kvstore::Value> value,
      kvstore::WriteOptions options) override;

  void ListImpl(kvstore::ListOptions options,
                kvstore::ListReceiver receiver) override;

  Future<const void> DeleteRange(KeyRange range) override;

  // Obtains a `SpecData` representation from an open `Driver`.
  absl::Status GetBoundSpecData(GcsGrpcKeyValueStoreSpecData& spec) const;

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return kvstore::SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
           kvstore::SupportedFeatures::kAtomicWriteWithoutOverwrite;
  }

  const Executor& executor() const {
    return spec_.data_copy_concurrency->executor;
  }

  // Bucket names are in the form: 'projects/{project-id}/buckets/{bucket-id}'
  std::string bucket_name() { return bucket_; }

  using StubInterface = ::google::storage::v2::Storage::StubInterface;
  std::shared_ptr<StubInterface> get_stub() {
    return storage_stub_pool_->get_next_stub();
  }

  Future<std::shared_ptr<grpc::ClientContext>> AllocateContext();

  // Apply default backoff/retry logic to the task.
  // Returns whether the task will be retried. On false, max retries have
  // been met or exceeded.  On true, `task->Retry()` will be scheduled to run
  // after a suitable backoff period.
  absl::Status BackoffForAttemptAsync(
      absl::Status status, int attempt, absl::AnyInvocable<void() &&> task,
      SourceLocation loc = SourceLocation::current());

  GcsGrpcKeyValueStoreSpecData spec_;
  std::string bucket_;
  std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy_;
  std::shared_ptr<StorageStubPool> storage_stub_pool_;
};

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    ::tensorstore::internal_gcs_grpc::GcsGrpcKeyValueStore)

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_GCS_GRPC_H_
