// Copyright 2022 The TensorStore Authors
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

// TODO(jbms): Add test coverage of error paths.

#include "tensorstore/kvstore/ocdbt/driver.h"

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/ref_counted_string.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security_registry.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/read.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_ocdbt {
namespace {

using ::tensorstore::kvstore::ListReceiver;

constexpr absl::Duration kDefaultLeaseDuration = absl::Seconds(10);

constexpr size_t kDefaultTargetBufferSize = 2u << 30;  // 2GB

struct OcdbtCoordinatorResourceTraits
    : public internal::ContextResourceTraits<OcdbtCoordinatorResource> {
  using Spec = OcdbtCoordinatorResource::Spec;
  using Resource = OcdbtCoordinatorResource::Resource;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member("address", jb::Projection<&Spec::address>()),
        jb::Member("lease_duration", jb::Projection<&Spec::lease_duration>()),
        jb::Member("security", jb::Projection<&Spec::security>(
                                   RpcSecurityMethodJsonBinder)));
  }
  static Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) {
    return spec;
  }

  static Spec GetSpec(const Resource& resource,
                      const internal::ContextSpecBuilder& builder) {
    return resource;
  }
};

const internal::ContextResourceRegistration<OcdbtCoordinatorResourceTraits>
    registration;

auto& ocdbt_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/read", "OCDBT driver kvstore::Read calls");

auto& ocdbt_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/write", "OCDBT driver kvstore::Write calls");

auto& ocdbt_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/delete_range",
    "OCDBT driver kvstore::DeleteRange calls");

auto& ocdbt_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/list", "OCDBT driver kvstore::List calls");

}  // namespace
namespace jb = ::tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    OcdbtDriverSpecData,
    jb::Object(
        jb::Member("base", jb::Projection<&OcdbtDriverSpecData::base>()),
        jb::Initialize([](auto* obj) {
          internal::EnsureDirectoryPath(obj->base.path);
          return absl::OkStatus();
        }),
        jb::Member("config", jb::Projection<&OcdbtDriverSpecData::config>(
                                 jb::DefaultInitializedValue())),
        jb::Member(
            "experimental_read_coalescing_threshold_bytes",
            jb::Projection<&OcdbtDriverSpecData::
                               experimental_read_coalescing_threshold_bytes>()),
        jb::Member(
            "experimental_read_coalescing_merged_bytes",
            jb::Projection<&OcdbtDriverSpecData::
                               experimental_read_coalescing_merged_bytes>()),
        jb::Member(
            "experimental_read_coalescing_interval",
            jb::Projection<
                &OcdbtDriverSpecData::experimental_read_coalescing_interval>()),
        jb::Member(
            "target_data_file_size",
            jb::Projection<&OcdbtDriverSpecData::target_data_file_size>()),
        jb::Member("coordinator",
                   jb::Projection<&OcdbtDriverSpecData::coordinator>()),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection<&OcdbtDriverSpecData::cache_pool>()),
        jb::Member(
            internal::DataCopyConcurrencyResource::id,
            jb::Projection<&OcdbtDriverSpecData::data_copy_concurrency>())));

Result<kvstore::Spec> OcdbtDriverSpec::GetBase(std::string_view path) const {
  return data_.base;
}

Future<kvstore::DriverPtr> OcdbtDriverSpec::DoOpen() const {
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const OcdbtDriverSpec>(this)](
          kvstore::KvStore& base_kvstore) -> Result<kvstore::DriverPtr> {
        auto driver = internal::MakeIntrusivePtr<OcdbtDriver>();
        driver->base_ = std::move(base_kvstore);

        auto supported_manifest_features =
            driver->base_.driver->GetSupportedFeatures(KeyRange::Prefix(
                tensorstore::StrCat(driver->base_.path, "manifest.")));

        driver->cache_pool_ = spec->data_.cache_pool;
        driver->data_copy_concurrency_ = spec->data_.data_copy_concurrency;
        driver->experimental_read_coalescing_threshold_bytes_ =
            spec->data_.experimental_read_coalescing_threshold_bytes;
        driver->experimental_read_coalescing_merged_bytes_ =
            spec->data_.experimental_read_coalescing_merged_bytes;
        driver->experimental_read_coalescing_interval_ =
            spec->data_.experimental_read_coalescing_interval;
        driver->target_data_file_size_ = spec->data_.target_data_file_size;

        std::optional<ReadCoalesceOptions> read_coalesce_options;
        if (driver->experimental_read_coalescing_threshold_bytes_ ||
            driver->experimental_read_coalescing_merged_bytes_ ||
            driver->experimental_read_coalescing_interval_) {
          read_coalesce_options.emplace();
          read_coalesce_options->max_overhead_bytes_per_request =
              static_cast<int64_t>(
                  driver->experimental_read_coalescing_threshold_bytes_
                      .value_or(0));
          read_coalesce_options->max_merged_bytes_per_request =
              static_cast<int64_t>(
                  driver->experimental_read_coalescing_merged_bytes_.value_or(
                      0));
          read_coalesce_options->max_interval =
              driver->experimental_read_coalescing_interval_.value_or(
                  absl::ZeroDuration());
        }

        driver->io_handle_ = internal_ocdbt::MakeIoHandle(
            driver->data_copy_concurrency_, driver->cache_pool_->get(),
            driver->base_,
            internal::MakeIntrusivePtr<ConfigState>(
                spec->data_.config, supported_manifest_features),
            driver->target_data_file_size_.value_or(kDefaultTargetBufferSize),
            std::move(read_coalesce_options));
        driver->btree_writer_ =
            MakeNonDistributedBtreeWriter(driver->io_handle_);
        driver->coordinator_ = spec->data_.coordinator;
        if (!driver->coordinator_->address) {
          driver->btree_writer_ =
              MakeNonDistributedBtreeWriter(driver->io_handle_);
          return driver;
        }

        // Finish initialization for distributed mode.
        DistributedBtreeWriterOptions options;
        options.io_handle = driver->io_handle_;
        options.coordinator_address = *driver->coordinator_->address;
        options.security = driver->coordinator_->security;
        if (!options.security) {
          options.security = GetInsecureRpcSecurityMethod();
        }
        options.lease_duration = driver->coordinator_->lease_duration.value_or(
            kDefaultLeaseDuration);

        // Compute unique identifier for the base kvstore to use with
        // coordinator.
        TENSORSTORE_ASSIGN_OR_RETURN(auto base_spec,
                                     driver->base_.spec(MinimalSpec{}));
        TENSORSTORE_ASSIGN_OR_RETURN(auto base_spec_json, base_spec.ToJson());
        options.storage_identifier = base_spec_json.dump();
        driver->btree_writer_ = MakeDistributedBtreeWriter(std::move(options));
        return driver;
      },
      kvstore::Open(data_.base));
}

absl::Status OcdbtDriverSpec::ApplyOptions(
    kvstore::DriverSpecOptions&& options) {
  if (options.minimal_spec) {
    data_.config = {};
  }
  return data_.base.driver.Set(std::move(options));
}

absl::Status OcdbtDriver::GetBoundSpecData(OcdbtDriverSpecData& spec) const {
  TENSORSTORE_ASSIGN_OR_RETURN(spec.base.driver, base_.driver->GetBoundSpec());
  spec.base.path = base_.path;
  spec.data_copy_concurrency = data_copy_concurrency_;
  spec.cache_pool = cache_pool_;
  spec.config = io_handle_->config_state->GetConstraints();
  spec.experimental_read_coalescing_threshold_bytes =
      experimental_read_coalescing_threshold_bytes_;
  spec.experimental_read_coalescing_merged_bytes =
      experimental_read_coalescing_merged_bytes_;
  spec.experimental_read_coalescing_interval =
      experimental_read_coalescing_interval_;
  spec.target_data_file_size = target_data_file_size_;
  spec.coordinator = coordinator_;
  return absl::Status();
}

kvstore::SupportedFeatures OcdbtDriver::GetSupportedFeatures(
    const KeyRange& key_range) const {
  return kvstore::SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
         kvstore::SupportedFeatures::kAtomicWriteWithoutOverwrite;
}

Future<kvstore::ReadResult> OcdbtDriver::Read(kvstore::Key key,
                                              kvstore::ReadOptions options) {
  ocdbt_read.Increment();
  return internal_ocdbt::NonDistributedRead(io_handle_, std::move(key),
                                            std::move(options));
}

void OcdbtDriver::ListImpl(kvstore::ListOptions options,
                           ListReceiver receiver) {
  ocdbt_list.Increment();
  return internal_ocdbt::NonDistributedList(io_handle_, std::move(options),
                                            std::move(receiver));
}

Future<TimestampedStorageGeneration> OcdbtDriver::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  ocdbt_write.Increment();
  return btree_writer_->Write(std::move(key), std::move(value),
                              std::move(options));
}

Future<const void> OcdbtDriver::DeleteRange(KeyRange range) {
  ocdbt_delete_range.Increment();
  return btree_writer_->DeleteRange(std::move(range));
}

Future<const void> OcdbtDriver::ExperimentalCopyRangeFrom(
    const internal::OpenTransactionPtr& transaction, const KvStore& source,
    std::string target_prefix, kvstore::CopyRangeOptions options) {
  if (typeid(*source.driver) == typeid(OcdbtDriver)) {
    auto& source_driver = static_cast<OcdbtDriver&>(*source.driver);
    if (source.transaction != no_transaction || transaction) {
      return absl::UnimplementedError("Transactions not supported");
    }
    if (source_driver.base_.driver == base_.driver &&
        absl::StartsWith(source_driver.base_.path, base_.path)) {
      auto [promise, future] = PromiseFuturePair<void>::Make();
      auto manifest_future =
          source_driver.io_handle_->GetManifest(options.source_staleness_bound);
      LinkValue(
          [self = internal::IntrusivePtr<OcdbtDriver>(this),
           target_prefix = std::move(target_prefix),
           data_path_prefix =
               source_driver.base_.path.substr(base_.path.size()),
           source_range =
               KeyRange::AddPrefix(source.path, options.source_range),
           source_prefix_length = source.path.size()](
              Promise<void> promise,
              ReadyFuture<const ManifestWithTime> future) mutable {
            auto& manifest_with_time = future.value();
            if (!manifest_with_time.manifest) {
              // Source is empty.
              promise.SetResult(absl::OkStatus());
              return;
            }
            auto& manifest = *manifest_with_time.manifest;
            auto& latest_version = manifest.latest_version();
            if (latest_version.root.location.IsMissing()) {
              // Source is empty.
              promise.SetResult(absl::OkStatus());
              return;
            }
            BtreeWriter::CopySubtreeOptions copy_node_options;
            copy_node_options.node = latest_version.root;
            if (!data_path_prefix.empty()) {
              auto& base_path =
                  copy_node_options.node.location.file_id.base_path;
              internal::RefCountedStringWriter base_path_writer(
                  data_path_prefix.size() + base_path.size());
              std::memcpy(base_path_writer.data(), data_path_prefix.data(),
                          data_path_prefix.size());
              std::memcpy(base_path_writer.data() + data_path_prefix.size(),
                          base_path.data(), base_path.size());
              base_path = std::move(base_path_writer);
            }
            copy_node_options.node_height = latest_version.root_height;
            copy_node_options.range = std::move(source_range);
            copy_node_options.strip_prefix_length = source_prefix_length;
            copy_node_options.add_prefix = std::move(target_prefix);
            LinkResult(std::move(promise), self->btree_writer_->CopySubtree(
                                               std::move(copy_node_options)));
          },
          std::move(promise), std::move(manifest_future));
      return std::move(future);
    }
  }
  return kvstore::Driver::ExperimentalCopyRangeFrom(
      transaction, source, std::move(target_prefix), std::move(options));
}

std::string OcdbtDriver::DescribeKey(std::string_view key) {
  return tensorstore::StrCat(tensorstore::QuoteString(key),
                             " in OCDBT database at ",
                             io_handle_->DescribeLocation());
}

Result<KvStore> OcdbtDriver::GetBase(std::string_view path,
                                     const Transaction& transaction) const {
  return base_;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_ocdbt::OcdbtDriverSpec>
    registration;
}  // namespace
