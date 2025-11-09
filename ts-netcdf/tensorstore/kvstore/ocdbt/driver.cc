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

#include <cassert>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
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
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/auto_detect.h"
#include "tensorstore/kvstore/common_metrics.h"
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
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/read.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/transactional_btree_writer.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_variant.h"  // IWYU pragma: keep

using ::tensorstore::kvstore::ListReceiver;

namespace tensorstore {
namespace internal_ocdbt {
namespace {

namespace jb = ::tensorstore::internal_json_binding;

struct OcdbtMetrics : public internal_kvstore::CommonReadMetrics,
                      public internal_kvstore::CommonWriteMetrics {};

auto ocdbt_metrics = []() -> OcdbtMetrics {
  return {TENSORSTORE_KVSTORE_COMMON_READ_METRICS(ocdbt),
          TENSORSTORE_KVSTORE_COMMON_WRITE_METRICS(ocdbt)};
}();

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

constexpr auto VersionSpecJsonBinder = [](auto is_loading, const auto& options,
                                          auto* obj, auto* j) {
  if constexpr (is_loading) {
    if (auto* s = j->template get_ptr<const std::string*>()) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto commit_time,
                                   ParseCommitTimeFromUrl(*s));
      *obj = CommitTimeUpperBound{commit_time};
      return absl::OkStatus();
    } else {
      return jb::Integer<GenerationNumber>(1)(
          is_loading, options, &obj->template emplace<GenerationNumber>(), j);
    }
  } else {
    if (auto* v = std::get_if<GenerationNumber>(obj)) {
      *j = *v;
    } else {
      *j = FormatVersionSpecForUrl(*obj);
    }
    return absl::OkStatus();
  }
};

constexpr std::string_view kDefaultDataPrefix = "d/";

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    OcdbtDriverSpecData,
    jb::Object(
        jb::Member("base", jb::Projection<&OcdbtDriverSpecData::base>()),
        jb::Member("manifest",
                   jb::Projection<&OcdbtDriverSpecData::manifest>()),
        jb::Initialize([](auto* obj) {
          internal::EnsureDirectoryPath(obj->base.path);
          if (obj->manifest) {
            internal::EnsureDirectoryPath(obj->manifest->path);
          }
          return absl::OkStatus();
        }),
        jb::Member("config", jb::Projection<&OcdbtDriverSpecData::config>(
                                 jb::DefaultInitializedValue())),
        jb::Projection<&OcdbtDriverSpecData::data_file_prefixes>(jb::Sequence(
            jb::Member("value_data_prefix",
                       jb::Projection<&DataFilePrefixes::value>(
                           jb::DefaultValue([](auto* v) {
                             *v = kDefaultDataPrefix;
                           }))),
            jb::Member("btree_node_data_prefix",
                       jb::Projection<&DataFilePrefixes::btree_node>(
                           jb::DefaultValue([](auto* v) {
                             *v = kDefaultDataPrefix;
                           }))),
            jb::Member("version_tree_node_data_prefix",
                       jb::Projection<&DataFilePrefixes::version_tree_node>(
                           jb::DefaultValue([](auto* v) {
                             *v = kDefaultDataPrefix;
                           }))))),
        jb::Member("assume_config",
                   jb::Projection<&OcdbtDriverSpecData::assume_config>(
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
            jb::Projection<&OcdbtDriverSpecData::data_copy_concurrency>()),
        jb::Member("version",
                   jb::Projection<&OcdbtDriverSpecData::version_spec>(
                       jb::Optional(VersionSpecJsonBinder)))));

Result<kvstore::Spec> OcdbtDriverSpec::GetBase(std::string_view path) const {
  return data_.base;
}

Result<std::string> OcdbtDriverSpec::ToUrl(std::string_view path) const {
  if (data_.manifest) {
    return absl::InvalidArgumentError(
        "OCDBT URL syntax not supported with separate manifest kvstore");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto base_url,
                               data_.base.driver->ToUrl(data_.base.path));
  std::string version_string;
  if (data_.version_spec) {
    version_string = FormatVersionSpecForUrl(*data_.version_spec);
  }
  return absl::StrCat(base_url, "|", id, ":", version_string.empty() ? "" : "@",
                      version_string, version_string.empty() ? "" : "/",
                      internal::PercentEncodeKvStoreUriPath(path));
}

Future<kvstore::DriverPtr> OcdbtDriverSpec::DoOpen() const {
  auto base_kvstore_future = kvstore::Open(data_.base);
  Future<kvstore::KvStore> manifest_kvstore_future =
      data_.manifest ? kvstore::Open(*data_.manifest)
                     : Future<kvstore::KvStore>(kvstore::KvStore{});
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const OcdbtDriverSpec>(this)](
          kvstore::KvStore& base_kvstore,
          kvstore::KvStore& manifest_kvstore) -> Result<kvstore::DriverPtr> {
        auto driver = internal::MakeIntrusivePtr<OcdbtDriver>();
        driver->base_ = std::move(base_kvstore);
        driver->manifest_kvstore_ = std::move(manifest_kvstore);

        auto supported_manifest_features =
            driver->base_.driver->GetSupportedFeatures(KeyRange::Prefix(
                tensorstore::StrCat(driver->base_.path, "manifest.")));

        driver->cache_pool_ = spec->data_.cache_pool;
        driver->data_copy_concurrency_ = spec->data_.data_copy_concurrency;
        driver->data_file_prefixes_ = spec->data_.data_file_prefixes;
        driver->experimental_read_coalescing_threshold_bytes_ =
            spec->data_.experimental_read_coalescing_threshold_bytes;
        driver->experimental_read_coalescing_merged_bytes_ =
            spec->data_.experimental_read_coalescing_merged_bytes;
        driver->experimental_read_coalescing_interval_ =
            spec->data_.experimental_read_coalescing_interval;
        driver->target_data_file_size_ = spec->data_.target_data_file_size;
        driver->version_spec_ = spec->data_.version_spec;

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

        TENSORSTORE_ASSIGN_OR_RETURN(
            auto config_state,
            ConfigState::Make(spec->data_.config, supported_manifest_features,
                              spec->data_.assume_config));

        driver->io_handle_ = internal_ocdbt::MakeIoHandle(
            driver->data_copy_concurrency_, driver->cache_pool_->get(),
            driver->base_,
            driver->manifest_kvstore_.driver ? driver->manifest_kvstore_
                                             : driver->base_,
            std::move(config_state), driver->data_file_prefixes_,
            driver->target_data_file_size_.value_or(kDefaultTargetBufferSize),
            std::move(read_coalesce_options));
        driver->coordinator_ = spec->data_.coordinator;
        if (!driver->coordinator_->address || driver->version_spec_) {
          if (!driver->version_spec_) {
            driver->btree_writer_ =
                MakeNonDistributedBtreeWriter(driver->io_handle_);
          }
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
      std::move(base_kvstore_future), std::move(manifest_kvstore_future));
}

absl::Status OcdbtDriverSpec::ApplyOptions(
    kvstore::DriverSpecOptions&& options) {
  if (options.minimal_spec) {
    data_.config = {};
    data_.assume_config = false;
  }
  return data_.base.driver.Set(std::move(options));
}

absl::Status OcdbtDriver::GetBoundSpecData(OcdbtDriverSpecData& spec) const {
  TENSORSTORE_ASSIGN_OR_RETURN(spec.base.driver, base_.driver->GetBoundSpec());
  spec.base.path = base_.path;
  if (manifest_kvstore_.driver) {
    auto& manifest_spec = spec.manifest.emplace();
    TENSORSTORE_ASSIGN_OR_RETURN(manifest_spec.driver,
                                 base_.driver->GetBoundSpec());
    manifest_spec.path = manifest_kvstore_.path;
  }
  spec.data_copy_concurrency = data_copy_concurrency_;
  spec.cache_pool = cache_pool_;
  spec.config = io_handle_->config_state->GetConstraints();
  spec.assume_config = io_handle_->config_state->assume_config();
  spec.data_file_prefixes = data_file_prefixes_;
  spec.experimental_read_coalescing_threshold_bytes =
      experimental_read_coalescing_threshold_bytes_;
  spec.experimental_read_coalescing_merged_bytes =
      experimental_read_coalescing_merged_bytes_;
  spec.experimental_read_coalescing_interval =
      experimental_read_coalescing_interval_;
  spec.target_data_file_size = target_data_file_size_;
  spec.coordinator = coordinator_;
  spec.version_spec = version_spec_;
  return absl::Status();
}

kvstore::SupportedFeatures OcdbtDriver::GetSupportedFeatures(
    const KeyRange& key_range) const {
  return kvstore::SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
         kvstore::SupportedFeatures::kAtomicWriteWithoutOverwrite;
}

Future<kvstore::ReadResult> OcdbtDriver::Read(kvstore::Key key,
                                              kvstore::ReadOptions options) {
  ocdbt_metrics.read.Increment();
  return internal_ocdbt::NonDistributedRead(io_handle_, version_spec_,
                                            std::move(key), std::move(options));
}

void OcdbtDriver::ListImpl(kvstore::ListOptions options,
                           ListReceiver receiver) {
  ocdbt_metrics.list.Increment();
  return internal_ocdbt::NonDistributedList(
      io_handle_, version_spec_, std::move(options), std::move(receiver));
}

namespace {
absl::Status GetReadOnlyError(OcdbtDriver& driver) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Writing is not supported with version=",
      FormatVersionSpecForUrl(*driver.version_spec_), " specified"));
}
}  // namespace

Future<TimestampedStorageGeneration> OcdbtDriver::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  if (version_spec_) {
    return GetReadOnlyError(*this);
  }
  ocdbt_metrics.write.Increment();
  return btree_writer_->Write(std::move(key), std::move(value),
                              std::move(options));
}

Future<const void> OcdbtDriver::DeleteRange(KeyRange range) {
  if (version_spec_) {
    return GetReadOnlyError(*this);
  }
  ocdbt_metrics.delete_range.Increment();
  return btree_writer_->DeleteRange(std::move(range));
}

Future<const void> OcdbtDriver::ExperimentalCopyRangeFrom(
    const internal::OpenTransactionPtr& transaction, const KvStore& source,
    std::string target_prefix, kvstore::CopyRangeOptions options) {
  if (version_spec_) {
    return GetReadOnlyError(*this);
  }
  if (typeid(*source.driver) == typeid(OcdbtDriver)) {
    auto& source_driver = static_cast<OcdbtDriver&>(*source.driver);
    if (source.transaction != no_transaction) {
      return absl::UnimplementedError("Source transactions not supported");
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
           source_prefix_length = source.path.size(),
           transaction = std::move(transaction)](
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
            LinkResult(std::move(promise),
                       transaction ? internal_ocdbt::AddCopySubtree(
                                         &*self, *self->io_handle_, transaction,
                                         std::move(copy_node_options))
                                   : self->btree_writer_->CopySubtree(
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
  return tensorstore::StrCat(
      tensorstore::QuoteString(key), " in ",
      version_spec_
          ? tensorstore::StrCat("version ",
                                FormatVersionSpecForUrl(*version_spec_), " of ")
          : std::string{},
      "OCDBT database at ", io_handle_->DescribeLocation());
}

Result<KvStore> OcdbtDriver::GetBase(std::string_view path,
                                     const Transaction& transaction) const {
  return base_;
}

absl::Status OcdbtDriver::ReadModifyWrite(
    internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
    ReadModifyWriteSource& source) {
  if (version_spec_) {
    return GetReadOnlyError(*this);
  }
  if (!transaction || !transaction->atomic() || coordinator_->address) {
    return kvstore::Driver::ReadModifyWrite(transaction, phase, std::move(key),
                                            source);
  }
  return internal_ocdbt::AddReadModifyWrite(this, *io_handle_, transaction,
                                            phase, std::move(key), source);
}

absl::Status OcdbtDriver::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  if (version_spec_) {
    return GetReadOnlyError(*this);
  }
  if (!transaction->atomic() || coordinator_->address) {
    return kvstore::Driver::TransactionalDeleteRange(transaction,
                                                     std::move(range));
  }
  return internal_ocdbt::AddDeleteRange(this, *io_handle_, transaction,
                                        std::move(range));
}

void OcdbtDriver::TransactionalListImpl(
    const internal::OpenTransactionPtr& transaction,
    kvstore::ListOptions options, kvstore::ListReceiver receiver) {
  if (!transaction->atomic() || coordinator_->address) {
    return kvstore::Driver::TransactionalListImpl(
        transaction, std::move(options), std::move(receiver));
  }
  return internal_ocdbt::TransactionalListImpl(
      this, transaction, std::move(options), std::move(receiver));
}

Future<kvstore::ReadResult> OcdbtDriver::TransactionalRead(
    const internal::OpenTransactionPtr& transaction, Key key,
    ReadOptions options) {
  if (!transaction->atomic() || coordinator_->address) {
    return kvstore::Driver::TransactionalRead(transaction, std::move(key),
                                              std::move(options));
  }
  return internal_ocdbt::TransactionalReadImpl(
      this, *io_handle_, transaction, std::move(key), std::move(options));
}

namespace {
Result<kvstore::Spec> ParseOcdbtUrl(std::string_view url, kvstore::Spec base) {
  auto parsed = internal::ParseGenericUri(url);
  if (parsed.scheme != OcdbtDriverSpec::id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Scheme \"", OcdbtDriverSpec::id, ":\" not present in url"));
  }
  TENSORSTORE_RETURN_IF_ERROR(internal::EnsureNoQueryOrFragment(parsed));
  std::string_view encoded_path = parsed.path;
  std::optional<VersionSpec> version_spec;
  if (!encoded_path.empty() && encoded_path[0] == '@') {
    size_t version_end = encoded_path.find('/');
    std::string_view version_string = encoded_path.substr(1, version_end - 1);
    TENSORSTORE_ASSIGN_OR_RETURN(version_spec,
                                 ParseVersionSpecFromUrl(version_string));
    encoded_path = (version_end == std::string_view::npos)
                       ? std::string_view{}
                       : encoded_path.substr(version_end + 1);
  }
  std::string path = internal::PercentDecode(encoded_path);
  auto driver_spec = internal::MakeIntrusivePtr<OcdbtDriverSpec>();
  internal::EnsureDirectoryPath(base.path);
  driver_spec->data_.base = std::move(base);
  driver_spec->data_.cache_pool =
      Context::Resource<internal::CachePoolResource>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<internal::DataCopyConcurrencyResource>::DefaultSpec();
  driver_spec->data_.coordinator =
      Context::Resource<OcdbtCoordinatorResource>::DefaultSpec();
  driver_spec->data_.version_spec = version_spec;
  driver_spec->data_.data_file_prefixes.value = kDefaultDataPrefix;
  driver_spec->data_.data_file_prefixes.btree_node = kDefaultDataPrefix;
  driver_spec->data_.data_file_prefixes.version_tree_node = kDefaultDataPrefix;
  return {std::in_place, std::move(driver_spec), std::move(path)};
}
}  // namespace

}  // namespace internal_ocdbt
}  // namespace tensorstore

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_ocdbt::OcdbtDriverSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{tensorstore::internal_ocdbt::OcdbtDriverSpec::id,
                            tensorstore::internal_ocdbt::ParseOcdbtUrl};

const tensorstore::internal_kvstore::AutoDetectRegistration
    auto_detect_registration{
        tensorstore::internal_kvstore::AutoDetectDirectorySpec::SingleFile(
            tensorstore::internal_ocdbt::OcdbtDriverSpec::id,
            tensorstore::internal_ocdbt::kManifestFilename)};
}  // namespace
