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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DRIVER_H_
#define TENSORSTORE_KVSTORE_OCDBT_DRIVER_H_

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

// specializations
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_variant.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_variant.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_ocdbt {

struct OcdbtCoordinatorResource
    : public internal::ContextResourceTraits<OcdbtCoordinatorResource> {
  static constexpr char id[] = "ocdbt_coordinator";
  struct Spec {
    std::optional<std::string> address;
    std::optional<absl::Duration> lease_duration;
    RpcSecurityMethod::Ptr security;
  };
  using Resource = Spec;
};

struct OcdbtDriverSpecData {
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  kvstore::Spec base;
  ConfigConstraints config;
  std::optional<size_t> experimental_read_coalescing_threshold_bytes;
  Context::Resource<OcdbtCoordinatorResource> coordinator;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OcdbtDriverSpecData,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults,
                                          ::nlohmann::json::object_t)

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base, x.config, x.cache_pool, x.data_copy_concurrency,
             x.experimental_read_coalescing_threshold_bytes, x.coordinator);
  };
};

class OcdbtDriverSpec
    : public internal_kvstore::RegisteredDriverSpec<OcdbtDriverSpec,
                                                    OcdbtDriverSpecData> {
 public:
  static constexpr char id[] = "ocdbt";

  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status ApplyOptions(kvstore::DriverSpecOptions&& options) override;
};

class OcdbtDriver
    : public internal_kvstore::RegisteredDriver<OcdbtDriver, OcdbtDriverSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<const void> DeleteRange(KeyRange range) override;

  std::string DescribeKey(std::string_view key) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  absl::Status GetBoundSpecData(OcdbtDriverSpecData& spec) const;

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final;

  const Executor& executor() { return data_copy_concurrency_->executor; }

  IoHandle::Ptr io_handle_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  kvstore::KvStore base_;
  BtreeWriterPtr btree_writer_;
  std::optional<size_t> experimental_read_coalescing_threshold_bytes_;
  Context::Resource<OcdbtCoordinatorResource> coordinator_;
};

}  // namespace internal_ocdbt

namespace garbage_collection {
template <>
struct GarbageCollection<internal_ocdbt::OcdbtDriver> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const internal_ocdbt::OcdbtDriver& value) {
    garbage_collection::GarbageCollectionVisit(visitor, value.base_);
  }
};
}  // namespace garbage_collection

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DRIVER_H_
