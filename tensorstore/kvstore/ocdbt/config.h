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

#ifndef TENSORSTORE_KVSTORE_OCDBT_CONFIG_H_
#define TENSORSTORE_KVSTORE_OCDBT_CONFIG_H_

#include <atomic>
#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/supported_features.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Constraints on the configuration that may be indicated by the kvstore spec.
struct ConfigConstraints {
  ConfigConstraints() = default;
  explicit ConfigConstraints(const Config& config);

  std::optional<Uuid> uuid;
  std::optional<ManifestKind> manifest_kind;
  std::optional<uint32_t> max_inline_value_bytes;
  std::optional<uint32_t> max_decoded_node_bytes;
  std::optional<uint8_t> version_tree_arity_log2;
  std::optional<Config::Compression> compression;

  friend bool operator==(const ConfigConstraints& a,
                         const ConfigConstraints& b);
  friend bool operator!=(const ConfigConstraints& a,
                         const ConfigConstraints& b) {
    return !(a == b);
  }

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ConfigConstraints,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults)

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.uuid, x.manifest_kind, x.max_inline_value_bytes,
             x.max_decoded_node_bytes, x.version_tree_arity_log2,
             x.compression);
  };
};

/// Tracks the configuration for an open database.
///
/// Initially, when the database is opened, the manifest has not yet been read
/// (and may not even exist) and therefore the configuration is not known.
///
/// The configuration is considered *known* once it has been successfully read
/// or written.
///
/// Once the configuration is known, it is an error for it to change.
///
/// FIXME(jbms): Because of the open kvstore cache, there is a potential for
/// this caching of the configuration to cause problems in the case that the
/// ocdbt kvstore is opened, then deleted from its underlying store, then
/// recreated, then opened again with the same cache.  Instead of the current
/// behavior, the caching of the configuration should take into account a spec
/// option like `recheck_cached_metadata`.
class ConfigState : public internal::AtomicReferenceCount<ConfigState> {
 public:
  ConfigState();
  explicit ConfigState(
      const ConfigConstraints& constraints,
      kvstore::SupportedFeatures supported_features_for_manifest);
  absl::Status ValidateNewConfig(const Config& config);
  const Config* GetExistingConfig() const;
  Result<Config> CreateNewConfig();
  ConfigConstraints GetConstraints() const;

 private:
  mutable absl::Mutex mutex_;
  ConfigConstraints constraints_;
  Config config_;
  kvstore::SupportedFeatures supported_features_for_manifest_;
  std::atomic<bool> config_set_{false};
};

using ConfigStatePtr = internal::IntrusivePtr<ConfigState>;

absl::Status ValidateConfig(const Config& config,
                            const ConfigConstraints& constraints);

absl::Status CreateConfig(const ConfigConstraints& constraints,
                          kvstore::SupportedFeatures supported_features,
                          Config& config);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_CONFIG_H_
