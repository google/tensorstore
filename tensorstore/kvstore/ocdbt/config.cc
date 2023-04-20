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

#include "tensorstore/kvstore/ocdbt/config.h"

#include <atomic>
#include <type_traits>
#include <variant>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/raw_bytes_hex.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_binding/std_variant.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace jb = ::tensorstore::internal_json_binding;

constexpr auto NoCompressionJsonBinder = jb::Constant([] { return nullptr; });
constexpr auto ZstdCompressionJsonBinder = jb::Object(
    jb::Member("id", jb::Constant([] { return "zstd"; })),
    jb::Member(
        "level",
        jb::Projection<&Config::ZstdCompression::level>(
            jb::DefaultInitializedValue(jb::Integer<int32_t>(
                riegeli::ZstdWriterBase::Options::kMinCompressionLevel,
                riegeli::ZstdWriterBase::Options::kMaxCompressionLevel)))));
constexpr auto ConfigCompressionJsonBinder =
    jb::Variant(NoCompressionJsonBinder, ZstdCompressionJsonBinder);

constexpr auto ManifestKindJsonBinder = [](auto is_loading, const auto& options,
                                           auto* obj, auto* j) {
  // This is defined as a lambda that forwards to the function returned by
  // `jb::Enum` to workaround a constexpr issue on MSVC 14.35.
  return jb::Enum<ManifestKind, std::string_view>({
      {ManifestKind::kSingle, "single"},
      {ManifestKind::kNumbered, "numbered"},
  })(is_loading, options, obj, j);
};

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ConfigConstraints,
    jb::Object(
        jb::Member("uuid", jb::Projection<&ConfigConstraints::uuid>(
                               jb::Optional(jb::RawBytesHex))),
        jb::Member("manifest_kind",
                   jb::Projection<&ConfigConstraints::manifest_kind>(
                       jb::Optional(ManifestKindJsonBinder))),
        jb::Member(
            "max_inline_value_bytes",
            jb::Projection<&ConfigConstraints::max_inline_value_bytes>(
                jb::Optional(jb::Integer<uint32_t>(0, kMaxInlineValueLength)))),
        jb::Member(
            "max_decoded_node_bytes",
            jb::Projection<&ConfigConstraints::max_decoded_node_bytes>()),
        jb::Member("version_tree_arity_log2",
                   jb::Projection<&ConfigConstraints::version_tree_arity_log2>(
                       jb::Optional(
                           jb::Integer<uint8_t>(1, kMaxVersionTreeArityLog2)))),
        jb::Member("compression",
                   jb::Projection<&ConfigConstraints::compression>(
                       jb::Optional(ConfigCompressionJsonBinder)))))

void to_json(::nlohmann::json& j, const Config::Compression& compression) {
  ConfigCompressionJsonBinder(/*is_loading=*/std::false_type{},
                              /*options=*/IncludeDefaults{true}, &compression,
                              &j)
      .IgnoreError();
}

void to_json(::nlohmann::json& j, const Uuid& value) {
  jb::RawBytesHex(/*is_loading=*/std::false_type{},
                  /*options=*/jb::NoOptions{}, &value, &j)
      .IgnoreError();
}

void to_json(::nlohmann::json& j, ManifestKind value) {
  ManifestKindJsonBinder(/*is_loading=*/std::false_type{},
                         /*options=*/jb::NoOptions{}, &value, &j)
      .IgnoreError();
}

absl::Status ValidateConfig(const Config& config,
                            const ConfigConstraints& constraints) {
  const auto validate = [&](const char* name, const auto& config_value,
                            const auto& constraint_value) -> absl::Status {
    if (constraint_value && *constraint_value != config_value) {
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Configuration mismatch on ", name, ": expected ",
          ::nlohmann::json(*constraint_value).dump(), " but received ",
          ::nlohmann::json(config_value).dump()));
    }
    return absl::OkStatus();
  };
#define TENSORTORE_INTERNAL_DO_VALIDATE(X)                            \
  TENSORSTORE_RETURN_IF_ERROR(validate(#X, config.X, constraints.X)); \
  /**/

  TENSORTORE_INTERNAL_DO_VALIDATE(uuid)
  TENSORTORE_INTERNAL_DO_VALIDATE(manifest_kind)
  TENSORTORE_INTERNAL_DO_VALIDATE(max_inline_value_bytes)
  TENSORTORE_INTERNAL_DO_VALIDATE(max_decoded_node_bytes)
  TENSORTORE_INTERNAL_DO_VALIDATE(version_tree_arity_log2)
  TENSORTORE_INTERNAL_DO_VALIDATE(compression)

#undef TENSORTORE_INTERNAL_DO_VALIDATE

  return absl::OkStatus();
}

absl::Status CreateConfig(const ConfigConstraints& constraints,
                          kvstore::SupportedFeatures supported_features,
                          Config& config) {
  Config default_config;
  config.uuid = constraints.uuid ? *constraints.uuid : Uuid::Generate();
  if (constraints.manifest_kind) {
    config.manifest_kind = *constraints.manifest_kind;
  } else {
    using kvstore::SupportedFeatures;
    if ((supported_features &
         SupportedFeatures::kSingleKeyAtomicReadModifyWrite) !=
        SupportedFeatures{}) {
      config.manifest_kind = ManifestKind::kSingle;
    } else if ((supported_features &
                SupportedFeatures::kAtomicWriteWithoutOverwrite) !=
               SupportedFeatures{}) {
      config.manifest_kind = ManifestKind::kNumbered;
    } else {
#if 0
      return absl::InvalidArgumentError(
          "Cannot choose OCDBT manifest_kind automatically because no kind is "
          "known to be safe with the underlying key-value store");
#else
      // For compatibility with existing code, temporarily default to
      // single-file manifest even if unsafe.
      config.manifest_kind = ManifestKind::kSingle;
#endif
    }
  }
  config.max_inline_value_bytes = constraints.max_inline_value_bytes.value_or(
      default_config.max_inline_value_bytes);
  config.max_decoded_node_bytes = constraints.max_decoded_node_bytes.value_or(
      default_config.max_decoded_node_bytes);
  config.version_tree_arity_log2 = constraints.version_tree_arity_log2.value_or(
      default_config.version_tree_arity_log2);
  config.compression =
      constraints.compression.value_or(default_config.compression);
  return absl::OkStatus();
}

ConfigConstraints::ConfigConstraints(const Config& config)
    : uuid(config.uuid),
      max_inline_value_bytes(config.max_inline_value_bytes),
      max_decoded_node_bytes(config.max_decoded_node_bytes),
      version_tree_arity_log2(config.version_tree_arity_log2),
      compression(config.compression) {}

ConfigState::ConfigState()
    : supported_features_for_manifest_{kvstore::SupportedFeatures::kNone} {}

ConfigState::ConfigState(
    const ConfigConstraints& constraints,
    kvstore::SupportedFeatures supported_features_for_manifest)
    : constraints_(constraints),
      supported_features_for_manifest_(supported_features_for_manifest) {}

absl::Status ConfigState::ValidateNewConfig(const Config& config) {
  if (!config_set_.load(std::memory_order_acquire)) {
    absl::MutexLock lock(&mutex_);
    TENSORSTORE_RETURN_IF_ERROR(ValidateConfig(config, constraints_));
    config_ = config;
    constraints_ = ConfigConstraints(config);
    config_set_.store(true, std::memory_order_release);
    return absl::OkStatus();
  }
  return ValidateConfig(config, constraints_);
}

const Config* ConfigState::GetExistingConfig() const {
  if (!config_set_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  return &config_;
}

Result<Config> ConfigState::CreateNewConfig() {
  if (!config_set_.load(std::memory_order_acquire)) {
    absl::MutexLock lock(&mutex_);
    Config config;
    TENSORSTORE_RETURN_IF_ERROR(
        CreateConfig(constraints_, supported_features_for_manifest_, config));
    return config;
  }
  // Note: This function will only be called with `config_set_ == true` in the
  // case of multiple concurrent local attempts to create the initial manifest.
  //
  // TODO(jbms): Consider returning an error instead in this case.
  return config_;
}

ConfigConstraints ConfigState::GetConstraints() const {
  if (!config_set_.load(std::memory_order_acquire)) {
    absl::MutexLock lock(&mutex_);
    return constraints_;
  }
  return constraints_;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
