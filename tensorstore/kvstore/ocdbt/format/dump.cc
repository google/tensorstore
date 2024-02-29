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

#include "tensorstore/kvstore/ocdbt/format/dump.h"

#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include <nlohmann/json.hpp>
#include "re2/re2.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_variant.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

Result<LabeledIndirectDataReference> LabeledIndirectDataReference::Parse(
    std::string_view s) {
  LabeledIndirectDataReference r;
  static LazyRE2 kPattern = {"([^:]+):([^:]*):([^:]*):([0-9]+):([0-9]+)"};
  std::string_view encoded_base_path, encoded_relative_path;
  if (!RE2::FullMatch(s, *kPattern, &r.label, &encoded_base_path,
                      &encoded_relative_path, &r.location.offset,
                      &r.location.length)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid indirect data reference: ", tensorstore::QuoteString(s)));
  }
  r.location.file_id.base_path = internal::PercentDecode(encoded_base_path);
  r.location.file_id.relative_path =
      internal::PercentDecode(encoded_relative_path);
  TENSORSTORE_RETURN_IF_ERROR(r.location.Validate(/*allow_missing=*/false));
  return r;
}

namespace {

namespace jb = tensorstore::internal_json_binding;

constexpr auto ConfigBinder = jb::Compose<ConfigConstraints>(
    [](auto is_loading, const auto& options, auto* obj, auto* constraints) {
      if constexpr (is_loading) {
        CreateConfig(constraints, *obj);
        if (ConfigConstraints(*obj) != *constraints) {
          return absl::InvalidArgumentError("Config is not fully specified");
        }
      } else {
        *constraints = ConfigConstraints(*obj);
      }
      return absl::OkStatus();
    });

static inline constexpr internal::AsciiSet
    kLabeledIndirectDataReferenceUnreservedChars{
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_./"};

constexpr auto LabeledIndirectDataReferenceBinder = [](auto is_loading,
                                                       const auto& options,
                                                       auto* obj, auto* j) {
  if constexpr (is_loading) {
    if (auto* s = j->template get_ptr<const std::string*>()) {
      TENSORSTORE_ASSIGN_OR_RETURN(*obj,
                                   LabeledIndirectDataReference::Parse(*s));
    } else {
      return internal_json::ExpectedError(*j, "string");
    }
  } else {
    if (obj->location.IsMissing()) {
      *j = ::nlohmann::json::value_t::discarded;
    } else {
      *j = tensorstore::StrCat(
          obj->label, ":",
          internal::PercentEncodeReserved(
              obj->location.file_id.base_path,
              kLabeledIndirectDataReferenceUnreservedChars),
          ":",
          internal::PercentEncodeReserved(
              obj->location.file_id.relative_path,
              kLabeledIndirectDataReferenceUnreservedChars),
          ":", obj->location.offset, ":", obj->location.length);
    }
  }
  return absl::OkStatus();
};

constexpr auto IndirectDataReferenceBinder(std::string_view label) {
  return jb::Compose<LabeledIndirectDataReference>(
      [label](auto is_loading, const auto& options, auto* obj, auto* j) {
        if constexpr (is_loading) {
          *obj = j->location;
        } else {
          j->location = *obj;
          j->label = label;
        }
        return absl::OkStatus();
      },
      LabeledIndirectDataReferenceBinder);
}

constexpr auto CommitTimeBinder = jb::Projection<&CommitTime::value>();

constexpr auto BtreeNodeStatisticsBinder = jb::Object(
    jb::Member(
        "num_indirect_value_bytes",
        jb::Projection<&BtreeNodeStatistics::num_indirect_value_bytes>()),
    jb::Member("num_tree_bytes",
               jb::Projection<&BtreeNodeStatistics::num_tree_bytes>()),
    jb::Member("num_keys", jb::Projection<&BtreeNodeStatistics::num_keys>()));

constexpr auto BtreeNodeReferenceBinder = jb::Object(
    jb::Member("location", jb::Projection<&BtreeNodeReference::location>(
                               IndirectDataReferenceBinder("btreenode"))),
    jb::Member("statistics", jb::Projection<&BtreeNodeReference::statistics>(
                                 BtreeNodeStatisticsBinder)));

constexpr auto BtreeGenerationReferenceBinder = jb::Object(
    jb::Member("root", jb::Projection<&BtreeGenerationReference::root>(
                           BtreeNodeReferenceBinder)),
    jb::Member("generation_number",
               jb::Projection<&BtreeGenerationReference::generation_number>()),
    jb::Member("root_height",
               jb::Projection<&BtreeGenerationReference::root_height>()),
    jb::Member("commit_time",
               jb::Projection<&BtreeGenerationReference::commit_time>(
                   CommitTimeBinder)));

constexpr auto VersionNodeReferenceBinder = jb::Object(
    jb::Member("location", jb::Projection<&VersionNodeReference::location>(
                               IndirectDataReferenceBinder("versionnode"))),
    jb::Member("generation_number",
               jb::Projection<&VersionNodeReference::generation_number>()),
    jb::Member("height", jb::Projection<&VersionNodeReference::height>()),
    jb::Member("num_generations",
               jb::Projection<&VersionNodeReference::num_generations>()),
    jb::Member(
        "commit_time",
        jb::Projection<&VersionNodeReference::commit_time>(CommitTimeBinder)));

constexpr auto ManifestBinder = jb::Object(
    jb::Member("config", jb::Projection<&Manifest::config>(ConfigBinder)),
    jb::Member("versions", jb::Projection<&Manifest::versions>(
                               jb::Array(BtreeGenerationReferenceBinder))),
    jb::Member("version_tree_nodes",
               jb::Projection<&Manifest::version_tree_nodes>(
                   jb::Array(VersionNodeReferenceBinder))));

constexpr auto BinaryCordBinder = [](auto is_loading, const auto& options,
                                     auto* obj, auto* j) {
  if constexpr (is_loading) {
    if (auto* b = j->template get_ptr<const ::nlohmann::json::binary_t*>()) {
      *obj = absl::Cord(std::string_view(
          reinterpret_cast<const char*>(b->data()), b->size()));
      return absl::OkStatus();
    } else if (auto* s = j->template get_ptr<const std::string*>()) {
      *obj = absl::Cord(*s);
      return absl::OkStatus();
    } else {
      return internal_json::ExpectedError(*j, "string or byte string");
    }
  } else {
    ::nlohmann::json::binary_t v;
    v.reserve(obj->size());
    for (std::string_view chunk : obj->Chunks()) {
      v.insert(v.end(), chunk.begin(), chunk.end());
    }
    *j = std::move(v);
    return absl::OkStatus();
  }
};

constexpr auto LeafNodeValueReferenceBinder = jb::Variant(
    jb::Member("inline_value", BinaryCordBinder),
    jb::Member("indirect_value", IndirectDataReferenceBinder("value")));

constexpr auto BtreeLeafNodeEntryBinder(std::string_view key_prefix) {
  return
      [=](std::false_type is_loading, const auto& options, auto* obj, auto* j) {
        ::nlohmann::json::binary_t key;
        key.insert(key.end(), key_prefix.begin(), key_prefix.end());
        key.insert(key.end(), obj->key.begin(), obj->key.end());
        ::nlohmann::json::object_t x{{"key", key}};
        TENSORSTORE_RETURN_IF_ERROR(LeafNodeValueReferenceBinder(
            std::false_type{}, IncludeDefaults{}, &obj->value_reference, &x));
        *j = std::move(x);
        return absl::OkStatus();
      };
}

constexpr auto BtreeInteriorNodeEntryBinder(std::string_view key_prefix) {
  return [=](std::false_type is_loading, const auto& options, auto* obj,
             auto* j) {
    ::nlohmann::json::binary_t key;
    key.insert(key.end(), key_prefix.begin(), key_prefix.end());
    key.insert(key.end(), obj->key.begin(), obj->key.end());
    auto common_prefix = key;
    common_prefix.resize(obj->subtree_common_prefix_length + key_prefix.size());
    ::nlohmann::json::object_t x;
    TENSORSTORE_RETURN_IF_ERROR(BtreeNodeReferenceBinder(
        std::false_type{}, IncludeDefaults{}, &obj->node, &x));
    x["key"] = key;
    x["subtree_common_prefix"] = common_prefix;
    *j = std::move(x);
    return absl::OkStatus();
  };
}

constexpr auto BtreeNodeBinder = jb::Object(
    jb::Member("height", jb::Projection<&BtreeNode::height>()),
    jb::Member("entries",
               [](auto is_loading, const auto& options, auto* obj, auto* j) {
                 return jb::Variant(
                     jb::Array(BtreeLeafNodeEntryBinder(obj->key_prefix)),
                     jb::Array(BtreeInteriorNodeEntryBinder(obj->key_prefix)))(
                     is_loading, options, &obj->entries, j);
               }));

constexpr auto VersionTreeNodeBinder = jb::Object(
    jb::Member("height", jb::Projection<&VersionTreeNode::height>()),
    jb::Member("version_tree_arity_log2",
               jb::Projection<&VersionTreeNode::version_tree_arity_log2>()),
    jb::Member("entries", jb::Projection<&VersionTreeNode::entries>(jb::Variant(
                              jb::Array(BtreeGenerationReferenceBinder),
                              jb::Array(VersionNodeReferenceBinder)))));

}  // namespace

::nlohmann::json Dump(const Manifest& manifest) {
  return jb::ToJson(manifest, ManifestBinder).value();
}

::nlohmann::json Dump(const BtreeNode& node) {
  return jb::ToJson(node, BtreeNodeBinder).value();
}

::nlohmann::json Dump(const VersionTreeNode& node) {
  return jb::ToJson(node, VersionTreeNodeBinder).value();
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
