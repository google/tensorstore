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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_DUMP_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_DUMP_H_

/// \file
///
/// Debugging tool for printing data structures.

#include <string>
#include <string_view>

#include <nlohmann/json.hpp>
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Combines an IndirectDataReference with a label indicating the value type.
struct LabeledIndirectDataReference {
  IndirectDataKind kind;
  IndirectDataReference location;

  /// Parses from a string of the form `"<label>:<file_id>:<offset>:<length>"`.
  ///
  /// The `<label>` indicates the type of the referenced value, and must be one
  /// of:
  ///
  /// - "value": raw value for a key stored indirectly
  /// - "btreenode": B+Tree node
  /// - "versionnode": Version tree node
  static Result<LabeledIndirectDataReference> Parse(std::string_view s);
};

/// Dumps a manifest to a JSON representation.
::nlohmann::json Dump(const Manifest& manifest);

/// Dumps a manifest to a JSON representation.
///
/// Note: The returned JSON value may contain `::nlohmann::json::binary_t`
/// string values that will not be printed in a nice way in JSON format.  For
/// better output, `internal_python::PrettyPrintJsonAsPython` may be used to
/// print the result instead.
::nlohmann::json Dump(const BtreeNode& node);

/// Dumps a manifest to a JSON representation.
::nlohmann::json Dump(const VersionTreeNode& node);

/// Converts a JSON manifest dump to the on-disk manifest format.
///
/// \param dumped_manifest The JSON dump of the manifest.
///
/// \return The on-disk representation of the manifest.
Result<absl::Cord> UndumpManifest(::nlohmann::json dumped_manifest);

/// Converts a JSON version tree node dump to the on-disk version tree node
/// format.
///
/// \param dumped_config The JSON dump of the version tree node's config.
/// \param dumped_node The JSON dump of the version tree node.
///
/// \return The on-disk representation of the version tree node.
Result<absl::Cord> UndumpVersionTreeNode(::nlohmann::json dumped_config,
                                         ::nlohmann::json dumped_node);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_DUMP_H_
