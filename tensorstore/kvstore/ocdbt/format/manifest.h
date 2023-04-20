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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_MANIFEST_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_MANIFEST_H_

/// \file
///
/// Defines the in-memory representation of the manifest.
///
/// The top-level data structure of the OCDBT database is the manifest,
/// represented by the `Manifest` class.  It contains references to encoded:
///
/// - version tree nodes, represented by the `VersionTreeNode` class, and
/// - b+tree roots, represented by the `BtreeNode` class.

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Returns the path to the manifest given a base directory path.
///
/// Equal to `base_path + "manifest"`.
std::string GetManifestPath(std::string_view base_path);

/// Returns the path to the numbered manifest given a base directory path.
///
/// Equal to `base_path + "manifest.XXXXXXXXXXXXXXXX"`.
std::string GetNumberedManifestPath(std::string_view base_path,
                                    GenerationNumber generation_number);

// Number of manifests to keep.
//
// TODO(jbms): Add time-based criteria in addition to this.
constexpr GenerationNumber kNumNumberedManifestsToKeep = 128;

/// In-memory representation of a manifest.
struct Manifest {
  /// Database configuration.
  Config config;

  /// Versions stored inline in the manifest, with consecutive increasing
  /// version numbers.
  VersionTreeNode::LeafNodeEntries versions;

  /// Version tree nodes stored out-of-line.
  ///
  /// `version_tree_nodes[i]` references a node of height
  /// `version_tree_nodes.size() - i`.  Version tree nodes of height 0 are never
  /// directly referenced by the manifest.
  VersionTreeNode::InteriorNodeEntries version_tree_nodes;

  /// Returns the latest b+tree generation reference.
  const BtreeGenerationReference& latest_version() const {
    return versions.back();
  }

  /// Returns the latest generation number.
  GenerationNumber latest_generation() const {
    return latest_version().generation_number;
  }

  friend bool operator==(const Manifest& a, const Manifest& b);
  friend bool operator!=(const Manifest& a, const Manifest& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const Manifest& e);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.config, x.versions, x.version_tree_nodes);
  };
};

inline GenerationNumber GetLatestGeneration(const Manifest* manifest) {
  return manifest ? manifest->latest_generation() : 0;
}

/// Pairs a `Manifest` with its storage timestamp.
struct ManifestWithTime {
  std::shared_ptr<const Manifest> manifest;
  absl::Time time;
};

/// Decodes the manifest.
Result<Manifest> DecodeManifest(const absl::Cord& encoded);

/// Encodes the manifest.
Result<absl::Cord> EncodeManifest(const Manifest& manifest,
                                  bool encode_as_single = false);

/// Iterates over the version tree nodes that may be referenced from the
/// manifest with the given latest `generation_number`.
///
/// The `callback` is invoked with consecutive values of `height`, starting at
/// `1`, with the allowed range of generations for each version tree node.  The
/// manifest must not reference any version tree nodes with heights that are not
/// passed to `callback`, and must reference at most one version tree node for
/// each height that is passed to `callback`.
void ForEachManifestVersionTreeNodeRef(
    GenerationNumber generation_number, uint8_t version_tree_arity_log2,
    absl::FunctionRef<void(GenerationNumber min_generation_number,
                           GenerationNumber max_generation_number,
                           VersionTreeHeight height)>
        callback);

#ifndef NDEBUG
/// Checks invariants.
///
/// These invariants are all verified by `DecodeManifest` using a separate code
/// path.  However, this is used in debug mode by `EncodeManifest` to verify
/// invariants before writing.
void CheckManifestInvariants(const Manifest& manifest,
                             bool assume_single = false);
#endif  // NDEBUG

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_MANIFEST_H_
