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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_H_

/// \file
///
/// Defines types related to the in-memory representation of the version tree.
///
/// The version tree nodes contain references to:
///
/// - other version tree nodes (represented by `VersionTreeNode`), and
/// - root b+tree node (represented by `BtreeNode`).

#include <cstdint>
#include <limits>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Representation of a generation number (commit number).
using GenerationNumber = uint64_t;

/// Checks if a generation number is value.
inline constexpr bool IsValidGenerationNumber(
    GenerationNumber generation_number) {
  return generation_number != 0;
}

/// In-memory representation of a version tree node height.
using VersionTreeHeight = uint8_t;

using VersionTreeArityLog2 = uint8_t;

constexpr VersionTreeArityLog2 kMaxVersionTreeArityLog2 = 16;

/// In-memory representation of a b+tree version commit timestamp.
struct CommitTime {
  CommitTime() = default;

  constexpr explicit CommitTime(uint64_t ns_since_unix_epoch)
      : value(ns_since_unix_epoch) {}

  static Result<CommitTime> FromAbslTime(absl::Time time);

  explicit operator absl::Time() const;

  /// Nanoseconds since Unix epoch.
  using Value = uint64_t;
  Value value;
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.value);
  };

  constexpr static CommitTime min() { return CommitTime{0}; }
  constexpr static CommitTime max() {
    return CommitTime{std::numeric_limits<Value>::max()};
  }

  friend bool operator==(CommitTime a, CommitTime b) {
    return a.value == b.value;
  }
  friend bool operator<(CommitTime a, CommitTime b) {
    return a.value < b.value;
  }
  friend bool operator<=(CommitTime a, CommitTime b) {
    return a.value <= b.value;
  }
  friend bool operator>(CommitTime a, CommitTime b) {
    return a.value > b.value;
  }
  friend bool operator>=(CommitTime a, CommitTime b) {
    return a.value >= b.value;
  }
  friend bool operator!=(CommitTime a, CommitTime b) { return !(a == b); }
  friend std::ostream& operator<<(std::ostream& os, CommitTime x);

  template <typename Sink>
  friend void AbslStringify(Sink&& sink, CommitTime x) {
    sink.Append(absl::FormatTime(static_cast<absl::Time>(x)));
  }
};

/// In-memory representation of a reference to a b+tree version.
struct BtreeGenerationReference {
  /// Reference to the root node.
  BtreeNodeReference root;

  /// Generation number of the root.  Must be non-zero.
  GenerationNumber generation_number;

  /// Height of the root node.  This is stored explicitly for the root node; for
  /// all other references to sub-trees, the height is implicitly equal to one
  /// less than the height of the parent.
  BtreeNodeHeight root_height;

  /// Time at which this version was created.
  CommitTime commit_time;

  friend bool operator==(const BtreeGenerationReference& a,
                         const BtreeGenerationReference& b);
  friend bool operator!=(const BtreeGenerationReference& a,
                         const BtreeGenerationReference& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const BtreeGenerationReference& x);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.root, x.generation_number, x.root_height, x.commit_time);
  };
};

/// In-memory representation of a reference to a version tree node.
struct VersionNodeReference {
  /// Location of the encoded tree node.
  IndirectDataReference location;

  /// Last generation included in the node.
  GenerationNumber generation_number;

  /// Height of referenced subtree.
  VersionTreeHeight height;

  /// Total number of B+tree roots referenced within this subtree.
  GenerationNumber num_generations;

  /// First commit time reachable from this version tree node.
  CommitTime commit_time;

  friend bool operator==(const VersionNodeReference& a,
                         const VersionNodeReference& b);
  friend bool operator!=(const VersionNodeReference& a,
                         const VersionNodeReference& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const VersionNodeReference& e);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.location, x.generation_number, x.commit_time);
  };
};

/// In-memory representation of a version tree node.
struct VersionTreeNode {
  /// Height of the version tree node.
  ///
  /// Leaf nodes have a height of 0.
  VersionTreeHeight height;

  /// Log base-2 of the max arity of the version tree.
  VersionTreeArityLog2 version_tree_arity_log2;

  /// Latest generation number referenced by this subtree.
  GenerationNumber generation_number() const;

  using LeafNodeEntries = std::vector<BtreeGenerationReference>;
  using InteriorNodeEntries = std::vector<VersionNodeReference>;
  using Entries = std::variant<LeafNodeEntries, InteriorNodeEntries>;

  /// Child entries.
  Entries entries;

  friend bool operator==(const VersionTreeNode& a, const VersionTreeNode& b);
  friend bool operator!=(const VersionTreeNode& a, const VersionTreeNode& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const VersionTreeNode& e);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.height, x.version_tree_arity_log2, x.entries);
  };
};

/// Returns the maximum version tree height.
inline VersionTreeHeight GetMaxVersionTreeHeight(
    VersionTreeArityLog2 version_tree_arity_log2) {
  return 63 / version_tree_arity_log2 - 1;
}

/// Returns the min/max range of generation numbers that may be stored in the
/// same version tree leaf node as `generation_number`.
std::pair<GenerationNumber, GenerationNumber>
GetVersionTreeLeafNodeRangeContainingGeneration(
    VersionTreeArityLog2 version_tree_arity_log2,
    GenerationNumber generation_number);

/// Validates invariants for interior node entries.
///
/// This is called internally by `DecodeVersionTreeNode`, but may be called
/// directly for testing purposes.
absl::Status ValidateVersionTreeInteriorNodeEntries(
    VersionTreeArityLog2 version_tree_arity_log2, VersionTreeHeight height,
    const VersionTreeNode::InteriorNodeEntries& entries);

/// Validates invariants for leaf node entries.
///
/// This is called internally by `DecodeVersionTreeNode`, but may be called
/// directly for testing purposes.
absl::Status ValidateVersionTreeLeafNodeEntries(
    VersionTreeArityLog2 version_tree_arity_log2,
    const VersionTreeNode::LeafNodeEntries& entries);

/// Decodes a version tree node, and validates invariants.
Result<VersionTreeNode> DecodeVersionTreeNode(const absl::Cord& encoded,
                                              const BasePath& base_path);

/// Encodes a version tree node.
///
/// If `NDEBUG` is not defined, also CHECKs invariants.
Result<absl::Cord> EncodeVersionTreeNode(const Config& config,
                                         const VersionTreeNode& node);

/// Validates that a version tree node has the expected generation number,
/// height and configuration.
///
/// TODO(jbms): Also validate `num_generations`, `commit_time`
absl::Status ValidateVersionTreeNodeReference(
    const VersionTreeNode& node, const Config& config,
    GenerationNumber last_generation_number, VersionTreeHeight height);

// Specifies an inclusive upper bound on a version's commit time.
struct CommitTimeUpperBound {
  CommitTime commit_time;
};

// Specifies a version.
//
// - `GenerationNumber` indicates an exact generation number.
//
// - `CommitTime` indicates an exact commit time.
//
// - `CommitTimeUpperBound` indicates an inclusive upper bound on the commit
// time.
using VersionSpec =
    std::variant<GenerationNumber, CommitTime, CommitTimeUpperBound>;

inline bool IsVersionSpecExact(VersionSpec version_spec) {
  return !std::holds_alternative<CommitTimeUpperBound>(version_spec);
}

std::string FormatVersionSpec(VersionSpec version_spec);

// Compares `version_spec` to `ref`.
//
// Returns:
//
//   `-1` if `version_spec` is older than `ref`.
//   `0` if `version_spec` exactly matches `ref`.
//   `1` if `version_spec` is newer than `ref`.
//
// This function treats `CommitTimeUpperBound` the same as `CommitTime`.
int CompareVersionSpecToVersion(VersionSpec version_spec,
                                const BtreeGenerationReference& ref);

inline int CompareVersionSpecToVersion(GenerationNumber generation_number,
                                       const BtreeGenerationReference& ref) {
  return generation_number < ref.generation_number      ? -1
         : (generation_number == ref.generation_number) ? 0
                                                        : 1;
}

inline int CompareVersionSpecToVersion(CommitTime commit_time,
                                       const BtreeGenerationReference& ref) {
  return commit_time < ref.commit_time      ? -1
         : (commit_time == ref.commit_time) ? 0
                                            : 1;
}

inline int CompareVersionSpecToVersion(CommitTimeUpperBound commit_time,
                                       const BtreeGenerationReference& ref) {
  return CompareVersionSpecToVersion(commit_time.commit_time, ref);
}

/// Finds the generation with the specified `generation_number`.
///
/// Returns `nullptr` if not present.
const BtreeGenerationReference* FindVersion(
    span<const BtreeGenerationReference> versions,
    GenerationNumber generation_number);

const BtreeGenerationReference* FindVersion(
    span<const BtreeGenerationReference> versions, CommitTime commit_time);

const BtreeGenerationReference* FindVersion(
    span<const BtreeGenerationReference> versions,
    CommitTimeUpperBound commit_time);

const BtreeGenerationReference* FindVersion(
    span<const BtreeGenerationReference> versions, VersionSpec version_spec);

span<const BtreeGenerationReference>::iterator FindVersionLowerBound(
    span<const BtreeGenerationReference> versions,
    GenerationNumber generation_number);

inline span<const BtreeGenerationReference>::iterator FindVersionLowerBound(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const BtreeGenerationReference> versions,
    GenerationNumber generation_number) {
  return internal_ocdbt::FindVersionLowerBound(versions, generation_number);
}

span<const BtreeGenerationReference>::iterator FindVersionLowerBound(
    span<const BtreeGenerationReference> versions, CommitTime commit_time);

span<const BtreeGenerationReference>::iterator FindVersionUpperBound(
    span<const BtreeGenerationReference> versions,
    GenerationNumber generation_number);

span<const BtreeGenerationReference>::iterator FindVersionUpperBound(
    span<const BtreeGenerationReference> versions, CommitTime commit_time);

const VersionNodeReference* FindVersion(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children,
    GenerationNumber generation_number);

const VersionNodeReference* FindVersion(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children, CommitTime commit_time);

const VersionNodeReference* FindVersion(
    span<const VersionNodeReference> children, CommitTime commit_time);

inline const VersionNodeReference* FindVersion(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children, CommitTime commit_time) {
  return FindVersion(children, commit_time);
}

inline const VersionNodeReference* FindVersion(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children,
    CommitTimeUpperBound commit_time) {
  return FindVersion(children, commit_time.commit_time);
}

const VersionNodeReference* FindVersion(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children, VersionSpec version_spec);

span<const VersionNodeReference>::iterator FindVersionLowerBound(
    VersionTreeArityLog2 version_tree_arity_log2,
    span<const VersionNodeReference> children,
    GenerationNumber generation_number);

span<const VersionNodeReference>::iterator FindVersionLowerBound(
    span<const VersionNodeReference> children, CommitTime commit_time);

span<const VersionNodeReference>::iterator FindVersionUpperBound(
    span<const VersionNodeReference> children,
    GenerationNumber generation_number);

span<const VersionNodeReference>::iterator FindVersionUpperBound(
    span<const VersionNodeReference> children, CommitTime commit_time);

#ifndef NDEBUG
/// Checks invariants.
///
/// These invariants are all verified by `DecodeVersionTreeNode` using a
/// separate code path.  However, this is used in debug mode by
/// `EncodeVersionTreeNode` to verify invariants before writing.
void CheckVersionTreeNodeInvariants(const VersionTreeNode& node);
#endif  // NDEBUG

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_H_
