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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_H_

/// \file
///
/// Defines types related to the in-memory representation of the b+tree.
///
/// The b+tree nodes contain references to:
///
/// - other b+tree nodes (represented by `BtreeNode`), and
/// - raw values.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ocdbt {

/// In-memory representation of the length of a key.
using KeyLength = uint16_t;

constexpr KeyLength kMaxKeyLength = 65535;

/// In-memory representation of statistics over a b+tree subtree.
///
/// These are included with every stored reference to a b+tree node.
struct BtreeNodeStatistics {
  /// Sum of lengths of all indirect values within all of the leaf nodes
  /// reachable from this node.
  uint64_t num_indirect_value_bytes;

  /// Sum of encoded sizes of all btree nodes reachable from this node,
  /// including itself.
  uint64_t num_tree_bytes;

  /// Number of keys within all of the leaf nodes reachable from this node.
  uint64_t num_keys;

  BtreeNodeStatistics& operator+=(const BtreeNodeStatistics& other);
  friend bool operator==(const BtreeNodeStatistics& a,
                         const BtreeNodeStatistics& b);
  friend bool operator!=(const BtreeNodeStatistics& a,
                         const BtreeNodeStatistics& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const BtreeNodeStatistics& x);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.num_indirect_value_bytes, x.num_tree_bytes, x.num_keys);
  };
};

/// In-memory representation of a reference to a b+tree node.
struct BtreeNodeReference {
  /// Location of the encoded tree node.
  IndirectDataReference location;

  /// Statistics for the referenced sub-tree.
  BtreeNodeStatistics statistics;

  friend bool operator==(const BtreeNodeReference& a,
                         const BtreeNodeReference& b);
  friend bool operator!=(const BtreeNodeReference& a,
                         const BtreeNodeReference& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const BtreeNodeReference& x);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.location, x.statistics);
  };
};

/// In-memory representation of a b+tree node height.
using BtreeNodeHeight = uint8_t;

using LeafNodeValueReference = std::variant<absl::Cord, IndirectDataReference>;

using LeafNodeValueKind = uint8_t;
constexpr LeafNodeValueKind kInlineValue = 0;
constexpr LeafNodeValueKind kOutOfLineValue = 1;

/// In-memory representation of a single key/value entry in a b+tree leaf node.
struct LeafNodeEntry {
  /// The key associated with this entry.
  ///
  /// This key should be interpreted as a suffix appended to the prefix
  /// specified by the parent of the containing node.
  ///
  /// This references a substring of the `BtreeNode::key_buffer` of the
  /// containing node.
  std::string_view key;

  /// Either the value associated with this entry, or a reference to the value.
  LeafNodeValueReference value_reference;

  /// The size of the stored value.
  uint64_t value_size() const {
    struct LeafNodeSizeVisitor {
      uint64_t operator()(const absl::Cord& direct) const {
        return direct.size();
      }
      uint64_t operator()(const IndirectDataReference& ref) const {
        return ref.length;
      }
    };
    return std::visit(LeafNodeSizeVisitor{}, value_reference);
  }

  friend bool operator==(const LeafNodeEntry& a, const LeafNodeEntry& b);
  friend bool operator!=(const LeafNodeEntry& a, const LeafNodeEntry& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const LeafNodeEntry& e);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.key, x.value_reference);
  };
};

/// In-memory representation of a single child entry in a b+tree interior
/// (non-leaf) node.
template <typename Key>
struct InteriorNodeEntryData {
  static_assert(std::is_same_v<Key, std::string> ||
                std::is_same_v<Key, std::string_view>);
  /// An inclusive lower bound for the keys contained within the child node.
  ///
  /// This key should be interpreted as a suffix appended to the prefix
  /// specified by the parent of the containing node.
  ///
  /// When `Key == std::string_view`, this references a substring of the
  /// `key_buffer` of the containing node.
  Key key;

  /// Length of prefix of `inclusive_min_key` that is common to all keys within
  /// the subtree rooted at the child node.  This portion of the key will not be
  /// stored within descendant nodes.
  ///
  /// This length EXCLUDES the length of `BtreeNode::key_prefix` that is part of
  /// the common prefix.
  KeyLength subtree_common_prefix_length;

  std::string_view key_suffix() const {
    return std::string_view(key).substr(subtree_common_prefix_length);
  }

  /// Reference to the child node.
  BtreeNodeReference node;

  friend bool operator==(const InteriorNodeEntryData& a,
                         const InteriorNodeEntryData& b) {
    return a.key == b.key &&
           a.subtree_common_prefix_length == b.subtree_common_prefix_length &&
           a.node == b.node;
  }
  friend bool operator!=(const InteriorNodeEntryData& a,
                         const InteriorNodeEntryData& b) {
    return !(a == b);
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.key, x.subtree_common_prefix_length, x.node);
  };
};

struct InteriorNodeEntry : public InteriorNodeEntryData<std::string_view> {
  friend std::ostream& operator<<(std::ostream& os, const InteriorNodeEntry& e);
};

/// In-memory representation of a b+tree node.
struct BtreeNode {
  /// Height of the sub-tree rooted at this node.
  ///
  /// Equal to 0 for leaf nodes.
  BtreeNodeHeight height;

  /// Common prefix for every entry in `entries`.
  ///
  /// References a substring of `key_buffer`.
  std::string_view key_prefix;

  using LeafNodeEntries = std::vector<LeafNodeEntry>;
  using InteriorNodeEntries = std::vector<InteriorNodeEntry>;
  using Entries = std::variant<LeafNodeEntries, InteriorNodeEntries>;

  /// Child entries.
  Entries entries;

  struct KeyBuffer {
    KeyBuffer() = default;
    KeyBuffer(size_t size) : data(new char[size]), size(size) {}
    std::shared_ptr<char[]> data;
    size_t size = 0;
  };

  /// Concatenated key data, referenced by `key_prefix` and `entries`.
  KeyBuffer key_buffer;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.height, x.key_prefix, x.entries, x.key_buffer);
  };
};

/// Functions for estimating the size of the in-memory representation of a
/// B+tree node entry.
///
/// This is used to calculate when to split nodes.  For that reason, it should
/// be stable, and is therefore hard coded rather than simply set to the actual
/// in-memory size.
///
/// This counts the full size of each `DataFileId`, even though the `DataFileId`
/// may often be shared, to simplify the splitting calculations.

/// Approximate size in bytes of the in-memory representation of a leaf node
/// entry, excluding the variable-length key and value data.
constexpr size_t kLeafNodeFixedSize = 8     // key pointer
                                      + 8;  // key size

constexpr size_t kInteriorNodeFixedSize =
    +8                              // key pointer
    + 8                             // key size
    + 8                             // child data length
    + 8                             // child data offset
    + sizeof(BtreeNodeStatistics);  // statistics

/// Estimates the approximate size in bytes on the in-memory representation of
/// just the value associated with a leaf node entry.
inline size_t GetLeafNodeDataSize(const LeafNodeEntry& entry) {
  if (auto* value = std::get_if<absl::Cord>(&entry.value_reference)) {
    return value->size();
  } else {
    auto& ref = std::get<IndirectDataReference>(entry.value_reference);
    return 8    // offset
           + 8  // length
           + ref.file_id.size();
  }
}

/// Estimates the approximate size in bytes of the in-memory representation of
/// a leaf node entry.
inline size_t EstimateDecodedEntrySizeExcludingKey(const LeafNodeEntry& entry) {
  return kLeafNodeFixedSize + GetLeafNodeDataSize(entry);
}

/// Estimates the approximate size in bytes of the in-memory representation of
/// an interior node entry.
inline size_t EstimateDecodedEntrySizeExcludingKey(
    const InteriorNodeEntry& entry) {
  return kInteriorNodeFixedSize + entry.node.location.file_id.size();
}

/// Validates that a b+tree node has the expected height and min key.
///
/// TODO(jbms): Also validate `BtreeNodeStatistics`
absl::Status ValidateBtreeNodeReference(const BtreeNode& node,
                                        BtreeNodeHeight height,
                                        std::string_view inclusive_min_key);

/// Decodes a b+tree node.
Result<BtreeNode> DecodeBtreeNode(const absl::Cord& encoded,
                                  const BasePath& base_path);

/// Function object where `ComparePrefixedKeyToUnprefixedKey{prefix}(a, b)`
/// returns `(prefix + a).compare(b)`.
///
/// This is used when binary searching for a key in a `BtreeNode`.
struct ComparePrefixedKeyToUnprefixedKey {
  std::string_view prefix;
  int operator()(std::string_view prefixed, std::string_view unprefixed) const {
    auto unprefixed_prefix =
        unprefixed.substr(0, std::min(unprefixed.size(), prefix.size()));
    int c = prefix.compare(unprefixed_prefix);
    if (c != 0) return c;
    return prefixed.compare(unprefixed.substr(prefix.size()));
  }
};

/// Returns the entry with key equal to `key`, or `nullptr` if there is no such
/// entry.
const LeafNodeEntry* FindBtreeEntry(span<const LeafNodeEntry> entries,
                                    std::string_view key);

/// Returns a pointer to the first entry with a key not less than
/// `inclusive_min`, or a pointer one past the end of `entries` if there is no
/// such entry.
const LeafNodeEntry* FindBtreeEntryLowerBound(span<const LeafNodeEntry> entries,
                                              std::string_view inclusive_min);

/// Returns the sub-span of entries with keys greater or equal to
/// `inclusive_min` and less than `exclusive_max` (where, as for `KeyRange`, an
/// empty string for `exclusive_max` indicates no upper bound).
span<const LeafNodeEntry> FindBtreeEntryRange(span<const LeafNodeEntry> entries,
                                              std::string_view inclusive_min,
                                              std::string_view exclusive_max);

/// Returns a pointer to the entry whose subtree may contain `key`, or `nullptr`
/// if no entry has a subtree that may contain `key`.
const InteriorNodeEntry* FindBtreeEntry(span<const InteriorNodeEntry> entries,
                                        std::string_view key);

/// Returns a pointer to the first entry whose key range intersects the set of
/// keys that are not less than `inclusive_min`.
const InteriorNodeEntry* FindBtreeEntryLowerBound(
    span<const InteriorNodeEntry> entries, std::string_view inclusive_min);

/// Returns the sub-span of entries whose subtrees intersect the key range
/// `[inclusive_min, exclusive_max)` (where, as for `KeyRange`, an empty string
/// for `exclusive_max` indicates no upper bound).
span<const InteriorNodeEntry> FindBtreeEntryRange(
    span<const InteriorNodeEntry> entries, std::string_view inclusive_min,
    std::string_view exclusive_max);

#ifndef NDEBUG
/// Checks invariants.
///
/// These invariants are all verified by `DecodeBtreeNode` using a separate code
/// path.  But this may be used for testing.
void CheckBtreeNodeInvariants(const BtreeNode& node);
#endif  // NDEBUG

}  // namespace internal_ocdbt

namespace internal {
template <>
struct HeapUsageEstimator<internal_ocdbt::BtreeNode::KeyBuffer> {
  static size_t EstimateHeapUsage(
      const internal_ocdbt::BtreeNode::KeyBuffer& key_buffer,
      size_t max_depth) {
    return key_buffer.size;
  }
};
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_H_
