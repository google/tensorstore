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

#include "tensorstore/kvstore/ocdbt/format/btree.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference_codec.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {

bool ReadKeyPrefixLengths(riegeli::Reader& reader,
                          span<KeyLength> prefix_lengths,
                          KeyLength& common_prefix_length) {
  KeyLength min_prefix_length = kMaxKeyLength;
  for (auto& prefix_length : prefix_lengths) {
    if (!KeyLengthCodec{}(reader, prefix_length)) return false;
    min_prefix_length = std::min(min_prefix_length, prefix_length);
  }
  common_prefix_length = min_prefix_length;
  return true;
}

bool ReadKeySuffixLengths(riegeli::Reader& reader,
                          span<KeyLength> suffix_lengths) {
  for (auto& length : suffix_lengths) {
    if (!KeyLengthCodec{}(reader, length)) return false;
  }
  return true;
}

// Read the key-related field columns for `ocdbt-btree-leaf-node-entry-array`
// and `ocdbt-btree-interior-node-entry-array`.
//
// See the format documentation in `index.rst`.  The corresponding write
// function is `WriteKeys` in `btree_node_encoder.cc`.
template <typename Entry>
bool ReadKeys(riegeli::Reader& reader, std::string_view& common_prefix,
              BtreeNode::KeyBuffer& key_buffer, span<Entry> entries) {
  const size_t num_entries = entries.size();
  KeyLength common_prefix_length;
  std::vector<KeyLength> key_length_buffer(num_entries * 2);
  span<KeyLength> prefix_lengths(key_length_buffer.data(), num_entries);
  span<KeyLength> suffix_lengths(key_length_buffer.data() + num_entries,
                                 num_entries);
  // Read `key_prefix_length` column.
  if (!ReadKeyPrefixLengths(reader, prefix_lengths.subspan(1),
                            common_prefix_length)) {
    return false;
  }
  // Read `key_suffix_length` column.
  if (!ReadKeySuffixLengths(reader, suffix_lengths)) return false;
  if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
    // Read `subtree_common_prefix_length` column.
    for (auto& entry : entries) {
      if (!KeyLengthCodec{}(reader, entry.subtree_common_prefix_length)) {
        return false;
      }
      common_prefix_length =
          std::min(common_prefix_length, entry.subtree_common_prefix_length);
    }
  }
  common_prefix_length = std::min(suffix_lengths[0], common_prefix_length);

  size_t key_buffer_size = common_prefix_length;

  // Verify that lengths are valid, calculate the size required for the
  // `key_buffer`, and for interior nodes, adjust `subtree_common_prefix_length`
  // to exclude `common_prefix_length`.
  for (size_t i = 0, prev_length = 0; i < num_entries; ++i) {
    size_t prefix_length = prefix_lengths[i];
    if (prefix_length > prev_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "Child %d: Prefix length of %d exceeds previous key length %d", i,
          prefix_length, prev_length)));
      return false;
    }
    size_t suffix_length = suffix_lengths[i];
    size_t key_length = prefix_length + suffix_length;
    if (key_length > kMaxKeyLength) {
      reader.Fail(absl::DataLossError(
          absl::StrFormat("Child %d: Key length %d exceeds limit of %d", i,
                          key_length, kMaxKeyLength)));
      return false;
    }
    if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
      auto& entry = entries[i];
      if (entry.subtree_common_prefix_length > key_length) {
        reader.Fail(absl::DataLossError(absl::StrFormat(
            "Key %d: subtree common prefix length of %d exceeds key length of "
            "%d",
            i, entry.subtree_common_prefix_length, key_length)));
        return false;
      }
      assert(entry.subtree_common_prefix_length >= common_prefix_length);
      entry.subtree_common_prefix_length -= common_prefix_length;
    }
    prev_length = key_length;
    key_buffer_size += key_length - common_prefix_length;
  }

  key_buffer = BtreeNode::KeyBuffer(key_buffer_size);

  char* key_buffer_ptr = key_buffer.data.get();

  const auto append_key_data = [&](auto... parts) {
    std::string_view s(key_buffer_ptr, (parts.size() + ...));
    (static_cast<void>(std::memcpy(key_buffer_ptr, parts.data(), parts.size()),
                       key_buffer_ptr += parts.size()),
     ...);
    return s;
  };

  // Read first `key_suffix` and extract common prefix.
  {
    size_t key_length = suffix_lengths[0];
    if (!reader.Pull(key_length)) return false;
    auto full_first_key =
        append_key_data(std::string_view(reader.cursor(), key_length));
    common_prefix = full_first_key.substr(0, common_prefix_length);
    entries[0].key = full_first_key.substr(common_prefix_length);
    reader.move_cursor(key_length);
  }

  // Read remaining `key_suffix` values.
  for (size_t i = 1; i < num_entries; ++i) {
    size_t prefix_length = prefix_lengths[i] - common_prefix_length;
    size_t suffix_length = suffix_lengths[i];
    if (!reader.Pull(suffix_length)) return false;
    auto prev_key = std::string_view(entries[i - 1].key);
    auto suffix = std::string_view(reader.cursor(), suffix_length);
    if (prev_key.substr(prefix_length) >= suffix) {
      reader.Fail(absl::DataLossError("Invalid key order"));
      return false;
    }
    entries[i].key = append_key_data(prev_key.substr(0, prefix_length), suffix);
    reader.move_cursor(suffix_length);
  }

  return true;
}

template <typename Entry>
bool ReadBtreeNodeEntries(riegeli::Reader& reader,
                          const DataFileTable& data_file_table,
                          uint64_t num_entries, BtreeNode& node) {
  auto& entries = node.entries.emplace<std::vector<Entry>>();
  entries.resize(num_entries);
  if (!ReadKeys<Entry>(reader, node.key_prefix, node.key_buffer, entries)) {
    return false;
  }
  if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
    return BtreeNodeReferenceArrayCodec{data_file_table,
                                        [](auto& entry) -> decltype(auto) {
                                          return (entry.node);
                                        }}(reader, entries);
  } else {
    return LeafNodeValueReferenceArrayCodec{data_file_table,
                                            [](auto& entry) -> decltype(auto) {
                                              return (entry.value_reference);
                                            }}(reader, entries);
  }
}

}  // namespace

Result<BtreeNode> DecodeBtreeNode(const absl::Cord& encoded,
                                  const BasePath& base_path) {
  BtreeNode node;
  auto status = DecodeWithOptionalCompression(
      encoded, kBtreeNodeMagic, kBtreeNodeFormatVersion,
      [&](riegeli::Reader& reader, uint32_t version) -> bool {
        if (!reader.ReadByte(node.height)) return false;
        DataFileTable data_file_table;
        if (!ReadDataFileTable(reader, base_path, data_file_table)) {
          return false;
        }
        uint32_t num_entries;
        if (!ReadVarintChecked(reader, num_entries)) return false;
        if (num_entries == 0) {
          reader.Fail(absl::DataLossError("Empty b-tree node"));
          return false;
        }
        if (num_entries > kMaxNodeArity) {
          reader.Fail(absl::DataLossError(absl::StrFormat(
              "B-tree node has arity %d, which exceeds limit of %d",
              num_entries, kMaxNodeArity)));
          return false;
        }
        if (node.height == 0) {
          return ReadBtreeNodeEntries<LeafNodeEntry>(reader, data_file_table,
                                                     num_entries, node);
        } else {
          return ReadBtreeNodeEntries<InteriorNodeEntry>(
              reader, data_file_table, num_entries, node);
        }
      });
  if (!status.ok()) {
    return tensorstore::MaybeAnnotateStatus(status,
                                            "Error decoding b-tree node");
  }
#ifndef NDEBUG
  CheckBtreeNodeInvariants(node);
#endif
  return node;
}

absl::Status ValidateBtreeNodeReference(const BtreeNode& node,
                                        BtreeNodeHeight height,
                                        std::string_view inclusive_min_key) {
  if (node.height != height) {
    return absl::DataLossError(absl::StrFormat(
        "Expected height of %d but received: %d", height, node.height));
  }

  return std::visit(
      [&](auto& entries) {
        if (ComparePrefixedKeyToUnprefixedKey{node.key_prefix}(
                entries.front().key, inclusive_min_key) < 0) {
          return absl::DataLossError(
              tensorstore::StrCat("First key ",
                                  tensorstore::QuoteString(tensorstore::StrCat(
                                      node.key_prefix, entries.front().key)),
                                  " is less than inclusive_min ",
                                  tensorstore::QuoteString(inclusive_min_key),
                                  " specified by parent node"));
        }
        return absl::OkStatus();
      },
      node.entries);
}

bool operator==(const BtreeNodeStatistics& a, const BtreeNodeStatistics& b) {
  return a.num_indirect_value_bytes == b.num_indirect_value_bytes &&
         a.num_tree_bytes == b.num_tree_bytes && a.num_keys == b.num_keys;
}

std::ostream& operator<<(std::ostream& os, const BtreeNodeStatistics& x) {
  return os << "{num_indirect_value_bytes=" << x.num_indirect_value_bytes
            << ", num_tree_bytes=" << x.num_tree_bytes
            << ", num_keys=" << x.num_keys << "}";
}

BtreeNodeStatistics& BtreeNodeStatistics::operator+=(
    const BtreeNodeStatistics& other) {
  num_indirect_value_bytes = internal::AddSaturate(
      num_indirect_value_bytes, other.num_indirect_value_bytes);
  num_tree_bytes = internal::AddSaturate(num_tree_bytes, other.num_tree_bytes);
  num_keys = internal::AddSaturate(num_keys, other.num_keys);
  return *this;
}

bool operator==(const LeafNodeEntry& a, const LeafNodeEntry& b) {
  return a.key == b.key && a.value_reference == b.value_reference;
}

std::ostream& operator<<(std::ostream& os, const LeafNodeValueReference& x) {
  if (auto* value = std::get_if<absl::Cord>(&x)) {
    return os << tensorstore::QuoteString(std::string(*value));
  } else {
    return os << std::get<IndirectDataReference>(x);
  }
}

std::ostream& operator<<(std::ostream& os, const LeafNodeEntry& e) {
  return os << "{key=" << tensorstore::QuoteString(e.key)
            << ", value_reference=" << e.value_reference << "}";
}

bool operator==(const BtreeNodeReference& a, const BtreeNodeReference& b) {
  return a.location == b.location && a.statistics == b.statistics;
}

std::ostream& operator<<(std::ostream& os, const BtreeNodeReference& x) {
  return os << "{location=" << x.location << ", statistics=" << x.statistics
            << "}";
}

std::ostream& operator<<(std::ostream& os, const InteriorNodeEntry& e) {
  return os << "{key=" << tensorstore::QuoteString(e.key)
            << ", subtree_common_prefix_length="
            << e.subtree_common_prefix_length << ", node=" << e.node << "}";
}

const LeafNodeEntry* FindBtreeEntry(span<const LeafNodeEntry> entries,
                                    std::string_view key) {
  const LeafNodeEntry* entry = FindBtreeEntryLowerBound(entries, key);
  if (entry == entries.data() + entries.size() || entry->key != key) {
    return nullptr;
  }
  return entry;
}

const LeafNodeEntry* FindBtreeEntryLowerBound(span<const LeafNodeEntry> entries,
                                              std::string_view inclusive_min) {
  return std::lower_bound(
      entries.data(), entries.data() + entries.size(), inclusive_min,
      [](const LeafNodeEntry& entry, std::string_view inclusive_min) {
        return entry.key < inclusive_min;
      });
}

span<const LeafNodeEntry> FindBtreeEntryRange(span<const LeafNodeEntry> entries,
                                              std::string_view inclusive_min,
                                              std::string_view exclusive_max) {
  const LeafNodeEntry* lower = FindBtreeEntryLowerBound(entries, inclusive_min);
  const LeafNodeEntry* upper = entries.data() + entries.size();
  if (!exclusive_max.empty()) {
    upper = std::lower_bound(
        lower, upper, exclusive_max,
        [](const LeafNodeEntry& entry, std::string_view exclusive_max) {
          return entry.key < exclusive_max;
        });
  }
  return {lower, upper};
}

const InteriorNodeEntry* FindBtreeEntry(span<const InteriorNodeEntry> entries,
                                        std::string_view key) {
  auto it = std::lower_bound(
      entries.data(), entries.data() + entries.size(), key,
      [](const InteriorNodeEntry& entry, std::string_view inclusive_min) {
        return entry.key <= inclusive_min;
      });
  if (it == entries.data()) {
    // Key not present.
    return nullptr;
  }
  return it - 1;
}

const InteriorNodeEntry* FindBtreeEntryLowerBound(
    span<const InteriorNodeEntry> entries, std::string_view inclusive_min) {
  auto it = std::lower_bound(
      entries.data(), entries.data() + entries.size(), inclusive_min,
      [](const InteriorNodeEntry& entry, std::string_view inclusive_min) {
        return entry.key <= inclusive_min;
      });
  if (it != entries.data()) --it;
  return it;
}

span<const InteriorNodeEntry> FindBtreeEntryRange(
    span<const InteriorNodeEntry> entries, std::string_view inclusive_min,
    std::string_view exclusive_max) {
  const InteriorNodeEntry* lower =
      FindBtreeEntryLowerBound(entries, inclusive_min);
  const InteriorNodeEntry* upper = entries.data() + entries.size();
  if (!exclusive_max.empty()) {
    upper = std::lower_bound(
        lower, upper, exclusive_max,
        [](const InteriorNodeEntry& entry, std::string_view exclusive_max) {
          return entry.key < exclusive_max;
        });
  }
  return {lower, upper};
}

#ifndef NDEBUG
void CheckBtreeNodeInvariants(const BtreeNode& node) {
  if (node.height == 0) {
    assert(std::holds_alternative<BtreeNode::LeafNodeEntries>(node.entries));
    auto& entries = std::get<BtreeNode::LeafNodeEntries>(node.entries);
    assert(!entries.empty());
    assert(entries.size() <= kMaxNodeArity);
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries[i];
      if (auto* location =
              std::get_if<IndirectDataReference>(&entry.value_reference)) {
        assert(!location->IsMissing());
      }
      if (i != 0) {
        assert(entry.key > entries[i - 1].key);
      }
    }
  } else {
    assert(
        std::holds_alternative<BtreeNode::InteriorNodeEntries>(node.entries));
    auto& entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);
    assert(!entries.empty());
    assert(entries.size() <= kMaxNodeArity);
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries[i];
      assert(entry.subtree_common_prefix_length <= entry.key.size());
      assert(!entry.node.location.IsMissing());
      if (i != 0) {
        assert(entry.key > entries[i - 1].key);
      }
    }
  }
}
#endif  // NDEBUG

}  // namespace internal_ocdbt
}  // namespace tensorstore
