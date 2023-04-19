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
  KeyLength min_prefix_length = std::numeric_limits<KeyLength>::max();
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

template <typename Entry>
bool ReadKeys(riegeli::Reader& reader, std::string_view& common_prefix,
              BtreeNode::KeyBuffer& key_buffer, span<Entry> entries,
              KeyLength max_common_prefix_length) {
  const size_t num_entries = entries.size();
  KeyLength common_prefix_length;
  std::vector<KeyLength> key_length_buffer(num_entries * 2 - 1);
  span<KeyLength> prefix_lengths(key_length_buffer.data() + num_entries,
                                 num_entries - 1);
  span<KeyLength> suffix_lengths(key_length_buffer.data(), num_entries);
  if (!ReadKeyPrefixLengths(reader, prefix_lengths, common_prefix_length)) {
    return false;
  }
  if (!ReadKeySuffixLengths(reader, suffix_lengths)) return false;
  common_prefix_length =
      std::min(common_prefix_length, max_common_prefix_length);
  common_prefix_length = std::min(suffix_lengths[0], common_prefix_length);

  size_t key_buffer_size = suffix_lengths[0];

  // First verify that lengths are valid.
  for (size_t i = 1, prev_length = suffix_lengths[0]; i < num_entries; ++i) {
    size_t prefix_length = prefix_lengths[i - 1];
    if (prefix_length > prev_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "Child %d: Prefix length of %d exceeds previous key length %d", i,
          prefix_length, prev_length)));
      return false;
    }
    size_t suffix_length = suffix_lengths[i];
    if (prefix_length + suffix_length > std::numeric_limits<KeyLength>::max()) {
      reader.Fail(absl::DataLossError(
          absl::StrFormat("Child %d: Key length %d exceeds limit of %d", i,
                          prefix_length + suffix_length,
                          std::numeric_limits<KeyLength>::max())));
      return false;
    }
    prev_length = prefix_length + suffix_length;
    key_buffer_size += prev_length - common_prefix_length;
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

  // Read first key and extract common prefix.
  {
    size_t key_length = suffix_lengths[0];
    if (!reader.Pull(key_length)) return false;
    auto full_first_key =
        append_key_data(std::string_view(reader.cursor(), key_length));
    common_prefix = full_first_key.substr(0, common_prefix_length);
    entries[0].key = full_first_key.substr(common_prefix_length);
    reader.move_cursor(key_length);
  }

  // Read remaining keys.
  for (size_t i = 1; i < num_entries; ++i) {
    size_t prefix_length = prefix_lengths[i - 1] - common_prefix_length;
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

bool ReadBtreeLeafNode(riegeli::Reader& reader,
                       const DataFileTable& data_file_table,
                       std::string_view& prefix,
                       BtreeNode::KeyBuffer& key_buffer,
                       BtreeNode::LeafNodeEntries& entries,
                       KeyLength max_common_prefix_length) {
  uint32_t num_entries;
  if (!ReadVarintChecked(reader, num_entries)) return false;
  if (num_entries == 0) {
    reader.Fail(absl::DataLossError("Empty b-tree node"));
    return false;
  }
  if (num_entries > kMaxNodeArity) {
    reader.Fail(absl::DataLossError(
        absl::StrFormat("B-tree node has arity %d, which exceeds limit of %d",
                        num_entries, kMaxNodeArity)));
    return false;
  }
  entries.resize(num_entries);
  if (!ReadKeys<LeafNodeEntry>(reader, prefix, key_buffer, entries,
                               max_common_prefix_length)) {
    return false;
  }

  return LeafNodeValueReferenceArrayCodec{data_file_table,
                                          [](auto& entry) -> decltype(auto) {
                                            return (entry.value_reference);
                                          }}(reader, entries);
}

bool ReadBtreeInteriorNode(riegeli::Reader& reader,
                           const DataFileTable& data_file_table,
                           std::string_view& prefix,
                           BtreeNode::KeyBuffer& key_buffer,
                           BtreeNode::InteriorNodeEntries& entries,
                           KeyLength max_common_prefix_length) {
  uint32_t num_entries;
  if (!ReadVarintChecked(reader, num_entries)) return false;
  if (num_entries == 0) {
    reader.Fail(absl::DataLossError("Empty b-tree node"));
    return false;
  }
  if (num_entries > kMaxNodeArity) {
    reader.Fail(absl::DataLossError(
        absl::StrFormat("B-tree node has arity %d, which exceeds limit of %d",
                        num_entries, kMaxNodeArity)));
    return false;
  }
  entries.resize(num_entries);
  KeyLength min_subtree_common_prefix_length = max_common_prefix_length;
  for (auto& entry : entries) {
    if (!KeyLengthCodec{}(reader, entry.subtree_common_prefix_length)) {
      return false;
    }
    min_subtree_common_prefix_length = std::min(
        min_subtree_common_prefix_length, entry.subtree_common_prefix_length);
  }
  if (!ReadKeys<InteriorNodeEntry>(reader, prefix, key_buffer, entries,
                                   min_subtree_common_prefix_length)) {
    return false;
  }
  size_t common_prefix_length = prefix.size();
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries[i];
    KeyLength key_length = entry.key.size() + common_prefix_length;
    if (entry.subtree_common_prefix_length > key_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "Key %d: subtree common prefix length of %d exceeds key length of %d",
          i, entry.subtree_common_prefix_length, key_length)));
      return false;
    }
    assert(entry.subtree_common_prefix_length >= common_prefix_length);
    entry.subtree_common_prefix_length -= common_prefix_length;
  }

  if (!BtreeNodeReferenceArrayCodec{data_file_table,
                                    [](auto& entry) -> decltype(auto) {
                                      return (entry.node);
                                    }}(reader, entries)) {
    return false;
  }

  return true;
}

[[nodiscard]] bool ReadBtreeNodeInner(
    riegeli::Reader& reader, const BasePath& base_path, BtreeNode& node,
    KeyLength max_common_prefix_length =
        std::numeric_limits<KeyLength>::max()) {
  DataFileTable data_file_table;
  if (!ReadDataFileTable(reader, base_path, data_file_table)) return false;
  if (node.height == 0) {
    return ReadBtreeLeafNode(reader, data_file_table, node.key_prefix,
                             node.key_buffer,
                             node.entries.emplace<BtreeNode::LeafNodeEntries>(),
                             max_common_prefix_length);
  } else {
    return ReadBtreeInteriorNode(
        reader, data_file_table, node.key_prefix, node.key_buffer,
        node.entries.emplace<BtreeNode::InteriorNodeEntries>(),
        max_common_prefix_length);
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
        return ReadBtreeNodeInner(reader, base_path, node);
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
