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

#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/debug_defines.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

size_t FindExistingNotExistingCommonPrefixLength(
    std::string_view existing_prefix, std::string_view existing_key,
    std::string_view new_key) {
  size_t prefix_length = FindCommonPrefixLength(existing_prefix, new_key);
  if (prefix_length == existing_prefix.size()) {
    prefix_length += FindCommonPrefixLength(
        existing_key, new_key.substr(existing_prefix.size()));
  }
  return prefix_length;
}

size_t GetCommonPrefixLength(std::string_view existing_prefix, bool a_existing,
                             std::string_view a_key, bool b_existing,
                             std::string_view b_key) {
  if (a_existing) {
    if (b_existing) {
      return existing_prefix.size() + FindCommonPrefixLength(a_key, b_key);
    } else {
      return FindExistingNotExistingCommonPrefixLength(existing_prefix, a_key,
                                                       b_key);
    }
  } else {
    if (b_existing) {
      return FindExistingNotExistingCommonPrefixLength(existing_prefix, b_key,
                                                       a_key);
    } else {
      return FindCommonPrefixLength(a_key, b_key);
    }
  }
}

template <typename Entry>
void GetCommonPrefixLengthOfEntries(
    KeyLength& excluded_prefix_length,
    span<typename BtreeNodeEncoder<Entry>::BufferedEntry> entries,
    std::string_view existing_prefix) {
  auto& first = entries.front();
  auto& last = entries.back();
  excluded_prefix_length =
      GetCommonPrefixLength(existing_prefix, first.existing, first.entry.key,
                            last.existing, last.entry.key);
}

// Write the key-related field columns for `ocdbt-btree-leaf-node-entry-array`
// and `ocdbt-btree-interior-node-entry-array`.
//
// See the format documentation in `index.rst`.  The corresponding read function
// is `ReadKeys` in `btree.cc`.  Refer to the
template <typename Entry>
bool WriteKeys(riegeli::Writer& writer, KeyLength excluded_prefix_length,
               span<typename BtreeNodeEncoder<Entry>::BufferedEntry> entries,
               std::string_view existing_prefix) {
  // Write `key_prefix_length` column
  for (auto& entry : entries.subspan(1)) {
    if (!riegeli::WriteVarint32(
            entry.common_prefix_with_next_entry_length - excluded_prefix_length,
            writer)) {
      return false;
    }
  }

  // Write `key_suffix_length` column
  for (auto& e : entries) {
    size_t key_length =
        (e.existing ? existing_prefix.size() : 0) + e.entry.key.size();
    assert(key_length >= e.common_prefix_with_next_entry_length);
    if (!riegeli::WriteVarint32(
            key_length - e.common_prefix_with_next_entry_length, writer)) {
      return false;
    }
  }

  if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
    // Write `subtree_common_prefix_length` column
    for (auto& e : entries) {
      KeyLength subtree_common_prefix_length =
          e.entry.subtree_common_prefix_length +
          (e.existing ? existing_prefix.size() : 0) - excluded_prefix_length;
      if (!riegeli::WriteVarint32(subtree_common_prefix_length, writer)) {
        return false;
      }
    }
  }

  // Write `key_suffix` column
  for (auto& e : entries) {
    if (e.existing) {
      size_t existing_prefix_skip = std::min(
          e.common_prefix_with_next_entry_length, existing_prefix.size());
      if (!writer.Write(existing_prefix.substr(existing_prefix_skip)) ||
          !writer.Write(e.entry.key.substr(
              e.common_prefix_with_next_entry_length - existing_prefix_skip))) {
        return false;
      }
    } else {
      if (!writer.Write(
              e.entry.key.substr(e.common_prefix_with_next_entry_length))) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

template <typename Entry>
BtreeNodeEncoder<Entry>::BtreeNodeEncoder(const Config& config,
                                          BtreeNodeHeight height,
                                          std::string_view existing_prefix)
    : config_(config), height_(height), existing_prefix_(existing_prefix) {
  if constexpr (std::is_same_v<Entry, LeafNodeEntry>) {
    assert(height_ == 0);
  }
}

template <typename Entry>
void BtreeNodeEncoder<Entry>::AddEntry(bool existing, Entry&& entry) {
  size_t key_length =
      (existing ? existing_prefix_.size() : 0) + entry.key.size();
  size_t additional_size =
      EstimateDecodedEntrySizeExcludingKey(entry) + key_length;
  if (buffered_entries_.empty()) {
    common_prefix_length_ = key_length;
    buffered_entries_.push_back(
        BufferedEntry{0, existing, std::move(entry), additional_size});
    return;
  }
  size_t prefix_length = GetCommonPrefixLength(
      existing_prefix_, buffered_entries_.back().existing,
      buffered_entries_.back().entry.key, existing, entry.key);
  common_prefix_length_ = std::min(common_prefix_length_, prefix_length);
  buffered_entries_.push_back(BufferedEntry{
      prefix_length, existing, std::move(entry),
      buffered_entries_.back().cumulative_size + additional_size});
}

namespace {
template <typename Entry>
bool EncodeEntriesInner(
    riegeli::Writer& writer, BtreeNodeHeight height,
    std::string_view existing_prefix,
    span<typename BtreeNodeEncoder<Entry>::BufferedEntry> entries, bool is_root,
    EncodedNodeInfo& info) {
  info.statistics = {};

  if constexpr (std::is_same_v<Entry, LeafNodeEntry>) {
    info.statistics.num_keys = entries.size();
  } else {
    for (const auto& entry : entries) {
      info.statistics += entry.entry.node.statistics;
    }
  }

  DataFileTableBuilder data_file_table;
  for (const auto& entry : entries) {
    internal_ocdbt::AddDataFiles(data_file_table, entry.entry);
  }

  if (!data_file_table.Finalize(writer)) return false;

  // num_entries
  if (!riegeli::WriteVarint32(entries.size(), writer)) return false;

  KeyLength max_excluded_prefix_length =
      is_root ? 0 : std::numeric_limits<KeyLength>::max();

  if (!is_root) {
    auto& first = entries[0];
    tensorstore::StrAppend(
        &info.inclusive_min_key,
        first.existing ? existing_prefix : std::string_view{}, first.entry.key);
  }

  if constexpr (std::is_same_v<Entry, InteriorNodeEntry>) {
    for (auto& entry : entries) {
      KeyLength subtree_common_prefix_length =
          entry.entry.subtree_common_prefix_length +
          (entry.existing ? existing_prefix.size() : 0);
      max_excluded_prefix_length =
          std::min(max_excluded_prefix_length, subtree_common_prefix_length);
    }
  }

  if (max_excluded_prefix_length != 0) {
    auto& first = entries.front();
    auto& last = entries.back();
    info.excluded_prefix_length = std::min<KeyLength>(
        max_excluded_prefix_length,
        GetCommonPrefixLength(existing_prefix, first.existing, first.entry.key,
                              last.existing, last.entry.key));
  } else {
    info.excluded_prefix_length = 0;
  }

  entries.front().common_prefix_with_next_entry_length =
      info.excluded_prefix_length;

  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Encoding node: height=" << static_cast<int>(height)
      << ", inclusive_min_key="
      << tensorstore::QuoteString(info.inclusive_min_key)
      << ", excluded_prefix_length=" << info.excluded_prefix_length;

  if (ocdbt_logging.Level(1)) {
    for (auto& entry : entries) {
      ABSL_LOG(INFO) << "  Entry: key="
                     << tensorstore::QuoteString(tensorstore::StrCat(
                            entry.existing ? existing_prefix
                                           : std::string_view(),
                            entry.entry.key));
    }
  }

  // Keys
  if (!WriteKeys<Entry>(writer, info.excluded_prefix_length, entries,
                        existing_prefix)) {
    return false;
  }

  if constexpr (std::is_same_v<Entry, LeafNodeEntry>) {
    if (!LeafNodeValueReferenceArrayCodec{data_file_table,
                                          [](auto& e) -> decltype(auto) {
                                            return (e.entry.value_reference);
                                          }}(writer, entries)) {
      return false;
    }

    for (auto& entry : entries) {
      if (auto* data_ref = std::get_if<IndirectDataReference>(
              &entry.entry.value_reference)) {
        info.statistics.num_indirect_value_bytes = internal::AddSaturate(
            info.statistics.num_indirect_value_bytes, data_ref->length);
      }
    }
  } else {
    if (!BtreeNodeReferenceArrayCodec{data_file_table,
                                      [](auto& e) -> decltype(auto) {
                                        return (e.entry.node);
                                      }}(writer, entries)) {
      return false;
    }
  }
  return true;
}
}  // namespace

template <typename Entry>
Result<EncodedNode> EncodeEntries(
    const Config& config, BtreeNodeHeight height,
    std::string_view existing_prefix,
    span<typename BtreeNodeEncoder<Entry>::BufferedEntry> entries,
    bool is_root) {
  EncodedNode encoded;
  auto result = EncodeWithOptionalCompression(
      config, kBtreeNodeMagic, kBtreeNodeFormatVersion,
      [&](riegeli::Writer& writer) -> bool {
        // height
        if (!writer.WriteByte(height)) return false;
        return EncodeEntriesInner<Entry>(writer, height, existing_prefix,
                                         entries, is_root, encoded.info);
      });
  TENSORSTORE_ASSIGN_OR_RETURN(
      encoded.encoded_node, std::move(result),
      tensorstore::MaybeAnnotateStatus(_, "Error encoding b-tree node"));
  encoded.info.statistics.num_tree_bytes += encoded.encoded_node.size();
  return encoded;
}

template <typename Entry>
Result<std::vector<EncodedNode>> BtreeNodeEncoder<Entry>::Finalize(
    bool may_be_root) {
#ifdef TENSORSTORE_INTERNAL_OCDBT_DEBUG
  // Verify that entries are sorted.
  for (size_t i = 1; i < buffered_entries_.size(); ++i) {
    auto& a = buffered_entries_[i - 1];
    auto& b = buffered_entries_[i];
    if (a.existing == b.existing) {
      ABSL_DCHECK_LT(a.entry.key, b.entry.key);
    } else if (a.existing) {
      ABSL_DCHECK_LT(ComparePrefixedKeyToUnprefixedKey{existing_prefix_}(
                         a.entry.key, b.entry.key),
                     0);
    } else {
      ABSL_DCHECK_GT(ComparePrefixedKeyToUnprefixedKey{existing_prefix_}(
                         b.entry.key, a.entry.key),
                     0);
    }
  }
#endif  //  TENSORSTORE_INTERNAL_OCDBT_DEBUG
  std::vector<EncodedNode> encoded_nodes;

  constexpr size_t kMinArity = std::is_same_v<Entry, LeafNodeEntry> ? 1 : 2;

  size_t start_i = 0;
  size_t prev_size_estimate = 0;

  const auto get_range_size = [&](size_t end_i) {
    return (buffered_entries_[end_i - 1].cumulative_size - prev_size_estimate) -
           (end_i - start_i - 1) * common_prefix_length_;
  };

  while (start_i < buffered_entries_.size()) {
    size_t size_upper_bound = get_range_size(buffered_entries_.size());
    size_t num_nodes = tensorstore::CeilOfRatio<size_t>(
        buffered_entries_.size() - start_i, kMaxNodeArity);
    if (config_.max_decoded_node_bytes != 0) {
      num_nodes = std::max(
          num_nodes, tensorstore::CeilOfRatio<size_t>(
                         size_upper_bound, config_.max_decoded_node_bytes));
    }
    size_t target_size = tensorstore::CeilOfRatio(size_upper_bound, num_nodes);
    size_t end_i;
    for (end_i = start_i + 1; end_i < buffered_entries_.size(); ++end_i) {
      if (end_i - start_i >= kMaxNodeArity) break;
      size_t size = get_range_size(end_i);
      if (size >= target_size && end_i >= start_i + kMinArity) {
        if (size > config_.max_decoded_node_bytes &&
            end_i > start_i + kMinArity) {
          --end_i;
        }
        break;
      }
    }
    assert(end_i > start_i);
    assert(end_i - start_i <= kMaxNodeArity);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto encoded_node,
        EncodeEntries<Entry>(
            config_, height_, existing_prefix_,
            span(buffered_entries_.data() + start_i, end_i - start_i),
            may_be_root && start_i == 0 && end_i == buffered_entries_.size()));
    encoded_nodes.push_back(std::move(encoded_node));
    start_i = end_i;
    prev_size_estimate = buffered_entries_[end_i - 1].cumulative_size;
  }
  return encoded_nodes;
}

void AddNewInteriorEntry(BtreeNodeEncoder<InteriorNodeEntry>& encoder,
                         const InteriorNodeEntryData<std::string>& entry) {
  InteriorNodeEntry new_entry;
  new_entry.key = entry.key;
  new_entry.subtree_common_prefix_length = entry.subtree_common_prefix_length;
  new_entry.node = entry.node;
  encoder.AddEntry(/*existing=*/false, std::move(new_entry));
}

template class BtreeNodeEncoder<LeafNodeEntry>;
template class BtreeNodeEncoder<InteriorNodeEntry>;

}  // namespace internal_ocdbt
}  // namespace tensorstore
