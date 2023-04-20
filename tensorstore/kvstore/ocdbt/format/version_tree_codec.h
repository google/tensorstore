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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_CODEC_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_CODEC_H_

/// \file
///
/// Internal codecs for version tree related data structures, used by the
/// version tree and manifest codec.

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference_codec.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"

namespace tensorstore {
namespace internal_ocdbt {

constexpr uint32_t kVersionTreeNodeMagic = 0x0cdb1234;
constexpr uint8_t kVersionTreeNodeFormatVersion = 0;

using GenerationNumberCodec = VarintCodec<GenerationNumber>;

using VersionTreeNodeNumEntriesCodec = VarintCodec<uint64_t>;

struct VersionTreeHeightCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                VersionTreeHeight& value) const {
    return reader.ReadByte(value);
  }

  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                VersionTreeHeight value) const {
    return writer.WriteByte(value);
  }
};

using BtreeNodeHeightCodec = VersionTreeHeightCodec;

struct CommitTimeCodec {
  template <typename IO, typename T>
  [[nodiscard]] bool operator()(IO& io, T&& value) const {
    return LittleEndianCodec<CommitTime::Value>{}(io, value.value);
  }
};

struct VersionTreeNumEntriesCodec {
  size_t max_num_entries;

  template <typename Entry>
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                std::vector<Entry>& value) const {
    uint64_t num_entries;
    if (!VersionTreeNodeNumEntriesCodec{}(reader, num_entries)) return false;
    if (num_entries > max_num_entries) {
      reader.Fail(absl::DataLossError(
          absl::StrFormat("Stored num_children=%d exceeds maximum of %d",
                          num_entries, max_num_entries)));
      return false;
    }
    value.resize(static_cast<size_t>(num_entries));
    return true;
  }

  template <typename Entry>
  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                const std::vector<Entry>& value) const {
    assert(value.size() <= max_num_entries);
    return VersionTreeNodeNumEntriesCodec{}(writer, value.size());
  }
};

template <typename DataFileTable>
struct VersionTreeLeafNodeEntryArrayCodec {
  const DataFileTable& data_file_table;
  size_t max_num_entries;
  template <typename IO, typename T>
  [[nodiscard]] bool operator()(IO& io, T&& value) const {
    if (!VersionTreeNumEntriesCodec{max_num_entries}(io, value)) {
      return false;
    }

    for (auto& entry : value) {
      if (!GenerationNumberCodec{}(io, entry.generation_number)) {
        return false;
      }
    }

    for (auto& entry : value) {
      if (!BtreeNodeHeightCodec{}(io, entry.root_height)) {
        return false;
      }
    }

    if (!BtreeNodeReferenceArrayCodec{
            data_file_table, [](auto& v) -> decltype(auto) { return (v.root); },
            /*allow_missing=*/true}(io, value)) {
      return false;
    }

    for (auto& v : value) {
      if (!CommitTimeCodec{}(io, v.commit_time)) return false;
    }
    return true;
  }
};

template <typename DataFileTable>
struct VersionTreeInteriorNodeEntryArrayCodec {
  const DataFileTable& data_file_table;
  size_t max_num_entries;
  bool include_entry_height = false;

  template <typename IO, typename T>
  [[nodiscard]] bool operator()(IO& io, T&& value) const {
    if (!VersionTreeNumEntriesCodec{max_num_entries}(io, value)) {
      return false;
    }

    for (auto& v : value) {
      if (!GenerationNumberCodec{}(io, v.generation_number)) {
        return false;
      }
    }

    if (!IndirectDataReferenceArrayCodec{data_file_table,
                                         [](auto& entry) -> decltype(auto) {
                                           return (entry.location);
                                         }}(io, value)) {
      return false;
    }

    for (auto& v : value) {
      if (!GenerationNumberCodec{}(io, v.num_generations)) return false;
    }

    for (auto& v : value) {
      if (!CommitTimeCodec{}(io, v.commit_time)) return false;
    }

    if (include_entry_height) {
      for (auto& v : value) {
        if (!VersionTreeHeightCodec{}(io, v.height)) return false;
      }
    }
    return true;
  }
};

struct VersionTreeArityLog2Codec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                VersionTreeArityLog2& value) const;

  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                VersionTreeArityLog2 value) const {
    return writer.WriteByte(value);
  }
};

[[nodiscard]] bool ReadVersionTreeLeafNode(
    VersionTreeArityLog2 version_tree_arity_log2, riegeli::Reader& reader,
    const DataFileTable& data_file_table,
    VersionTreeNode::LeafNodeEntries& entries);

[[nodiscard]] bool WriteVersionTreeNodeEntries(
    const Config& config, riegeli::Writer& writer,
    const DataFileTableBuilder& data_file_table,
    const VersionTreeNode::LeafNodeEntries& entries);

inline void AddDataFiles(DataFileTableBuilder& data_file_table,
                         const BtreeGenerationReference& entry) {
  internal_ocdbt::AddDataFiles(data_file_table, entry.root.location);
}

inline void AddDataFiles(DataFileTableBuilder& data_file_table,
                         const VersionNodeReference& entry) {
  internal_ocdbt::AddDataFiles(data_file_table, entry.location);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_VERSION_TREE_CODEC_H_
