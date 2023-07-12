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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_CODEC_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_CODEC_H_

/// \file
///
/// Internal codecs for b+tree-node related data structures, used by the b+tree,
/// version tree and manifest codecs.

#include <cstdint>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference_codec.h"

namespace tensorstore {
namespace internal_ocdbt {

constexpr uint32_t kBtreeNodeMagic = 0x0cdb20de;
constexpr uint8_t kBtreeNodeFormatVersion = 0;
constexpr size_t kMaxNodeArity = 1024 * 1024;

using NumIndirectValueBytesCodec = VarintCodec<uint64_t>;
using NumTreeBytesCodec = VarintCodec<uint64_t>;
using NumKeysCodec = VarintCodec<uint64_t>;

template <typename Getter>
struct BtreeNodeStatisticsArrayCodec {
  Getter getter;
  template <typename IO, typename Vec>
  [[nodiscard]] bool operator()(IO& io, Vec&& vec) const {
    static_assert(std::is_same_v<IO, riegeli::Reader> ||
                  std::is_same_v<IO, riegeli::Writer>);
    for (auto& entry : vec) {
      if (!NumKeysCodec{}(io, getter(entry).num_keys)) return false;
    }

    for (auto& entry : vec) {
      if (!NumTreeBytesCodec{}(io, getter(entry).num_tree_bytes)) return false;
    }

    for (auto& entry : vec) {
      if (!NumIndirectValueBytesCodec{}(
              io, getter(entry).num_indirect_value_bytes)) {
        return false;
      }
    }
    return true;
  }
};

template <typename Getter>
BtreeNodeStatisticsArrayCodec(Getter) -> BtreeNodeStatisticsArrayCodec<Getter>;

using KeyLengthCodec = VarintCodec<KeyLength>;

template <typename DataFileTable, typename Getter>
struct BtreeNodeReferenceArrayCodec {
  const DataFileTable& data_file_table;
  Getter getter;
  bool allow_missing = false;
  template <typename IO, typename Vec>
  [[nodiscard]] bool operator()(IO& io, Vec&& vec) const {
    if (!IndirectDataReferenceArrayCodec{data_file_table,
                                         [&](auto& entry) -> decltype(auto) {
                                           return (getter(entry).location);
                                         },
                                         allow_missing}(io, vec)) {
      return false;
    }

    if (!BtreeNodeStatisticsArrayCodec{
            [&](auto& entry) -> decltype(auto) {
              return (getter(entry).statistics);
            },
        }(io, vec)) {
      return false;
    }

    return true;
  }
};

template <typename DataFileTable, typename Getter>
BtreeNodeReferenceArrayCodec(const DataFileTable&, Getter,
                             bool allow_missing = false)
    -> BtreeNodeReferenceArrayCodec<DataFileTable, Getter>;

template <typename DataFileTable, typename Getter>
struct LeafNodeValueReferenceArrayCodec {
  const DataFileTable& data_file_table;
  Getter getter;
  template <typename Entries>
  [[nodiscard]] bool operator()(riegeli::Reader& reader, Entries&& entries) {
    std::vector<uint64_t> value_lengths(entries.size());
    // First read lengths.
    for (auto& length : value_lengths) {
      if (!DataFileLengthCodec{}(reader, length)) return false;
    }

    // Read value kinds
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries[i];
      auto& value_reference = getter(entry);
      LeafNodeValueKind value_kind;
      if (!reader.ReadByte(value_kind)) return false;
      if (value_kind > kOutOfLineValue) {
        reader.Fail(absl::DataLossError(absl::StrFormat(
            "value_kind[%d]=%d is outside valid range [0, 1]", i, value_kind)));
        return false;
      }
      if (value_kind == kInlineValue) {
        if (value_lengths[i] > kMaxInlineValueLength) {
          reader.Fail(absl::DataLossError(absl::StrFormat(
              "value_length[%d]=%d exceeds maximum of %d for an inline value",
              i, value_lengths[i], kMaxInlineValueLength)));
          return false;
        }
        value_reference.template emplace<absl::Cord>();
      } else {
        auto& data_ref =
            value_reference.template emplace<IndirectDataReference>();
        data_ref.length = value_lengths[i];
      }
    }

    // Read file_ids for indirect values.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto* data_ref = std::get_if<IndirectDataReference>(&value_reference);
      if (!data_ref) continue;
      if (!DataFileIdCodec<riegeli::Reader>{data_file_table}(
              reader, data_ref->file_id)) {
        return false;
      }
    }

    // Read offsets for indirect values.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto* data_ref = std::get_if<IndirectDataReference>(&value_reference);
      if (!data_ref) continue;
      if (!DataFileOffsetCodec{}(reader, data_ref->offset)) return false;
      TENSORSTORE_RETURN_IF_ERROR(data_ref->Validate(/*allow_missing=*/false),
                                  (reader.Fail(_), false));
    }

    // Read values for direct values.
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries[i];
      auto& value_reference = getter(entry);
      auto* value = std::get_if<absl::Cord>(&value_reference);
      if (!value) continue;
      if (!reader.Read(value_lengths[i], *value)) return false;
    }
    return true;
  }

  template <typename Entries>
  [[nodiscard]] bool operator()(riegeli::Writer& writer, Entries&& entries) {
    // length
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      uint64_t length;
      if (auto* data_ref =
              std::get_if<IndirectDataReference>(&value_reference)) {
        length = data_ref->length;
      } else {
        length = std::get<absl::Cord>(value_reference).size();
      }
      if (!DataFileLengthCodec{}(writer, length)) return false;
    }

    // value kind
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      LeafNodeValueKind value_kind = value_reference.index();
      if (!writer.WriteByte(value_kind)) return false;
    }

    // file_ids for indirect values
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      if (auto* data_ref =
              std::get_if<IndirectDataReference>(&value_reference)) {
        if (!DataFileIdCodec<riegeli::Writer>{data_file_table}(
                writer, data_ref->file_id)) {
          return false;
        }
      }
    }

    // offsets for indirect values
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      if (auto* data_ref =
              std::get_if<IndirectDataReference>(&value_reference)) {
        if (!DataFileOffsetCodec{}(writer, data_ref->offset)) return false;
      }
    }

    // direct values
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      if (auto* data = std::get_if<absl::Cord>(&value_reference)) {
        if (!writer.Write(*data)) return false;
      }
    }
    return true;
  }
};

template <typename DataFileTable, typename Getter>
LeafNodeValueReferenceArrayCodec(const DataFileTable&, Getter)
    -> LeafNodeValueReferenceArrayCodec<DataFileTable, Getter>;

template <typename Key>
void AddDataFiles(DataFileTableBuilder& data_file_table,
                  const InteriorNodeEntryData<Key>& entry) {
  internal_ocdbt::AddDataFiles(data_file_table, entry.node.location);
}

inline void AddDataFiles(DataFileTableBuilder& data_file_table,
                         const LeafNodeValueReference& value) {
  auto* location = std::get_if<IndirectDataReference>(&value);
  if (location) internal_ocdbt::AddDataFiles(data_file_table, *location);
}

inline void AddDataFiles(DataFileTableBuilder& data_file_table,
                         const LeafNodeEntry& entry) {
  internal_ocdbt::AddDataFiles(data_file_table, entry.value_reference);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_CODEC_H_
