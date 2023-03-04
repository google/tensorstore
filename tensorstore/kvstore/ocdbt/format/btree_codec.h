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

template <typename Getter>
struct BtreeNodeReferenceArrayCodec {
  Getter getter;
  bool allow_missing = false;
  template <typename IO, typename Vec>
  [[nodiscard]] bool operator()(IO& io, Vec&& vec) const {
    if (!IndirectDataReferenceArrayCodec{[&](auto& entry) -> decltype(auto) {
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

template <typename Getter>
BtreeNodeReferenceArrayCodec(Getter, bool allow_missing = false)
    -> BtreeNodeReferenceArrayCodec<Getter>;

template <typename Getter>
struct LeafNodeValueReferenceArrayCodec {
  Getter getter;
  template <typename Entries>
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                const Entries& entries) {
    // First read lengths.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto& data_ref =
          value_reference.template emplace<IndirectDataReference>();
      if (!DataFileLengthCodec{}(reader, data_ref.length)) return false;
      if ((data_ref.length & 1) == 0 &&
          (data_ref.length >> 1) > kMaxInlineValueLength) {
        reader.Fail(absl::DataLossError(
            absl::StrFormat("Inline value length of %d exceeds maximum of %d",
                            data_ref.length >> 1, kMaxInlineValueLength)));
        return false;
      }
    }

    // Read file_ids for indirect values.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto& data_ref = std::get<IndirectDataReference>(value_reference);
      if (data_ref.length & 1) {
        if (!DataFileIdCodec{}(reader, data_ref.file_id)) return false;
      }
    }

    // Read offsets for indirect values.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto& data_ref = std::get<IndirectDataReference>(value_reference);
      if (data_ref.length & 1) {
        if (!DataFileOffsetCodec{}(reader, data_ref.offset)) return false;
      }
    }

    // Read values for direct values.
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      auto& data_ref = std::get<IndirectDataReference>(value_reference);
      uint64_t length = data_ref.length >> 1;
      if (data_ref.length & 1) {
        data_ref.length = length;
        continue;
      }
      auto& value = value_reference.template emplace<absl::Cord>();
      if (!reader.Read(length, value)) return false;
    }
    return true;
  }

  template <typename Entries>
  [[nodiscard]] bool operator()(riegeli::Writer& writer, Entries&& entries) {
    // length
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      uint64_t encoded_length;
      if (auto* data_ref =
              std::get_if<IndirectDataReference>(&value_reference)) {
        encoded_length = (data_ref->length << 1) | 1;

      } else {
        uint64_t length = std::get<absl::Cord>(value_reference).size();
        encoded_length = length << 1;
      }
      if (!DataFileLengthCodec{}(writer, encoded_length)) return false;
    }

    // file_ids for indirect values
    for (auto& entry : entries) {
      auto& value_reference = getter(entry);
      if (auto* data_ref =
              std::get_if<IndirectDataReference>(&value_reference)) {
        if (!DataFileIdCodec{}(writer, data_ref->file_id)) return false;
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

template <typename Getter>
LeafNodeValueReferenceArrayCodec(Getter)
    -> LeafNodeValueReferenceArrayCodec<Getter>;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_BTREE_CODEC_H_
