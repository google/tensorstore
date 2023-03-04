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

#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/serialization/riegeli_delimited.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {
struct KeyCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                std::string& value) const {
    KeyLength length;
    if (!KeyLengthCodec{}(reader, length)) return false;
    value.resize(length);
    return reader.Read(length, value);
  }
  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                std::string_view value) const {
    assert(value.size() <= std::numeric_limits<KeyLength>::max());
    return KeyLengthCodec{}(writer, static_cast<KeyLength>(value.size())) &&
           writer.Write(value);
  }
};

struct KeyRangeCodec {
  template <typename IO, typename Value>
  [[nodiscard]] bool operator()(IO& io, Value&& value) const {
    static_assert(std::is_same_v<IO, riegeli::Writer> ||
                  std::is_same_v<IO, riegeli::Reader>);
    if (!KeyCodec{}(io, value.inclusive_min) ||
        !KeyCodec{}(io, value.exclusive_max)) {
      return false;
    }
    return true;
  }
};

struct SizeCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader, size_t& value) const {
    return serialization::ReadSize(reader, value);
  }
  [[nodiscard]] bool operator()(riegeli::Writer& writer, size_t value) const {
    return serialization::WriteSize(writer, value);
  }
};

struct GenerationCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                std::string& value) const {
    return serialization::ReadDelimited(reader, value);
  }
  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                std::string_view value) const {
    return serialization::WriteDelimited(writer, value);
  }
};

struct BtreeLeafNodeWriteMutationCodec {
  template <typename IO, typename Value>
  [[nodiscard]] bool operator()(IO& io, Value&& value) {
    static_assert(std::is_same_v<IO, riegeli::Writer> ||
                  std::is_same_v<IO, riegeli::Reader>);
    uint8_t has_new_entry;
    if constexpr (std::is_same_v<IO, riegeli::Writer>) {
      has_new_entry = value.new_entry.has_value();
    }
    if (!KeyCodec{}(io, value.key) ||
        !GenerationCodec{}(io, value.existing_generation.value) ||
        !LittleEndianCodec<uint8_t>{}(io, has_new_entry)) {
      return false;
    }
    if (!has_new_entry) return true;
    if constexpr (std::is_same_v<IO, riegeli::Reader>) {
      value.new_entry.emplace();
    }
    // Reuse the `LeafNodeValueReferenceArrayCodec` to write a single
    // `LeafNodeValueReference`.
    return LeafNodeValueReferenceArrayCodec{[](auto& e) -> decltype(auto) {
      return (e.value_reference);
    }}(io, span(&*value.new_entry, 1));
  }
};

struct BtreeInteriorNodeWriteMutationCodec {
  template <typename IO, typename Value>
  [[nodiscard]] bool operator()(IO& io, Value&& value) {
    static_assert(std::is_same_v<IO, riegeli::Writer> ||
                  std::is_same_v<IO, riegeli::Reader>);
    size_t num_entries;
    if constexpr (std::is_same_v<IO, riegeli::Writer>) {
      num_entries = value.new_entries.size();
    }
    if (!KeyRangeCodec{}(io, value.existing_range) ||
        !GenerationCodec{}(io, value.existing_generation.value) ||
        !SizeCodec{}(io, num_entries)) {
      return false;
    }

    if constexpr (std::is_same_v<IO, riegeli::Reader>) {
      if (value.existing_range.empty()) {
        io.Fail(absl::InvalidArgumentError("empty key range"));
        return false;
      }
    }

    constexpr size_t kMaxInitialSize = 1000;
    if constexpr (std::is_same_v<IO, riegeli::Reader>) {
      value.new_entries.resize(std::min(num_entries, kMaxInitialSize));
    }

    for (size_t i = 0; i < num_entries; ++i) {
      if constexpr (std::is_same_v<IO, riegeli::Reader>) {
        // Incrementally increase size rather than preallocating, to avoid
        // crashing if an invalid size is received.
        if (i >= kMaxInitialSize) {
          value.new_entries.emplace_back();
        }
      }
      auto& entry = value.new_entries[i];
      if (!KeyCodec{}(io, entry.key) ||
          !KeyLengthCodec{}(io, entry.subtree_common_prefix_length)) {
        return false;
      }

      if constexpr (std::is_same_v<IO, riegeli::Reader>) {
        if (entry.subtree_common_prefix_length > entry.key.size()) {
          io.Fail(absl::InvalidArgumentError(
              "subtree_common_prefix_length must not exceed key length"));
          return false;
        }
      }

      if (!BtreeNodeReferenceArrayCodec{[](auto& e) -> decltype(auto) {
            return (e.node);
          }}(io, value.new_entries)) {
        return false;
      }
    }
    return true;
  }
};
}  // namespace

absl::Status BtreeLeafNodeWriteMutation::DecodeFrom(riegeli::Reader& reader) {
  return internal_ocdbt::FinalizeReader(
      reader, BtreeLeafNodeWriteMutationCodec{}(reader, *this));
}

absl::Status BtreeLeafNodeWriteMutation::EncodeTo(
    riegeli::Writer&& writer) const {
  return internal_ocdbt::FinalizeWriter(
      writer, BtreeLeafNodeWriteMutationCodec{}(writer, *this));
}

absl::Status BtreeInteriorNodeWriteMutation::DecodeFrom(
    riegeli::Reader& reader) {
  return internal_ocdbt::FinalizeReader(
      reader, BtreeInteriorNodeWriteMutationCodec{}(reader, *this));
}

absl::Status BtreeInteriorNodeWriteMutation::EncodeTo(
    riegeli::Writer&& writer) const {
  return internal_ocdbt::FinalizeWriter(
      writer, BtreeInteriorNodeWriteMutationCodec{}(writer, *this));
}

bool AddNewEntries(BtreeNodeEncoder<LeafNodeEntry>& encoder,
                   const BtreeLeafNodeWriteMutation& mutation) {
  if (!mutation.new_entry) return false;
  auto& new_entry = *mutation.new_entry;
  LeafNodeEntry entry;
  entry.key = mutation.key;
  entry.value_reference = new_entry.value_reference;
  encoder.AddEntry(/*existing=*/false, std::move(entry));
  return true;
}

bool AddNewEntries(BtreeNodeEncoder<InteriorNodeEntry>& encoder,
                   const BtreeInteriorNodeWriteMutation& mutation) {
  for (const auto& new_entry : mutation.new_entries) {
    AddNewInteriorEntry(encoder, new_entry);
  }
  return !mutation.new_entries.empty();
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
