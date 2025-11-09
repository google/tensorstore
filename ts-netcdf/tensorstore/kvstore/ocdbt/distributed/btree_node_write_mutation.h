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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_WRITE_MUTATION_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_WRITE_MUTATION_H_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"

namespace tensorstore {
namespace internal_ocdbt {

struct BtreeNodeWriteMutation
    : public internal::AtomicReferenceCount<BtreeNodeWriteMutation> {
  virtual ~BtreeNodeWriteMutation() = default;
  using Ptr = internal::IntrusivePtr<const BtreeNodeWriteMutation>;
  virtual absl::Status EncodeTo(riegeli::Writer&& writer) const = 0;

  enum Mode : uint8_t {
    // Retain existing value.  This mutation merely checks that the
    // `existing_generation` matches.
    kRetainExisting = 0,

    // Delete existing value.
    kDeleteExisting = 1,

    // Replace existing value with new entry/entries.
    kAddNew = 2,
  };
  Mode mode;
};

struct BtreeLeafNodeWriteMutation : public BtreeNodeWriteMutation {
  std::string key;
  StorageGeneration existing_generation;
  struct NewEntry {
    LeafNodeValueReference value_reference;
  };

  // `new_entry` is meaningful only if
  // `BtreeNodeWriteMutation::mode == kAddNew`.
  NewEntry new_entry;

  std::string_view inclusive_min() const { return key; }
  std::string_view key_or_range() const { return key; }

  absl::Status DecodeFrom(riegeli::Reader& reader);
  absl::Status EncodeTo(riegeli::Writer&& writer) const override;
};

struct BtreeInteriorNodeWriteMutation : public BtreeNodeWriteMutation {
  KeyRange existing_range;
  StorageGeneration existing_generation;

  // Invariant: `new_entries` is non-empty, if and only if,
  // `BtreeNodeWriteMutation::mode == kAddNew`.
  std::vector<InteriorNodeEntryData<std::string>> new_entries;

  std::string_view inclusive_min() const {
    return existing_range.inclusive_min;
  }
  const KeyRange& key_or_range() const { return existing_range; }

  absl::Status DecodeFrom(riegeli::Reader& reader);
  absl::Status EncodeTo(riegeli::Writer&& writer) const override;
};

// Adds all new entries specified by `mutation` to `encoder`.
//
// Returns `true` if any entries were added.
bool AddNewEntries(BtreeNodeEncoder<LeafNodeEntry>& encoder,
                   const BtreeLeafNodeWriteMutation& mutation);

// Adds all new entries specified by `mutation` to `encoder`.
//
// Returns `true` if any entries were added.
bool AddNewEntries(BtreeNodeEncoder<InteriorNodeEntry>& encoder,
                   const BtreeInteriorNodeWriteMutation& mutation);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_WRITE_MUTATION_H_
