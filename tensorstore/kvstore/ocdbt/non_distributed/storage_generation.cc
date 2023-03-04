// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"

#include <cstring>
#include <string>
#include <string_view>
#include <variant>

#include "absl/base/internal/endian.h"
#include "absl/strings/cord.h"
#include <blake3.h>
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {
void EncodeIndirectDataReference(const IndirectDataReference& ref,
                                 unsigned char buffer[32]) {
  std::memcpy(&buffer[0], &ref.file_id, sizeof(DataFileId));
  absl::little_endian::Store64(&buffer[16], ref.offset);
  absl::little_endian::Store64(&buffer[24], ref.length);
}

void UpdateBlake3FromCord(blake3_hasher& hasher, const absl::Cord& cord) {
  for (std::string_view chunk : cord.Chunks()) {
    blake3_hasher_update(&hasher, chunk.data(), chunk.size());
  }
}
}  // namespace

StorageGeneration ComputeStorageGeneration(const LeafNodeValueReference& ref) {
  StorageGeneration generation;
  if (auto* location = std::get_if<IndirectDataReference>(&ref)) {
    generation.value.resize(33);
    EncodeIndirectDataReference(
        *location, reinterpret_cast<uint8_t*>(generation.value.data()));
    generation.value[32] = StorageGeneration::kBaseGeneration;
  } else {
    generation.value.resize(21);
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    UpdateBlake3FromCord(hasher, std::get<absl::Cord>(ref));
    blake3_hasher_finalize(
        &hasher, reinterpret_cast<uint8_t*>(generation.value.data()), 20);
    generation.value[20] = StorageGeneration::kBaseGeneration;
  }
  return generation;
}

StorageGeneration ComputeStorageGeneration(
    const IndirectDataReference& location,
    std::string_view subtree_common_prefix) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  unsigned char buffer[32];
  EncodeIndirectDataReference(location, buffer);
  blake3_hasher_update(&hasher, buffer, 32);
  blake3_hasher_update(&hasher, subtree_common_prefix.data(),
                       subtree_common_prefix.size());
  StorageGeneration generation;
  generation.value.resize(21);
  blake3_hasher_finalize(
      &hasher, reinterpret_cast<uint8_t*>(generation.value.data()), 20);
  generation.value[20] = StorageGeneration::kBaseGeneration;
  return generation;
}

StorageGeneration ComputeStorageGeneration(
    const InteriorNodeEntry& entry, std::string_view subtree_common_prefix) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  unsigned char buffer[32];
  EncodeIndirectDataReference(entry.node.location, buffer);
  blake3_hasher_update(&hasher, buffer, 32);
  blake3_hasher_update(&hasher, subtree_common_prefix.data(),
                       subtree_common_prefix.size());
  blake3_hasher_update(&hasher, entry.key.data(),
                       entry.subtree_common_prefix_length);
  StorageGeneration generation;
  generation.value.resize(21);
  blake3_hasher_finalize(
      &hasher, reinterpret_cast<uint8_t*>(generation.value.data()), 20);
  generation.value[20] = StorageGeneration::kBaseGeneration;
  return generation;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
