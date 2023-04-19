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
void EncodeIndirectDataReference(blake3_hasher& hasher,
                                 const IndirectDataReference& ref) {
  char header[32];
  absl::little_endian::Store64(&header[0], ref.offset);
  absl::little_endian::Store64(&header[8], ref.length);
  const size_t base_path_size = ref.file_id.base_path.size();
  absl::little_endian::Store64(&header[16], base_path_size);
  const size_t relative_path_size = ref.file_id.relative_path.size();
  absl::little_endian::Store64(&header[24], relative_path_size);
  blake3_hasher_update(&hasher, header, sizeof(header));
  blake3_hasher_update(&hasher, ref.file_id.base_path.data(), base_path_size);
  blake3_hasher_update(&hasher, ref.file_id.relative_path.data(),
                       relative_path_size);
}

void UpdateBlake3FromCord(blake3_hasher& hasher, const absl::Cord& cord) {
  for (std::string_view chunk : cord.Chunks()) {
    blake3_hasher_update(&hasher, chunk.data(), chunk.size());
  }
}
StorageGeneration GenerationFromHasher(blake3_hasher& hasher) {
  StorageGeneration generation;
  generation.value.resize(21);
  generation.value[20] = StorageGeneration::kBaseGeneration;
  blake3_hasher_finalize(
      &hasher, reinterpret_cast<uint8_t*>(generation.value.data()), 20);
  return generation;
}
}  // namespace

StorageGeneration ComputeStorageGeneration(const LeafNodeValueReference& ref) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  char mode;
  if (auto* location = std::get_if<IndirectDataReference>(&ref)) {
    mode = 0;
    blake3_hasher_update(&hasher, &mode, 1);
    EncodeIndirectDataReference(hasher, *location);
  } else {
    mode = 1;
    blake3_hasher_update(&hasher, &mode, 1);
    UpdateBlake3FromCord(hasher, std::get<absl::Cord>(ref));
  }
  return GenerationFromHasher(hasher);
}

StorageGeneration ComputeStorageGeneration(
    const IndirectDataReference& location,
    std::string_view subtree_common_prefix) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  EncodeIndirectDataReference(hasher, location);
  blake3_hasher_update(&hasher, subtree_common_prefix.data(),
                       subtree_common_prefix.size());
  return GenerationFromHasher(hasher);
}

StorageGeneration ComputeStorageGeneration(
    const InteriorNodeEntry& entry, std::string_view subtree_common_prefix) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  EncodeIndirectDataReference(hasher, entry.node.location);
  blake3_hasher_update(&hasher, subtree_common_prefix.data(),
                       subtree_common_prefix.size());
  blake3_hasher_update(&hasher, entry.key.data(),
                       entry.subtree_common_prefix_length);
  return GenerationFromHasher(hasher);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
