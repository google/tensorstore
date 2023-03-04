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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_STORAGE_GENERATION_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_STORAGE_GENERATION_H_

#include <string_view>

#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"

namespace tensorstore {
namespace internal_ocdbt {

// Computes a storage generation from a leaf node value reference.
//
// - For value stored inline inline in the leaf node, the generation is derived
//   from a cryptographic hash of the value.
//
// - For values stored out-of-line, the generation is derived from the location
//   of the value only.
StorageGeneration ComputeStorageGeneration(const LeafNodeValueReference& ref);

// Computes a storage generation from a node location and its subtree key
// prefix.
StorageGeneration ComputeStorageGeneration(
    const IndirectDataReference& location,
    std::string_view subtree_common_prefix);

// Computes a storage generation for a B+tree node entry.
//
// - For leaf node entries, this is derived only from the
//   `LeafNodeEntry::value_reference`.
//
// - For interior node entries, this is derived from both the child node
//   location and the `subtree_common_prefix`.
inline StorageGeneration ComputeStorageGeneration(
    const LeafNodeEntry& entry, std::string_view subtree_common_prefix) {
  return ComputeStorageGeneration(entry.value_reference);
}
StorageGeneration ComputeStorageGeneration(
    const InteriorNodeEntry& entry, std::string_view subtree_common_prefix);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_STORAGE_GENERATION_H_
