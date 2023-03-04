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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_WRITE_NODES_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_WRITE_NODES_H_

#include <string>
#include <vector>

#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

std::vector<InteriorNodeEntryData<std::string>> WriteNodes(
    const IoHandle& io_handle, FlushPromise& flush_promise,
    std::vector<EncodedNode> encoded_nodes);

Result<BtreeGenerationReference> WriteRootNode(
    const IoHandle& io_handle, FlushPromise& flush_promise,
    BtreeNodeHeight height,
    std::vector<InteriorNodeEntryData<std::string>> new_entries);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_WRITE_NODES_H_
