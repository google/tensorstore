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

#include "tensorstore/kvstore/ocdbt/non_distributed/write_nodes.h"

#include <cassert>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

std::vector<InteriorNodeEntryData<std::string>> WriteNodes(
    const IoHandle& io_handle, FlushPromise& flush_promise,
    std::vector<EncodedNode> encoded_nodes) {
  std::vector<InteriorNodeEntryData<std::string>> new_entries(
      encoded_nodes.size());
  for (size_t i = 0; i < encoded_nodes.size(); ++i) {
    auto& encoded_node = encoded_nodes[i];
    auto& new_entry = new_entries[i];
    flush_promise.Link(io_handle.WriteData(std::move(encoded_node.encoded_node),
                                           new_entry.node.location));
    new_entry.key = std::move(encoded_node.info.inclusive_min_key);
    new_entry.node.statistics = encoded_node.info.statistics;
    new_entry.subtree_common_prefix_length =
        encoded_node.info.excluded_prefix_length;
  }

  return new_entries;
}

Result<BtreeGenerationReference> WriteRootNode(
    const IoHandle& io_handle, FlushPromise& flush_promise,
    BtreeNodeHeight height,
    std::vector<InteriorNodeEntryData<std::string>> new_entries) {
  while (true) {
    if (new_entries.size() <= 1) {
      BtreeGenerationReference new_generation;
      if (new_entries.empty()) {
        new_generation.root_height = 0;
        new_generation.root.statistics = {};
        new_generation.root.location = IndirectDataReference::Missing();
      } else {
        new_generation.root_height = height;
        new_generation.root = new_entries[0].node;
      }
      return new_generation;
    }

    // Create additional level.
    if (height == std::numeric_limits<BtreeNodeHeight>::max()) {
      return absl::DataLossError("Maximum B+tree height exceeded");
    }
    ++height;
    auto* config = io_handle.config_state->GetExistingConfig();
    assert(config);
    BtreeInteriorNodeEncoder node_encoder(*config, height,
                                          /*existing_prefix=*/{});
    for (auto& entry : new_entries) {
      internal_ocdbt::AddNewInteriorEntry(node_encoder, entry);
    }
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded_nodes,
                                 node_encoder.Finalize(/*may_be_root=*/true));
    new_entries = internal_ocdbt::WriteNodes(io_handle, flush_promise,
                                             std::move(encoded_nodes));
  }
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
