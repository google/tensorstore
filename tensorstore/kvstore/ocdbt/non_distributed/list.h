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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_

#include <stddef.h>

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ocdbt {

void NonDistributedList(ReadonlyIoHandle::Ptr io_handle,
                        std::optional<VersionSpec> version_spec,
                        kvstore::ListOptions options,
                        kvstore::ListReceiver&& receiver);

void NonDistributedListSubtree(
    ReadonlyIoHandle::Ptr io_handle, const BtreeNodeReference& node_ref,
    BtreeNodeHeight node_height, std::string subtree_key_prefix,
    KeyRange&& key_range,
    AnyFlowReceiver<absl::Status, std::string_view, span<const LeafNodeEntry>>&&
        receiver);

/// Returns `add_prefix + (key_prefix + key_suffix).substr(strip_prefix_length)`
/// without materializing the intermediate concatenation.
///
/// This is used to compute the output key for a leaf entry emitted by
/// `NonDistributedListSubtree` as `(key_prefix, entry)`.
std::string GetListedKey(std::string_view add_prefix,
                         size_t strip_prefix_length,
                         std::string_view key_prefix,
                         std::string_view key_suffix);

/// Returns `GetListedKey(add_prefix, strip_prefix_length, key_prefix,
/// entry.key)` for each of `entries`.
///
/// This computes the target keys for `BtreeWriter::CopySubtree`.
///
/// \error `absl::StatusCode::kInvalidArgument` if any target key exceeds
///     `kMaxKeyLength`.
Result<std::vector<std::string>> GetCopySubtreeTargetKeys(
    std::string_view add_prefix, size_t strip_prefix_length,
    std::string_view key_prefix, span<const LeafNodeEntry> entries);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_
