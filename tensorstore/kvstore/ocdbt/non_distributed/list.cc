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

#include "tensorstore/kvstore/ocdbt/non_distributed/list.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;

// Asynchronous operation state used to implement `internal_ocdbt::List`.
//
// The list operation is implemented as follows:
//
// 1. Resolve the root b+tree node by reading the manifest.
//
// 2. Recursively descend the tree in parallel, reading all nodes that
//    intersect the key range specified in `list_options`.
//
// 3. Emit matching leaf-node keys to the receiver.
//
// TODO(jbms): Currently memory usage is not bounded.  That needs to be
// addressed, e.g. by limiting the number of in-flight nodes.
struct ListOperation
    : public internal::FlowSenderOperationState<std::string_view,
                                                span<const LeafNodeEntry>> {
  using Ptr = internal::IntrusivePtr<ListOperation>;
  using Base = internal::FlowSenderOperationState<std::string_view,
                                                  span<const LeafNodeEntry>>;

  using Base::Base;

  ReadonlyIoHandle::Ptr io_handle;
  KeyRange range;

  // Prepares the asynchronous list operation.
  //
  // Args:
  //   io_handle: I/O handle to use.
  //   range: Key range constraint.
  //   receiver: Receiver of the results.
  static Ptr Initialize(ReadonlyIoHandle::Ptr&& io_handle, KeyRange&& range,
                        BaseReceiver&& receiver) {
    auto op = internal::MakeIntrusivePtr<ListOperation>(std::move(receiver));
    op->io_handle = std::move(io_handle);
    op->range = std::move(range);
    return op;
  }

  // Called when the manifest lookup has completed.
  struct ManifestReadyCallback {
    ListOperation::Ptr op;
    void operator()(Promise<void> promise,
                    ReadyFuture<const ManifestWithTime> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto manifest_with_time,
                                   read_future.result(), op->SetError(_));
      const auto* manifest = manifest_with_time.manifest.get();
      if (!manifest || manifest->latest_version().root.location.IsMissing()) {
        // Manifest not present or btree is empty.
        return;
      }
      auto& latest_version = manifest->versions.back();
      VisitSubtree(std::move(op), latest_version.root,
                   latest_version.root_height,
                   /*inclusive_min_key=*/{},
                   /*subtree_common_prefix_length=*/0);
    }
  };

  // Emit all matches within a subtree.
  //
  // Args:
  //   op: List operation state.
  //   node_height: Height of the node.
  //   inclusive_min_key: Full inclusive min key for the node.
  //   prefix_length: Length of the prefix of `inclusive_min_key` that specifies
  //     the implicit prefix that is excluded from the encoded representation of
  //     the node.
  static void VisitSubtree(ListOperation::Ptr op,
                           const BtreeNodeReference& node_ref,
                           BtreeNodeHeight node_height,
                           std::string inclusive_min_key,
                           KeyLength subtree_common_prefix_length) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "List: node=" << node_ref
        << ", node_height=" << static_cast<int>(node_height)
        << ", subtree_common_prefix_length=" << subtree_common_prefix_length
        << ", inclusive_min_key=" << tensorstore::QuoteString(inclusive_min_key)
        << ", key_range=" << op->range;
    auto* op_ptr = op.get();
    Link(WithExecutor(op_ptr->io_handle->executor,
                      NodeReadyCallback{std::move(op), node_height,
                                        std::move(inclusive_min_key),
                                        subtree_common_prefix_length}),
         op_ptr->promise, op_ptr->io_handle->GetBtreeNode(node_ref.location));
  }

  // Called when a B+tree node lookup completes.
  struct NodeReadyCallback {
    ListOperation::Ptr op;

    BtreeNodeHeight node_height;

    // Full inclusive min key for `node_cache_entry`.
    std::string inclusive_min_key;

    // Specifies the length of the implicit prefix that is excluded from the
    // encoded representation of the node specified by `node_cache_entry`.  The
    // prefix is equal to
    // `inclusive_min_key.substr(subtree_common_prefix_length)`.
    KeyLength subtree_common_prefix_length;

    void operator()(
        Promise<void> promise,
        ReadyFuture<const std::shared_ptr<const BtreeNode>> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto node, read_future.result(),
                                   op->SetError(_));
      if (op->cancelled()) return;
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateBtreeNodeReference(*node, node_height,
                                     std::string_view(inclusive_min_key)
                                         .substr(subtree_common_prefix_length)),
          op->SetError(_));
      auto& subtree_key_prefix = inclusive_min_key;
      subtree_key_prefix.resize(subtree_common_prefix_length);
      subtree_key_prefix += node->key_prefix;
      auto key_range = KeyRange::RemovePrefix(subtree_key_prefix, op->range);

      if (node->height > 0) {
        VisitInteriorNode(std::move(op), *node, subtree_key_prefix, key_range);
      } else {
        VisitLeafNode(std::move(op), *node, subtree_key_prefix, key_range);
      }
    }
  };

  // Recursively visits matching children.
  static void VisitInteriorNode(ListOperation::Ptr op, const BtreeNode& node,
                                std::string_view subtree_key_prefix,
                                const KeyRange& key_range) {
    auto& all_entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);
    auto entries = FindBtreeEntryRange(all_entries, key_range.inclusive_min,
                                       key_range.exclusive_max);
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "VisitInteriorNode: subtree_key_prefix="
        << tensorstore::QuoteString(subtree_key_prefix)
        << ", key_range=" << key_range << ", first node key="
        << tensorstore::QuoteString(all_entries.front().key)
        << ", last node key="
        << tensorstore::QuoteString(all_entries.back().key)
        << ", num matches=" << entries.size();
    // Note: It is safe to access `all_entries.front()` and `all_entries.back()`
    // because B+tree nodes are guaranteed to have at least one entry.
    for (const auto& entry : entries) {
      VisitSubtree(op, entry.node, node.height - 1,
                   /*inclusive_min_key=*/
                   tensorstore::StrCat(subtree_key_prefix, entry.key),
                   /*subtree_common_prefix_length=*/subtree_key_prefix.size() +
                       entry.subtree_common_prefix_length);
    }
  }

  // Emits matches in the leaf node.
  static void VisitLeafNode(ListOperation::Ptr op, const BtreeNode& node,
                            std::string_view subtree_key_prefix,
                            const KeyRange& key_range) {
    auto& all_entries = std::get<BtreeNode::LeafNodeEntries>(node.entries);
    auto entries = FindBtreeEntryRange(all_entries, key_range.inclusive_min,
                                       key_range.exclusive_max);
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "VisitLeafNode: subtree_key_prefix="
        << tensorstore::QuoteString(subtree_key_prefix)
        << ", key_range=" << key_range << ", first node key="
        << tensorstore::QuoteString(all_entries.front().key)
        << ", last node key="
        << tensorstore::QuoteString(all_entries.back().key)
        << ", num matches=" << entries.size();
    // Note: It is safe to access `all_entries.front()` and `all_entries.back()`
    // because B+tree nodes are guaranteed to have at least one entry.
    if (entries.empty()) return;
    execution::set_value(op->shared_receiver->receiver, subtree_key_prefix,
                         entries);
  }
};

// Adapts a kvstore List receiver into the receiver type expected by
// `ListOperation`.
struct KeyReceiverAdapter {
  ListReceiver receiver;
  size_t strip_prefix_length = 0;

  void set_done() { execution::set_done(receiver); }

  void set_error(absl::Status&& error) {
    execution::set_error(receiver, std::move(error));
  }

  void set_value(std::string_view key_prefix,
                 span<const LeafNodeEntry> entries) {
    for (const auto& entry : entries) {
      auto key = tensorstore::StrCat(
          std::string_view(key_prefix)
              .substr(std::min(key_prefix.size(), strip_prefix_length)),
          std::string_view(entry.key).substr(
              std::min(entry.key.size(),
                       strip_prefix_length -
                           std::min(strip_prefix_length, key_prefix.size()))));
      execution::set_value(receiver,
                           ListEntry{
                               std::move(key),
                               ListEntry::checked_size(entry.value_size()),
                           });
    }
  }

  template <typename Cancel>
  void set_starting(Cancel&& cancel) {
    execution::set_starting(receiver, std::forward<Cancel>(cancel));
  }

  void set_stopping() { execution::set_stopping(receiver); }
};

}  // namespace

void NonDistributedList(ReadonlyIoHandle::Ptr io_handle,
                        kvstore::ListOptions options, ListReceiver&& receiver) {
  auto op = ListOperation::Initialize(
      std::move(io_handle), std::move(options.range),
      KeyReceiverAdapter{std::move(receiver), options.strip_prefix_length});
  auto* op_ptr = op.get();
  Link(WithExecutor(op_ptr->io_handle->executor,
                    ListOperation::ManifestReadyCallback{std::move(op)}),
       op_ptr->promise,
       op_ptr->io_handle->GetManifest(options.staleness_bound));
}

void NonDistributedListSubtree(
    ReadonlyIoHandle::Ptr io_handle, const BtreeNodeReference& node_ref,
    BtreeNodeHeight node_height, std::string subtree_key_prefix,
    KeyRange&& key_range,
    AnyFlowReceiver<absl::Status, std::string_view, span<const LeafNodeEntry>>&&
        receiver) {
  auto op = ListOperation::Initialize(
      std::move(io_handle), std::move(key_range), std::move(receiver));
  const size_t subtree_common_prefix_length = subtree_key_prefix.size();
  ListOperation::VisitSubtree(std::move(op), node_ref, node_height,
                              std::move(subtree_key_prefix),
                              subtree_common_prefix_length);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
