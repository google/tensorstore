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

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
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
struct ListOperation : public internal::AtomicReferenceCount<ListOperation> {
  using Ptr = internal::IntrusivePtr<ListOperation>;
  ReadonlyIoHandle::Ptr io_handle;
  kvstore::ListOptions list_options;
  AnyFlowReceiver<absl::Status, kvstore::Key> receiver;

  // Initiates the asynchronous list operation.
  //
  // Args:
  //   io_handle: I/O handle to use.
  //   options: List options.
  //   receiver: Receiver of the results.
  static void Start(ReadonlyIoHandle::Ptr&& io_handle,
                    kvstore::ListOptions&& options,
                    AnyFlowReceiver<absl::Status, kvstore::Key>&& receiver) {
    auto [cancel_promise, cancel_future] =
        PromiseFuturePair<void>::Make(absl::OkStatus());
    execution::set_starting(receiver, [cancel_promise = cancel_promise] {
      SetDeferredResult(cancel_promise, absl::CancelledError(""));
    });
    auto op = internal::MakeIntrusivePtr<ListOperation>();
    op->io_handle = std::move(io_handle);
    op->receiver = std::move(receiver);
    op->list_options = std::move(options);
    cancel_future.ExecuteWhenReady([op](ReadyFuture<void> f) {
      if (f.status().ok() || absl::IsCancelled(f.status())) {
        execution::set_done(op->receiver);
      } else {
        execution::set_error(op->receiver, f.status());
      }
      execution::set_stopping(op->receiver);
    });

    auto [list_promise, list_future] =
        PromiseFuturePair<void>::Make(absl::OkStatus());
    Link([](Promise<void> promise,
            ReadyFuture<void> future) { promise.SetResult(future.result()); },
         std::move(cancel_promise), std::move(list_future));

    auto* op_ptr = op.get();

    auto manifest_future =
        op_ptr->io_handle->GetManifest(op->list_options.staleness_bound);
    Link(WithExecutor(op_ptr->io_handle->executor,
                      ListOperation::ManifestReadyCallback{std::move(op)}),
         std::move(list_promise), std::move(manifest_future));
  }

  // Called when the manifest lookup has completed.
  struct ManifestReadyCallback {
    ListOperation::Ptr op;
    void operator()(Promise<void> promise,
                    ReadyFuture<const ManifestWithTime> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto manifest_with_time, read_future.result(),
          static_cast<void>(SetDeferredResult(promise, _)));
      const auto* manifest = manifest_with_time.manifest.get();
      if (!manifest || manifest->latest_version().root.location.IsMissing()) {
        // Manifest not present or btree is empty.
        return;
      }
      auto& latest_version = manifest->versions.back();
      VisitSubtree(std::move(op), std::move(promise), latest_version.root,
                   latest_version.root_height,
                   /*inclusive_min_key=*/{},
                   /*subtree_common_prefix_length=*/0);
    }
  };

  // Emit all matches within a subtree.
  //
  // Args:
  //   op: List operation state.
  //   promise: Promise to be resolved once the operation completes.
  //   node_height: Height of the node.
  //   inclusive_min_key: Full inclusive min key for the node.
  //   prefix_length: Length of the prefix of `inclusive_min_key` that specifies
  //     the implicit prefix that is excluded from the encoded representation of
  //     the node.
  static void VisitSubtree(ListOperation::Ptr op, Promise<void> promise,
                           const BtreeNodeReference& node_ref,
                           BtreeNodeHeight node_height,
                           std::string inclusive_min_key,
                           KeyLength subtree_common_prefix_length) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "List: "
        << "subtree_common_prefix_length=" << subtree_common_prefix_length
        << ", node_height=" << static_cast<int>(node_height)
        << ", inclusive_min_key="
        << tensorstore::QuoteString(inclusive_min_key);
    auto read_future = op->io_handle->GetBtreeNode(node_ref.location);
    auto executor = op->io_handle->executor;
    Link(WithExecutor(std::move(executor),
                      NodeReadyCallback{std::move(op), node_height,
                                        std::move(inclusive_min_key),
                                        subtree_common_prefix_length}),
         std::move(promise), std::move(read_future));
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
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto node, read_future.result(),
          static_cast<void>(SetDeferredResult(promise, _)));
      if (!promise.result_needed()) return;
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateBtreeNodeReference(*node, node_height,
                                     std::string_view(inclusive_min_key)
                                         .substr(subtree_common_prefix_length)),
          static_cast<void>(SetDeferredResult(promise, _)));
      auto& subtree_key_prefix = inclusive_min_key;
      subtree_key_prefix.resize(subtree_common_prefix_length);
      subtree_key_prefix += node->key_prefix;
      auto key_range =
          KeyRange::RemovePrefix(subtree_key_prefix, op->list_options.range);

      if (node->height > 0) {
        VisitInteriorNode(std::move(op), std::move(promise), *node,
                          subtree_key_prefix, key_range);
      } else {
        VisitLeafNode(std::move(op), std::move(promise), *node,
                      subtree_key_prefix, key_range);
      }
    }
  };

  template <typename Entry>
  static span<const Entry> GetMatchingEntryRange(span<const Entry> entries,
                                                 const KeyRange& key_range) {
    auto lower = std::lower_bound(
        entries.data(), entries.data() + entries.size(),
        key_range.inclusive_min,
        [](const Entry& entry, std::string_view inclusive_min) {
          return entry.key < inclusive_min;
        });
    auto upper = std::upper_bound(
        entries.begin(), entries.end(), key_range.exclusive_max,
        [](std::string_view exclusive_max, const Entry& entry) {
          return KeyRange::CompareExclusiveMaxAndKey(exclusive_max, entry.key) <
                 0;
        });
    return {lower, upper};
  }

  // Recursively visits matching children.
  static void VisitInteriorNode(ListOperation::Ptr op, Promise<void> promise,
                                const BtreeNode& node,
                                std::string_view subtree_key_prefix,
                                const KeyRange& key_range) {
    auto entries = GetMatchingEntryRange<InteriorNodeEntry>(
        std::get<BtreeNode::InteriorNodeEntries>(node.entries), key_range);
    for (const auto& entry : entries) {
      VisitSubtree(op, promise, entry.node, node.height - 1,
                   /*inclusive_min_key=*/
                   tensorstore::StrCat(subtree_key_prefix, entry.key),
                   /*subtree_common_prefix_length=*/subtree_key_prefix.size() +
                       entry.subtree_common_prefix_length);
    }
  }

  // Emits matches in the leaf node.
  static void VisitLeafNode(ListOperation::Ptr op, Promise<void> promise,
                            const BtreeNode& node,
                            std::string_view subtree_key_prefix,
                            const KeyRange& key_range) {
    auto entries = GetMatchingEntryRange<LeafNodeEntry>(
        std::get<BtreeNode::LeafNodeEntries>(node.entries), key_range);
    const size_t strip_prefix_length = op->list_options.strip_prefix_length;
    for (const auto& entry : entries) {
      auto key = tensorstore::StrCat(
          std::string_view(subtree_key_prefix)
              .substr(std::min(subtree_key_prefix.size(), strip_prefix_length)),
          std::string_view(entry.key).substr(std::min(
              entry.key.size(),
              strip_prefix_length -
                  std::min(strip_prefix_length, subtree_key_prefix.size()))));
      execution::set_value(op->receiver, std::move(key));
    }
  }
};
}  // namespace

void NonDistributedList(ReadonlyIoHandle::Ptr io_handle,
                        kvstore::ListOptions options,
                        AnyFlowReceiver<absl::Status, kvstore::Key> receiver) {
  ListOperation::Start(std::move(io_handle), std::move(options),
                       std::move(receiver));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
