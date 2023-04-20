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

#include "tensorstore/kvstore/ocdbt/non_distributed/read.h"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/storage_generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {

// Asynchronous operation state used to implement
// `internal_ocdbt::NonDistributedRead`.
//
// The read operation is implemented as follows:
//
// 1. Resolve the root b+tree node by reading the manifest.
//
// 2. Descend the tree along the path to the requested key.
//
// 3. Stop and return a missing value indication as soon as the key is
//    determined to be missing.
//
// 4. Once the leaf node containing the key is reached, either return the value
//    directly (if stored inline), or read it via
//    `ReadonlyIoHandle::ReadIndirectData` (if stored separately).
struct ReadOperation : public internal::AtomicReferenceCount<ReadOperation> {
  using Ptr = internal::IntrusivePtr<ReadOperation>;
  ReadonlyIoHandle::Ptr io_handle;
  StorageGeneration if_not_equal;
  StorageGeneration if_equal;
  OptionalByteRangeRequest byte_range;
  absl::Time time;
  // Full key being read.
  std::string key;
  // Length of the prefix of `key` that corresponds to the implicit prefix of
  // the node currently being processed.
  size_t matched_length = 0;

  // Generation of indirect value.
  StorageGeneration generation;

  // Initiates the asynchronous read operation.
  //
  // Args:
  //   promise: Promise on which to set read result when ready.
  //   io_handle: I/O handle to use.
  //   key: Key to read.
  //   options: Additional read options.
  //
  // Returns:
  //   Futures that resolves when the read completes.
  static Future<kvstore::ReadResult> Start(ReadonlyIoHandle::Ptr io_handle,
                                           kvstore::Key&& key,
                                           kvstore::ReadOptions&& options) {
    auto op = internal::MakeIntrusivePtr<ReadOperation>();
    op->io_handle = std::move(io_handle);
    op->if_not_equal = std::move(options.if_not_equal);
    op->if_equal = std::move(options.if_equal);
    op->byte_range = options.byte_range;
    op->key = std::move(key);
    auto* op_ptr = op.get();
    return PromiseFuturePair<kvstore::ReadResult>::LinkValue(
               WithExecutor(
                   op_ptr->io_handle->executor,
                   [op = std::move(op)](
                       Promise<kvstore::ReadResult> promise,
                       ReadyFuture<const ManifestWithTime> future) mutable {
                     ManifestReady(std::move(op), std::move(promise),
                                   future.value());
                   }),
               op_ptr->io_handle->GetManifest(options.staleness_bound))
        .future;
  }

  // Called when the manifest lookup has completed.
  static void ManifestReady(ReadOperation::Ptr op,
                            Promise<kvstore::ReadResult> promise,
                            const ManifestWithTime& manifest_with_time) {
    op->time = manifest_with_time.time;
    auto* manifest = manifest_with_time.manifest.get();
    if (!manifest || manifest->latest_version().root.location.IsMissing()) {
      // Manifest not preset or btree is empty.
      op->KeyNotPresent(promise);
      return;
    }
    auto& latest_version = manifest->versions.back();
    LookupNodeReference(std::move(op), std::move(promise), latest_version.root,
                        latest_version.root_height,
                        /*inclusive_min_key=*/{});
  }

  // Completes the read request, indicating that the key is missing.
  void KeyNotPresent(const Promise<kvstore::ReadResult>& promise) {
    promise.SetResult(
        std::in_place, kvstore::ReadResult::kMissing, absl::Cord(),
        TimestampedStorageGeneration{StorageGeneration::NoValue(), time});
  }

  // Recursively descends the B+tree.
  //
  // Args:
  //   op: Operation state.
  //   promise: Promise on which to set the read result when ready.
  //   node_ref: Reference to the B+tree node to descend.
  //   node_height: Height of the node specified by `node_ref`.
  //   inclusive_min_key: Lower bound of the key range corresponding to
  //     `node_ref`.
  static void LookupNodeReference(ReadOperation::Ptr op,
                                  Promise<kvstore::ReadResult> promise,
                                  const BtreeNodeReference& node_ref,
                                  BtreeNodeHeight node_height,
                                  std::string_view inclusive_min_key) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "Read: key=" << tensorstore::QuoteString(op->key)
        << ", matched_length=" << op->matched_length
        << ", node_height=" << static_cast<int>(node_height)
        << ", inclusive_min_key="
        << tensorstore::QuoteString(inclusive_min_key);

    auto read_future = op->io_handle->GetBtreeNode(node_ref.location);
    auto executor = op->io_handle->executor;
    LinkValue(WithExecutor(std::move(executor),
                           NodeReadyCallback{std::move(op), node_height,
                                             std::string(inclusive_min_key)}),
              std::move(promise), std::move(read_future));
  }

  struct NodeReadyCallback {
    ReadOperation::Ptr op;
    BtreeNodeHeight node_height;
    std::string inclusive_min_key;
    void operator()(
        Promise<kvstore::ReadResult> promise,
        ReadyFuture<const std::shared_ptr<const BtreeNode>> read_future) {
      auto node = read_future.value();
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateBtreeNodeReference(*node, node_height, inclusive_min_key),
          static_cast<void>(promise.SetResult(_)));
      auto unmatched_key_suffix =
          std::string_view(op->key).substr(op->matched_length);
      std::string_view node_key_prefix = node->key_prefix;
      if (!absl::StartsWith(unmatched_key_suffix, node_key_prefix)) {
        op->KeyNotPresent(promise);
        return;
      }
      unmatched_key_suffix.remove_prefix(node_key_prefix.size());
      op->matched_length += node_key_prefix.size();

      if (node->height > 0) {
        VisitInteriorNode(std::move(op), *node, std::move(promise),
                          unmatched_key_suffix);
      } else {
        VisitLeafNode(std::move(op), *node, std::move(promise),
                      unmatched_key_suffix);
      }
    }
  };

  static void VisitInteriorNode(ReadOperation::Ptr op, const BtreeNode& node,
                                Promise<kvstore::ReadResult> promise,
                                std::string_view unmatched_key_suffix) {
    auto& entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);
    auto it =
        std::upper_bound(entries.begin(), entries.end(), unmatched_key_suffix,
                         [](std::string_view unmatched_key_suffix,
                            const InteriorNodeEntry& entry) {
                           return unmatched_key_suffix < entry.key;
                         });
    if (it == entries.begin()) {
      // Less than first key in node, which indicates key is not present.
      op->KeyNotPresent(promise);
      return;
    }
    --it;
    std::string_view subtree_common_prefix =
        std::string_view(it->key).substr(0, it->subtree_common_prefix_length);
    if (!absl::StartsWith(unmatched_key_suffix, subtree_common_prefix)) {
      // Remaining unmatched portion of key does not match subtree common prefix
      // of the child that must contain it.  Therefore, the key is not present.
      op->KeyNotPresent(promise);
      return;
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "Read: key=" << tensorstore::QuoteString(op->key)
        << ", matched_length=" << op->matched_length
        << ", node.height=" << static_cast<int>(node.height)
        << ", node.key_prefix=" << tensorstore::QuoteString(node.key_prefix)
        << ", key=" << tensorstore::QuoteString(it->key)
        << ", subtree_common_prefix_length="
        << it->subtree_common_prefix_length;

    op->matched_length += it->subtree_common_prefix_length;
    LookupNodeReference(std::move(op), std::move(promise), it->node,
                        node.height - 1, it->key_suffix());
  }

  static void VisitLeafNode(ReadOperation::Ptr op, const BtreeNode& node,
                            Promise<kvstore::ReadResult> promise,
                            std::string_view unmatched_key_suffix) {
    auto& entries = std::get<BtreeNode::LeafNodeEntries>(node.entries);
    auto it = std::lower_bound(
        entries.begin(), entries.end(), unmatched_key_suffix,
        [](const LeafNodeEntry& entry, std::string_view unmatched_key_suffix) {
          return entry.key < unmatched_key_suffix;
        });
    if (it == entries.end() || it->key != unmatched_key_suffix) {
      // Key not present.
      op->KeyNotPresent(promise);
      return;
    }

    auto generation =
        internal_ocdbt::ComputeStorageGeneration(it->value_reference);

    // Check if_equal and if_not_equal conditions.
    if (!StorageGeneration::EqualOrUnspecified(generation, op->if_equal) ||
        !StorageGeneration::NotEqualOrUnspecified(generation,
                                                  op->if_not_equal)) {
      promise.SetResult(std::in_place, TimestampedStorageGeneration{
                                           std::move(generation), op->time});
      return;
    }

    if (auto* direct_value = std::get_if<absl::Cord>(&it->value_reference)) {
      // Value stored directly in btree node.
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto byte_range, op->byte_range.Validate(direct_value->size()),
          static_cast<void>(promise.SetResult(_)));
      promise.SetResult(
          std::in_place, kvstore::ReadResult::kValue,
          internal::GetSubCord(*direct_value, byte_range),
          TimestampedStorageGeneration{std::move(generation), op->time});
      return;
    }

    // Value stored indirectly.
    auto& indirect_ref = std::get<IndirectDataReference>(it->value_reference);
    kvstore::ReadOptions read_options;
    read_options.byte_range = op->byte_range;
    op->generation = std::move(generation);
    auto read_future =
        op->io_handle->ReadIndirectData(indirect_ref, std::move(read_options));
    LinkValue(
        [op = std::move(op)](Promise<kvstore::ReadResult> promise,
                             ReadyFuture<kvstore::ReadResult> read_future) {
          kvstore::ReadResult read_result;
          read_result.state = kvstore::ReadResult::kValue;
          read_result.value = std::move(read_future.result()->value);
          read_result.stamp.time = op->time;
          read_result.stamp.generation = std::move(op->generation);
          promise.SetResult(std::move(read_result));
        },
        std::move(promise), std::move(read_future));
  }
};

}  // namespace

Future<kvstore::ReadResult> NonDistributedRead(ReadonlyIoHandle::Ptr io_handle,
                                               kvstore::Key key,
                                               kvstore::ReadOptions options) {
  return ReadOperation::Start(std::move(io_handle), std::move(key),
                              std::move(options));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
