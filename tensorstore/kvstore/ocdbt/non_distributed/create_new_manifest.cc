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

// Adds a new B+tree generation to the manifest.
//
// In the simple (common) case, the new generation can simply be added to the
// version tree leaf node that is stored inline in the manifest.  In the general
// case, one or more version tree nodes will have to be written as well.
//
// The structure of the version tree is determined entirely from the latest
// generation number and the `version_tree_arity_log2`.  The version tree may
// exclude some generations, but skipped generations do not impact the structure
// of the version tree, except that entirely empty subtrees are excluded.

#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

struct CreateNewManifestOperation
    : public internal::AtomicReferenceCount<CreateNewManifestOperation> {
  using Ptr = internal::IntrusivePtr<CreateNewManifestOperation>;
  IoHandle::Ptr io_handle;
  FlushPromise flush_promise;
  std::shared_ptr<Manifest> new_manifest;
};

using ManifestWithFlushFuture =
    std::pair<std::shared_ptr<Manifest>, Future<const void>>;

struct ExistingVersionTreeNodeReady {
  CreateNewManifestOperation::Ptr op_;
  size_t i_;
  const VersionNodeReference* existing_node_ref_;
  std::shared_ptr<Manifest> new_manifest_;
  VersionNodeReference new_child_ref_;

  void operator()(
      Promise<void> promise,
      ReadyFuture<const std::shared_ptr<const VersionTreeNode>> future) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto existing_node, future.result(),
        static_cast<void>(SetDeferredResult(promise, _)));
    auto& new_node_ref = new_manifest_->version_tree_nodes[i_];

    TENSORSTORE_RETURN_IF_ERROR(
        ValidateVersionTreeNodeReference(*existing_node, new_manifest_->config,
                                         existing_node_ref_->generation_number,
                                         existing_node_ref_->height),
        static_cast<void>(SetDeferredResult(promise, _)));

    VersionTreeNode new_node;
    new_node.height = existing_node_ref_->height;
    new_node.version_tree_arity_log2 =
        new_manifest_->config.version_tree_arity_log2;
    auto& existing_children =
        std::get<VersionTreeNode::InteriorNodeEntries>(existing_node->entries);
    auto& new_children =
        new_node.entries.emplace<VersionTreeNode::InteriorNodeEntries>();
    new_children.reserve(existing_children.size() + 1);
    new_children.insert(new_children.begin(), existing_children.begin(),
                        existing_children.end());
    new_children.push_back(new_child_ref_);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto encoded, EncodeVersionTreeNode(new_manifest_->config, new_node),
        static_cast<void>(SetDeferredResult(promise, _)));
    new_node_ref.commit_time = new_children.front().commit_time;
    op_->flush_promise.Link(
        op_->io_handle->WriteData(std::move(encoded), new_node_ref.location));
  }
};

}  // namespace

Future<std::pair<std::shared_ptr<Manifest>, Future<const void>>>
CreateNewManifest(IoHandle::Ptr io_handle,
                  std::shared_ptr<const Manifest> existing_manifest,
                  BtreeGenerationReference new_generation) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "CreateNewManifest";
  auto commit_time = absl::Now();
  if (existing_manifest) {
    // Ensure that `commit_time` is monotonically increasing.
    commit_time = std::max(
        commit_time, static_cast<absl::Time>(
                         existing_manifest->latest_version().commit_time) +
                         absl::Nanoseconds(1));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      new_generation.commit_time, CommitTime::FromAbslTime(commit_time),
      internal::ConvertInvalidArgumentToFailedPrecondition(_));
  GenerationNumber existing_generation =
      existing_manifest ? existing_manifest->latest_generation() : 0;

  if (existing_generation >= std::numeric_limits<GenerationNumber>::max() - 1) {
    // Should not occur.
    return absl::FailedPreconditionError(
        absl::StrFormat("Existing generation number is already at maximum: %d",
                        existing_generation));
  }

  GenerationNumber new_generation_number = new_generation.generation_number =
      existing_generation + 1;

  auto new_manifest = std::make_shared<Manifest>();
  if (!existing_manifest) {
    TENSORSTORE_ASSIGN_OR_RETURN(new_manifest->config,
                                 io_handle->config_state->CreateNewConfig());
    new_manifest->versions.push_back(new_generation);
    return ManifestWithFlushFuture{new_manifest, {}};
  }

  new_manifest->config = existing_manifest->config;

  // Check if the generation range of the existing inline leaf node can fit the
  // new generation.
  if (GetVersionTreeLeafNodeRangeContainingGeneration(
          existing_manifest->config.version_tree_arity_log2,
          new_generation_number - 1)
          .second >= new_generation_number) {
    // Can just append new version to manifest without creating any new
    // version tree nodes.
    new_manifest->versions.reserve(existing_manifest->versions.size() + 1);
    new_manifest->versions.insert(new_manifest->versions.end(),
                                  existing_manifest->versions.begin(),
                                  existing_manifest->versions.end());
    new_manifest->versions.push_back(new_generation);
    new_manifest->version_tree_nodes = existing_manifest->version_tree_nodes;
    return ManifestWithFlushFuture{new_manifest, {}};
  }

  // The new generation does not fit in the existing inline leaf node.  The
  // generations referenced from the existing inline leaf node must be moved to
  // a new regular leaf node.  The key range of the inline leaf node advances by
  // `1 << version_tree_arity_log2`, with the new generation the only generation
  // referenced by it.

  new_manifest->versions.push_back(new_generation);

  // Need to write new version tree nodes.
  auto op = internal::MakeIntrusivePtr<CreateNewManifestOperation>();
  op->io_handle = std::move(io_handle);
  op->new_manifest = new_manifest;

  // Create `promise`/`future` pair to track of each version tree node write.
  // This `promise` won't resolve until all pending operations have completed.
  auto [promise, future] = PromiseFuturePair<void>::Make(absl::OkStatus());
  auto op_value_future = MapFuture(
      InlineExecutor{},
      [op = op](const Result<void>& result) -> Result<ManifestWithFlushFuture> {
        TENSORSTORE_RETURN_IF_ERROR(result);
        return ManifestWithFlushFuture{std::move(op->new_manifest),
                                       std::move(op->flush_promise).future()};
      },
      std::move(future));

  // Create new height 0 (leaf) version tree node containing the content of the
  // inline leaf node in the existing manifest.  This must always be written,
  // along with a new height 1 node that references it.  If the key range of the
  // existing height 1 node referenced from the manifest cannot fit this new
  // leaf node, then a reference to the existing height 1 must be added to a
  // height 2 node, etc.
  VersionNodeReference new_height0_node_ref;
  {
    VersionTreeNode new_node;
    new_node.height = 0;
    new_node.version_tree_arity_log2 =
        new_manifest->config.version_tree_arity_log2;
    auto& children = new_node.entries.emplace<VersionTreeNode::LeafNodeEntries>(
        existing_manifest->versions);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto encoded, EncodeVersionTreeNode(new_manifest->config, new_node));
    new_height0_node_ref.generation_number = children.back().generation_number;
    new_height0_node_ref.num_generations = children.size();
    new_height0_node_ref.commit_time = children.front().commit_time;
    new_height0_node_ref.height = 0;
    op->flush_promise.Link(op->io_handle->WriteData(
        std::move(encoded), new_height0_node_ref.location));
  }

  // Determine which existing version tree nodes contribute to the new version
  // tree nodes.
  //
  // Specifies list of `(existing_node, new_child)`.  Initially in order of
  // increasing height, but will be reversed.
  std::vector<
      std::pair<const VersionNodeReference*, const VersionNodeReference*>>
      new_version_tree_nodes;
  {
    size_t existing_version_tree_node_i =
        existing_manifest->version_tree_nodes.size();
    const VersionNodeReference* prev_child_ref = &new_height0_node_ref;
    // Determine which height/generation number range combinations may be
    // referenced from the new manifest.
    ForEachManifestVersionTreeNodeRef(
        new_generation_number,
        existing_manifest->config.version_tree_arity_log2,
        [&](GenerationNumber min_generation_number,
            GenerationNumber max_generation_number, VersionTreeHeight height) {
          // At the start of each invocation of this callback, `prev_child_ref`
          // points to the version tree node of height `height - 1` from the
          // existing manifest.

          // Determine the `VersionNodeReference` for `height` in the existing
          // manifest.
          const VersionNodeReference* old_node_ref = nullptr;
          if (existing_version_tree_node_i > 0) {
            if (auto* node =
                    &existing_manifest
                         ->version_tree_nodes[existing_version_tree_node_i - 1];
                node->height == height) {
              --existing_version_tree_node_i;
              old_node_ref = node;
            }
          }

          // Determine contents of new node:

          // 1. If there is an existing node of the same height that falls
          //    within `[min_generation_number, max_generation_number]`, then
          //    all of its children will be in the new node (the new node just
          //    extends the existing node).
          bool has_existing_children =
              (old_node_ref &&
               old_node_ref->generation_number >= min_generation_number &&
               old_node_ref->generation_number <= max_generation_number);

          // 2. If there is an existing node of `height-1` that falls within
          //    `[min_generation_number, max_generation_number]`, then it will
          //    be added as a child of the new node.
          bool has_new_child =
              (prev_child_ref &&
               prev_child_ref->generation_number >= min_generation_number &&
               prev_child_ref->generation_number <= max_generation_number);

          if (has_existing_children || has_new_child) {
            new_version_tree_nodes.emplace_back(
                has_existing_children ? old_node_ref : nullptr,
                has_new_child ? prev_child_ref : nullptr);
          }
          prev_child_ref = old_node_ref;
        });
    std::reverse(new_version_tree_nodes.begin(), new_version_tree_nodes.end());
  }
  new_manifest->version_tree_nodes.resize(new_version_tree_nodes.size());

  for (size_t i = 0; i < new_version_tree_nodes.size(); ++i) {
    auto [existing_node_ref, new_child_ref] = new_version_tree_nodes[i];
    assert(existing_node_ref || new_child_ref);
    auto& new_node_ref = new_manifest->version_tree_nodes[i];
    if (!existing_node_ref) {
      // No existing children to read, write new node immediately with a single
      // child.
      VersionTreeNode new_node;
      new_node.height = new_child_ref->height + 1;
      new_node.version_tree_arity_log2 =
          new_manifest->config.version_tree_arity_log2;
      auto& children =
          new_node.entries.emplace<VersionTreeNode::InteriorNodeEntries>();
      children.push_back(*new_child_ref);
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto encoded, EncodeVersionTreeNode(new_manifest->config, new_node),
          (static_cast<void>(SetDeferredResult(promise, _)), op_value_future));
      new_node_ref.commit_time = new_child_ref->commit_time;
      new_node_ref.generation_number = new_child_ref->generation_number;
      new_node_ref.num_generations = new_child_ref->num_generations;
      new_node_ref.height = new_node.height;
      op->flush_promise.Link(
          op->io_handle->WriteData(std::move(encoded), new_node_ref.location));
      continue;
    }

    if (!new_child_ref) {
      // Just store reference to existing node.
      new_node_ref = *existing_node_ref;
      continue;
    }

    // Must read existing version tree node and append additional child to it.
    new_node_ref.generation_number = new_child_ref->generation_number;
    new_node_ref.num_generations =
        existing_node_ref->num_generations + new_child_ref->num_generations;
    new_node_ref.commit_time = existing_node_ref->commit_time;
    new_node_ref.height = existing_node_ref->height;
    auto read_future =
        op->io_handle->GetVersionTreeNode(existing_node_ref->location);
    Link(WithExecutor(
             op->io_handle->executor,
             ExistingVersionTreeNodeReady{op, i, existing_node_ref,
                                          new_manifest, *new_child_ref}),
         promise, std::move(read_future));
  }
  return op_value_future;
}

Future<absl::Time> EnsureExistingManifest(IoHandle::Ptr io_handle) {
  auto read_future = io_handle->GetManifest(absl::InfinitePast());
  auto [promise, future] = PromiseFuturePair<absl::Time>::Make();
  LinkValue(
      [io_handle = std::move(io_handle)](
          Promise<absl::Time> promise,
          ReadyFuture<const ManifestWithTime> future) mutable {
        auto& manifest_with_time = future.value();
        if (manifest_with_time.manifest) {
          promise.SetResult(manifest_with_time.time);
          return;
        }

        auto new_manifest = std::make_shared<Manifest>();
        TENSORSTORE_ASSIGN_OR_RETURN(new_manifest->config,
                                     io_handle->config_state->CreateNewConfig(),
                                     static_cast<void>(promise.SetResult(_)));
        BtreeGenerationReference ref;
        ref.root_height = 0;
        ref.root.statistics = {};
        ref.root.location = IndirectDataReference::Missing();
        TENSORSTORE_ASSIGN_OR_RETURN(
            ref.commit_time, CommitTime::FromAbslTime(absl::Now()),
            static_cast<void>(promise.SetResult(
                internal::ConvertInvalidArgumentToFailedPrecondition(_))));
        ref.generation_number = 1;
        new_manifest->versions.push_back(ref);

        auto update_future = io_handle->TryUpdateManifest(
            /*old_manifest=*/{}, /*new_manifest=*/std::move(new_manifest),
            /*time=*/absl::Now());
        LinkValue(
            [io_handle = std::move(io_handle)](
                Promise<absl::Time> promise,
                ReadyFuture<TryUpdateManifestResult> future) mutable {
              auto& result = future.value();
              promise.SetResult(result.time);
            },
            std::move(promise), std::move(update_future));
      },
      std::move(promise), std::move(read_future));
  return std::move(future);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
