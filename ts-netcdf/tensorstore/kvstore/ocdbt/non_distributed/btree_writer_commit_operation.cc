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

#include "tensorstore/kvstore/ocdbt/non_distributed/btree_writer_commit_operation.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/write_nodes.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

void BtreeWriterCommitOperationBase::ReadManifest() {
  Future<const ManifestWithTime> read_future;

  if (io_handle_->config_state->GetAssumedOrExistingConfig()) {
    read_future = io_handle_->GetManifest(staleness_bound_);
  } else {
    auto ensure_future = internal_ocdbt::EnsureExistingManifest(io_handle_);
    read_future = PromiseFuturePair<ManifestWithTime>::LinkValue(
                      [this](Promise<ManifestWithTime> promise,
                             ReadyFuture<const absl::Time> time) {
                        LinkResult(std::move(promise),
                                   io_handle_->GetManifest(std::max(
                                       staleness_bound_, time.value())));
                      },
                      std::move(ensure_future))
                      .future;
  }
  read_future.Force();
  read_future.ExecuteWhenReady(
      [this](ReadyFuture<const ManifestWithTime> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          this->Fail(r.status());
          return;
        }
        existing_manifest_ = r->manifest;
        staleness_bound_ = r->time;
        auto& executor = io_handle_->executor;
        executor([this]() mutable {
          WriteStager stager(*this);
          StagePending(stager);
          auto [promise, future] =
              PromiseFuturePair<void>::Make(absl::OkStatus());
          TraverseBtreeStartingFromRoot(std::move(promise));
          future.Force();
          future.ExecuteWhenReady([this](ReadyFuture<void> future) mutable {
            auto& r = future.result();
            if (!r.ok()) {
              if (absl::IsCancelled(r.status())) {
                // Out of date, retry.
                this->Retry();
                return;
              }
              Fail(r.status());
              return;
            }
            WriteNewManifest();
          });
        });
      });
}

void BtreeWriterCommitOperationBase::WriteStager::Stage(
    LeafNodeValueReference& value_ref) {
  if (auto* value_ptr = std::get_if<absl::Cord>(&value_ref)) {
    if (value_ptr->size() > config.max_inline_value_bytes) {
      auto value = std::move(*value_ptr);
      auto value_future =
          op.io_handle_->WriteData(IndirectDataKind::kValue, std::move(value),
                                   value_ref.emplace<IndirectDataReference>());
      op.flush_promise_.Link(std::move(value_future));
    }
  }
}

void BtreeWriterCommitOperationBase::RootNodeTraversalState::ApplyMutations() {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "ApplyMutations: height=" << static_cast<int>(this->height_)
      << ", num_mutations=" << this->mutations_.size();
  if (this->mutations_.empty()) {
    if (!this->writer_->existing_manifest_) {
      // Since there is no existing manifest, write an initial manifest.
      BtreeGenerationReference ref;
      ref.root_height = 0;
      ref.root.statistics = {};
      ref.root.location = IndirectDataReference::Missing();
      this->writer_->CreateNewManifest(std::move(this->promise_), ref);
      return;
    }
    // Leave manifest unchanged.
    this->writer_->new_manifest_ = this->writer_->existing_manifest_;
    this->writer_->NewManifestReady(std::move(this->promise_));
    return;
  }

  while (true) {
    // Mutations must be of the form: delete "", add ...
    [[maybe_unused]] auto& deletion_entry = this->mutations_.front();
    assert(!deletion_entry.add);
    assert(deletion_entry.entry.key.empty());

    if (this->mutations_.size() == 1) {
      // Root node is empty.
      BtreeGenerationReference ref;
      ref.root_height = 0;
      ref.root.statistics = {};
      ref.root.location = IndirectDataReference::Missing();
      this->writer_->CreateNewManifest(std::move(this->promise_), ref);
      return;
    }

    if (this->mutations_.size() == 2) {
      // Exactly one root node, no need to increase height.
      auto& new_root_mutation = this->mutations_[1];
      assert(new_root_mutation.add);
      assert(new_root_mutation.entry.key.empty());
      assert(new_root_mutation.entry.subtree_common_prefix_length == 0);
      assert(this->height_ > 0);
      BtreeGenerationReference ref;
      ref.root_height = this->height_ - 1;
      ref.root = new_root_mutation.entry.node;
      this->writer_->CreateNewManifest(std::move(this->promise_), ref);
      return;
    }

    // Need to add a level to the tree.
    auto mutations = std::exchange(this->mutations_, {});
    UpdateParent(*this, /*existing_relative_child_key=*/{},
                 EncodeUpdatedInteriorNodes(this->writer_->existing_config(),
                                            this->height_,
                                            /*existing_prefix=*/{},
                                            /*existing_entries=*/{}, mutations,
                                            /*may_be_root=*/true));
    ++this->height_;
  }
}

void BtreeWriterCommitOperationBase::InteriorNodeTraversalState::
    ApplyMutations() {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "ApplyMutations: existing inclusive_min="
      << tensorstore::QuoteString(
             tensorstore::StrCat(parent_state_->existing_subtree_key_prefix_,
                                 existing_relative_child_key_))
      << ", height=" << static_cast<int>(this->height_)
      << ", num_mutations=" << this->mutations_.size();
  if (this->mutations_.empty()) {
    // There are no mutations to the key range of this node.  Therefore,
    // this node can remain referenced unmodified from its parent.
    return;
  }

  UpdateParent(
      *parent_state_, existing_relative_child_key_,
      EncodeUpdatedInteriorNodes(
          this->writer_->existing_config(), this->height_,
          this->existing_subtree_key_prefix_,
          std::get<BtreeNode::InteriorNodeEntries>(existing_node_->entries),
          this->mutations_,
          /*may_be_root=*/parent_state_->is_root_parent()));
}

void BtreeWriterCommitOperationBase::UpdateParent(
    NodeTraversalState& parent_state,
    std::string_view existing_relative_child_key,
    Result<std::vector<EncodedNode>>&& encoded_nodes_result) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto encoded_nodes, std::move(encoded_nodes_result),
      static_cast<void>(SetDeferredResult(parent_state.promise_, _)));

  auto new_entries = internal_ocdbt::WriteNodes(
      *parent_state.writer_->io_handle_, parent_state.writer_->flush_promise_,
      std::move(encoded_nodes));

  {
    absl::MutexLock lock(&parent_state.mutex_);

    // Remove `existing_relative_child_key` from the parent node.
    {
      auto& mutation = parent_state.mutations_.emplace_back();
      mutation.add = false;
      mutation.entry.key =
          tensorstore::StrCat(parent_state.existing_subtree_key_prefix_,
                              existing_relative_child_key);
    }

    // Add `new_entries` in its place.
    for (auto& new_entry : new_entries) {
      auto& mutation = parent_state.mutations_.emplace_back();
      mutation.add = true;
      mutation.entry = std::move(new_entry);
    }
  }
}

Result<std::vector<EncodedNode>>
BtreeWriterCommitOperationBase::EncodeUpdatedInteriorNodes(
    const Config& config, BtreeNodeHeight height,
    std::string_view existing_prefix,
    span<const InteriorNodeEntry> existing_entries,
    span<InteriorNodeMutation> mutations, bool may_be_root) {
  // Sort by key order, with deletions before additions, which allows the code
  // below to remove and add the same key without additional checks.
  std::sort(mutations.begin(), mutations.end(),
            [](const InteriorNodeMutation& a, const InteriorNodeMutation& b) {
              int c = a.entry.key.compare(b.entry.key);
              if (c != 0) return c < 0;
              return a.add < b.add;
            });

  BtreeInteriorNodeEncoder encoder(config, height, existing_prefix);
  auto existing_it = existing_entries.begin();
  auto mutation_it = mutations.begin();

  ComparePrefixedKeyToUnprefixedKey compare_existing_and_new_keys{
      existing_prefix};

  // Merge existing entries with mutations.
  while (existing_it != existing_entries.end() ||
         mutation_it != mutations.end()) {
    int c = existing_it == existing_entries.end() ? 1
            : mutation_it == mutations.end()
                ? -1
                : compare_existing_and_new_keys(existing_it->key,
                                                mutation_it->entry.key);
    if (c < 0) {
      // Existing key comes before mutation.
      encoder.AddEntry(/*existing=*/true, InteriorNodeEntry(*existing_it));
      ++existing_it;
      continue;
    }

    if (c == 0) {
      // Mutation replaces or deletes existing key.
      ++existing_it;
    }

    if (mutation_it->add) {
      internal_ocdbt::AddNewInteriorEntry(encoder, mutation_it->entry);
    }
    ++mutation_it;
  }

  return encoder.Finalize(may_be_root);
}

void BtreeWriterCommitOperationBase::CreateNewManifest(
    Promise<void> promise, const BtreeGenerationReference& new_generation) {
  auto future = internal_ocdbt::CreateNewManifest(
      io_handle_, existing_manifest_, new_generation);
  LinkValue(
      [this](
          Promise<void> promise,
          ReadyFuture<std::pair<std::shared_ptr<Manifest>, Future<const void>>>
              future) mutable {
        auto& create_result = future.value();
        flush_promise_.Link(std::move(create_result.second));
        new_manifest_ = std::move(create_result.first);
        auto executor = io_handle_->executor;
        executor([this, promise = std::move(promise)]() mutable {
          NewManifestReady(std::move(promise));
        });
      },
      std::move(promise), std::move(future));
}

void BtreeWriterCommitOperationBase::NewManifestReady(Promise<void> promise) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "NewManifestReady";
  auto flush_future = std::move(flush_promise_).future();
  if (flush_future.null()) {
    return;
  }
  flush_future.Force();
  LinkError(std::move(promise), std::move(flush_future));
}

void BtreeWriterCommitOperationBase::WriteNewManifest() {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "WriteNewManifest: existing_generation="
      << GetLatestGeneration(existing_manifest_.get())
      << ", new_generation=" << GetLatestGeneration(new_manifest_.get());
  auto update_future = io_handle_->TryUpdateManifest(
      existing_manifest_, new_manifest_, absl::Now());
  update_future.Force();
  update_future.ExecuteWhenReady(
      WithExecutor(io_handle_->executor,
                   [this](ReadyFuture<TryUpdateManifestResult> future) {
                     auto& r = future.result();
                     ABSL_LOG_IF(INFO, ocdbt_logging)
                         << "Manifest written: " << r.status()
                         << ", success=" << (r.ok() ? r->success : false);
                     if (!r.ok()) {
                       Fail(r.status());
                       return;
                     }
                     if (!r->success) {
                       ABSL_CHECK_GE(r->time, staleness_bound_);
                       staleness_bound_ = r->time;
                       Retry();
                       return;
                     }
                     CommitSuccessful(r->time);
                   }));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
