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

#include "tensorstore/kvstore/ocdbt/non_distributed/list_versions.h"

#include <variant>

#include "absl/log/absl_log.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {

// Asynchronous operation state used to implement
// `internal_ocdbt::ListVersions`.
//
// The list operation is implemented as follows:
//
// 1. Resolve the root b+tree node by reading the manifest.
//
// 2. Recursively descend the tree in parallel, reading all nodes that intersect
//    the key range specified in `options`.
//
// 3. Emit matching leaf-node keys to the receiver.
//
// TODO(jbms): Currently memory usage is not bounded.  That needs to be
// addressed, e.g. by limiting the number of in-flight nodes.
struct ListVersionsOperation
    : public internal::AtomicReferenceCount<ListVersionsOperation> {
  using Ptr = internal::IntrusivePtr<ListVersionsOperation>;
  ReadonlyIoHandle::Ptr io_handle;
  ListVersionsOptions options;
  AnyFlowReceiver<absl::Status, std::vector<BtreeGenerationReference>> receiver;

  // Initiates the asynchronous list operation.
  //
  // Args:
  //   io_handle: I/O handle to use.
  //   options: List options.
  //   receiver: Receiver of the results.
  static void Start(
      ReadonlyIoHandle::Ptr&& io_handle, const ListVersionsOptions& options,
      AnyFlowReceiver<absl::Status, std::vector<BtreeGenerationReference>>&&
          receiver) {
    auto [cancel_promise, cancel_future] =
        PromiseFuturePair<void>::Make(absl::OkStatus());
    execution::set_starting(receiver, [cancel_promise = cancel_promise] {
      SetDeferredResult(cancel_promise, absl::CancelledError(""));
    });
    auto op = internal::MakeIntrusivePtr<ListVersionsOperation>();
    op->io_handle = std::move(io_handle);
    op->receiver = std::move(receiver);
    op->options = options;
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
        op_ptr->io_handle->GetManifest(op->options.staleness_bound);
    Link(WithExecutor(
             op_ptr->io_handle->executor,
             ListVersionsOperation::ManifestReadyCallback{std::move(op)}),
         std::move(list_promise), std::move(manifest_future));
  }

  // Called when the manifest lookup has completed.
  struct ManifestReadyCallback {
    Ptr op;
    void operator()(Promise<void> promise,
                    ReadyFuture<const ManifestWithTime> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto manifest_with_time, read_future.result(),
          static_cast<void>(SetDeferredResult(promise, _)));
      const auto* manifest = manifest_with_time.manifest.get();
      if (!manifest) {
        // Manifest not present.
        return;
      }

      VisitEntries(*op, promise, manifest->version_tree_nodes);
      VisitEntries(*op, promise, manifest->versions);
    }
  };

  // Emit all matches within a subtree.
  //
  // Args:
  //   op: List operation state.
  //   promise: Promise to be resolved once the operation completes.
  //   node_ref: Node reference.
  static void VisitSubtree(Ptr op, Promise<void> promise,
                           const VersionNodeReference& node_ref) {
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "ListVersions: "
        << "node_ref=" << node_ref;
    auto read_future = op->io_handle->GetVersionTreeNode(node_ref.location);
    auto executor = op->io_handle->executor;
    Link(WithExecutor(
             std::move(executor),
             NodeReadyCallback{std::move(op), node_ref.generation_number,
                               node_ref.height}),
         std::move(promise), std::move(read_future));
  }

  // Called when a version tree node lookup completes.
  struct NodeReadyCallback {
    Ptr op;
    GenerationNumber generation_number;
    VersionTreeHeight height;

    void operator()(
        Promise<void> promise,
        ReadyFuture<const std::shared_ptr<const VersionTreeNode>> read_future) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto node, read_future.result(),
          static_cast<void>(SetDeferredResult(promise, _)));
      if (!promise.result_needed()) return;
      auto* config = op->io_handle->config_state->GetExistingConfig();
      assert(config);
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateVersionTreeNodeReference(*node, *config, generation_number,
                                           height),
          static_cast<void>(SetDeferredResult(promise, _)));

      std::visit(
          [&](const auto& entries) { VisitEntries(*op, promise, entries); },
          node->entries);
    }
  };

  // Recursively visits matching child subtrees.
  static void VisitEntries(ListVersionsOperation& op,
                           const Promise<void>& promise,
                           span<const VersionNodeReference> entries) {
    auto matching_entries = GetMatches(op, entries);
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "ListVersions: Visiting " << matching_entries.size() << "/"
        << entries.size() << " children of interior node";
    for (const auto& entry : matching_entries) {
      VisitSubtree(Ptr(&op), promise, entry);
    }
  }

  // Emits matching versions.
  static void VisitEntries(ListVersionsOperation& op,
                           const Promise<void>& promise,
                           span<const BtreeGenerationReference> entries) {
    auto matching_entries = GetMatches(op, entries);
    ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
        << "ListVersions: Emitting " << matching_entries.size() << "/"
        << entries.size() << " versions";
    if (!matching_entries.empty()) {
      execution::set_value(
          op.receiver, std::vector<BtreeGenerationReference>(
                           matching_entries.begin(), matching_entries.end()));
    }
  }

  template <typename Entry>
  static span<const Entry> GetMatches(ListVersionsOperation& op,
                                      span<const Entry> entries) {
    auto* config = op.io_handle->config_state->GetExistingConfig();
    assert(config);

    const auto& options = op.options;

    if (options.min_generation_number != 0) {
      entries = {internal_ocdbt::FindVersionLowerBound(
                     config->version_tree_arity_log2, entries,
                     options.min_generation_number),
                 entries.end()};
    }
    if (options.max_generation_number !=
        std::numeric_limits<GenerationNumber>::max()) {
      entries = {entries.begin(), internal_ocdbt::FindVersionUpperBound(
                                      entries, options.min_generation_number)};
    }
    if (options.min_commit_time != CommitTime::min()) {
      entries = {internal_ocdbt::FindVersionLowerBound(entries,
                                                       options.min_commit_time),
                 entries.end()};
    }
    if (options.max_commit_time != CommitTime::max()) {
      entries = {entries.begin(), internal_ocdbt::FindVersionUpperBound(
                                      entries, options.max_commit_time)};
    }
    return entries;
  }
};
}  // namespace

void ListVersions(
    ReadonlyIoHandle::Ptr io_handle, const ListVersionsOptions& options,
    AnyFlowReceiver<absl::Status, std::vector<BtreeGenerationReference>>
        receiver) {
  ListVersionsOperation::Start(std::move(io_handle), options,
                               std::move(receiver));
}

namespace {
struct ListVersionsFutureReceiver {
  Promise<std::vector<BtreeGenerationReference>> promise;
  std::vector<BtreeGenerationReference> refs;
  FutureCallbackRegistration cancel_registration;

  void set_value(std::vector<BtreeGenerationReference> value) {
    if (refs.empty()) {
      refs = std::move(value);
    } else {
      refs.insert(refs.end(), value.begin(), value.end());
    }
  }

  void set_error(absl::Status status) { promise.SetResult(std::move(status)); }

  void set_done() {
    std::sort(refs.begin(), refs.end(),
              [](const BtreeGenerationReference& a,
                 const BtreeGenerationReference& b) {
                return a.generation_number < b.generation_number;
              });
    promise.SetResult(std::move(refs));
  }

  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration = promise.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_stopping() { cancel_registration.Unregister(); }
};
}  // namespace

Future<std::vector<BtreeGenerationReference>> ListVersionsFuture(
    ReadonlyIoHandle::Ptr io_handle, const ListVersionsOptions& options) {
  auto [promise, future] =
      PromiseFuturePair<std::vector<BtreeGenerationReference>>::Make();
  ListVersions(
      std::move(io_handle), options,
      SyncFlowReceiver<ListVersionsFutureReceiver>{{std::move(promise)}});
  return std::move(future);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
