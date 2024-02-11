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

#include "tensorstore/kvstore/ocdbt/non_distributed/read_version.h"

#include <cassert>
#include <memory>
#include <utility>
#include <variant>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

// Asynchronous operation state used to implement `internal_ocdbt::ReadVersion`.
//
// The read version operation is implemented as follows:
//
// 1. Read the manifest.
//
// 2. If the specified version is newer than the current manifest, re-read the
//    manifest.
//
// 3. Descend the version tree along the path to the requested version.
//
// 4. Stop and return a missing version indication as soon as the version is
//    determined to be missing.
//
// 5. Return once the leaf node containing the version is reached.
struct ReadVersionOperation
    : public internal::AtomicReferenceCount<ReadVersionOperation> {
  using Ptr = internal::IntrusivePtr<ReadVersionOperation>;
  using PromiseType = Promise<BtreeGenerationReference>;
  ReadonlyIoHandle::Ptr io_handle;
  VersionSpec version_spec;
  absl::Time staleness_bound;

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
  static Future<BtreeGenerationReference> Start(ReadonlyIoHandle::Ptr io_handle,
                                                VersionSpec version_spec,
                                                absl::Time staleness_bound) {
    auto op = internal::MakeIntrusivePtr<ReadVersionOperation>();
    op->io_handle = std::move(io_handle);
    op->version_spec = version_spec;
    op->staleness_bound = staleness_bound;
    auto [promise, future] =
        PromiseFuturePair<BtreeGenerationReference>::Make();
    // First attempt to use any cached manifest.  If the result cannot be
    // determined definitively from the cached manifest, take `staleness_bound`
    // into account.
    RequestManifest(std::move(op), std::move(promise), absl::InfinitePast());
    return std::move(future);
  }

  static void RequestManifest(ReadVersionOperation::Ptr op, PromiseType promise,
                              absl::Time staleness_bound) {
    auto* op_ptr = op.get();
    LinkValue(
        WithExecutor(op_ptr->io_handle->executor,
                     [op = std::move(op)](
                         PromiseType promise,
                         ReadyFuture<const ManifestWithTime> future) mutable {
                       ManifestReady(std::move(op), std::move(promise),
                                     future.value());
                     }),
        std::move(promise), op_ptr->io_handle->GetManifest(staleness_bound));
  }

  // Called when the manifest lookup has completed.
  static void ManifestReady(ReadVersionOperation::Ptr op, PromiseType promise,
                            const ManifestWithTime& manifest_with_time) {
    if (!manifest_with_time.manifest ||
        CompareVersionSpecToVersion(
            op->version_spec, manifest_with_time.manifest->latest_version()) >
            0) {
      if (manifest_with_time.time < op->staleness_bound) {
        // Retry
        auto staleness_bound = op->staleness_bound;
        RequestManifest(std::move(op), std::move(promise), staleness_bound);
        return;
      }
      if (!manifest_with_time.manifest ||
          IsVersionSpecExact(op->version_spec)) {
        op->VersionNotPresent(promise);
        return;
      }
    }

    const auto& manifest = *manifest_with_time.manifest;

    if (CompareVersionSpecToVersion(op->version_spec,
                                    manifest.versions.front()) >= 0) {
      // Generation is inline in manifest if present.
      if (auto* ref = internal_ocdbt::FindVersion(manifest.versions,
                                                  op->version_spec)) {
        promise.SetResult(*ref);
        return;
      }
      op->VersionNotPresent(promise);
      return;
    }

    auto* ref = internal_ocdbt::FindVersion(
        manifest.config.version_tree_arity_log2, manifest.version_tree_nodes,
        op->version_spec);
    if (!ref) {
      op->VersionNotPresent(promise);
      return;
    }

    LookupNodeReference(std::move(op), std::move(promise), *ref);
  }

  // Completes the read request, indicating that the version is missing.
  void VersionNotPresent(const PromiseType& promise) {
    promise.SetResult(absl::NotFoundError(absl::StrFormat(
        "Version where %s not present", FormatVersionSpec(version_spec))));
  }

  // Recursively descends the version tree.
  //
  // Args:
  //   op: Operation state.
  //   promise: Promise on which to set the read result when ready.
  //   node_ref: Reference to the B+tree node to descend.
  //   node_height: Height of the node specified by `node_ref`.
  //   inclusive_min_key: Lower bound of the key range corresponding to
  //     `node_ref`.
  static void LookupNodeReference(ReadVersionOperation::Ptr op,
                                  PromiseType promise,
                                  const VersionNodeReference& node_ref) {
    ABSL_LOG_IF(INFO, ocdbt_logging)
        << "ReadVersion: " << FormatVersionSpec(op->version_spec)
        << ", node_ref=" << node_ref;

    auto read_future = op->io_handle->GetVersionTreeNode(node_ref.location);
    auto executor = op->io_handle->executor;
    LinkValue(WithExecutor(std::move(executor),
                           NodeReadyCallback{std::move(op), node_ref}),
              std::move(promise), std::move(read_future));
  }

  struct NodeReadyCallback {
    ReadVersionOperation::Ptr op;
    VersionNodeReference node_ref;
    void operator()(
        PromiseType promise,
        ReadyFuture<const std::shared_ptr<const VersionTreeNode>> read_future) {
      auto node = read_future.value();
      auto* config = op->io_handle->config_state->GetExistingConfig();
      assert(config);
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateVersionTreeNodeReference(
              *node, *config, node_ref.generation_number, node_ref.height),
          static_cast<void>(promise.SetResult(_)));
      if (node->height > 0) {
        VisitInteriorNode(std::move(op), *node, std::move(promise));
      } else {
        VisitLeafNode(std::move(op), *node, std::move(promise));
      }
    }
  };

  static void VisitInteriorNode(ReadVersionOperation::Ptr op,
                                const VersionTreeNode& node,
                                PromiseType promise) {
    auto& entries =
        std::get<VersionTreeNode::InteriorNodeEntries>(node.entries);
    auto* config = op->io_handle->config_state->GetExistingConfig();
    assert(config);
    auto* node_ref = internal_ocdbt::FindVersion(
        config->version_tree_arity_log2, entries, op->version_spec);
    if (!node_ref) {
      op->VersionNotPresent(std::move(promise));
      return;
    }
    LookupNodeReference(std::move(op), std::move(promise), *node_ref);
  }

  static void VisitLeafNode(ReadVersionOperation::Ptr op,
                            const VersionTreeNode& node, PromiseType promise) {
    auto& entries = std::get<VersionTreeNode::LeafNodeEntries>(node.entries);
    auto* ref = internal_ocdbt::FindVersion(entries, op->version_spec);
    if (!ref) {
      op->VersionNotPresent(std::move(promise));
      return;
    }
    promise.SetResult(*ref);
  }
};

}  // namespace

Future<BtreeGenerationReference> ReadVersion(ReadonlyIoHandle::Ptr io_handle,
                                             VersionSpec version_spec,
                                             absl::Time staleness_bound) {
  if (const GenerationNumber* generation_number =
          std::get_if<GenerationNumber>(&version_spec)) {
    if (*generation_number == 0) {
      return absl::InvalidArgumentError("Generation number must be positive");
    }
  }
  return ReadVersionOperation::Start(std::move(io_handle), version_spec,
                                     std::move(staleness_bound));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
