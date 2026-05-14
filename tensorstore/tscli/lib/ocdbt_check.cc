// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/tscli/lib/ocdbt_check.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/read_version.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/lib/ocdbt_check_reporter.h"
#include "tensorstore/tscli/lib/ocdbt_file_usage_tracker.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

namespace tensorstore {
namespace cli {
namespace {

using ::tensorstore::QuoteString;
using ::tensorstore::internal_ocdbt::BtreeNode;
using ::tensorstore::internal_ocdbt::BtreeNodeHeight;
using ::tensorstore::internal_ocdbt::BtreeNodeReference;
using ::tensorstore::internal_ocdbt::BtreeNodeStatistics;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::ConfigState;
using ::tensorstore::internal_ocdbt::FormatVersionSpec;
using ::tensorstore::internal_ocdbt::GenerationNumber;
using ::tensorstore::internal_ocdbt::GetBtreeNodeLocalStatistics;
using ::tensorstore::internal_ocdbt::IndirectDataReference;
using ::tensorstore::internal_ocdbt::IoHandle;
using ::tensorstore::internal_ocdbt::kMaxNodeArity;
using ::tensorstore::internal_ocdbt::Manifest;
using ::tensorstore::internal_ocdbt::ParseVersionSpecFromUrl;
using ::tensorstore::internal_ocdbt::ValidateBtreeNodeReference;
using ::tensorstore::internal_ocdbt::ValidateVersionTreeInteriorNodeEntries;
using ::tensorstore::internal_ocdbt::ValidateVersionTreeLeafNodeEntries;
using ::tensorstore::internal_ocdbt::ValidateVersionTreeNodeReference;
using ::tensorstore::internal_ocdbt::VersionNodeReference;
using ::tensorstore::internal_ocdbt::VersionSpec;
using ::tensorstore::internal_ocdbt::VersionTreeHeight;
using ::tensorstore::internal_ocdbt::VersionTreeNode;

struct VisitedBtreeNodeInfo {
  BtreeNodeStatistics statistics;
  BtreeNodeHeight height;
  std::string first_key;
};

struct VisitedVersionNodeInfo {
  GenerationNumber generation_number;
  VersionTreeHeight height;
  GenerationNumber num_generations;
  CommitTime commit_time;
};

struct BtreeTask {
  BtreeNodeReference ref;
  BtreeNodeHeight height;
  std::string inclusive_min_key;
};

class OcdbtCheckRunner {
 public:
  OcdbtCheckRunner(IoHandle::Ptr io_handle,
                   std::optional<VersionSpec> version_spec,
                   kvstore::KvStore base_kvstore,
                   const OcdbtCheckOptions& options,
                   OcdbtCheckReporter& reporter)
      : io_handle_(std::move(io_handle)),
        version_spec_(version_spec),
        base_kvstore_(base_kvstore),
        detailed_(options.detailed),
        reporter_(reporter),
        tracker_(),
        alignment_(options.alignment),
        concurrency_(options.concurrency) {}

  absl::Status Run() {
    reporter_.ReportInfo("Starting OCDBT integrity check...");
    reporter_.ResetProgress();

    PopulateOnDiskFiles();

    if (version_spec_) {
      CheckSpecificVersion();
    } else {
      CheckAllVersions();
    }

    absl::Status status = RunDfs();
    if (!status.ok()) {
      PrintSummary();
      return status;
    }

    RunFileAccessChecks();

    tracker_.CheckOrphanedFiles(reporter_);
    tracker_.CheckUnusedRanges(reporter_, alignment_);

    PrintSummary();
    return absl::OkStatus();
  }

  void PrintSummary() {
    reporter_.ReportInfo(
        "OCDBT integrity check completed.\nTotal errors found: %d\nTotal "
        "warnings found: %d",
        reporter_.error_count(), reporter_.warning_count());
  }

  void PopulateOnDiskFiles() {
    auto list_future = kvstore::ListFuture(base_kvstore_);
    auto list_result = list_future.result();
    if (!list_result.ok()) {
      reporter_.ReportWarning(
          "Warning: Failed to list base kvstore, orphaned files check "
          "will be skipped: %s",
          list_result.status().ToString());
    } else {
      absl::flat_hash_map<std::string, int64_t> files;
      if (detailed_) reporter_.ReportInfo("On-disk files found:");
      for (const auto& entry : *list_result) {
        if (detailed_) {
          reporter_.ReportInfo("  %s (%d bytes)", entry.key, entry.size);
        }
        files[entry.key] = entry.size;
      }
      tracker_.SetOnDiskFiles(std::move(files));
    }
  }

 private:
  void CheckSpecificVersion() {
    reporter_.ReportInfo("Checking specific version: %s",
                         FormatVersionSpec(*version_spec_));
    auto response_future =
        internal_ocdbt::ReadVersion(io_handle_, version_spec_);
    auto response_result = response_future.result();
    if (!response_result.ok()) {
      reporter_.ReportError(tensorstore::StatusBuilder(response_result.status())
                                .Format("Error resolving version"));
      return;
    }
    auto& response = *response_result;
    if (!response.manifest_with_time.manifest) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Error: Manifest not found"));
      return;
    }
    if (!response.generation) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Error: Version %s not found",
                      FormatVersionSpec(*version_spec_)));
      return;
    }

    const Manifest& manifest = *response.manifest_with_time.manifest;
    config_ = manifest.config;

    {
      absl::MutexLock lock(mutex_);
      EnqueueBtreeTaskLocked(BtreeTask{response.generation->root,
                                       response.generation->root_height, ""});
    }
  }

  void CheckAllVersions() {
    auto manifest_with_time_future = io_handle_->GetManifest(absl::Now());
    auto manifest_with_time_result = manifest_with_time_future.result();
    if (!manifest_with_time_result.ok()) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(manifest_with_time_result.status())
              .Format("Error reading manifest"));
      return;
    }

    auto manifest_with_time = *manifest_with_time_result;
    if (!manifest_with_time.manifest) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Error: Manifest not found"));
      return;
    }

    const Manifest& manifest = *manifest_with_time.manifest;
    reporter_.ReportInfo("Manifest loaded successfully.");
    reporter_.ReportInfo("Config: %v", manifest.config);
    config_ = manifest.config;

    {
      absl::MutexLock lock(mutex_);
      for (const auto& version_ref : manifest.versions) {
        EnqueueBtreeTaskLocked(
            BtreeTask{version_ref.root, version_ref.root_height, ""});
      }
      for (const auto& version_node_ref : manifest.version_tree_nodes) {
        EnqueueVersionTaskLocked(version_node_ref);
      }
    }
  }

  absl::Status RunDfs() {
    while (true) {
      std::variant<std::monostate, BtreeTask, VersionNodeReference> task;

      {
        absl::MutexLock lock(mutex_);

        // Wait until either a slot opens up or work becomes available.
        mutex_.Await(absl::Condition(
            +[](OcdbtCheckRunner* self)
                 ABSL_EXCLUSIVE_LOCKS_REQUIRED(self->mutex_) {
                   // Resume when: (below concurrency limit AND work is
                   // available), OR all outstanding reads are done (terminal
                   // condition).
                   return (self->active_reads_ < self->concurrency_ &&
                           (!self->btree_stack_.empty() ||
                            !self->version_stack_.empty())) ||
                          self->active_reads_ == 0;
                 },
            this));

        // Terminal condition: no tasks and no outstanding reads.
        if (btree_stack_.empty() && version_stack_.empty() &&
            active_reads_ == 0) {
          return status_;
        }

        // Get the next task to process from the stacks.
        if (!btree_stack_.empty()) {
          task = std::move(btree_stack_.back());
          btree_stack_.pop_back();
          active_reads_++;
        } else if (!version_stack_.empty()) {
          task = std::move(version_stack_.back());
          version_stack_.pop_back();
          active_reads_++;
        }
      }

      std::visit(absl::Overload([](std::monostate&&) {},
                                [this](BtreeTask&& btree) {
                                  ProcessBtreeTaskAsync(std::move(btree));
                                },
                                [this](VersionNodeReference&& vnode) {
                                  ProcessVersionTaskAsync(std::move(vnode));
                                }),
                 std::move(task));
    }
  }

  void RunFileAccessChecks() {
    reporter_.ReportInfo("Starting File access checks...");
    reporter_.ResetProgress(tracker_.referenced_files_size());

    tracker_.ForEachFileUsedRange([this](std::string_view file_path,
                                         const auto&) {
      {
        absl::MutexLock lock(mutex_);
        while (active_reads_ >= concurrency_) {
          cond_.Wait(&mutex_);
        }
        active_reads_++;
      }

      kvstore::ReadOptions read_options;
      read_options.byte_range = OptionalByteRangeRequest::Stat();

      reporter_.PrintProgress("Stating file: %s", file_path);

      kvstore::Read(base_kvstore_, file_path, read_options)
          .ExecuteWhenReady([this, file_path = std::string(file_path)](auto f) {
            auto result = f.result();
            absl::Status file_status;
            if (!result.ok()) {
              file_status =
                  tensorstore::StatusBuilder(file_status)
                      .Format("Error stat'ing indirect value %v", file_path);
              reporter_.ReportError(file_status);
            } else if (result->state == kvstore::ReadResult::State::kMissing) {
              file_status =
                  tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                      .Format("File %v is missing", file_path);
              reporter_.ReportError(file_status);
            }
            absl::MutexLock lock(mutex_);
            active_reads_--;
            cond_.SignalAll();
            status_.Update(file_status);
          });
    });

    // Wait for all active reads to finish.
    {
      absl::MutexLock lock(mutex_);
      while (active_reads_ > 0) {
        cond_.Wait(&mutex_);
      }
    }
  }

  void EnqueueBtreeTaskLocked(BtreeTask task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (task.ref.location.IsMissing()) return;

    if (auto it = visited_btree_nodes_.find(task.ref.location);
        it != visited_btree_nodes_.end()) {
      ValidateAlreadyVisitedBtreeNodeLocked(task, it->second);
      return;
    }
    btree_stack_.push_back(task);
  }

  void ProcessBtreeTaskAsync(BtreeTask task) {
    {
      absl::MutexLock lock(mutex_);

      // If this node is already visited, validate against the cached result
      // and return immediately without issuing another read.
      if (auto it = visited_btree_nodes_.find(task.ref.location);
          it != visited_btree_nodes_.end()) {
        ValidateAlreadyVisitedBtreeNodeLocked(task, it->second);
        active_reads_--;
        cond_.SignalAll();
        return;
      }

      // If this node is currently being fetched by another task, queue this
      // task as a pending waiter.  It will be validated once the in-flight read
      // completes, without blocking an executor thread.
      auto [it, inserted] = pending_btree_tasks_.try_emplace(task.ref.location);
      if (!inserted) {
        it->second.push_back(std::move(task));
        active_reads_--;
        cond_.SignalAll();
        return;
      }
    }

    reporter_.PrintProgress("Registering B-tree node range: %s %v",
                            task.ref.location.file_id.FullPath(),
                            task.ref.location);
    tracker_.RegisterUsedRange(task.ref.location.file_id.FullPath(),
                               task.ref.location.offset,
                               task.ref.location.length);

    io_handle_->GetBtreeNode(task.ref.location)
        .ExecuteWhenReady([this, task](auto f) {
          absl::Status handle_status = absl::OkStatus();
          auto result = f.result();
          if (!result.ok()) {
            reporter_.ReportError(
                tensorstore::StatusBuilder(result.status())
                    .Format("Error reading B-tree node %v", task.ref.location));
          } else {
            handle_status = HandleBtreeNode(task, **result);
          }

          absl::MutexLock lock(mutex_);
          active_reads_--;

          // Process any tasks that were waiting for this node to be read.
          auto it = pending_btree_tasks_.find(task.ref.location);
          CHECK(it != pending_btree_tasks_.end());
          if (auto vit = visited_btree_nodes_.find(task.ref.location);
              vit != visited_btree_nodes_.end()) {
            for (const auto& pending : it->second) {
              ValidateAlreadyVisitedBtreeNodeLocked(pending, vit->second);
            }
          }
          pending_btree_tasks_.erase(it);

          if (!handle_status.ok()) status_.Update(handle_status);
          cond_.SignalAll();
        });
  }

  absl::Status HandleBtreeNode(BtreeTask task, const BtreeNode& node) {
    auto validation_status =
        ValidateBtreeNodeReference(node, task.height, task.inclusive_min_key);
    if (!validation_status.ok()) {
      reporter_.ReportError(tensorstore::StatusBuilder(validation_status)
                                .Format("Validation error for B-tree node %v",
                                        task.ref.location));
    }

    // Inline B-tree invariant checks
    if (node.height == 0) {
      auto& entries = std::get<BtreeNode::LeafNodeEntries>(node.entries);
      if (entries.empty()) {
        reporter_.ReportError(
            tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                .Format("Leaf B-tree node %v is empty", task.ref.location));
      } else if (entries.size() > kMaxNodeArity) {
        reporter_.ReportError(
            tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                .Format("Leaf B-tree node %v has too many entries: %d "
                        "(max %d)",
                        task.ref.location, entries.size(), kMaxNodeArity));
      }
      for (size_t i = 0; i < entries.size(); ++i) {
        // Entry keys are relative to node.key_prefix. Within a single node all
        // entries share the same prefix, so comparing the suffixes is
        // equivalent to comparing the full keys — but the error message reports
        // the full reconstructed key so it is meaningful to the user.
        if (i != 0 && entries[i].key <= entries[i - 1].key) {
          reporter_.ReportError(
              tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                  .Format("Leaf B-tree node %v keys are not strictly "
                          "increasing: key[%d]=%v <= key[%d]=%v",
                          task.ref.location, i,
                          QuoteString(
                              absl::StrCat(node.key_prefix, entries[i].key)),
                          i - 1,
                          QuoteString(absl::StrCat(node.key_prefix,
                                                   entries[i - 1].key))));
        }
      }
    } else {
      auto& entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);
      if (entries.empty()) {
        reporter_.ReportError(
            tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                .Format("Interior B-tree node %v is empty", task.ref.location));
      } else if (entries.size() > kMaxNodeArity) {
        reporter_.ReportError(
            tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                .Format("Interior B-tree node %v has too many entries: %d "
                        "(max %d)",
                        task.ref.location, entries.size(), kMaxNodeArity));
      }
      for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].subtree_common_prefix_length > entries[i].key.size()) {
          reporter_.ReportError(
              tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                  .Format("Interior B-tree node %v entry %d has "
                          "subtree_common_prefix_length %d > key size %d",
                          task.ref.location, i,
                          entries[i].subtree_common_prefix_length,
                          entries[i].key.size()));
        }
        if (i != 0 && entries[i].key <= entries[i - 1].key) {
          reporter_.ReportError(
              tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
                  .Format("Interior B-tree node %v keys are not strictly "
                          "increasing: key[%d]=%v <= key[%d]=%v",
                          task.ref.location, i,
                          QuoteString(
                              absl::StrCat(node.key_prefix, entries[i].key)),
                          i - 1,
                          QuoteString(absl::StrCat(node.key_prefix,
                                                   entries[i - 1].key))));
        }
      }
    }

    auto calculated_stats =
        GetBtreeNodeLocalStatistics(node, task.ref.location.length);
    if (calculated_stats != task.ref.statistics) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Statistics mismatch for B-tree node %v\n  Expected:   "
                      "%v\n  Calculated: %v",
                      task.ref.location, task.ref.statistics,
                      calculated_stats));
    }

    std::string first_key;
    std::visit(
        [&](const auto& entries) {
          if (!entries.empty()) {
            first_key = absl::StrCat(node.key_prefix, entries.front().key);
          }
        },
        node.entries);

    {
      absl::MutexLock lock(mutex_);
      if (node.height > 0) {
        auto& entries = std::get<BtreeNode::InteriorNodeEntries>(node.entries);
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
          const auto& child_entry = *it;
          EnqueueBtreeTaskLocked(BtreeTask{
              child_entry.node, static_cast<BtreeNodeHeight>(task.height - 1),
              std::string(child_entry.key_suffix())});
        }
      } else {
        auto& entries = std::get<BtreeNode::LeafNodeEntries>(node.entries);
        for (const auto& leaf_entry : entries) {
          if (std::holds_alternative<IndirectDataReference>(
                  leaf_entry.value_reference)) {
            auto& value_ref =
                std::get<IndirectDataReference>(leaf_entry.value_reference);
            reporter_.PrintProgress("Registering value range: %s [%u, %u)",
                                    value_ref.file_id.FullPath(),
                                    value_ref.offset, value_ref.length);
            tracker_.RegisterUsedRange(value_ref.file_id.FullPath(),
                                       value_ref.offset, value_ref.length);
          }
        }
      }

      visited_btree_nodes_[task.ref.location] = VisitedBtreeNodeInfo{
          calculated_stats, node.height, std::move(first_key)};
    }
    return absl::OkStatus();
  }

  void EnqueueVersionTaskLocked(const VersionNodeReference& ref)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (ref.location.IsMissing()) return;
    if (auto it = visited_version_nodes_.find(ref.location);
        it != visited_version_nodes_.end()) {
      ValidateAlreadyVisitedVersionNodeLocked(ref, it->second);
      return;
    }
    version_stack_.push_back(ref);
  }

  void ProcessVersionTaskAsync(VersionNodeReference ref) {
    // Short-circuit if the task is missing or already visited/in progress.
    {
      absl::MutexLock lock(mutex_);

      // If this node is already visited, validate against the cached result
      // and return immediately without issuing another read.
      if (auto it = visited_version_nodes_.find(ref.location);
          it != visited_version_nodes_.end()) {
        ValidateAlreadyVisitedVersionNodeLocked(ref, it->second);
        active_reads_--;
        cond_.SignalAll();
        return;
      }

      // If this node is currently being fetched by another task, queue this
      // task as a pending waiter.  It will be validated once the in-flight read
      // completes, without blocking an executor thread.
      auto [it, inserted] = pending_version_tasks_.try_emplace(ref.location);
      if (!inserted) {
        it->second.push_back(std::move(ref));
        active_reads_--;
        cond_.SignalAll();
        return;
      }
    }

    reporter_.PrintProgress("Registering Version node: %s %v",
                            ref.location.file_id.FullPath(), ref.location);
    tracker_.RegisterUsedRange(ref.location.file_id.FullPath(),
                               ref.location.offset, ref.location.length);

    io_handle_->GetVersionTreeNode(ref.location)
        .ExecuteWhenReady([this, ref](auto f) {
          absl::Status handle_status = absl::OkStatus();
          auto result = f.result();
          if (!result.ok()) {
            reporter_.ReportError(
                tensorstore::StatusBuilder(result.status())
                    .Format("Error reading version tree node %v",
                            ref.location));
          } else {
            handle_status = HandleVersionNode(ref, **result);
          }

          absl::MutexLock lock(mutex_);
          active_reads_--;

          auto it = pending_version_tasks_.find(ref.location);
          CHECK(it != pending_version_tasks_.end());
          if (auto vit = visited_version_nodes_.find(ref.location);
              vit != visited_version_nodes_.end()) {
            for (const auto& pending : it->second) {
              ValidateAlreadyVisitedVersionNodeLocked(pending, vit->second);
            }
          }
          pending_version_tasks_.erase(it);

          if (!handle_status.ok()) status_.Update(handle_status);
          cond_.SignalAll();
        });
  }

  absl::Status HandleVersionNode(VersionNodeReference ref,
                                 const VersionTreeNode& node) {
    auto validation_status = ValidateVersionTreeNodeReference(
        node, config_, ref.generation_number, ref.height);
    if (!validation_status.ok()) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(validation_status)
              .Format("Validation error for version tree node %v",
                      ref.location));
    }

    // Entry-level validation
    if (node.height == 0) {
      auto& entries = std::get<VersionTreeNode::LeafNodeEntries>(node.entries);
      auto entries_status = ValidateVersionTreeLeafNodeEntries(
          node.version_tree_arity_log2, entries);
      if (!entries_status.ok()) {
        reporter_.ReportError(
            tensorstore::StatusBuilder(entries_status)
                .Format(
                    "Validation error for version tree leaf node %v entries",
                    ref.location));
      }
    } else {
      auto& entries =
          std::get<VersionTreeNode::InteriorNodeEntries>(node.entries);
      auto entries_status = ValidateVersionTreeInteriorNodeEntries(
          node.version_tree_arity_log2, node.height, entries);
      if (!entries_status.ok()) {
        reporter_.ReportError(tensorstore::StatusBuilder(entries_status)
                                  .Format("Validation error for version tree "
                                          "interior node %v entries",
                                          ref.location));
      }
    }

    {
      absl::MutexLock lock(mutex_);
      visited_version_nodes_[ref.location] =
          VisitedVersionNodeInfo{ref.generation_number, ref.height,
                                 ref.num_generations, ref.commit_time};
      if (node.height > 0) {
        auto& entries =
            std::get<VersionTreeNode::InteriorNodeEntries>(node.entries);
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
          EnqueueVersionTaskLocked(*it);
        }
      } else {
        auto& entries =
            std::get<VersionTreeNode::LeafNodeEntries>(node.entries);
        for (auto i = entries.rbegin(); i != entries.rend(); ++i) {
          const auto& btree_ref = *i;
          EnqueueBtreeTaskLocked(
              BtreeTask{btree_ref.root, btree_ref.root_height, ""});
        }
      }
    }
    return absl::OkStatus();
  }

  void ValidateAlreadyVisitedBtreeNodeLocked(const BtreeTask& task,
                                             const VisitedBtreeNodeInfo& info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (info.statistics != task.ref.statistics) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Statistics mismatch for cached B-tree node %v\n  "
                      "Expected:   %v\n  Cached:     %v",
                      task.ref.location, task.ref.statistics, info.statistics));
    }
    if (info.height != task.height) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Height mismatch for cached B-tree node %v\n  Expected: "
                      "%d\n  Cached:   %d",
                      task.ref.location, task.height, info.height));
    }
    if (info.first_key < task.inclusive_min_key) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("First key %v of cached B-tree node %v is less than "
                      "inclusive_min %v specified by parent node",
                      QuoteString(info.first_key), task.ref.location,
                      QuoteString(task.inclusive_min_key)));
    }
  }

  void ValidateAlreadyVisitedVersionNodeLocked(
      const VersionNodeReference& ref, const VisitedVersionNodeInfo& info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (info.generation_number != ref.generation_number) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Generation number mismatch for cached version tree node "
                      "%v\n  Expected: %d\n  Cached:   %d",
                      ref.location, ref.generation_number,
                      info.generation_number));
    }
    if (info.height != ref.height) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Height mismatch for cached version tree node %v\n  "
                      "Expected: %d\n  Cached:   %d",
                      ref.location, ref.height, info.height));
    }
    if (info.num_generations != ref.num_generations) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Num generations mismatch for cached version tree node "
                      "%v\n  Expected: %d\n  Cached:   %d",
                      ref.location, ref.num_generations, info.num_generations));
    }
    if (info.commit_time != ref.commit_time) {
      reporter_.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Commit time mismatch for cached version tree node %v\n  "
                      "Expected: %v\n  Cached:   %v",
                      ref.location, ref.commit_time, info.commit_time));
    }
  }

  IoHandle::Ptr io_handle_;
  std::optional<VersionSpec> version_spec_;
  kvstore::KvStore base_kvstore_;
  bool detailed_;
  OcdbtCheckReporter& reporter_;
  OcdbtFileUsageTracker tracker_;
  uint64_t alignment_;
  size_t concurrency_;
  Config config_;

  // Traversal state
  std::vector<BtreeTask> btree_stack_ ABSL_GUARDED_BY(mutex_);
  std::vector<VersionNodeReference> version_stack_ ABSL_GUARDED_BY(mutex_);
  size_t active_reads_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
  absl::CondVar cond_;

  // Tasks deferred because their target node is currently in-flight.
  // Validated once the corresponding read completes.
  absl::flat_hash_map<IndirectDataReference, std::vector<BtreeTask>>
      pending_btree_tasks_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<IndirectDataReference, std::vector<VersionNodeReference>>
      pending_version_tasks_ ABSL_GUARDED_BY(mutex_);

  // Data structures for tracking visited nodes.
  absl::flat_hash_map<IndirectDataReference, VisitedVersionNodeInfo>
      visited_version_nodes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<IndirectDataReference, VisitedBtreeNodeInfo>
      visited_btree_nodes_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace

absl::Status OcdbtCheck(Context context, tensorstore::kvstore::Spec source_spec,
                        std::ostream& output, OcdbtCheckOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto base,
                               kvstore::Open(source_spec, context).result());

  tensorstore::internal::EnsureDirectoryPath(base.path);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto data_copy_concurrency_resource,
      context
          .GetResource<tensorstore::internal::DataCopyConcurrencyResource>());
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto cache_pool_resource,
      context.GetResource<tensorstore::internal::CachePoolResource>());

  auto io_handle = tensorstore::internal_ocdbt::MakeIoHandle(
      data_copy_concurrency_resource, cache_pool_resource->get(), base, base,
      /*config_state=*/
      ConfigState::Make().value(), /*data_file_prefixes=*/{});

  std::optional<VersionSpec> version_spec;
  if (options.version) {
    TENSORSTORE_ASSIGN_OR_RETURN(version_spec,
                                 ParseVersionSpecFromUrl(*options.version));
  }

  // Use at least 4 threads for file I/O.
  if (options.concurrency < 4) {
    options.concurrency = 4;
  }

  OcdbtCheckReporter reporter(output, options.detailed);
  OcdbtCheckRunner runner(std::move(io_handle), version_spec, base, options,
                          reporter);
  TENSORSTORE_RETURN_IF_ERROR(runner.Run());

  if (reporter.error_count() > 0) {
    return tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
        .Format("Found %d integrity errors", reporter.error_count());
  }

  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
