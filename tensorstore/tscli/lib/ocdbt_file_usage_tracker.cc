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

#include "tensorstore/tscli/lib/ocdbt_file_usage_tracker.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/tscli/lib/ocdbt_check_reporter.h"
#include "tensorstore/util/status_builder.h"

namespace tensorstore {
namespace cli {
namespace {

using Interval = OcdbtFileUsageTracker::Interval;

std::vector<OcdbtFileUsageTracker::Interval> MergeIntervals(
    const std::vector<Interval>& intervals) {
  if (intervals.empty()) {
    return {};
  }
  std::vector<Interval> sorted_intervals = intervals;
  std::sort(sorted_intervals.begin(), sorted_intervals.end());

  std::vector<Interval> merged;
  merged.push_back(sorted_intervals[0]);

  for (size_t i = 1; i < sorted_intervals.size(); ++i) {
    Interval& last = merged.back();
    const Interval& curr = sorted_intervals[i];

    uint64_t last_end = last.offset + last.length;

    // If the current interval starts at or before the end of the last
    // interval, merge them.
    if (curr.offset <= last_end) {
      // The merged interval starts at last.offset. Its new end is the max of
      // the original ends.
      uint64_t new_end = std::max(last_end, curr.offset + curr.length);
      last.length = new_end - last.offset;
    } else {
      // Otherwise, add the current interval as a new, separate interval.
      merged.push_back(curr);
    }
  }
  return merged;
}

// Computes the unused ranges (gaps) within a file, given a set of merged
// used intervals and the total actual size of the file.
// The final gap at the end of the file is aligned to the provided alignment.
std::vector<Interval> ComputeUnusedRanges(
    const std::vector<Interval>& merged_intervals, uint64_t actual_size,
    uint64_t alignment) {
  std::vector<Interval> unused;
  uint64_t current_pos = 0;

  for (const auto& used : merged_intervals) {
    // Found a gap between the end of the previous merged block (current_pos)
    // and the start of the current used block.
    if (used.offset > current_pos) {
      unused.push_back(Interval{current_pos, used.offset - current_pos});
    }
    // Update current_pos to the end of the current used interval.
    current_pos = std::max(current_pos, used.offset + used.length);
  }

  // Check for a gap at the end of the file, aligned to the provided
  // alignment.
  uint64_t aligned_end =
      ((current_pos + alignment - 1) / alignment) * alignment;
  if (aligned_end < actual_size) {
    unused.push_back(Interval{aligned_end, actual_size - aligned_end});
  }
  return unused;
}
}  // namespace

void OcdbtFileUsageTracker::SetOnDiskFiles(
    absl::flat_hash_map<std::string, int64_t> files) {
  absl::MutexLock lock(mutex_);
  on_disk_files_ = std::move(files);
}

void OcdbtFileUsageTracker::RegisterUsedRange(std::string_view file_path,
                                              uint64_t offset,
                                              uint64_t length) {
  absl::MutexLock lock(mutex_);
  std::string path(file_path);
  referenced_files_[path].push_back(Interval{offset, length});
}

void OcdbtFileUsageTracker::ForEachFileUsedRange(
    absl::FunctionRef<void(std::string_view file_path,
                           const std::vector<Interval>& intervals)>
        func) {
  auto referenced_files = [this] {
    absl::MutexLock lock(mutex_);
    return referenced_files_;
  }();
  for (const auto& [file_path, intervals] : referenced_files) {
    func(file_path, intervals);
  }
}

void OcdbtFileUsageTracker::CheckOrphanedFiles(OcdbtCheckReporter& reporter) {
  std::vector<std::string> orphans;
  {
    absl::MutexLock lock(mutex_);
    if (!on_disk_files_.empty()) {
      for (const auto& pair : on_disk_files_) {
        const auto& file = pair.first;
        if (file == "manifest.ocdbt" || absl::StartsWith(file, "manifest.")) {
          continue;
        }
        if (referenced_files_.contains(file)) {
          continue;
        }
        orphans.push_back(file);
      }
    }
  }
  if (orphans.empty()) return;

  if (reporter.detailed()) {
    reporter.ReportWarning("Warning: Found %d orphaned files on disk",
                           orphans.size());
    for (const auto& file : orphans) {
      reporter.ReportInfo("  %s", file);
    }
  } else {
    reporter.ReportWarning(
        "Warning: Found %d orphaned files on disk (run with --detailed to "
        "list them).",
        orphans.size());
  }
}

void OcdbtFileUsageTracker::CheckUnusedRanges(OcdbtCheckReporter& reporter,
                                              uint64_t alignment) {
  uint64_t global_total_unused_bytes = 0;
  uint64_t global_total_size = 0;
  size_t files_with_unused_count = 0;

  absl::MutexLock lock(mutex_);
  for (auto& [file_path, intervals] : referenced_files_) {
    int64_t file_size = -1;
    if (auto it = on_disk_files_.find(file_path); it != on_disk_files_.end()) {
      file_size = it->second;
    }
    if (file_size < 0) {
      reporter.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Error: File %s is referenced but not "
                      "present in kvstore listing",
                      file_path));
      continue;
    }

    uint64_t actual_size = static_cast<uint64_t>(file_size);
    global_total_size += actual_size;

    auto merged_intervals = MergeIntervals(intervals);
    uint64_t max_reached =
        merged_intervals.empty()
            ? 0
            : merged_intervals.back().offset + merged_intervals.back().length;

    if (actual_size < max_reached) {
      reporter.ReportError(
          tensorstore::StatusBuilder(absl::StatusCode::kDataLoss)
              .Format("Error reading indirect value (truncation "
                      "check): File %s is truncated. Size on "
                      "disk: %d, expected at least: %d (max "
                      "referenced offset + length)",
                      file_path, actual_size, max_reached));
      continue;
    }

    auto unused = ComputeUnusedRanges(merged_intervals, actual_size, alignment);
    if (!unused.empty()) {
      files_with_unused_count++;
      uint64_t total_unused_bytes = 0;
      std::string gaps_str;
      for (const auto& gap : unused) {
        total_unused_bytes += gap.length;
        if (reporter.detailed()) {
          absl::StrAppendFormat(&gaps_str, "\n    [%d, %d)", gap.offset,
                                gap.offset + gap.length);
        }
      }
      global_total_unused_bytes += total_unused_bytes;
      if (reporter.detailed()) {
        double percentage =
            (static_cast<double>(total_unused_bytes) / actual_size) * 100.0;
        reporter.ReportInfo(
            "File %s has %d unused bytes (%.2f%%) across %d gaps:%s", file_path,
            total_unused_bytes, percentage, unused.size(), gaps_str);
      }
    }
  }

  if (global_total_unused_bytes > 0 && !reporter.detailed()) {
    double global_percentage = 0.0;
    if (global_total_size > 0) {
      global_percentage =
          (static_cast<double>(global_total_unused_bytes) / global_total_size) *
          100.0;
    }
    reporter.ReportInfo(
        "Database fragmentation: %d unused bytes (%.2f%%) across "
        "%d files (run with --detailed to view per-file gaps).",
        global_total_unused_bytes, global_percentage, files_with_unused_count);
  }
}

}  // namespace cli
}  // namespace tensorstore
