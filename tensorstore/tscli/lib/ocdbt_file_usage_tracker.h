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

#ifndef TENSORSTORE_TSCLI_LIB_OCDBT_FILE_USAGE_TRACKER_H_
#define TENSORSTORE_TSCLI_LIB_OCDBT_FILE_USAGE_TRACKER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/tscli/lib/ocdbt_check_reporter.h"

namespace tensorstore {
namespace cli {

class OcdbtFileUsageTracker {
 public:
  struct Interval {
    uint64_t offset;
    uint64_t length;
    bool operator<(const Interval& other) const {
      return offset < other.offset;
    }
  };

  OcdbtFileUsageTracker() = default;

  size_t on_disk_files_size() const {
    absl::MutexLock lock(mutex_);
    return on_disk_files_.size();
  }

  size_t referenced_files_size() const {
    absl::MutexLock lock(mutex_);
    return referenced_files_.size();
  }

  void SetOnDiskFiles(absl::flat_hash_map<std::string, int64_t> files);

  void RegisterUsedRange(std::string_view file_path, uint64_t offset,
                         uint64_t length);

  void ForEachFileUsedRange(
      absl::FunctionRef<void(std::string_view file_path,
                             const std::vector<Interval>& intervals)>
          func);

  void CheckOrphanedFiles(OcdbtCheckReporter& reporter);
  void CheckUnusedRanges(OcdbtCheckReporter& reporter, uint64_t alignment);

 private:
  absl::flat_hash_map<std::string, int64_t> on_disk_files_
      ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, std::vector<Interval>> referenced_files_
      ABSL_GUARDED_BY(mutex_);
  mutable absl::Mutex mutex_;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_OCDBT_FILE_USAGE_TRACKER_H_
