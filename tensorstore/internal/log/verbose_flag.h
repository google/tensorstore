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

#ifndef TENSORSTORE_INTERNAL_LOG_VERBOSE_FLAG_H_
#define TENSORSTORE_INTERNAL_LOG_VERBOSE_FLAG_H_

#include <stddef.h>

#include <atomic>
#include <limits>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"

namespace tensorstore {
namespace internal_log {

/// Set the verbose logging flags.  `input` is a comma separated string list of
/// names, or name=level values, which are reflected in the VerboseFlag objects.
/// Including the special value "all" in the list sets the global default,
/// enabling all verbose log sites.
void UpdateVerboseLogging(std::string_view input, bool overwrite);

/// VerboseFlag is used for verbose logging. It must be initialized by a
/// constant string, which is used in conjunction with the flags
/// --tensorstore_verbose_logging and/or environment variable
/// TENSORSTORE_VERBOSE_LOGGING to enable verbose logging for the name.
///
/// When adding a new VerboseFlag, also add it to docs/environment.rst
/// There is only a single supported way to use VerboseFlag:
///
///   namespace {
///     ABSL_CONST_INIT internal_log::VerboseFlag role("lady_macbeth");
///   }
///   ABSL_LOG_IF(INFO, role) << "What's done can't be undone.";
///
class VerboseFlag {
 public:
  constexpr static int kValueUninitialized = std::numeric_limits<int>::max();

  // VerboseFlag never be deallocated. name must never be deallocated.
  explicit constexpr VerboseFlag(const char* name)
      : value_(kValueUninitialized), name_(name), next_(nullptr) {}

  VerboseFlag(const VerboseFlag&) = delete;
  VerboseFlag& operator=(const VerboseFlag&) = delete;

  /// Returns whether logging is enabled for the flag at the given level.
  /// `level` must be >= 0; -1 is used to indicate a disabled flag.
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  bool Level(int level) {
    int v = value_.load(std::memory_order_relaxed);
    if (ABSL_PREDICT_TRUE(level > v)) {
      return false;
    }
    return VerboseFlagSlowPath(this, v, level);
  }

  /// Returns whether logging is enabled for the flag at level 0.
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  operator bool() {
    int v = value_.load(std::memory_order_relaxed);
    if (ABSL_PREDICT_TRUE(0 > v)) {
      return false;
    }
    return VerboseFlagSlowPath(this, v, 0);
  }

 private:
  static bool VerboseFlagSlowPath(VerboseFlag* flag, int old_v, int level);
  static int RegisterVerboseFlag(VerboseFlag* flag);

  std::atomic<int> value_;
  const char* const name_;
  VerboseFlag* next_;  // Read under global lock.

  friend void UpdateVerboseLogging(std::string_view, bool);
};

}  // namespace internal_log
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_LOG_VERBOSE_FLAG_H_
