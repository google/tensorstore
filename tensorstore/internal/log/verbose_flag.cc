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

#include "tensorstore/internal/log/verbose_flag.h"

#include <stddef.h>

#include <atomic>
#include <cassert>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/no_destructor.h"

ABSL_FLAG(std::string, tensorstore_verbose_logging, {},
          "comma-separated list of tensorstore verbose logging flags")
    .OnUpdate([]() {
      if (!absl::GetFlag(FLAGS_tensorstore_verbose_logging).empty()) {
        tensorstore::internal_log::UpdateVerboseLogging(
            absl::GetFlag(FLAGS_tensorstore_verbose_logging), true);
      }
    });

namespace tensorstore {
namespace internal_log {
namespace {

// Guards the current log levels and the head of the list of installed sites.
ABSL_CONST_INIT absl::Mutex g_mutex(absl::kConstInit);

// Head of an intrusive slist of VerboseFlag*.  When a new flag instance is
// first seen, it is added to this list (under lock), so that when flags are
// updated the flag instance can be updated as well. List traversal is done
// without locks since the slist cannot change once initialized.
ABSL_CONST_INIT VerboseFlag* g_list_head ABSL_GUARDED_BY(g_mutex) = nullptr;

// The current log-levels config contains a mapping from value to level,
// which defaults to the "default_level"
struct LoggingLevelConfig {
  int default_level = -1;
  absl::flat_hash_map<std::string, int> levels;
};

void UpdateLoggingLevelConfig(std::string_view input,
                              LoggingLevelConfig& config) {
  auto& levels = config.levels;

  // Split input into name=value pairs.
  for (std::string_view flag : absl::StrSplit(input, ',', absl::SkipEmpty())) {
    const size_t eq = flag.rfind('=');
    if (eq == flag.npos) {
      levels.insert_or_assign(std::string(flag), 0);
      continue;
    }
    if (eq == 0) continue;
    int level;
    if (!absl::SimpleAtoi(flag.substr(eq + 1), &level)) continue;

    // Clamp level to [-1..1000].
    if (level < -1) {
      level = -1;
    } else if (level > 1000) {
      level = 1000;
    }
    levels.insert_or_assign(std::string(flag.substr(0, eq)), level);
  }

  config.default_level = -1;
  if (auto it = levels.find("all"); it != levels.end()) {
    config.default_level = it->second;
  }
}

// Owns the current set of live/active logging config values.
LoggingLevelConfig& GetLoggingLevelConfig()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_mutex) {
  // The initial use installs the environment variable. This may slow down
  // the first log statement.
  static internal::NoDestructor<LoggingLevelConfig> flags{[] {
    LoggingLevelConfig config;
    if (auto env = internal::GetEnv("TENSORSTORE_VERBOSE_LOGGING"); env) {
      UpdateLoggingLevelConfig(*env, config);
    }
    return config;
  }()};

  return *flags;
}

}  // namespace

void UpdateVerboseLogging(std::string_view input, bool overwrite)
    ABSL_LOCKS_EXCLUDED(g_mutex) {
  ABSL_LOG(INFO) << "--tensorstore_verbose_logging=" << input;
  LoggingLevelConfig config;
  UpdateLoggingLevelConfig(input, config);

  // Update the global map and all registered flag values under lock.
  // This can stall logging on first-seen VerboseFlags, so if VerboseFlag
  // are frequently declared inline in each ABSL_LOG statement, such as in
  // an implementation detail of a macro, perhaps, revisit this.
  absl::MutexLock lock(&g_mutex);

  VerboseFlag* slist = g_list_head;
  LoggingLevelConfig& global_config = GetLoggingLevelConfig();

  // Update the global map with the new values.
  std::swap(global_config.levels, config.levels);
  std::swap(global_config.default_level, config.default_level);
  if (!overwrite) {
    if (global_config.levels.count("all")) {
      global_config.default_level = config.default_level;
    }
    // merge old entries into the new map.
    global_config.levels.merge(config.levels);
  }

  // Update all registered named flags.
  std::string_view last;
  int last_level = 0;
  while (slist != nullptr) {
    std::string_view current(slist->name_);
    if (current != last) {
      last = current;
      auto it = global_config.levels.find(current);
      if (it != global_config.levels.end()) {
        last_level = it->second;
      } else {
        last_level = global_config.default_level;
      }
    }
    slist->value_.store(last_level, std::memory_order_seq_cst);
    slist = slist->next_;
  }
}

/* static */
int VerboseFlag::RegisterVerboseFlag(VerboseFlag* flag) {
  std::string_view flag_name(flag->name_);

  absl::MutexLock lock(&g_mutex);
  int old_v = flag->value_.load(std::memory_order_relaxed);
  if (old_v == kValueUninitialized) {
    // If the value was uninitialized, this is the first time the registration
    // code is running. Set the verbosity level and add to the linked list.
    const auto& global_config = GetLoggingLevelConfig();
    if (auto it = global_config.levels.find(flag_name);
        it != global_config.levels.end()) {
      old_v = it->second;
    } else {
      old_v = global_config.default_level;
    }
    flag->value_.store(old_v, std::memory_order_relaxed);
    flag->next_ = std::exchange(g_list_head, flag);
  }
  return old_v;
}

/* static */
bool VerboseFlag::VerboseFlagSlowPath(VerboseFlag* flag, int old_v, int level) {
  if (ABSL_PREDICT_TRUE(old_v != kValueUninitialized)) {
    return level >= 0;
  }

  // This is the first time the flag is seen, so add it to the global
  // tracking list and initialize the value.
  old_v = RegisterVerboseFlag(flag);
  return ABSL_PREDICT_FALSE(old_v >= level);
}

static_assert(std::is_trivially_destructible<VerboseFlag>::value,
              "VerboseFlag must be trivially destructible");

}  // namespace internal_log
}  // namespace tensorstore
