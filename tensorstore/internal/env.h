// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_ENV_H_
#define TENSORSTORE_INTERNAL_ENV_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/strings/numbers.h"

namespace tensorstore {
namespace internal {

// Returns the parsed environment variables
absl::flat_hash_map<std::string, std::string> GetEnvironmentMap();

// Returns the value of an environment variable or empty.
std::optional<std::string> GetEnv(char const* variable);

// Sets environment variable `variable` to `value`.
void SetEnv(const char* variable, const char* value);

// Removes environment variable `variable`.
void UnsetEnv(const char* variable);

// Returns the parsed value of an environment variable or empty.
template <typename T>
std::optional<T> GetEnvValue(const char* variable) {
  auto env = internal::GetEnv(variable);
  if (!env) return std::nullopt;
  if constexpr (std::is_same_v<bool, T>) {
    T n;
    if (absl::SimpleAtob(*env, &n)) return n;
  } else if constexpr (std::is_same_v<float, T>) {
    T n;
    if (absl::SimpleAtof(*env, &n)) return n;
  } else if constexpr (std::is_same_v<double, T>) {
    T n;
    if (absl::SimpleAtod(*env, &n)) return n;
  } else {
    T n;
    if (absl::SimpleAtoi(*env, &n)) return n;
  }
  ABSL_LOG(INFO) << "Failed to parse" << variable << " as a value: " << *env;
  return std::nullopt;
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ENV_H_
