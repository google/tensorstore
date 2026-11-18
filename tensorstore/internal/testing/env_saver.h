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

#ifndef TENSORSTORE_INTERNAL_TESTING_ENV_SAVER_H_
#define TENSORSTORE_INTERNAL_TESTING_ENV_SAVER_H_

#include <string>

#include "absl/container/flat_hash_map.h"

namespace tensorstore {
namespace internal_testing {

// Saves and restores environment variables within a scope (similar to
// `absl::FlagSaver`).
//
// NOTE: Modifying environment variables is not thread-safe. Only use `EnvSaver`
// in single-threaded contexts such as unit test setup/teardown.
class EnvSaver {
 public:
  EnvSaver();
  ~EnvSaver();

  EnvSaver(const EnvSaver&) = delete;
  EnvSaver& operator=(const EnvSaver&) = delete;

 private:
  absl::flat_hash_map<std::string, std::string> saved_env_;
};

}  // namespace internal_testing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TESTING_ENV_SAVER_H_
