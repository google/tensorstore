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

#include "tensorstore/internal/testing/random_seed.h"

#include <random>

#include "absl/log/absl_log.h"
#include "absl/strings/numbers.h"
#include "tensorstore/internal/env.h"

namespace tensorstore {
namespace internal_testing {

unsigned int GetRandomSeedForTest(const char* env_var) {
  unsigned int seed;
  if (auto env_seed = internal::GetEnv(env_var)) {
    if (absl::SimpleAtoi(*env_seed, &seed)) {
      ABSL_LOG(INFO) << "Using deterministic random seed " << env_var << "="
                     << seed;
      return seed;
    }
  }
  seed = std::random_device()();
  ABSL_LOG(INFO) << "Define environment variable " << env_var << "=" << seed
                 << " for deterministic seeding";
  return seed;
}

}  // namespace internal_testing
}  // namespace tensorstore
