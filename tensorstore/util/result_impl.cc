// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/util/result_impl.h"

#include <stdlib.h>

#include <cstdio>
#include <exception>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"

namespace tensorstore {
namespace internal_result {

void HandleInvalidResultCtor(absl::Status* absl_nonnull status) {
  std::fputs("An OK status is not a valid constructor argument to Result<T>",
             stderr);
  std::fflush(stderr);
  std::terminate();

  // NOTE: In release mode we could assign an absl::InternalError to status here
  // instead of terminating the process.
}

void CrashOnResultNotOk(const absl::Status& status) {
  std::fprintf(
      stderr,
      "Attempting to fetch Result<T> value instead of handling error %s",
      status.ToString(absl::StatusToStringMode::kWithEverything).c_str());
  std::fflush(stderr);
  std::terminate();
}

}  // namespace internal_result
}  // namespace tensorstore
