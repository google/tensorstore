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

#ifndef TENSORSTORE_UTIL_STATUS_IMPL_H_
#define TENSORSTORE_UTIL_STATUS_IMPL_H_

#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal_status {

// Add the SourceLocation to the Status.
void MaybeAddSourceLocationImpl(absl::Status& status, SourceLocation loc);

// Copy the SourceLocation payloads from `source` to `dest`.
void CopyPayloadsImpl(absl::Status& dest, const absl::Status& source,
                      SourceLocation loc);

// Create a new Status with the given code, message, and SourceLocation.
absl::Status StatusWithSourceLocation(absl::StatusCode code,
                                      std::string_view message,
                                      SourceLocation loc);

}  // namespace internal_status
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STATUS_IMPL_H_
