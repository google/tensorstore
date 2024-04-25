// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_HTTP_BYTE_RANGE_UTIL_H_
#define TENSORSTORE_KVSTORE_HTTP_BYTE_RANGE_UTIL_H_

#include <stdint.h>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/byte_range.h"

namespace tensorstore {
namespace internal_http {

/// Validates that `response` is consistent with `byte_range_request`.
///
/// Assigns the validated content to `value`, the corresponding byte range to
/// `byte_range`, and the total size (or `-1` if unknown) to `total_size`.`
absl::Status ValidateResponseByteRange(
    const HttpResponse& response,
    const OptionalByteRangeRequest& byte_range_request, absl::Cord& value,
    ByteRange& byte_range, int64_t& total_size);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_HTTP_BYTE_RANGE_UTIL_H_
