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

#include "tensorstore/kvstore/http/byte_range_util.h"

#include <stdint.h>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

absl::Status ValidateResponseByteRange(
    const HttpResponse& response,
    const OptionalByteRangeRequest& byte_range_request, absl::Cord& value,
    ByteRange& byte_range, int64_t& total_size) {
  value = response.payload;
  if (response.status_code != 206) {
    // This may or may not have been a range request; attempt to validate.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto resolved_byte_range,
        byte_range_request.Validate(response.payload.size()));

    if (resolved_byte_range.size() != response.payload.size()) {
      // Byte range was requested and the server sent the entire object
      // which was larger than the requested range. This likely indicates
      // that a `Content-Encoding` header was set on the object. Fail the
      // request to indicate to the user that the request was inefficient.
      return absl::FailedPreconditionError(
          tensorstore::StrCat("Requested byte range ", byte_range_request,
                              " was ignored by server"));
    }
    total_size = response.payload.size();
    byte_range = {0, total_size};
  } else {
    // Server should return a parseable content-range header.
    TENSORSTORE_ASSIGN_OR_RETURN(auto content_range_info,
                                 ParseContentRangeHeader(response));
    byte_range = {content_range_info.inclusive_min,
                  content_range_info.exclusive_max};
    total_size = content_range_info.total_size;
    if (auto request_size = byte_range_request.size();
        (byte_range_request.inclusive_min >= 0 &&
         byte_range_request.inclusive_min != byte_range.inclusive_min) ||
        (byte_range_request.exclusive_max != -1 &&
         byte_range_request.exclusive_max != byte_range.exclusive_max) ||
        (request_size >= 0 && request_size != value.size()) ||
        (total_size != -1 && byte_range_request.exclusive_max == -1 &&
         byte_range.exclusive_max != total_size)) {
      // Return an error when the response does not start at the requested
      // offset or when the response is smaller than the desired size.
      return absl::OutOfRangeError(
          tensorstore::StrCat("Requested byte range ", byte_range_request,
                              " was not satisfied by response with byte range ",
                              byte_range, " and total size ", total_size));
    }
  }
  return absl::OkStatus();
}

}  // namespace internal_http
}  // namespace tensorstore
