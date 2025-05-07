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

#include "tensorstore/internal/os/error_code.h"

#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal {

absl::Status StatusFromOsError(OsErrorCode error_code, std::string_view a,
                               std::string_view b, std::string_view c,
                               std::string_view d, std::string_view e,
                               std::string_view f, SourceLocation loc) {
  return StatusFromOsError(internal::GetOsErrorStatusCode(error_code),
                           error_code, a, b, c, d, e, f, loc);
}

}  // namespace internal
}  // namespace tensorstore
