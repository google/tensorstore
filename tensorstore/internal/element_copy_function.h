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

#ifndef TENSORSTORE_INTERNAL_ELEMENT_COPY_FUNCTION_H_
#define TENSORSTORE_INTERNAL_ELEMENT_COPY_FUNCTION_H_

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Returns the number of elements successfully copied.  If not equal to the
/// number requested, the absl::Status out parameter may be used to indicate an
/// error.
///
/// Elements should either be successfully copied, or be left untouched.
using ElementCopyFunction = internal::ElementwiseFunction<2, absl::Status*>;

inline absl::Status GetElementCopyErrorStatus(absl::Status status) {
  return status.ok() ? absl::UnknownError("Data conversion failure.") : status;
}

inline absl::Status GetElementCopyErrorStatus(
    Result<ArrayIterateResult>&& iterate_result, absl::Status&& status) {
  return !iterate_result.ok()
             ? iterate_result.status()
             : (iterate_result->success
                    ? absl::Status()
                    : GetElementCopyErrorStatus(std::move(status)));
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ELEMENT_COPY_FUNCTION_H_
