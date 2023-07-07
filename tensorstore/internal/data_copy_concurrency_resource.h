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

#ifndef TENSORSTORE_INTERNAL_DATA_COPY_CONCURRENCY_RESOURCE_H_
#define TENSORSTORE_INTERNAL_DATA_COPY_CONCURRENCY_RESOURCE_H_

#include "tensorstore/internal/concurrency_resource.h"

namespace tensorstore {
namespace internal {

/// Context resource used for "data copying" or other CPU-bound tasks like
/// encoding/decoding chunk data.
struct DataCopyConcurrencyResource : public ConcurrencyResource {
  static constexpr char id[] = "data_copy_concurrency";
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DATA_COPY_CONCURRENCY_RESOURCE_H_
