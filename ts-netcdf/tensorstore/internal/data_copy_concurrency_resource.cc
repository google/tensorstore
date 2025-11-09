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

#include "tensorstore/internal/data_copy_concurrency_resource.h"

#include <algorithm>
#include <thread>  // NOLINT

#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource_provider.h"

namespace tensorstore {
namespace internal {

namespace {
struct DataCopyConcurrencyResourceTraits
    : public ConcurrencyResourceTraits,
      public ContextResourceTraits<DataCopyConcurrencyResource> {
  DataCopyConcurrencyResourceTraits()
      : ConcurrencyResourceTraits(
            // This resource is for CPU-bound tasks.  Therefore, there is no
            // advantage in oversubscribing the number of available CPU cores.
            //
            // Always use at least 1 thread in case
            // `std::thread::hardware_concurrency()` returns 0 (due to being
            // unable to determine number of cpu cores).
            std::max(size_t(1), size_t(std::thread::hardware_concurrency()))) {}
};

const ContextResourceRegistration<DataCopyConcurrencyResourceTraits>
    registration;

}  // namespace
}  // namespace internal
}  // namespace tensorstore
