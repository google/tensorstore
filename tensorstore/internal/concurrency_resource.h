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

#ifndef TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_H_
#define TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_H_

#include <cstddef>
#include <optional>

#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

/// Base resource provider class for defining context resource types that
/// represent thread pools, where the default value is a shared thread pool
/// (shared even across multiple independent `Context` objects) of a given size.
///
/// This is used to define `data_copy_concurrency`, `file_io_concurrency`,
/// `gcs_request_concurrency`.
///
/// Normally resources aren't shared between multiple `Context` objects.
/// However, this sharing is done for thread pools because they inherently
/// represent global resources.  However, this sharing can be overridden by
/// specifying an explicit limit in the resource specification rather than
/// relying on the default.
///
/// To define a derived concurrency resource type:
///
/// 1. Define a class that inherits from `ConcurrencyResource` with an `id`
///    member.
///
/// 2. Define a `Traits` type that inherits from `ConcurrencyResourceTraits`
///    (defined in `concurrency_resource_provider.h`) and from
///    `ContextResourceTraits<Traits>` with a default constructor that specifies
///    the shared thread pool size to the `ConcurrencyResourceTraits`
///    constructor.
///
/// 3. Register the `Traits` type using a `ContextResourceRegistration` object.
struct ConcurrencyResource {
  using Spec = std::optional<size_t>;
  struct Resource {
    // If equal to `nullopt`, indicates that the shared executor is used.
    Spec spec;
    Executor executor;
  };
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_H_
