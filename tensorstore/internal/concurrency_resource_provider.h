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

#ifndef TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_PROVIDER_H_
#define TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_PROVIDER_H_

#include <optional>

#include "absl/base/call_once.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Base traits class for registering context resource types that represent
/// thread pools.
class ConcurrencyResourceTraits {
 public:
  using Spec = typename ConcurrencyResource::Spec;
  using Resource = typename ConcurrencyResource::Resource;
  ConcurrencyResourceTraits(size_t shared_limit)
      : shared_limit_(shared_limit) {}

  static Spec Default() { return std::nullopt; }

  static AnyContextResourceJsonBinder<Spec> JsonBinder();

  Result<Resource> Create(const Spec& spec,
                          ContextResourceCreationContext context) const;
  Spec GetSpec(const Resource& value, const ContextSpecBuilder& builder) const;

 private:
  /// Size of thread pool referenced by `shared_executor_`.
  size_t shared_limit_;
  /// Protects initialization of `shared_executor_`.
  mutable absl::once_flag shared_executor_once_;
  /// Lazily-initialization shared thread pool used in the case of a default
  /// resource specification.
  mutable Executor shared_executor_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONCURRENCY_RESOURCE_PROVIDER_H_
