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

/// \file Implementation of `ConcurrencyResourceTraits` defined in
/// `concurrency_resource_provider.h`.

#include "tensorstore/internal/concurrency_resource.h"

#include <optional>

#include "absl/base/call_once.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource_provider.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/internal/thread_pool.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

AnyContextResourceJsonBinder<ConcurrencyResource::Spec>
ConcurrencyResourceTraits::JsonBinder() {
  namespace jb = tensorstore::internal::json_binding;
  constexpr auto get_default = [](auto* obj) { *obj = std::nullopt; };
  return jb::DefaultValue(
      get_default,
      jb::Object(jb::Member(
          "limit", jb::DefaultValue(get_default,
                                    jb::Optional(jb::Integer<size_t>(1),
                                                 [] { return "shared"; })))));
}

Result<ConcurrencyResource::Resource> ConcurrencyResourceTraits::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  Resource value;
  value.spec = spec;
  if (spec) {
    value.executor = DetachedThreadPool(*spec);
  } else {
    absl::call_once(shared_executor_once_, [&] {
      shared_executor_ = DetachedThreadPool(shared_limit_);
    });
    value.executor = shared_executor_;
  }
  return value;
}

ConcurrencyResource::Spec ConcurrencyResourceTraits::GetSpec(
    const Resource& value, const ContextSpecBuilder& builder) const {
  return value.spec;
}

}  // namespace internal
}  // namespace tensorstore
