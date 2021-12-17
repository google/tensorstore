// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_CONTEXT_BINDING_VECTOR_H_
#define TENSORSTORE_INTERNAL_CONTEXT_BINDING_VECTOR_H_

#include <vector>

#include "absl/status/status.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

template <typename Spec>
struct ContextBindingTraits<std::vector<Spec>> {
  using BaseTraits = ContextBindingTraits<Spec>;

  static absl::Status Bind(std::vector<Spec>& spec, const Context& context) {
    for (auto& x : spec) {
      TENSORSTORE_RETURN_IF_ERROR(BaseTraits::Bind(x, context));
    }
    return absl::OkStatus();
  }

  static void Unbind(std::vector<Spec>& spec,
                     const ContextSpecBuilder& builder) {
    for (auto& x : spec) {
      BaseTraits::Unbind(x, builder);
    }
  }

  static void Strip(std::vector<Spec>& spec) {
    for (auto& x : spec) {
      BaseTraits::Strip(x);
    }
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONTEXT_BINDING_VECTOR_H_
