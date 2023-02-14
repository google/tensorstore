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

#ifndef THIRD_PARTY_PY_TENSORSTORE_CONTEXT_H_
#define THIRD_PARTY_PY_TENSORSTORE_CONTEXT_H_

/// \file Defines the `tensorstore.Context` and `tensorstore.ContextSpec` Python
/// classes.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/intrusive_ptr_holder.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "tensorstore/context.h"
#include "tensorstore/context_impl.h"

namespace tensorstore {
namespace internal_python {

// Type alias for use with `PYBIND11_DECLARE_HOLDER_TYPE` below.
//
// Because `PYBIND11_DECLARE_HOLDER_TYPE` expects a template, we have to
// register a template even though
// `T = tensorstore::internal_context::ResourceImplBase` (corresponding
// to `ResourceImplWeakPtr`).  The alias is needed to avoid having a
// comma inside the macro argument.
template <typename T>
using ContextResourcePtrHolderWorkaround =
    internal::IntrusivePtr<T, internal_context::ResourceImplWeakPtrTraits>;

inline Context WrapImpl(internal_context::ContextImplPtr impl) {
  Context context;
  internal_context::Access::impl(context) = std::move(impl);
  return context;
}

inline Context::Spec WrapImpl(internal_context::ContextSpecImplPtr impl) {
  Context::Spec spec;
  internal_context::Access::impl(spec) = std::move(impl);
  return spec;
}

}  // namespace internal_python
}  // namespace tensorstore

// Declare `tensorstore::internal_context::ResourceImplWeakPtr` a holder
// for `internal_context::ResourceImplBase`.  This allows pybind11 to
// directly hold a `ResourceImplWeakPtr`, rather than wrapping that
// pointer in a `unique_ptr`, which is needed for correct deduplication when
// pickling (and also improves efficiently generally).
PYBIND11_DECLARE_HOLDER_TYPE(
    T, ::tensorstore::internal_python::ContextResourcePtrHolderWorkaround<T>,
    /*always_construct_holder=*/true)

// Note that `ContextSpecImplPtr` and `ContextImplPtr` are also defined as
// holder types by the generic definition in `intrusive_ptr_holder.h`.

#endif  // THIRD_PARTY_PY_TENSORSTORE_CONTEXT_H_
