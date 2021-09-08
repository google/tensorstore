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
///
/// Pickling of `Context` objects is supported.  Pickling of other objects that
/// depend on Context resources, such as `TensorStore` objects, is also
/// supported.
///
/// In order to exactly preserve object identity equivalence among multiple
/// context resources that are pickled together, pickling is handled specially.
/// While the public C++ `tensorstore::Context` API natively supports a form of
/// persistence via conversion to/from JSON, and the
/// `tensorstore::internal::ContextSpecBuilder` API provides a way to preserve
/// object identity equivalence, Python's pickling mechanism does not directly
/// expose a way to manage per-session state (e.g. to hold a single
/// `ContextSpecBuilder` object used for all context resources pickled in a
/// given session).
///
/// Instead, we directly expose individual `ContextResource` objects to Python
/// and rely on the combination of pybind11's object identity preservation and
/// the object deduplication built into Python's pickling mechanism.

#include "python/tensorstore/intrusive_ptr_holder.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "pybind11/pybind11.h"
#include "tensorstore/context.h"
#include "tensorstore/context_impl.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

void RegisterContextBindings(pybind11::module m, Executor defer);

/// Pickles a `ContextSpecBuilder` by pickling the map of resource keys and
/// associated resources that it contains.
///
/// The returned tuple is of the form `key0, resource0, key1, resource1, ...`,
/// where the `key` values are strings and the `resource` values are the Python
/// `ContextResource` objects themselves (rather than their JSON
/// representation), in order to allow them to be properly deduplicated by
/// Python's pickling mechanism.
///
/// \param builder The builder to pickle, must be non-null and the last
///     reference.
pybind11::tuple PickleContextSpecBuilder(internal::ContextSpecBuilder builder);

/// Inverse of `PickleContextSpecBuilder`.
///
/// This is also used as an implementation detail when unpickling entire
/// `Context` objects (which use a separate pickling path from
/// `PickleContextSpecBuilder`).
///
/// \returns Non-null context object.
internal_context::ContextImplPtr UnpickleContextSpecBuilder(
    pybind11::tuple t, bool allow_key_mismatch, bool bind_partial);

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

/// Pickles a `Spec`-like type that supports a JSON representation with a nested
/// `context`.
///
/// This is the inverse of `UnpickleWithNestedContext`.
template <typename T>
pybind11::tuple PickleWithNestedContext(T spec) {
  auto builder = internal::ContextSpecBuilder::Make();
  internal::SetRecordBindingState(builder, true);
  spec.UnbindContext(builder);
  auto pickled_context =
      internal_python::PickleContextSpecBuilder(std::move(builder));
  JsonSerializationOptions json_serialization_options;
  json_serialization_options.preserve_bound_context_resources_ = true;
  auto json_spec = ValueOrThrow(spec.ToJson(json_serialization_options));
  return pybind11::make_tuple(pybind11::cast(json_spec),
                              std::move(pickled_context));
}

/// Unpickles a `Spec`-like type that supports a JSON representation with a
/// nested `context`.
///
/// This is the inverse of `PickleWithNestedContext`.
template <typename T>
T UnpickleWithNestedContext(pybind11::tuple t) {
  auto json_spec = pybind11::cast<::nlohmann::json>(t[0]);
  auto context = WrapImpl(internal_python::UnpickleContextSpecBuilder(
      t[1], /*allow_key_mismatch=*/true, /*bind_partial=*/true));
  JsonSerializationOptions json_serialization_options;
  json_serialization_options.preserve_bound_context_resources_ = true;
  T spec = ValueOrThrow(
      T::FromJson(std::move(json_spec), json_serialization_options));
  ThrowStatusException(spec.BindContext(context));
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
