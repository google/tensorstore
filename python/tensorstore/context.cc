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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <optional>
#include <utility>

#include "python/tensorstore/context.h"
#include "python/tensorstore/intrusive_ptr_holder.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/context.h"
#include "tensorstore/context_impl.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

using internal_context::Access;
using internal_context::ContextImpl;
using internal_context::ContextImplPtr;
using internal_context::ContextSpecImpl;
using internal_context::ContextSpecImplPtr;
using internal_context::ResourceImplBase;
using internal_context::ResourceImplWeakPtr;

using ContextCls = py::class_<ContextImpl, ContextImplPtr>;
using ContextSpecCls = py::class_<ContextSpecImpl, ContextSpecImplPtr>;

// `ResourceImplBase` represents a context resource.  It is exposed
// primarily for pickling and testing.  There isn't a whole lot that can be
// done with objects of this type, though their identity can be compared.
using ContextResourceCls = py::class_<ResourceImplBase, ResourceImplWeakPtr>;

ContextCls MakeContextClass(pybind11::module m) {
  return ContextCls(m, "Context",
                    R"(
Manages shared TensorStore :ref:`context resources<context>`, such as caches and credentials.

Group:
  Core

See also:
  :ref:`context`

)");
}

void DefineContextAttributes(ContextCls& cls) {
  cls.def(py::init([] { return Access::impl(Context::Default()); }),
          R"(
Constructs a default context.

Example:

    >>> context = ts.Context()
    >>> context.spec is None
    True

.. note::

   Each call to this constructor returns a unique default context instance, that
   does *not* share resources with other default context instances.  To share
   resources, you must use the same :py:obj:`Context` instance.

Overload:
  default
)");

  cls.def(py::init([](ContextSpecImplPtr spec,
                      std::optional<ContextImplPtr> parent) {
            if (!parent) parent.emplace();
            return Access::impl(Context(WrapImpl(std::move(spec)),
                                        WrapImpl(std::move(*parent))));
          }),
          R"(
Constructs a context from a parsed spec.

Args:
  spec: Parsed context spec.
  parent: Parent context from which to inherit.  Defaults to a new default
    context as returned by :python:`tensorstore.Context()`.

Overload:
  spec
)",
          py::arg("spec"), py::arg("parent") = std::nullopt);

  cls.def(
      py::init([](::nlohmann::json json, std::optional<ContextImplPtr> parent) {
        if (!parent) parent.emplace();
        return Access::impl(Context(ValueOrThrow(Context::Spec::FromJson(json)),
                                    WrapImpl(std::move(*parent))));
      }),
      R"(
Constructs a context from its :json:schema:`JSON representation<Context>`.

Example:

    >>> context = ts.Context({'cache_pool': {'total_bytes_limit': 5000000}})
    >>> context.spec
    Context.Spec({'cache_pool': {'total_bytes_limit': 5000000}})

Args:
  json: :json:schema:`JSON representation<Context>` of the context.
  parent: Parent context from which to inherit.  Defaults to a new default
    context as returned by :python:`tensorstore.Context()`.

Overload:
  json
)",
      py::arg("json"), py::arg("parent") = std::nullopt);

  cls.def_property_readonly(
      "parent", [](const ContextImpl& self) { return self.parent_; },
      R"(
Parent context from which this context inherits.

Example:

    >>> parent = ts.Context({
    ...     'cache_pool': {
    ...         'total_bytes_limit': 5000000
    ...     },
    ...     'file_io_concurrency': {
    ...         'limit': 10
    ...     }
    ... })
    >>> child = ts.Context({'cache_pool': {
    ...     'total_bytes_limit': 10000000
    ... }},
    ...                    parent=parent)
    >>> assert child.parent is parent
    >>> parent['cache_pool'].to_json()
    {'total_bytes_limit': 5000000}
    >>> child['cache_pool'].to_json()
    {'total_bytes_limit': 10000000}
    >>> child['file_io_concurrency'].to_json()
    {'limit': 10}

Group:
  Accessors
)");

  cls.def_property_readonly(
      "spec", [](const ContextImpl& self) { return self.spec_; },
      R"(
Spec from which this context was constructed.

Example:

    >>> parent = ts.Context({
    ...     'cache_pool': {
    ...         'total_bytes_limit': 5000000
    ...     },
    ...     'file_io_concurrency': {
    ...         'limit': 10
    ...     }
    ... })
    >>> child = ts.Context({'cache_pool': {
    ...     'total_bytes_limit': 10000000
    ... }},
    ...                    parent=parent)
    >>> child.spec
    Context.Spec({'cache_pool': {'total_bytes_limit': 10000000}})
    >>> child.parent.spec
    Context.Spec({
      'cache_pool': {'total_bytes_limit': 5000000},
      'file_io_concurrency': {'limit': 10},
    })

Group:
  Accessors
)");

  cls.def(
      "__getitem__",
      [](ContextImplPtr self, std::string key) {
        auto provider_id = internal_context::ParseResourceProvider(key);
        auto* provider = internal_context::GetProvider(provider_id);
        if (!provider) {
          ThrowStatusException(
              internal_context::ProviderNotRegisteredError(provider_id));
        }
        auto spec = ValueOrThrow(
            internal_context::ResourceSpecFromJson(provider_id, key, {}));
        return ValueOrThrow(
            internal_context::GetOrCreateResource(*self, *spec, nullptr));
      },
      R"(
Creates or retrieves the context resource for the given key.

This is primarily useful for introspection of a context.

Example:

    >>> context = ts.Context(
    ...     {'cache_pool#a': {
    ...         'total_bytes_limit': 10000000
    ...     }})
    >>> context['cache_pool#a']
    Context.Resource({'total_bytes_limit': 10000000})
    >>> context['cache_pool']
    Context.Resource({})

Args:
  key: Resource key, of the form :python:`'<resource-type>'` or
    :python:`<resource-type>#<id>`.

Returns:
  The resource handle.

Group:
  Accessors
)",
      py::arg("key"));

  EnablePicklingFromSerialization<ContextImplPtr>(
      cls, serialization::NonNullIndirectPointerSerializer<
               ContextImplPtr,
               internal_context::ContextImplPtrNonNullDirectSerializer>{});
}

ContextSpecCls MakeContextSpecClass(ContextCls& cls_context) {
  return ContextSpecCls(cls_context, "Spec", R"(
Parsed representation of a :json:schema:`JSON Context<Context>` specification.
)");
}

void DefineContextSpecAttributes(ContextSpecCls& cls) {
  cls.def(py::init([](const ::nlohmann::json& json) {
            return Access::impl(ValueOrThrow(Context::Spec::FromJson(json)));
          }),
          R"(
Creates a context specification from its :json:schema:`JSON representation<Context>`.
)",
          py::arg("json"));

  cls.def(
      "to_json",
      [](internal_context::ContextSpecImplPtr self, bool include_defaults) {
        return WrapImpl(std::move(self))
            .ToJson(IncludeDefaults{include_defaults});
      },
      R"(
Returns the :json:schema:`JSON representation<Context>`.

Args:
  include_defaults: Indicates whether to include members even if they are equal to the default value.

Group:
  Accessors
)",
      py::arg("include_defaults") = false);

  cls.def("__repr__", [](internal_context::ContextSpecImplPtr self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        WrapImpl(std::move(self)).ToJson(IncludeDefaults{false}),
        "Context.Spec(", ")");
  });

  EnablePicklingFromSerialization<ContextSpecImplPtr>(cls);
}

ContextResourceCls MakeContextResourceClass(ContextCls& cls_context) {
  return ContextResourceCls(cls_context, "Resource", R"(
Handle to a context resource.
)");
}

void DefineContextResourceAttributes(ContextResourceCls& cls) {
  cls.def(
      "to_json",
      [](ResourceImplWeakPtr self, bool include_defaults) {
        return ValueOrThrow(
            self->spec_->ToJson(IncludeDefaults{include_defaults}));
      },
      py::arg("include_defaults") = false,
      R"(
Returns the :json:schema:`JSON representation<ContextResource>` of the context resource.

Example:

    >>> context = ts.Context(
    ...     {'cache_pool#a': {
    ...         'total_bytes_limit': 10000000
    ...     }})
    >>> context['cache_pool#a'].to_json()
    {'total_bytes_limit': 10000000}

Group:
  Accessors
)");

  cls.def("__repr__", [](ResourceImplWeakPtr self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self->spec_->ToJson(IncludeDefaults{false}), "Context.Resource(", ")");
  });

  EnablePicklingFromSerialization<ResourceImplWeakPtr>(
      cls, serialization::NonNullIndirectPointerSerializer<
               ResourceImplWeakPtr,
               internal_context::
                   UntypedContextResourceImplPtrNonNullDirectSerializer>{});
}

void RegisterContextBindings(pybind11::module_ m, Executor defer) {
  auto cls_context = MakeContextClass(m);
  defer([cls_context]() mutable { DefineContextAttributes(cls_context); });

  defer([cls = MakeContextSpecClass(cls_context)]() mutable {
    DefineContextSpecAttributes(cls);
  });

  defer([cls = MakeContextResourceClass(cls_context)]() mutable {
    DefineContextResourceAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterContextBindings, /*priority=*/-750);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
