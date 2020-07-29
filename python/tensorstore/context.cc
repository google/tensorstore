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

#include "python/tensorstore/context.h"

#include <optional>
#include <utility>

#include "python/tensorstore/intrusive_ptr_holder.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/context.h"
#include "tensorstore/context_impl.h"
#include "tensorstore/internal/json_pprint_python.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

using internal::ContextSpecBuilder;
using internal_context::Access;
using internal_context::BuilderResourceSpec;
using internal_context::ContextImpl;
using internal_context::ContextImplPtr;
using internal_context::ContextResourceContainer;
using internal_context::ContextResourceImplBase;
using internal_context::ContextResourceImplWeakPtr;
using internal_context::ContextSpecImpl;
using internal_context::ContextSpecImplPtr;

[[noreturn]] void ThrowCorruptContextPickle() {
  throw py::value_error(
      "Corrupt pickle representation for tensorstore.Context");
}

/// Pickles a `Context` object.
///
/// Returns a tuple of the form:
///
/// `(parent, resources, spec_key0, spec_value0, spec_key1, spec_value1, ...)`
///
/// where:
///
/// `parent` is a Python `tensorstore.Context` object referring to the parent,
/// or `None` if there is no pointer;
///
/// `resources` is a tuple of the form `(key0, resource0, key1, resource1, ...)`
/// compatible with `UnpickleContextSpecBuilder` containing the resources of
/// `self` that have been successfully loaded, where `keyN` is a Python `str`
/// object and `resourceN` is a `tensorstore._ContextResource` object referring
/// to the resource;
///
/// `spec_keyN` is a Python `str` and `spec_valueN` is a Python JSON object.
///
/// The `spec_keyN` and `spec_valueN` values specify the remaining resources in
/// `self->spec_` that have not been successfully loaded.
constexpr auto PickleContext = [](ContextImplPtr self) -> py::tuple {
  py::object parent_obj;
  if (self->parent_) {
    parent_obj = py::cast(self->parent_);
  } else {
    parent_obj = py::none();
  }

  // Set of successfully loaded resources in `context`.  These are
  // pickled as `ContextResource` objects directly, in order to take
  // advantage of the deduplication built into Python's pickling
  // mechanism.
  struct ResourceToPickle {
    ContextResourceImplWeakPtr resource;
    bool exclude_from_spec;
  };
  absl::flat_hash_map<std::string_view, ResourceToPickle> resources;
  size_t num_unloaded_resources =
      self->spec_ ? self->spec_->resources_.size() : 0;
  {
    // First collect the resources in a separate map with the mutex held.  This
    // way we avoid invoking Python code with the mutex held.
    auto& mutex = self->root_->mutex_;
    absl::MutexLock lock(&mutex);
    resources.reserve(self->resources_.size());
    for (auto& container : self->resources_) {
      // Skip resources that are still loading since they can't be
      // referenced by anything else being pickled anyway.
      if (!container->result_.ok() || !container->result_->get()) {
        continue;
      }
      ContextResourceImplBase* resource = container->result_->get();
      // Skip resources that merely refer to other resources.  These
      // are guaranteed to be pickled separately.
      if (resource->spec_ != container->spec_) continue;

      std::string_view key = container->spec_->key_;
      bool exclude_from_spec = false;
      auto it = self->spec_->resources_.find(key);
      if (it == self->spec_->resources_.end()) {
        // `resource` was not defined in the spec.  This can only happen in the
        // case of a default resource accessed by the provider_id.
        assert(key == container->spec_->provider_->id_);
        assert(container->spec_->is_default_);
        exclude_from_spec = true;
      } else {
        assert(num_unloaded_resources > 0);
        --num_unloaded_resources;
      }
      resources.emplace(key,
                        ResourceToPickle{ContextResourceImplWeakPtr(resource),
                                         exclude_from_spec});
    }
  }

  // Encode `resources` in the same representation used by
  // `PickleContextSpecBuilder`.
  py::tuple pickled_resources(resources.size() * 2);
  {
    size_t i = 0;
    for (auto& [key, resource_to_pickle] : resources) {
      if (resource_to_pickle.exclude_from_spec) {
        // Store `None` in place of the key to indicate this resource should not
        // be added to the spec.
        pickled_resources[i++] = py::none();
      } else {
        // Store key as Python `str`.
        pickled_resources[i++] = py::cast(key);
      }
      // Store a Python `tensorstore._ContextResource` object that refers to
      // `resource`.  This causes Python to in turn pickle each unique
      // `_ContextResource` object once.
      pickled_resources[i++] = py::cast(resource_to_pickle.resource);
    }
  }

  // Pickle the JSON specs for any resources that have not yet been
  // loaded.
  py::tuple t(2 + num_unloaded_resources * 2);
  t[0] = parent_obj;
  t[1] = pickled_resources;
  if (self->spec_) {
    size_t i = 2;
    for (auto& resource : self->spec_->resources_) {
      // Only store this resource spec if the corresponding loaded resource is
      // not already included in `pickled_resources`.
      if (resources.count(resource->key_)) continue;
      // Store the key as a Python `str`.
      t[i++] = py::cast(resource->key_);
      auto json_spec = ValueOrThrow(resource->ToJson({}));
      // Store the spec as a Python JSON object.
      t[i++] = py::cast(json_spec);
    }
    TENSORSTORE_CHECK(i == t.size());
  }
  return t;
};

constexpr auto UnpickleContext = [](py::tuple t) -> ContextImplPtr {
  ContextImplPtr parent;
  if (auto parent_obj = t[0]; !parent_obj.is_none()) {
    parent = py::cast<ContextImplPtr>(parent_obj);
  }
  ContextImplPtr context =
      UnpickleContextSpecBuilder(py::cast<py::tuple>(t[1]));
  if (parent) {
    context->root_ = parent->root_;
    context->parent_ = std::move(parent);
  }
  // Unpickle the resources that were not loaded at the time of
  // pickling and were stored by JSON spec.
  for (size_t i = 2; i < t.size(); i += 2) {
    std::string key = py::cast<std::string>(t[i]);
    ::nlohmann::json json_spec = py::cast<::nlohmann::json>(t[i + 1]);
    auto spec =
        ValueOrThrow(internal_context::ContextResourceSpecFromJsonWithKey(
            key, json_spec, {}));
    if (!context->spec_->resources_.emplace(spec).second) {
      ThrowCorruptContextPickle();
    }
  }
  return context;
};

constexpr auto PickleContextResource =
    [](ContextResourceImplBase* self) -> py::tuple {
  auto builder = ContextSpecBuilder::Make();
  auto spec = self->spec_->provider_->GetSpec(self, builder);
  py::object pickled_context = PickleContextSpecBuilder(std::move(builder));
  // Pickle the resource spec itself, along with any resources it
  // depends on.
  return py::make_tuple(py::cast(spec->provider_->id_), py::cast(spec->key_),
                        py::cast(spec->is_default_),
                        py::cast(ValueOrThrow(spec->ToJson({}))),
                        pickled_context);
};

constexpr auto UnpickleContextResource =
    [](py::tuple t) -> ContextResourceImplWeakPtr {
  if (t.size() != 5) ThrowCorruptContextPickle();
  std::string provider_id = py::cast<std::string>(t[0]);
  std::string key = py::cast<std::string>(t[1]);
  bool is_default = py::cast<bool>(t[2]);
  auto json_spec = py::cast<::nlohmann::json>(t[3]);
  // Unpickle any resources it depends on.
  auto context_impl =
      UnpickleContextSpecBuilder(py::cast<py::tuple>(t[4]), false);
  if (!key.empty() &&
      internal_context::ParseResourceProvider(key) != provider_id) {
    ThrowCorruptContextPickle();
  }
  auto* provider = internal_context::GetProvider(provider_id);
  if (!provider) {
    ThrowStatusException(
        internal_context::ProviderNotRegisteredError(provider_id));
  }
  // Unpickle the resource spec itself.
  auto resource_spec =
      ValueOrThrow(internal_context::ContextResourceSpecFromJson(
          *provider, std::move(json_spec), {}));
  resource_spec->is_default_ = is_default;
  auto resource = ValueOrThrow(internal_context::GetResource(
      context_impl.get(), resource_spec.get(), nullptr));
  // Don't set key until after getting the resource, since
  // `GetResource` expects any `resource_spec` with a key to be
  // internal to the same `context_impl`.
  resource_spec->key_ = std::move(key);
  return resource;
};

}  // namespace

py::tuple PickleContextSpecBuilder(ContextSpecBuilder builder) {
  std::vector<std::pair<ContextResourceImplWeakPtr,
                        internal::IntrusivePtr<BuilderResourceSpec>>>
      deps;
  auto& resources = Access::impl(builder)->resources_;
  deps.reserve(resources.size());
  for (auto& [resource, entry] : resources) {
    deps.emplace_back(resource, entry.spec);
    entry.shared = true;
  }
  TENSORSTORE_CHECK(Access::impl(builder)->use_count() == 1);
  // Rely on builder's destructor to update all of the spec keys in
  // `deps`.
  builder = ContextSpecBuilder();
  py::tuple t(deps.size() * 2);
  for (size_t i = 0; i < deps.size(); ++i) {
    auto& [dep_resource, dep_spec] = deps[i];
    t[2 * i] = py::cast(dep_spec->underlying_spec_->key_);
    t[1 + 2 * i] = py::cast(dep_resource);
  }
  return t;
}

ContextImplPtr UnpickleContextSpecBuilder(py::tuple t,
                                          bool allow_key_mismatch) {
  if (t.size() % 2 != 0) {
    ThrowCorruptContextPickle();
  }
  ContextImplPtr context_impl(new ContextImpl);
  context_impl->spec_.reset(new ContextSpecImpl);
  context_impl->root_ = context_impl.get();
  for (size_t i = 0; i < t.size(); i += 2) {
    py::object key_obj = t[i];
    bool exclude_from_spec;
    std::string dep_key;
    auto resource = py::cast<ContextResourceImplWeakPtr>(t[i + 1]);
    if (!resource) ThrowCorruptContextPickle();
    if (key_obj.is_none()) {
      exclude_from_spec = true;
      if (!resource->spec_->is_default_) {
        ThrowCorruptContextPickle();
      }
      dep_key = resource->spec_->provider_->id_;
    } else {
      exclude_from_spec = false;
      dep_key = py::cast<std::string>(key_obj);
      if (resource->spec_->provider_->id_ !=
          internal_context::ParseResourceProvider(dep_key)) {
        ThrowCorruptContextPickle();
      }
    }
    auto container = std::make_unique<ContextResourceContainer>();
    if (resource->spec_->key_ != dep_key) {
      // Wrap the spec in a `BuilderResourceSpec` in order to allow it to be
      // stored under a different key.  When unpickling a `Context` object that
      // will be exposed, this should not occur except in the case of a corrupt
      // pickle representation.
      if (!allow_key_mismatch) ThrowCorruptContextPickle();
      container->spec_.reset(new BuilderResourceSpec);
      container->spec_->provider_ = resource->spec_->provider_;
      container->spec_->key_ = std::move(dep_key);
      static_cast<BuilderResourceSpec&>(*container->spec_).underlying_spec_ =
          resource->spec_;
    } else {
      container->spec_ = resource->spec_;
    }
    container->result_.emplace(resource.get());
    if (!exclude_from_spec &&
        !context_impl->spec_->resources_.emplace(container->spec_).second) {
      // Keys are not unique.
      ThrowCorruptContextPickle();
    }
    if (!context_impl->resources_.emplace(std::move(container)).second) {
      // Keys are not unique.
      ThrowCorruptContextPickle();
    }
  }
  return context_impl;
}

void RegisterContextBindings(pybind11::module m) {
  py::class_<ContextSpecImpl, ContextSpecImplPtr> cls_context_spec(
      m, "ContextSpec");
  cls_context_spec
      .def(py::init([](const ::nlohmann::json& json, bool allow_unregistered) {
             return Access::impl(ValueOrThrow(Context::Spec::FromJson(
                 json, AllowUnregistered{allow_unregistered})));
           }),
           "Creates a ContextSpec from a JSON object.", py::arg("json"),
           py::arg("allow_unregistered") = false)
      .def(
          "to_json",
          [](internal_context::ContextSpecImplPtr self, bool include_defaults) {
            return WrapImpl(std::move(self))
                .ToJson(IncludeDefaults{include_defaults});
          },
          py::arg("include_defaults") = false)
      .def("__repr__",
           [](internal_context::ContextSpecImplPtr self) {
             return internal_python::PrettyPrintJsonAsPythonRepr(
                 WrapImpl(std::move(self)).ToJson(IncludeDefaults{false}),
                 "ContextSpec(", ")");
           })
      .def(py::pickle(
          [](ContextSpecImplPtr self) {
            // Just pickles the JSON representation.
            return py::make_tuple(py::cast(internal_python::ValueOrThrow(
                WrapImpl(std::move(self)).ToJson(IncludeDefaults{false}))));
          },
          [](py::tuple t) {
            return Access::impl(internal_python::ValueOrThrow(
                Context::Spec::FromJson(py::cast<::nlohmann::json>(t[0]))));
          }));

  py::class_<ContextImpl, ContextImplPtr> cls_context(m, "Context",
                                                      R"(
Manages shared TensorStore resources, such as caches and credentials.
)");
  cls_context
      .def(py::init([] { return Access::impl(Context::Default()); }),
           "Returns a default context")
      .def(py::init([](ContextSpecImplPtr spec, ContextImplPtr parent) {
             return Access::impl(Context(WrapImpl(std::move(spec)),
                                         WrapImpl(std::move(parent))));
           }),
           "Constructs a context.", py::arg("spec"),
           py::arg("parent") = nullptr)
      .def(py::init([](::nlohmann::json json, ContextImplPtr parent,
                       bool allow_unregistered) {
             return Access::impl(
                 Context(ValueOrThrow(Context::Spec::FromJson(
                             json, AllowUnregistered{allow_unregistered})),
                         WrapImpl(std::move(parent))));
           }),
           "Constructs a context from a JSON spec.", py::arg("json"),
           py::arg("parent") = nullptr, py::arg("allow_unregistered") = false)
      .def_property_readonly(
          "parent", [](const ContextImpl& self) { return self.parent_; })
      .def_property_readonly("spec",
                             [](const ContextImpl& self) { return self.spec_; })
      .def(
          "__getitem__",
          [](ContextImplPtr self, std::string key) {
            auto provider_id = internal_context::ParseResourceProvider(key);
            auto* provider = internal_context::GetProvider(provider_id);
            if (!provider) {
              ThrowStatusException(
                  internal_context::ProviderNotRegisteredError(provider_id));
            }
            auto spec =
                ValueOrThrow(internal_context::ContextResourceSpecFromJson(
                    provider_id, key, {}));
            return ValueOrThrow(
                internal_context::GetResource(self.get(), spec.get(), nullptr));
          },
          py::arg("key"))
      .def(py::pickle(PickleContext, UnpickleContext));

  // `ContextResourceImplBase` represents a context resource.  It is not part of
  // the public API, and is exposed only for pickling and testing.  There isn't
  // a whole lot that can be done with objects of this type, though their
  // identity can be compared.
  py::class_<ContextResourceImplBase, ContextResourceImplWeakPtr>
      cls_context_resource(m, "_ContextResource");
  cls_context_resource
      .def(
          "to_json",
          [](ContextResourceImplWeakPtr self, bool include_defaults) {
            return ValueOrThrow(
                self->spec_->ToJson(IncludeDefaults{include_defaults}));
          },
          py::arg("include_defaults") = false,
          "Returns the JSON representation of the context resource.")
      .def("__repr__",
           [](ContextResourceImplWeakPtr self) {
             return internal_python::PrettyPrintJsonAsPythonRepr(
                 self->spec_->ToJson(IncludeDefaults{false}),
                 "_ContextResource(", ")");
           })
      .def(py::pickle(PickleContextResource, UnpickleContextResource));
}

}  // namespace internal_python
}  // namespace tensorstore
