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

#ifndef THIRD_PARTY_PY_TENSORSTORE_SPEC_H_
#define THIRD_PARTY_PY_TENSORSTORE_SPEC_H_

/// \file
///
/// Defines `tensorstore.Spec` and `tensorstore.Schema`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/kvstore.h"
#include "python/tensorstore/sequence_parameter.h"
#include "python/tensorstore/unit.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_python {

struct PythonSpecObject
    : public GarbageCollectedPythonObject<PythonSpecObject, Spec> {
  constexpr static const char python_type_name[] = "tensorstore.Spec";
  ~PythonSpecObject() = delete;
};

using PythonSpec = PythonSpecObject::Handle;

/// Wrapper type used to indicate parameters that may be specified either as
/// `tensorstore.Spec` objects or json values.
struct SpecLike {
  Spec spec;
  PythonObjectReferenceManager reference_manager;
};

// Keyword argument ParamDef types for `Schema`
namespace schema_setters {

struct SetDtype {
  using type = tensorstore::internal_python::DataTypeLike;
  constexpr static const char* name = "dtype";
  constexpr static const char* doc = R"(

Constrains the data type of the TensorStore.  If a data type has already been
set, it is an error to specify a different data type.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(value.value);
  }
};

struct SetRank {
  using type = DimensionIndex;
  constexpr static const char* name = "rank";
  constexpr static const char* doc = R"(

Constrains the rank of the TensorStore.  If there is an index transform, the
rank constraint must match the rank of the *input* space.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(RankConstraint{value});
  }
};

struct SetDomain {
  using type = IndexDomain<>;
  constexpr static const char* name = "domain";
  constexpr static const char* doc = R"(

Constrains the domain of the TensorStore.  If there is an existing
domain, the specified domain is merged with it as follows:

1. The rank must match the existing rank.

2. All bounds must match, except that a finite or explicit bound is permitted to
   match an infinite and implicit bound, and takes precedence.

3. If both the new and existing domain specify non-empty labels for a dimension,
   the labels must be equal.  If only one of the domains specifies a non-empty
   label for a dimension, the non-empty label takes precedence.

Note that if there is an index transform, the domain must match the *input*
space, not the output space.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, const type& value) {
    return self.Set(value);
  }
};

struct SetShape {
  using type = SequenceParameter<Index>;
  constexpr static const char* name = "shape";
  constexpr static const char* doc = R"(

Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
:py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

.. note::

   This option also constrains the origin of all dimensions to be zero.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, const type& value) {
    return self.Set(Schema::Shape(value));
  }
};

struct SetCodec {
  using type = internal::IntrusivePtr<internal::CodecDriverSpec>;
  constexpr static const char* name = "codec";
  constexpr static const char* doc = R"(

Constrains the codec.  If there is an existing codec constraint, the constraints
are merged.  If the constraints are incompatible, an error is raised.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(CodecSpec(std::move(value)));
  }
};

struct SetChunkLayout {
  using type = ChunkLayout;
  constexpr static const char* name = "chunk_layout";
  constexpr static const char* doc = R"(

Constrains the chunk layout.  If there is an existing chunk layout constraint,
the constraints are merged.  If the constraints are incompatible, an error
is raised.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(std::move(value));
  }
};

struct SetSchema {
  using type = Schema;
  constexpr static const char* name = "schema";
  constexpr static const char* doc = R"(

Additional schema constraints to merge with existing constraints.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(std::move(value));
  }
};

struct SetFillValue {
  using type = ArrayArgumentPlaceholder;
  constexpr static const char* name = "fill_value";
  constexpr static const char* doc = R"(

Specifies the fill value for positions that have not been written.

The fill value data type must be convertible to the actual data type, and the
shape must be :ref:`broadcast-compatible<index-domain-alignment>` with the
domain.

If an existing fill value has already been set as a constraint, it is an
error to specify a different fill value (where the comparison is done after
normalization by broadcasting).

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    SharedArray<const void> array;
    internal_python::ConvertToArray(value.value, &array, self.dtype());
    return self.Set(Schema::FillValue(std::move(array)));
  }
};

struct SetDimensionUnits {
  using type = SequenceParameter<std::optional<UnitLike>>;
  constexpr static const char* name = "dimension_units";
  constexpr static const char* doc = R"(

Specifies the physical units of each dimension of the domain.

The *physical unit* for a dimension is the physical quantity corresponding to a
single index increment along each dimension.

A value of :python:`None` indicates that the unit is unknown.  A dimension-less
quantity can be indicated by a unit of :python:`""`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    const size_t size = value.size();
    std::vector<std::optional<Unit>> units(size);
    for (size_t i = 0; i < size; ++i) {
      auto& unit = value[i];
      if (!unit) continue;
      units[i] = std::move(unit->value);
    }
    return self.Set(Schema::DimensionUnits(units));
  }
};

}  // namespace schema_setters

constexpr auto WithSchemaKeywordArguments = [](auto callback,
                                               auto... other_param) {
  using namespace schema_setters;
  callback(other_param..., SetRank{}, SetDtype{}, SetDomain{}, SetShape{},
           SetChunkLayout{}, SetCodec{}, SetFillValue{}, SetDimensionUnits{},
           SetSchema{});
};

namespace spec_setters {

template <auto Mode>
struct SetModeBase {
  using type = bool;
  template <typename Self>
  static absl::Status Apply(Self& self, bool value) {
    if (!value) return absl::OkStatus();
    return self.Set(Mode);
  }
};

struct SetOpen : public SetModeBase<OpenMode::open> {
  static constexpr const char* name = "open";
  static constexpr const char* doc = R"(

Allow opening an existing TensorStore.  Overrides the existing open mode.

)";
};

struct SetCreate : public SetModeBase<OpenMode::create> {
  static constexpr const char* name = "create";
  static constexpr const char* doc = R"(

Allow creating a new TensorStore.  Overrides the existing open mode.  To open or
create, specify :python:`create=True` and :python:`open=True`.

)";
};

struct SetDeleteExisting : public SetModeBase<OpenMode::delete_existing> {
  static constexpr const char* name = "delete_existing";
  static constexpr const char* doc = R"(

Delete any existing data before creating a new array.  Overrides the existing
open mode.  Must be specified in conjunction with :python:`create=True`.

)";
};

struct SetAssumeMetadata : public SetModeBase<OpenMode::assume_metadata> {
  static constexpr const char* name = "assume_metadata";
  static constexpr const char* doc = R"(

Skip reading the metadata if possible.  Instead, just assume any necessary
metadata based on constraints in the spec, using the same defaults for any
unspecified metadata as when creating a new TensorStore.  Overrides the existing
open mode.  Requires that :py:param:`.open` is `True` and
:py:param:`.delete_existing` is `False`.

.. warning::

   This option can lead to data corruption if the assumed metadata does
   not match the stored metadata, or multiple concurrent writers use
   different assumed metadata.

.. seealso:

   - :ref:`python-open-assume-metadata`
)";
};

struct SetMinimalSpec {
  using type = bool;
  static constexpr const char* name = "minimal_spec";
  static constexpr const char* doc = R"(

Indicates whether to include in the returned :py:obj:`~tensorstore.Spec` the
metadata necessary to re-create the :py:obj:`~tensorstore.TensorStore`.  By
default, the returned :py:obj:`~tensorstore.Spec` includes the full metadata,
but it is skipped if :py:param:`.minimal_spec` is set to :python:`True`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, bool value) {
    return self.Set(tensorstore::MinimalSpec{value});
  }
};

template <ContextBindingMode Mode>
struct SetContextBindingModeBase {
  using type = bool;
  template <typename Self>
  static absl::Status Apply(Self& self, bool value) {
    if (!value) return absl::OkStatus();
    return self.Set(Mode);
  }
};

struct SetRetainContext
    : SetContextBindingModeBase<ContextBindingMode::retain> {
  static constexpr const char* name = "retain_context";
  static constexpr const char* doc = R"(

Retain all bound context resources (e.g. specific concurrency pools, specific
cache pools).

The resultant :py:obj:`~tensorstore.Spec` may be used to re-open the
:py:obj:`~tensorstore.TensorStore` using the identical context resources.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetUnbindContext
    : SetContextBindingModeBase<ContextBindingMode::unbind> {
  static constexpr const char* name = "unbind_context";
  static constexpr const char* doc = R"(

Convert any bound context resources to context resource specs that fully capture
the graph of shared context resources and interdependencies.

Re-binding/re-opening the resultant spec will result in a new graph of new
context resources that is isomorphic to the original graph of context resources.
The resultant spec will not refer to any external context resources;
consequently, binding it to any specific context will have the same effect as
binding it to a default context.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetStripContext : SetContextBindingModeBase<ContextBindingMode::strip> {
  static constexpr const char* name = "strip_context";
  static constexpr const char* doc = R"(

Replace any bound context resources and unbound context resource specs by
default context resource specs.

If the resultant :py:obj:`~tensorstore.Spec` is re-opened with, or re-bound to,
a new context, it will use the default context resources specified by that
context.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetContext {
  using type = internal_context::ContextImplPtr;
  static constexpr const char* name = "context";
  static constexpr const char* doc = R"(

Bind any context resource specs using the specified shared resource context.

Any already-bound context resources remain unchanged.  Additionally, any context
resources specified by a nested :json:schema:`TensorStore.context` spec will be
created as specified, but won't be overridden by :py:param:`.context`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(WrapImpl(std::move(value)));
  }
};

struct SetKvstore {
  using type = PythonKvStoreSpecObject*;
  static constexpr const char* name = "kvstore";
  static constexpr const char* doc = R"(

Sets the associated key-value store used as the underlying storage.

If the :py:obj:`~tensorstore.Spec.kvstore` has already been set, it is
overridden.

It is an error to specify this if the TensorStore driver does not use a
key-value store.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(value->value);
  }
};

}  // namespace spec_setters

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from compatible Python objects to
/// `tensorstore::Spec` parameters of pybind11-exposed functions, via JSON
/// conversion.
template <>
struct type_caster<tensorstore::internal_python::SpecLike> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::SpecLike,
                       _("Union[tensorstore.Spec, Any]"));
  bool load(handle src, bool convert);
};

template <>
struct type_caster<tensorstore::internal_python::PythonSpecObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonSpecObject> {};

template <>
struct type_caster<tensorstore::Spec>
    : public tensorstore::internal_python::GarbageCollectedObjectCaster<
          tensorstore::internal_python::PythonSpecObject> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_SPEC_H_
