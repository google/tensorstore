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

#include <string>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/keyword_arguments.h"
#include "pybind11/pybind11.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_python {

void RegisterSpecBindings(pybind11::module m);

/// Wrapper type used to indicate parameters that may be specified either as
/// `tensorstore.Spec` objects or json values.
struct SpecLike {
  Spec value;
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
  using type = internal::IntrusivePtr<CodecSpec>;
  constexpr static const char* name = "codec";
  constexpr static const char* doc = R"(

Constrains the codec.  If there is an existing codec constraint, the constraints
are merged.  If the constraints are incompatible, an error is raised.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(CodecSpec::Ptr(std::move(value)));
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
    internal_python::ConvertToArray(value.obj, &array, self.dtype());
    return self.Set(Schema::FillValue(std::move(array)));
  }
};

}  // namespace schema_setters

constexpr auto WithSchemaKeywordArguments = [](auto callback,
                                               auto... other_param) {
  using namespace schema_setters;
  callback(other_param..., SetRank{}, SetDtype{}, SetDomain{}, SetShape{},
           SetChunkLayout{}, SetCodec{}, SetFillValue{}, SetSchema{});
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
  static handle cast(tensorstore::internal_python::SpecLike value,
                     return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_SPEC_H_
