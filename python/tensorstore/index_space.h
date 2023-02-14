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

#ifndef THIRD_PARTY_PY_TENSORSTORE_INDEX_SPACE_H_
#define THIRD_PARTY_PY_TENSORSTORE_INDEX_SPACE_H_

/// \file
///
/// Defines `tensorstore.IndexTransform`, `tensorstore.IndexDomain`,
/// `tensorstore.Dim` (corresponding to `tensorstore::IndexDomainDimension<>`),
/// `tensorstore.d` (for specifying dimension expressions),
/// `tensorstore.newaxis`, and `tensorstore.inf`.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <utility>

#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/homogeneous_tuple.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/numpy_indexing_spec.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/subscript_method.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_python {

/// Converts `v` to a homogeneous tuple of bool.
HomogeneousTuple<bool> GetBitVector(BitSpan<const std::uint64_t> v);

/// Represents a standalone `OutputIndexMap`, for use in initializing an
/// `IndexTransform`.
struct OutputIndexMap {
  OutputIndexMap()
      : method(OutputIndexMethod::constant),
        offset(0),
        stride(0),
        input_dimension(-1) {}
  OutputIndexMap(OutputIndexMapRef<> r);
  OutputIndexMethod method;
  Index offset;
  Index stride;
  DimensionIndex input_dimension;
  SharedArray<const Index> index_array;
  IndexInterval index_range;

  friend bool operator==(const OutputIndexMap& a, const OutputIndexMap& b);
  friend bool operator!=(const OutputIndexMap& a, const OutputIndexMap& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OutputIndexMap& m) {
    switch (m.method) {
      case OutputIndexMethod::constant:
        return H::combine(std::move(h), m.method, m.offset);
      case OutputIndexMethod::single_input_dimension:
        return H::combine(std::move(h), m.method, m.offset, m.stride,
                          m.input_dimension);
      case OutputIndexMethod::array:
        return H::combine(std::move(h), m.method, m.offset, m.stride,
                          m.index_array, m.index_range);
    }
  }
};

template <size_t NumAssign>
struct IndexTransformOperationDocstrings {
  NumpyIndexingMethodDocstrings<NumAssign> numpy_indexing;
  IndexingOperationDocstrings<NumAssign> index_transform;
  IndexingOperationDocstrings<NumAssign> index_domain;
  IndexingOperationDocstrings<NumAssign> dim_expression;
};

/// \param modify_transform Function with signature
///     `IndexTransform<> (IndexTransform<> orig, SubscriptType subscript)`.
template <typename SubscriptType, typename T, typename... ClassOptions,
          typename GetTransform, typename ModifyTransform,
          typename ApplyTransform, typename... Assign>
void DefineIndexingMethods(
    pybind11::class_<T, ClassOptions...>* cls, const char* subscript_name,
    IndexingOperationDocstrings<sizeof...(Assign)> doc_strings,
    GetTransform get_transform, ModifyTransform modify_transform,
    ApplyTransform apply_transform, Assign... assign) {
  namespace py = ::pybind11;
  using Self = typename FunctionArgType<
      0, py::detail::function_signature_t<ApplyTransform>>::type;

  cls->def(
      "__getitem__",
      [get_transform, modify_transform, apply_transform](
          Self self, SubscriptType subscript) {
        auto transform =
            modify_transform(get_transform(self), std::move(subscript));
        return apply_transform(std::forward<Self>(self), std::move(transform));
      },
      doc_strings[0], py::arg(subscript_name));

  size_t doc_string_index = 1;

  // Defined as separate function rather than expanded inline within the `,`
  // fold expression below to work around MSVC 2019 ICE.
  [[maybe_unused]] const auto DefineAssignMethod = [&](auto assign) {
    cls->def(
        "__setitem__",
        [get_transform, modify_transform, apply_transform, assign](
            Self self, SubscriptType subscript,
            typename FunctionArgType<1, py::detail::function_signature_t<
                                            decltype(assign)>>::type source) {
          auto transform =
              modify_transform(get_transform(self), std::move(subscript));
          return assign(
              apply_transform(std::forward<Self>(self), std::move(transform)),
              source);
        },
        doc_strings[doc_string_index++], py::arg("transform"),
        py::arg("source"));
  };

  (DefineAssignMethod(assign), ...);
}

/// Defines the common indexing operations supported by all
/// `tensorstore.Indexable` types.
///
/// \param cls The pybind11 class for which to define the indexing operations.
/// \param get_transform Function with signature `IndexTransform<> (Self self)`
///     that returns the index transform associated with `self`.
/// \param apply_transform Function with signature
///     `Self (Self self, IndexTransform<> new_transform)` that returns a copy
///     of `self` with its transform replaced by `new_transform`.
/// \param assign... Zero or more functions with signatures
///     `(Self self, IndexTransform<> new_transform, Source source)` to be
///     exposed as `__setitem__` overloads.
template <typename T, typename... ClassOptions, typename GetTransform,
          typename ApplyTransform, typename... Assign>
void DefineIndexTransformOperations(
    pybind11::class_<T, ClassOptions...>* cls,
    IndexTransformOperationDocstrings<sizeof...(Assign)> doc_strings,
    GetTransform get_transform, ApplyTransform apply_transform,
    Assign... assign) {
  namespace py = ::pybind11;
  using Self = typename FunctionArgType<
      0, py::detail::function_signature_t<ApplyTransform>>::type;

  DefineIndexingMethods<IndexTransform<>>(
      cls, "transform", doc_strings.index_transform,
      get_transform, /*modify_transform=*/
      [](IndexTransform<> transform, IndexTransform<> other_transform) {
        return ValueOrThrow(
            [&] {
              GilScopedRelease gil_release;
              return ComposeTransforms(transform, other_transform);
            }(),
            StatusExceptionPolicy::kIndexError);
      },
      apply_transform, assign...);
  DefineIndexingMethods<IndexDomain<>>(
      cls, "domain", doc_strings.index_domain,
      get_transform, /*modify_transform=*/
      [](IndexTransform<> transform, IndexDomain<> other_domain) {
        return ValueOrThrow(other_domain(std::move(transform)),
                            StatusExceptionPolicy::kIndexError);
      },
      apply_transform, assign...);

  DefineIndexingMethods<const PythonDimExpression&>(
      cls, "expr", doc_strings.dim_expression,
      get_transform, /*modify_transform=*/
      [](IndexTransform<> transform, const PythonDimExpression& expr) {
        return ValueOrThrow(
            [&] {
              GilScopedRelease gil_release;
              DimensionIndexBuffer dims;
              return expr.Apply(std::move(transform), &dims,
                                /*top_level=*/true, /*domain_only=*/false);
            }(),
            StatusExceptionPolicy::kIndexError);
      },
      apply_transform, assign...);

  // Defined as separate function rather than expanded inline within parameter
  // list below to work around MSVC 2019 ICE.
  [[maybe_unused]] const auto DirectAssignMethod = [get_transform,
                                                    apply_transform](
                                                       auto assign) {
    return [get_transform, apply_transform, assign](
               Self self, NumpyIndexingSpecPlaceholder spec_placeholder,
               typename FunctionArgType<
                   1, py::detail::function_signature_t<decltype(assign)>>::type
                   source) {
      IndexTransform<> transform = get_transform(self);
      transform = ValueOrThrow(
          [&]() -> Result<IndexTransform<>> {
            auto spec =
                spec_placeholder.Parse(NumpyIndexingSpec::Usage::kDirect);
            GilScopedRelease gil_release;
            TENSORSTORE_ASSIGN_OR_RETURN(
                auto spec_transform,
                ToIndexTransform(spec, transform.domain()));
            return ComposeTransforms(std::move(transform),
                                     std::move(spec_transform));
          }(),
          StatusExceptionPolicy::kIndexError);
      return assign(
          apply_transform(std::forward<Self>(self), std::move(transform)),
          source);
    };
  };
  DefineNumpyIndexingMethods(
      cls, doc_strings.numpy_indexing,
      [get_transform, apply_transform](
          Self self, NumpyIndexingSpecPlaceholder spec_placeholder) {
        IndexTransform<> transform = get_transform(self);
        transform = ValueOrThrow(
            [&]() -> Result<IndexTransform<>> {
              auto spec =
                  spec_placeholder.Parse(NumpyIndexingSpec::Usage::kDirect);
              GilScopedRelease gil_release;
              TENSORSTORE_ASSIGN_OR_RETURN(
                  auto spec_transform,
                  ToIndexTransform(spec, transform.domain()));
              return ComposeTransforms(std::move(transform),
                                       std::move(spec_transform));
            }(),
            StatusExceptionPolicy::kIndexError);
        return apply_transform(std::forward<Self>(self), std::move(transform));
      },
      DirectAssignMethod(assign)...);

  cls->def_property_readonly(
      "T",
      [get_transform, apply_transform](Self self) {
        IndexTransform<> transform = get_transform(self);
        const DimensionIndex rank = transform.input_rank();
        DimensionIndexBuffer reversed_dims(rank);
        for (DimensionIndex i = 0; i < rank; ++i) {
          reversed_dims[i] = rank - 1 - i;
        }
        return apply_transform(
            std::forward<Self>(self),
            ValueOrThrow(std::move(transform) |
                         tensorstore::Dims(reversed_dims).Transpose()));
      },
      R"(View with transposed domain (reversed dimension order).

This is equivalent to: :python:`self[ts.d[::-1].transpose[:]]`.

Group:
  Indexing

)");
  cls->def_property_readonly(
      "origin",
      [get_transform](const Self& self) {
        return SpanToHomogeneousTuple<Index>(
            get_transform(self).input_origin());
      },
      R"(Inclusive lower bound of the domain.

This is equivalent to :python:`self.domain.origin`.

Group:
  Accessors

)");
  cls->def_property_readonly(
      "shape",
      [get_transform](const Self& self) {
        return SpanToHomogeneousTuple<Index>(get_transform(self).input_shape());
      },
      R"(Shape of the domain.

This is equivalent to :python:`self.domain.shape`.

Group:
  Accessors

)");
  cls->def_property_readonly(
      "size",
      [get_transform](const Self& self) {
        return get_transform(self).domain().num_elements();
      },
      R"(Total number of elements in the domain.

This is equivalent to :python:`self.domain.size`.

Group:
  Accessors

)");
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_INDEX_SPACE_H_
