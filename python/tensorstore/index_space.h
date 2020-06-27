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

#include <string>
#include <utility>

#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/indexing_spec.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/subscript_method.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
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

/// Marks `array` readonly.
pybind11::array MakeArrayReadonly(pybind11::array array);

/// Converts `labels` to a Python tuple object of strings.
pybind11::tuple GetLabelsTuple(span<const std::string> labels);

/// Converts `v` to a NumPy bool array.
pybind11::array GetBitVector(BitSpan<const std::uint64_t> v);

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

void RegisterIndexSpaceBindings(pybind11::module m);

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
void DefineIndexTransformOperations(pybind11::class_<T, ClassOptions...>* cls,
                                    GetTransform get_transform,
                                    ApplyTransform apply_transform,
                                    Assign... assign) {
  namespace py = ::pybind11;
  using Self = typename FunctionArgType<
      0, py::detail::function_signature_t<ApplyTransform>>::type;
  cls->def(
      "__getitem__",
      [get_transform, apply_transform](Self self,
                                       IndexTransform<> other_transform) {
        IndexTransform<> transform = get_transform(self);
        transform = ValueOrThrow(
            [&] {
              py::gil_scoped_release gil_release;
              return ComposeTransforms(transform, other_transform);
            }(),
            StatusExceptionPolicy::kIndexError);
        return apply_transform(std::move(self), std::move(transform));
      },
      "Applies an IndexTransform.", py::arg("transform"));
  cls->def(
      "__getitem__",
      [get_transform, apply_transform](Self self, IndexDomain<> other_domain) {
        IndexTransform<> transform = get_transform(self);
        transform = ValueOrThrow(other_domain(std::move(transform)),
                                 StatusExceptionPolicy::kIndexError);
        return apply_transform(std::move(self), std::move(transform));
      },
      "Slices by IndexDomain.", py::arg("domain"));

  // Defined as separate function rather than expanded inline within the `,`
  // fold expression below to work around MSVC 2019 ICE.
  [[maybe_unused]] const auto DefineSetItemOperations = [&](auto assign) {
    cls->def(
        "__setitem__",
        [get_transform, apply_transform, assign](
            Self self, IndexTransform<> other_transform,
            typename FunctionArgType<1, py::detail::function_signature_t<
                                            decltype(assign)>>::type source) {
          IndexTransform<> transform = get_transform(self);
          transform = ValueOrThrow(
              [&] {
                py::gil_scoped_release gil_release;
                return ComposeTransforms(transform, other_transform);
              }(),
              StatusExceptionPolicy::kIndexError);
          return assign(apply_transform(std::move(self), std::move(transform)),
                        source);
        },
        "Assigns to the result of applying an IndexTransform",
        py::arg("transform"), py::arg("source"));

    cls->def(
        "__setitem__",
        [get_transform, apply_transform, assign](
            Self self, const PythonDimExpression& expr,
            typename FunctionArgType<1, py::detail::function_signature_t<
                                            decltype(assign)>>::type source) {
          IndexTransform<> transform = get_transform(self);
          DimensionIndexBuffer dims;
          transform = ValueOrThrow(
              [&] {
                py::gil_scoped_release gil_release;
                return expr.Apply(std::move(transform), &dims);
              }(),
              StatusExceptionPolicy::kIndexError);
          return assign(apply_transform(std::move(self), std::move(transform)),
                        source);
        },
        "Assigns to the result of applying a DimExpression",
        py::arg("transform"), py::arg("source"));
  };

  (DefineSetItemOperations(assign), ...);

  cls->def(
      "__getitem__",
      [get_transform, apply_transform](Self self,
                                       const PythonDimExpression& expr) {
        IndexTransform<> transform = get_transform(self);
        DimensionIndexBuffer dims;
        transform = ValueOrThrow(
            [&] {
              py::gil_scoped_release gil_release;
              return expr.Apply(std::move(transform), &dims);
            }(),
            StatusExceptionPolicy::kIndexError);
        return apply_transform(std::move(self), std::move(transform));
      },
      "Applies a DimExpression.", py::arg("expr"));

  // Defined as separate function rather than expanded inline within parameter
  // list below to work around MSVC 2019 ICE.
  [[maybe_unused]] const auto DirectAssignMethod = [get_transform,
                                                    apply_transform](
                                                       auto assign) {
    return [=](Self self, IndexingSpec spec,
               typename FunctionArgType<
                   1, py::detail::function_signature_t<decltype(assign)>>::type
                   source) {
      IndexTransform<> transform = get_transform(self);
      transform = ValueOrThrow(
          [&] {
            py::gil_scoped_release gil_release;
            return ComposeTransforms(
                std::move(transform),
                ToIndexTransform(spec, transform.domain()));
          }(),
          StatusExceptionPolicy::kIndexError);
      return assign(apply_transform(std::move(self), std::move(transform)),
                    source);
    };
  };
  DefineIndexingMethods<IndexingSpec::Usage::kDirect>(
      cls,
      [get_transform, apply_transform](Self self, IndexingSpec spec) {
        IndexTransform<> transform = get_transform(self);
        transform = ValueOrThrow(
            [&] {
              py::gil_scoped_release gil_release;
              return ComposeTransforms(
                  std::move(transform),
                  ToIndexTransform(spec, transform.domain()));
            }(),
            StatusExceptionPolicy::kIndexError);
        return apply_transform(std::move(self), std::move(transform));
      },
      DirectAssignMethod(assign)...);
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_INDEX_SPACE_H_
