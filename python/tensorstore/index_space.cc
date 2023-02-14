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

#include "python/tensorstore/numpy.h"
// numpy.h must be included first.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <algorithm>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include <nlohmann/json.hpp>
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/homogeneous_tuple.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/numpy_indexing_spec.h"
#include "python/tensorstore/python_imports.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/sequence_parameter.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

bool operator==(const OutputIndexMap& a, const OutputIndexMap& b) {
  if (a.method != b.method || a.offset != b.offset) return false;
  switch (a.method) {
    case OutputIndexMethod::constant:
      return true;
    case OutputIndexMethod::single_input_dimension:
      return a.stride == b.stride && a.input_dimension == b.input_dimension;
    case OutputIndexMethod::array:
      return a.stride == b.stride && a.index_array == b.index_array &&
             a.index_range == b.index_range;
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

HomogeneousTuple<Index> GetExclusiveMax(IndexDomainView<> domain) {
  const DimensionIndex rank = domain.rank();
  Index temp[kMaxRank];
  for (DimensionIndex i = 0; i < rank; ++i) {
    temp[i] = domain[i].exclusive_max();
  }
  return SpanToHomogeneousTuple<Index>({temp, rank});
}

HomogeneousTuple<Index> GetInclusiveMax(IndexDomainView<> domain) {
  const DimensionIndex rank = domain.rank();
  Index temp[kMaxRank];
  for (DimensionIndex i = 0; i < rank; ++i) {
    temp[i] = domain[i].inclusive_max();
  }
  return SpanToHomogeneousTuple<Index>({temp, rank});
}

HomogeneousTuple<bool> GetBitVector(DimensionSet v, DimensionIndex size) {
  py::tuple t(size);
  for (DimensionIndex i = 0; i < size; ++i) {
    t[i] = py::reinterpret_borrow<py::object>(v[i] ? Py_True : Py_False);
  }
  return HomogeneousTuple<bool>{std::move(t)};
}

OutputIndexMap::OutputIndexMap(OutputIndexMapRef<> r)
    : method(r.method()), offset(r.offset()), stride(r.stride()) {
  switch (r.method()) {
    case OutputIndexMethod::constant:
      input_dimension = -1;
      break;
    case OutputIndexMethod::single_input_dimension:
      input_dimension = r.input_dimension();
      break;
    case OutputIndexMethod::array: {
      input_dimension = -1;
      auto index_array = r.index_array();
      const DimensionIndex input_rank = index_array.rank();
      this->index_array.layout().set_rank(index_array.rank());
      for (DimensionIndex i = 0; i < input_rank; ++i) {
        Index byte_stride = index_array.byte_strides()[i];
        Index size = index_array.layout().shape()[i];
        if (byte_stride == 0 && size > 1) size = 1;
        if (size <= 1) byte_stride = 0;
        this->index_array.shape()[i] = size;
        this->index_array.byte_strides()[i] = byte_stride;
      }
      this->index_array.element_pointer() =
          AddByteOffset(index_array.element_pointer(),
                        index_array.layout().origin_byte_offset());
      this->index_range = index_array.index_range();
    } break;
  }
}

namespace {

DimensionIndex NormalizePythonDimensionIndex(PythonDimensionIndex i,
                                             DimensionIndex size) {
  if (i.value < -size || i.value >= size) {
    throw py::index_error(tensorstore::StrCat("Index ", i.value,
                                              " is outside valid range [",
                                              -size, ", ", size, ")"));
  }
  if (i.value < 0) i.value += size;
  return i.value;
}

/// Returns an `IndexTransformBuilder` with the domain set from the specified
/// arguments.
///
/// The `<field>_field_name` arguments specify the parameter name corresponding
/// to the `<field>` parameter for use in error messages.
///
/// If `output_rank` is `std::nullopt`, the output rank is equal to the input
/// rank.
IndexTransformBuilder<> InitializeIndexTransformBuilder(
    std::optional<DimensionIndex> input_rank, const char* input_rank_field_name,
    const std::optional<SequenceParameter<Index>>& input_inclusive_min,
    const char* input_inclusive_min_field_name,
    const std::optional<SequenceParameter<bool>>& implicit_lower_bounds,
    const std::optional<SequenceParameter<Index>>& input_exclusive_max,
    const char* input_exclusive_max_field_name,
    const std::optional<SequenceParameter<Index>>& input_inclusive_max,
    const char* input_inclusive_max_field_name,
    const std::optional<SequenceParameter<Index>>& input_shape,
    const char* input_shape_field_name,
    const std::optional<SequenceParameter<bool>>& implicit_upper_bounds,
    const std::optional<SequenceParameter<std::optional<std::string>>>&
        input_labels,
    const char* input_labels_field_name,
    std::optional<DimensionIndex> output_rank) {
  const char* input_rank_field = nullptr;
  if (input_rank) {
    if (!IsValidRank(*input_rank)) {
      throw py::value_error(tensorstore::StrCat(
          "Invalid ", input_rank_field_name, ": ", *input_rank));
    }
    input_rank_field = input_rank_field_name;
  }

  const auto check_rank = [&](DimensionIndex rank, const char* field_name) {
    if (!input_rank) {
      if (!IsValidRank(rank)) {
        throw py::value_error(
            tensorstore::StrCat("Rank specified by `", field_name, "` (", rank,
                                ") exceeds maximum rank of ", kMaxRank));
      }
      input_rank = rank;
      input_rank_field = field_name;
    } else if (*input_rank != rank) {
      throw py::value_error(
          tensorstore::StrCat("Rank specified by `", field_name, "` (", rank,
                              ") does not match rank specified by `",
                              input_rank_field, "` (", *input_rank, ")"));
    }
  };
  if (input_inclusive_min) {
    check_rank(input_inclusive_min->size(), input_inclusive_min_field_name);
  }
  if (implicit_lower_bounds) {
    check_rank(implicit_lower_bounds->size(), "implicit_lower_bounds");
  }
  const char* upper_bound_field = nullptr;
  const auto check_upper_bound = [&](DimensionIndex rank,
                                     const char* field_name) {
    if (upper_bound_field) {
      throw py::value_error(tensorstore::StrCat("Cannot specify both `",
                                                upper_bound_field, "` and `",
                                                field_name, "`"));
    } else {
      upper_bound_field = field_name;
    }
    check_rank(rank, field_name);
  };
  if (input_exclusive_max) {
    check_upper_bound(input_exclusive_max->size(),
                      input_exclusive_max_field_name);
  }
  if (input_inclusive_max) {
    check_upper_bound(input_inclusive_max->size(),
                      input_inclusive_max_field_name);
  }
  if (input_shape) {
    check_upper_bound(input_shape->size(), input_shape_field_name);
  }
  if (implicit_upper_bounds) {
    check_rank(implicit_upper_bounds->size(), "implicit_upper_bounds");
  }
  if (input_labels) {
    check_rank(input_labels->size(), input_labels_field_name);
  }
  if (!input_rank) {
    throw py::value_error(
        tensorstore::StrCat("Must specify `", input_rank_field_name, "`"));
  }
  if (output_rank && !IsValidRank(*output_rank)) {
    throw py::value_error(
        tensorstore::StrCat("Number of output dimensions (", *output_rank,
                            ") exceeds maximum rank of ", kMaxRank));
  }
  auto builder =
      IndexTransformBuilder<>(*input_rank, output_rank.value_or(*input_rank));
  if (input_inclusive_min) {
    builder.input_origin(*input_inclusive_min);
  }
  if (implicit_lower_bounds) {
    builder.implicit_lower_bounds(
        DimensionSet::FromRange(*implicit_lower_bounds));
  }
  if (input_exclusive_max) {
    builder.input_exclusive_max(*input_exclusive_max);
  }
  if (input_inclusive_max) {
    builder.input_inclusive_max(*input_inclusive_max);
  }
  if (input_shape) {
    builder.input_shape(*input_shape);
  }
  if (implicit_upper_bounds) {
    builder.implicit_upper_bounds(
        DimensionSet::FromRange(*implicit_upper_bounds));
  }
  if (input_labels) {
    auto builder_input_labels = builder.input_labels();
    for (DimensionIndex i = 0; i < *input_rank; ++i) {
      const auto& label = (*input_labels)[i];
      if (label) builder_input_labels[i] = *label;
    }
  }
  return builder;
}

void SetOutputIndexMaps(
    const std::optional<SequenceParameter<OutputIndexMap>>& output,
    IndexTransformBuilder<>* builder) {
  const DimensionIndex output_rank = builder->output_rank();
  if (!output) {
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      builder->output_single_input_dimension(output_dim, output_dim);
    }
  } else {
    assert(static_cast<DimensionIndex>(output->size()) == output_rank);
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      const auto& map = (*output)[output_dim];
      switch (map.method) {
        case OutputIndexMethod::constant:
          builder->output_constant(output_dim, map.offset);
          break;
        case OutputIndexMethod::single_input_dimension:
          builder->output_single_input_dimension(
              output_dim, map.offset, map.stride, map.input_dimension);
          break;
        case OutputIndexMethod::array:
          builder->output_index_array(output_dim, map.offset, map.stride,
                                      map.index_array, map.index_range);
          break;
      }
    }
  }
}

std::string OutputIndexMapToString(const OutputIndexMap& m) {
  switch (m.method) {
    case OutputIndexMethod::constant:
      return tensorstore::StrCat("OutputIndexMap(offset=", m.offset, ")");
    case OutputIndexMethod::single_input_dimension:
      return tensorstore::StrCat("OutputIndexMap(offset=", m.offset,
                                 ", stride=", m.stride,
                                 ", input_dimension=", m.input_dimension, ")");
    case OutputIndexMethod::array:
      return tensorstore::StrCat("OutputIndexMap(offset=", m.offset,
                                 ", stride=", m.stride,
                                 ", index_array=", m.index_array,
                                 ", index_range=", m.index_range, ")");
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

using OutputIndexMapRangeContainer =
    OutputIndexMapRange<dynamic_rank, dynamic_rank, container>;

auto MakeIndexDomainClass(py::module m) {
  return py::class_<IndexDomain<>>(m, "IndexDomain", R"(
:ref:`Domain<index-domain>` (including bounds and optional dimension labels) of an N-dimensional :ref:`index space<index-space>`.

Logically, an :py:class:`.IndexDomain` is the cartesian product of a sequence of `Dim` objects.

Note:

   Index domains are immutable, but
   :ref:`dimension expressions<python-dim-expressions>` may be applied using
   :py:obj:`.__getitem__(expr)` to obtain a modified domain.

See also:
  - :py:obj:`IndexTransform`, which define a class of functions for index domains.
  - The :json:schema:`JSON representation<IndexDomain>`.

Group:
  Indexing
)");
}

void DefineIndexDomainAttributes(py::class_<IndexDomain<>>& cls) {
  cls.def(
      py::init([](std::optional<DimensionIndex> rank,
                  std::optional<SequenceParameter<Index>> inclusive_min,
                  std::optional<SequenceParameter<bool>> implicit_lower_bounds,
                  std::optional<SequenceParameter<Index>> exclusive_max,
                  std::optional<SequenceParameter<Index>> inclusive_max,
                  std::optional<SequenceParameter<Index>> shape,
                  std::optional<SequenceParameter<bool>> implicit_upper_bounds,
                  std::optional<SequenceParameter<std::optional<std::string>>>
                      labels) -> IndexDomain<> {
        auto builder = InitializeIndexTransformBuilder(
            rank, "rank", inclusive_min, "inclusive_min", implicit_lower_bounds,
            exclusive_max, "exclusive_max", inclusive_max, "inclusive_max",
            shape, "shape", implicit_upper_bounds, labels, "labels",
            /*output_rank=*/0);
        return ValueOrThrow(builder.Finalize()).domain();
      }),
      R"(
Constructs an index domain from component vectors.

Args:
  rank: Number of dimensions.  Only required if no other parameter is specified.
  inclusive_min: Inclusive lower bounds for each dimension.  If not specified,
      defaults to all zero if ``shape`` is specified, otherwise unbounded.
  implicit_lower_bounds: Indicates whether each lower bound is
      :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
      ``inclusive_min`` or ``shape`` is specified, otherwise defaults to all
      implicit.
  exclusive_max: Exclusive upper bounds for each dimension.  At most one of
      ``exclusive_max``, ``inclusive_max``, and ``shape`` may be specified.
  inclusive_max: Inclusive upper bounds for each dimension.
  shape: Size for each dimension.
  implicit_upper_bounds: Indicates whether each upper bound is
      :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
      ``exclusive_max``, ``inclusive_max``, or ``shape`` is specified, otherwise
      defaults to all implicit.
  labels: :ref:`Dimension labels<dimension-labels>`.  Defaults to all unlabeled.

Examples:

    >>> ts.IndexDomain(rank=5)
    { (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*) }
    >>> ts.IndexDomain(shape=[2, 3])
    { [0, 2), [0, 3) }

Overload:
  components
)",
      py::arg("rank") = std::nullopt, py::kw_only(),
      py::arg("inclusive_min") = std::nullopt,
      py::arg("implicit_lower_bounds") = std::nullopt,
      py::arg("exclusive_max") = std::nullopt,
      py::arg("inclusive_max") = std::nullopt, py::arg("shape") = std::nullopt,
      py::arg("implicit_upper_bounds") = std::nullopt,
      py::arg("labels") = std::nullopt);

  cls.def(
      py::init([](const SequenceParameter<IndexDomainDimension<>>& dimensions) {
        const DimensionIndex rank = dimensions.size();
        auto builder = IndexTransformBuilder<>(rank, 0);
        auto origin = builder.input_origin();
        auto shape = builder.input_shape();
        auto labels = builder.input_labels();
        auto& implicit_lower_bounds = builder.implicit_lower_bounds();
        auto& implicit_upper_bounds = builder.implicit_upper_bounds();
        for (DimensionIndex i = 0; i < rank; ++i) {
          const auto& d = dimensions[i];
          origin[i] = d.inclusive_min();
          shape[i] = d.size();
          labels[i] = std::string(d.label());
          implicit_lower_bounds[i] = d.implicit_lower();
          implicit_upper_bounds[i] = d.implicit_upper();
        }
        return ValueOrThrow(builder.Finalize()).domain();
      }),
      R"(
Constructs an index domain from a :py:class`.Dim` sequence.

Args:
  dimensions: :py:obj:`Sequence<python:typing.Sequence>` of :py:class`.Dim` objects.

Examples:

    >>> ts.IndexDomain([ts.Dim(5), ts.Dim(6, label='y')])
    { [0, 5), "y": [0, 6) }

Overload:
  dimensions
)",
      py::arg("dimensions"));

  cls.def(py::init([](::nlohmann::json json) {
            return ValueOrThrow(ParseIndexDomain(json));
          }),
          R"(
Constructs an index domain from its :json:schema:`JSON representation<IndexDomain>`.

Examples:

    >>> ts.IndexDomain(
    ...     json={
    ...         "inclusive_min": ["-inf", 7, ["-inf"], [8]],
    ...         "exclusive_max": ["+inf", 10, ["+inf"], [17]],
    ...         "labels": ["x", "y", "z", ""]
    ...     })
    { "x": (-inf, +inf), "y": [7, 10), "z": (-inf*, +inf*), [8*, 17*) }

Overload:
  json
)",
          py::kw_only(), py::arg("json"));

  cls.def_property_readonly("rank", &IndexDomain<>::rank,
                            R"(
Number of dimensions in the index space.

Example:

  >>> domain = ts.IndexDomain(shape=[100, 200, 300])
  >>> domain.rank
  3

Group:
  Accessors
)");

  cls.def_property_readonly("ndim", &IndexDomain<>::rank,
                            R"(
Alias for :py:obj:`.rank`.

Example:

  >>> domain = ts.IndexDomain(shape=[100, 200, 300])
  >>> domain.ndim
  3

Group:
  Accessors
)");

  cls.def(
      "__len__", [](const IndexDomain<>& d) { return d.rank(); },
      R"(
Returns the number of dimensions (:py:obj:`.rank`).

Example:

  >>> domain = ts.IndexDomain(shape=[100, 200, 300])
  >>> len(domain)
  3

Group:
  Sequence accessors
)");

  cls.def(
      "__getitem__",
      [](const IndexDomain<>& self, const PythonDimensionIdentifier& identifier)
          -> IndexDomainDimension<> {
        return self[ValueOrThrow(
            NormalizeDimensionIdentifier(ToDimensionIdentifier(identifier),
                                         self.labels()),
            StatusExceptionPolicy::kIndexError)];
      },
      R"(
Returns the single dimension specified by :python:`identifier`.

Args:
  identifier: Specifies a dimension by integer index or label.  As with
      :py:obj:`python:list`, a negative index specifies a dimension starting
      from the last dimension.

Examples:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                         exclusive_max=[4, 5, 6],
    ...                         labels=['x', 'y', 'z'])
    >>> domain[0]
    Dim(inclusive_min=1, exclusive_max=4, label="x")
    >>> domain['y']
    Dim(inclusive_min=2, exclusive_max=5, label="y")
    >>> domain[-1]
    Dim(inclusive_min=3, exclusive_max=6, label="z")

Overload:
  identifier

Group:
  Sequence accessors
)",
      py::arg("identifier"));

  cls.def(
      "__getitem__",
      [](const IndexDomain<>& self, DimensionSelectionLike s) -> IndexDomain<> {
        DimensionIndexBuffer dims;
        ThrowStatusException(internal_index_space::GetDimensions(
            self.labels(), s.value.dims, &dims));
        return self[span<const DimensionIndex>(dims)];
      },
      R"(
Returns a new domain with a subset of the dimensions.

Args:

  selection: Specifies the dimensions to include, either by index or label.  May
      be any value or sequence of values convertible to a
      :ref:`dimension selection<python-dim-selections>`.

Raises:
   ValueError: If any dimension is specified more than once.

Examples:

    >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                    exclusive_max=[4, 5, 6],
    ...                    labels=['x', 'y', 'z'])
    >>> a[:2]
    { "x": [1, 4), "y": [2, 5) }
    >>> a[0, -1]
    { "x": [1, 4), "z": [3, 6) }
    >>> a['y', 'x']
    { "y": [2, 5), "x": [1, 4) }
    >>> a['y', 1]
    Traceback (most recent call last):
        ...
    ValueError: Input dimensions {1} specified more than once

Overload:
  selection

Group:
  Indexing
)",
      py::arg("selection"));

  cls.def(
      "__getitem__",
      [](const IndexDomain<>& self,
         const IndexDomain<>& other) -> IndexDomain<> {
        return ValueOrThrow(
                   other(
                       internal_index_space::TransformAccess::transform(self)),
                   StatusExceptionPolicy::kIndexError)
            .domain();
      },
      R"(
Slices this domain by another domain.

The result is determined by matching dimensions of :python:`other` to dimensions
of :python:`self` either by label or by index, according to one of the following
three cases:

.. list-table::
   :widths: auto

   * - :python:`other` is entirely unlabeled

     - Result is
       :python:`self[ts.d[:][other.inclusive_min:other.exclusive_max]`.
       It is an error if :python:`self.rank != other.rank`.

   * - :python:`self` is entirely unlabeled

     - Result is
       :python:`self[ts.d[:][other.inclusive_min:other.exclusive_max].labels[other.labels]`.
       It is an error if :python:`self.rank != other.rank`.

   * - Both :python:`self` and :python:`other` have at least one labeled dimension.

     - Result is
       :python:`self[ts.d[dims][other.inclusive_min:other.exclusive_max]`, where
       the sequence of :python:`other.rank` dimension identifiers :python:`dims`
       is determined as follows:

       1. If :python:`other.labels[i]` is specified (i.e. non-empty),
          :python:`dims[i] = self.labels.index(other.labels[i])`.  It is an
          error if no such dimension exists.

       2. Otherwise, ``i`` is the ``j``\ th unlabeled dimension of :python:`other`
          (left to right), and :python:`dims[i] = k`, where ``k`` is the ``j``\ th
          unlabeled dimension of :python:`self` (left to right).  It is an error
          if no such dimension exists.

       If any dimensions of :python:`other` are unlabeled, then it is an error
       if :python:`self.rank != other.rank`.  This condition is not strictly
       necessary but serves to avoid a discrepancy in behavior with normal
       :ref:`domain alignment<index-domain-alignment>`.

.. admonition:: Example with all unlabeled dimensions
   :class: example

   >>> a = ts.IndexDomain(inclusive_min=[0, 1], exclusive_max=[5, 7])
   >>> b = ts.IndexDomain(inclusive_min=[2, 3], exclusive_max=[4, 6])
   >>> a[b]
   { [2, 4), [3, 6) }

.. admonition:: Example with fully labeled dimensions
   :class: example

   >>> a = ts.IndexDomain(inclusive_min=[0, 1, 2],
   ...                    exclusive_max=[5, 7, 8],
   ...                    labels=["x", "y", "z"])
   >>> b = ts.IndexDomain(inclusive_min=[2, 3],
   ...                    exclusive_max=[6, 4],
   ...                    labels=["y", "x"])
   >>> a[b]
   { "x": [3, 4), "y": [2, 6), "z": [2, 8) }

.. admonition:: Example with mixed labeled and unlabeled dimensions
   :class: example

   >>> a = ts.IndexDomain(inclusive_min=[0, 0, 0, 0],
   ...                    exclusive_max=[10, 10, 10, 10],
   ...                    labels=["x", "", "", "y"])
   >>> b = ts.IndexDomain(inclusive_min=[1, 2, 3, 4],
   ...                    exclusive_max=[6, 7, 8, 9],
   ...                    labels=["y", "", "x", ""])
   >>> a[b]
   { "x": [3, 8), [2, 7), [4, 9), "y": [1, 6) }

Note:

  On :python:`other`, :ref:`implicit bounds<implicit-bounds>` indicators have no
  effect.

Overload:
  domain

Group:
  Indexing
)",
      py::arg("other"));

  cls.def(
      "__getitem__",
      [](const IndexDomain<>& self, const PythonDimExpression& expr) {
        GilScopedRelease gil_release;
        DimensionIndexBuffer dims;
        return ValueOrThrow(
                   expr.Apply(
                       internal_index_space::TransformAccess::transform(self),
                       &dims, /*top_level=*/true, /*domain_only=*/true),
                   StatusExceptionPolicy::kIndexError)
            .domain();
      },
      R"(
Transforms the domain by a :ref:`dimension expression<python-dim-expressions>`.

Args:
  expr: :ref:`Dimension expression<python-dim-expressions>` to apply.

Examples:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                         exclusive_max=[6, 7, 8],
    ...                         labels=['x', 'y', 'z'])
    >>> domain[ts.d[:].translate_by[5]]
    { "x": [6, 11), "y": [7, 12), "z": [8, 13) }
    >>> domain[ts.d['y'][3:5]]
    { "x": [1, 6), "y": [3, 5), "z": [3, 8) }
    >>> domain[ts.d['z'][5]]
    { "x": [1, 6), "y": [2, 7) }

Note:

   For the purpose of applying a dimension expression, an
   :py:class:`IndexDomain` behaves like an :py:class:`IndexTransform` with an
   output rank of 0.  Consequently, operations that primarily affect the output
   index mappings, like
   :ref:`integer array indexing<python-indexing-integer-array>`, are not very
   useful, though they are still permitted.

       >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
       ...                         exclusive_max=[6, 7, 8],
       ...                         labels=['x', 'y', 'z'])
       >>> domain[ts.d['z'][[3, 5, 7]]]
       { "x": [1, 6), "y": [2, 7), [0, 3) }

Overload:
  expr

Group:
  Indexing
)",
      py::arg("expr"));

  cls.def(
      "__getitem__",
      [](const IndexDomain<>& self, const IndexTransform<>& transform) {
        GilScopedRelease gil_release;
        return ValueOrThrow(self | transform,
                            StatusExceptionPolicy::kIndexError);
      },
      R"(
Transforms the domain using an explicit :ref:`index transform<index-transform>`.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                         exclusive_max=[6, 7, 8])
    >>> transform = ts.IndexTransform(
    ...     input_rank=4,
    ...     output=[
    ...         ts.OutputIndexMap(offset=5, input_dimension=3),
    ...         ts.OutputIndexMap(offset=-7, input_dimension=0),
    ...         ts.OutputIndexMap(offset=3, input_dimension=1),
    ...     ])
    >>> domain[transform]
    { [9, 14), [0, 5), (-inf*, +inf*), [-4, 1) }

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.

Returns:

  New domain of rank :python:`transform.input_rank`.

Note:

   This is equivalent to composing an identity transform over :python:`self`
   with :py:param:`.transform`,
   i.e. :python:`ts.IndexTransform(self)[transform].domain`.  Consequently,
   operations that primarily affect the output index mappings, like
   :ref:`integer array indexing<python-indexing-integer-array>`, are not very
   useful, though they are still permitted.

Overload:
  transform

Group:
  Indexing
)",
      py::arg("transform"));

  cls.def(
      "intersect",
      [](const IndexDomain<>& self, const IndexDomain<> b) {
        return tensorstore::IntersectIndexDomains(self, b);
      },
      R"(
Intersects with another domain.

The ``implicit`` flag that corresponds to the selected bound is propagated.

Args:
  other: Object to intersect with.

Example:

    >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                    exclusive_max=[4, 5, 6],
    ...                    labels=['x', 'y', ''])
    >>> a.intersect(ts.IndexDomain(shape=[2, 3, 4]))
    { "x": [1, 2), "y": [2, 3), [3, 4) }

Group:
  Geometric operations
)",
      py::arg("other"));

  cls.def(
      "hull",
      [](const IndexDomain<>& self, const IndexDomain<> b) {
        return tensorstore::HullIndexDomains(self, b);
      },
      R"(
Computes the hull (minimum containing box) with another domain.

The ``implicit`` flag that corresponds to the selected bound is propagated.

Args:
  other: Object to hull with.

Example:

    >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                    exclusive_max=[4, 5, 6],
    ...                    labels=['x', 'y', ''])
    >>> a.hull(ts.IndexDomain(shape=[2, 3, 4]))
    { "x": [0, 4), "y": [0, 5), [0, 6) }

Group:
  Geometric operations
)",
      py::arg("other"));

  cls.def_property_readonly(
      "origin",
      [](const IndexDomain<>& self) {
        return SpanToHomogeneousTuple<Index>(self.origin());
      },
      R"(
Inclusive lower bound of the domain.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.origin
    (1, 2, 3)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "inclusive_min",
      [](const IndexDomain<>& self) {
        return SpanToHomogeneousTuple<Index>(self.origin());
      },
      R"(
Inclusive lower bound of the domain, alias of :py:obj:`.origin`.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.inclusive_min
    (1, 2, 3)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "shape",
      [](const IndexDomain<>& self) {
        return SpanToHomogeneousTuple<Index>(self.shape());
      },
      R"(
Shape of the domain.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.shape
    (3, 4, 5)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "exclusive_max",
      [](const IndexDomain<>& self) { return GetExclusiveMax(self); },
      R"(
Exclusive upper bound of the domain.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.exclusive_max
    (4, 6, 8)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "inclusive_max",
      [](const IndexDomain<>& self) { return GetInclusiveMax(self); },
      R"(
Inclusive upper bound of the domain.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.inclusive_max
    (3, 5, 7)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "labels",
      [](const IndexDomain<>& d) { return SpanToHomogeneousTuple(d.labels()); },
      R"(
:ref:`Dimension labels<dimension-labels>` for each dimension.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.labels
    ('', '', '')
    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                         shape=[3, 4, 5],
    ...                         labels=['x', 'y', 'z'])
    >>> domain.labels
    ('x', 'y', 'z')

Group:
  Accessors
)");

  cls.def_property_readonly(
      "implicit_lower_bounds",
      [](const IndexDomain<>& d) {
        return GetBitVector(d.implicit_lower_bounds(), d.rank());
      },
      R"(
Indicates whether the lower bound of each dimension is :ref:`implicit or explicit<implicit-bounds>`.

Example:

    >>> domain = ts.IndexDomain(rank=3)
    >>> domain.implicit_lower_bounds
    (True, True, True)
    >>> domain = ts.IndexDomain(inclusive_min=[2, 3, 4])
    >>> domain.implicit_lower_bounds
    (False, False, False)
    >>> domain = ts.IndexDomain(exclusive_max=[2, 3, 4])
    >>> domain.implicit_lower_bounds
    (True, True, True)
    >>> domain = ts.IndexDomain(shape=[4, 5, 6])
    >>> domain.implicit_lower_bounds
    (False, False, False)
    >>> domain = ts.IndexDomain(inclusive_min=[4, 5, 6],
    ...                         implicit_lower_bounds=[False, True, False])
    >>> domain.implicit_lower_bounds
    (False, True, False)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "implicit_upper_bounds",
      [](const IndexDomain<>& d) {
        return GetBitVector(d.implicit_upper_bounds(), d.rank());
      },
      R"(
Indicates whether the upper bound of each dimension is :ref:`implicit or explicit<implicit-bounds>`.

Example:

    >>> domain = ts.IndexDomain(rank=3)
    >>> domain.implicit_upper_bounds
    (True, True, True)
    >>> domain = ts.IndexDomain(shape=[2, 3, 4])
    >>> domain.implicit_upper_bounds
    (False, False, False)
    >>> domain = ts.IndexDomain(inclusive_min=[4, 5, 6])
    >>> domain.implicit_upper_bounds
    (True, True, True)
    >>> domain = ts.IndexDomain(exclusive_max=[4, 5, 6],
    ...                         implicit_upper_bounds=[False, True, False])
    >>> domain.implicit_upper_bounds
    (False, True, False)

Group:
  Accessors
)");

  cls.def_property_readonly(
      "size", [](const IndexDomain<>& self) { return self.num_elements(); },
      R"(Total number of elements in the domain.

This is simply the product of the extents in :py:obj:`.shape`.

Example:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
    >>> domain.size
    60

Group:
  Accessors
)");

  cls.def_property_readonly(
      "index_exp",
      [](const IndexDomain<>& self) -> HomogeneousTuple<py::slice> {
        const DimensionIndex rank = self.rank();
        py::tuple t(rank);

        const auto get_bound = [&](Index value, Index inf) -> py::object {
          if (value == inf) return py::none();
          if (value < 0) {
            throw py::value_error(tensorstore::StrCat(
                "Cannot convert domain ", self,
                " with negative bounds to index expression"));
          }
          return py::int_(value);
        };

        for (DimensionIndex i = 0; i < rank; ++i) {
          IndexInterval interval = self[i];
          t[i] = py::slice(get_bound(interval.inclusive_min(), -kInfIndex),
                           get_bound(interval.exclusive_max(), +kInfIndex + 1),
                           py::none());
        }
        return {t};
      },
      R"(
Equivalent NumPy-compatible :py:obj:`index expression<numpy.s_>`.

The index expression consists of a :py:obj:`tuple` of :py:obj:`.rank`
:py:obj:`slice` objects that specify the lower and upper bounds for each
dimension, where an infinite bound in the domain corresponds to a bound of
:py:obj:`None` in the :py:obj:`slice` object.

The index expression can be used with this library as a
:ref:`NumPy-style indexing expression<python-numpy-style-indexing>` or to
directly index a `NumPy array<numpy.ndarray>`.

Example:

    >>> ts.IndexDomain(rank=2).index_exp
    (slice(None, None, None), slice(None, None, None))
    >>> ts.IndexDomain(inclusive_min=[1, 2], exclusive_max=[5, 10]).index_exp
    (slice(1, 5, None), slice(2, 10, None))
    >>> arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> domain = ts.IndexDomain(inclusive_min=[0, 2], shape=[3, 2])
    >>> arr[domain.index_exp]
    array([[3, 4],
           [8, 9]])

Raises:
  ValueError: If any finite bound in :py:obj:`.inclusive_min` or
    :py:obj:`.exclusive_max` is negative.  In this case the index expression
    would not actually NumPy-compatible since NumPy does not support actual
    negative indices, and instead interprets negative numbers as counting from
    the end.

Group:
  Accessors
)");

  cls.def(
      "__repr__", [](const IndexDomain<>& d) { return tensorstore::StrCat(d); },
      "Returns the string representation.");

  cls.def(
      "to_json",
      [](const IndexDomain<>& self) { return ::nlohmann::json(self); },
      R"(
Returns the :json:schema:`JSON representation<IndexDomain>`.

Group:
  Accessors
)");

  cls.def("__eq__", [](const IndexDomain<>& self, const IndexDomain<>& other) {
    return self == other;
  });

  cls.def("__copy__", [](const IndexDomain<>& self) { return self; });

  cls.def(
      "__deepcopy__",
      [](const IndexDomain<>& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(
      cls, internal_index_space::IndexDomainNonNullSerializer{});

  py::implicitly_convertible<std::vector<IndexDomainDimension<>>,
                             IndexDomain<>>();
}

auto MakeIndexTransformClass(py::module m) {
  return py::class_<IndexTransform<>>(m, "IndexTransform", R"(
Represents a transform from an input index space to an output space.

The :ref:`index transform abstraction<index-transform>` underlies all indexing
operations in the TensorStore library, and enables fully-composable virtual
views.  For many common use cases cases, however, it does not need to be used
directly; instead, it is used indirectly through
:ref:`indexing operations<python-indexing>` on the :py:class:`TensorStore` class
and other :py:class:`Indexable` types.

See also:
  - :py:obj:`IndexDomain`, which represents the domain of an index transform.
  - The :json:schema:`JSON representation<IndexTransform>`.

Group:
  Indexing

Constructors
============

Accessors
=========

Indexing
========

)");
}

void DefineIndexTransformAttributes(py::class_<IndexTransform<>>& cls) {
  cls.def(
      py::init([](std::optional<DimensionIndex> input_rank,
                  std::optional<SequenceParameter<Index>> input_inclusive_min,
                  std::optional<SequenceParameter<bool>> implicit_lower_bounds,
                  std::optional<SequenceParameter<Index>> input_exclusive_max,
                  std::optional<SequenceParameter<Index>> input_inclusive_max,
                  std::optional<SequenceParameter<Index>> input_shape,
                  std::optional<SequenceParameter<bool>> implicit_upper_bounds,
                  std::optional<SequenceParameter<std::optional<std::string>>>
                      input_labels,
                  std::optional<SequenceParameter<OutputIndexMap>> output)
                   -> IndexTransform<> {
        std::optional<DimensionIndex> output_rank_opt;
        if (output) output_rank_opt = output->size();
        auto builder = InitializeIndexTransformBuilder(
            input_rank, "input_rank", input_inclusive_min,
            "input_inclusive_min", implicit_lower_bounds, input_exclusive_max,
            "input_exclusive_max", input_inclusive_max, "input_inclusive_max",
            input_shape, "input_shape", implicit_upper_bounds, input_labels,
            "input_labels", output_rank_opt);
        SetOutputIndexMaps(output, &builder);
        return ValueOrThrow(builder.Finalize());
      }),
      R"(
Constructs an index transform from component vectors.

Args:
  input_rank: Number of input dimensions.  Only required if the input rank is
      not otherwise specified.
  input_inclusive_min: Inclusive lower bounds for each input dimension.  If not
      specified, defaults to all zero if ``input_shape`` is specified, otherwise
      unbounded.
  implicit_lower_bounds: Indicates whether each lower bound is
      :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
      ``input_inclusive_min`` or ``input_shape`` is specified, otherwise
      defaults to all implicit.
  input_exclusive_max: Exclusive upper bounds for each input dimension.  At most
      one of ``input_exclusive_max``, ``input_inclusive_max``, and
      ``input_shape`` may be specified.
  input_inclusive_max: Inclusive upper bounds for each input dimension.
  input_shape: Size for each input dimension.
  implicit_upper_bounds: Indicates whether each upper bound is
      :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
      ``input_exclusive_max``, ``input_inclusive_max``, or ``shape`` is
      specified, otherwise defaults to all implicit.
  input_labels: :ref:`Dimension labels<dimension-labels>` for each input
      dimension.  Defaults to all unlabeled.
  output: Sequence of output index maps, or :py:obj:`OutputIndexMaps` object
      from an existing transform.  If not specified, constructs an identity
      transform over the domain.

Examples:

    >>> # Identity transform of rank 3
    >>> ts.IndexTransform(3)
    Rank 3 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*)
        1: (-inf*, +inf*)
        2: (-inf*, +inf*)
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]
    >>> ts.IndexTransform(
    ...     input_shape=[3, 2],
    ...     output=[
    ...         ts.OutputIndexMap(offset=7, input_dimension=1),
    ...         ts.OutputIndexMap([[1, 2]], offset=2, stride=-1),
    ...         ts.OutputIndexMap(8),
    ...         ts.OutputIndexMap([[1, 2]],
    ...                           offset=2,
    ...                           stride=-1,
    ...                           index_range=ts.Dim(inclusive_min=0,
    ...                                              exclusive_max=8)),
    ...     ],
    ... )
    Rank 2 -> 4 index space transform:
      Input domain:
        0: [0, 3)
        1: [0, 2)
      Output index maps:
        out[0] = 7 + 1 * in[1]
        out[1] = 2 + -1 * bounded((-inf, +inf), array(in)), where array =
          {{1, 2}}
        out[2] = 8
        out[3] = 2 + -1 * bounded([0, 8), array(in)), where array =
          {{1, 2}}

Overload:
  components
)",
      py::arg("input_rank") = std::nullopt, py::kw_only(),
      py::arg("input_inclusive_min") = std::nullopt,
      py::arg("implicit_lower_bounds") = std::nullopt,
      py::arg("input_exclusive_max") = std::nullopt,
      py::arg("input_inclusive_max") = std::nullopt,
      py::arg("input_shape") = std::nullopt,
      py::arg("implicit_upper_bounds") = std::nullopt,
      py::arg("input_labels") = std::nullopt, py::arg("output") = std::nullopt);

  cls.def(py::init([](IndexDomain<> domain,
                      std::optional<SequenceParameter<OutputIndexMap>> output) {
            const DimensionIndex output_rank =
                output ? output->size() : domain.rank();
            IndexTransformBuilder<> builder(domain.rank(), output_rank);
            builder.input_domain(domain);
            SetOutputIndexMaps(output, &builder);
            return ValueOrThrow(builder.Finalize());
          }),
          R"(
Constructs an index transform from a domain and output index maps.

Args:
  domain: The domain of the index transform.
  output: Sequence of output index maps, or :py:obj:`OutputIndexMaps` object
      from an existing transform.  If not specified, constructs an identity
      transform over the domain.

Examples:

    >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
    ...                         exclusive_max=[4, 5, 6])
    >>> ts.IndexTransform(domain)
    Rank 3 -> 3 index space transform:
      Input domain:
        0: [1, 4)
        1: [2, 5)
        2: [3, 6)
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]
    >>> ts.IndexTransform(
    ...     domain,
    ...     output=[
    ...         ts.OutputIndexMap(offset=7),
    ...         ts.OutputIndexMap(input_dimension=0),
    ...     ],
    ... )
    Rank 3 -> 2 index space transform:
      Input domain:
        0: [1, 4)
        1: [2, 5)
        2: [3, 6)
      Output index maps:
        out[0] = 7
        out[1] = 0 + 1 * in[0]

Overload:
  domain
)",
          py::arg("domain"), py::arg("output") = std::nullopt);

  cls.def(py::init([](const ::nlohmann::json& json) {
            return ValueOrThrow(ParseIndexTransform(json));
          }),
          R"(
Constructs an index transform from its :json:schema:`JSON representation<IndexTransform>`.

Examples:

    >>> ts.IndexTransform(
    ...     json={
    ...         "input_inclusive_min": ["-inf", 7, ["-inf"], [8]],
    ...         "input_exclusive_max": ["+inf", 11, ["+inf"], [17]],
    ...         "input_labels": ["x", "y", "z", ""],
    ...         "output": [
    ...             {
    ...                 "offset": 3
    ...             },
    ...             {
    ...                 "stride": 2,
    ...                 "input_dimension": 2
    ...             },
    ...             {
    ...                 "offset": 7,
    ...                 "index_array": [[[[1]], [[2]], [[3]], [[4]]]],
    ...                 "index_array_bounds": [1, 4]
    ...             },
    ...         ],
    ...     })
    Rank 4 -> 3 index space transform:
      Input domain:
        0: (-inf, +inf) "x"
        1: [7, 11) "y"
        2: (-inf*, +inf*) "z"
        3: [8*, 17*)
      Output index maps:
        out[0] = 3
        out[1] = 0 + 2 * in[2]
        out[2] = 7 + 1 * bounded([1, 5), array(in)), where array =
          {{{{1}}, {{2}}, {{3}}, {{4}}}}

Overload:
  json
)",
          py::kw_only(), py::arg("json"));

  cls.def_property_readonly(
      "domain",
      [](const IndexTransform<>& t) -> IndexDomain<> { return t.domain(); },
      R"(
Input domain of the index transform.

Example:

    >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
    ...                               input_labels=["x", "y", "z"])
    >>> transform.domain
    { "x": [0, 3), "y": [0, 4), "z": [0, 5) }

Group:
  Accessors

)");

  cls.def_property_readonly("input_rank", &IndexTransform<>::input_rank,
                            R"(
Rank of the input space.

Example:

    >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
    ...                               input_labels=["x", "y", "z"])
    >>> transform.input_rank
    3

Group:
  Accessors

)");

  cls.def_property_readonly("output_rank", &IndexTransform<>::output_rank,
                            R"(
Rank of the output space.

Example:

    >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
    ...                               input_labels=["x", "y", "z"],
    ...                               output=[ts.OutputIndexMap(offset=5)])
    >>> transform.output_rank
    1

Group:
  Accessors

)");

  cls.def_property_readonly("ndim", &IndexTransform<>::input_rank,
                            R"(
Rank of the input space, alias for :py:obj:`.input_rank`.

Example:

    >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
    ...                               input_labels=["x", "y", "z"])
    >>> transform.ndim
    3

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_origin",
      [](const IndexTransform<>& t) {
        return SpanToHomogeneousTuple(t.input_origin());
      },
      R"(
Inclusive lower bound of the input domain.

Alias for the :py:obj:`~tensorstore.IndexDomain.origin` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
    ...                               input_shape=[3, 4, 5])
    >>> transform.input_origin
    (1, 2, 3)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_inclusive_min",
      [](const IndexTransform<>& t) {
        return SpanToHomogeneousTuple(t.input_origin());
      },
      R"(
Inclusive lower bound of the input domain, alias for :py:obj:`.input_origin`.

Alias for the :py:obj:`~tensorstore.IndexDomain.inclusive_min` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
    ...                               input_shape=[3, 4, 5])
    >>> transform.input_inclusive_min
    (1, 2, 3)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_shape",
      [](const IndexTransform<>& t) {
        return SpanToHomogeneousTuple(t.input_shape());
      },
      R"(
Shape of the input domain.

Alias for the :py:obj:`~tensorstore.IndexDomain.shape` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_shape=[3, 4, 5])
    >>> transform.input_shape
    (3, 4, 5)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_exclusive_max",
      [](const IndexTransform<>& self) {
        return GetExclusiveMax(self.domain());
      },
      R"(
Exclusive upper bound of the input domain.

Alias for the :py:obj:`~tensorstore.IndexDomain.exclusive_max` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
    ...                               input_shape=[3, 4, 5])
    >>> transform.input_exclusive_max
    (4, 6, 8)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_inclusive_max",
      [](const IndexTransform<>& self) {
        return GetInclusiveMax(self.domain());
      },
      R"(
Inclusive upper bound of the input domain.

Alias for the :py:obj:`~tensorstore.IndexDomain.inclusive_max` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
    ...                               input_shape=[3, 4, 5])
    >>> transform.input_inclusive_max
    (3, 5, 7)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "input_labels",
      [](const IndexTransform<>& t) {
        return SpanToHomogeneousTuple(t.input_labels());
      },
      R"(
:ref:`Dimension labels<dimension-labels>` for each input dimension.

Alias for the :py:obj:`~tensorstore.IndexDomain.labels` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
    ...                               input_shape=[3, 4, 5],
    ...                               input_labels=['x', 'y', 'z'])
    >>> transform.input_labels
    ('x', 'y', 'z')

Group:
  Accessors

)");

  cls.def_property_readonly(
      "implicit_lower_bounds",
      [](const IndexTransform<>& t) {
        return GetBitVector(t.implicit_lower_bounds(), t.input_rank());
      },
      R"(
Indicates whether the lower bound of each input dimension is :ref:`implicit or explicit<implicit-bounds>`.

Alias for the :py:obj:`~tensorstore.IndexDomain.implicit_lower_bounds` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_rank=3)
    >>> transform.implicit_lower_bounds
    (True, True, True)
    >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4])
    >>> transform.implicit_lower_bounds
    (False, False, False)
    >>> transform = ts.IndexTransform(input_exclusive_max=[2, 3, 4])
    >>> transform.implicit_lower_bounds
    (True, True, True)
    >>> transform = ts.IndexTransform(input_shape=[4, 5, 6])
    >>> transform.implicit_lower_bounds
    (False, False, False)
    >>> transform = ts.IndexTransform(
    ...     input_inclusive_min=[4, 5, 6],
    ...     implicit_lower_bounds=[False, True, False])
    >>> transform.implicit_lower_bounds
    (False, True, False)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "implicit_upper_bounds",
      [](const IndexTransform<>& t) {
        return GetBitVector(t.implicit_upper_bounds(), t.input_rank());
      },
      R"(
Indicates whether the upper bound of each input dimension is :ref:`implicit or explicit<implicit-bounds>`.

Alias for the :py:obj:`~tensorstore.IndexDomain.implicit_upper_bounds` property of the :py:obj:`.domain`.

Example:

    >>> transform = ts.IndexTransform(input_rank=3)
    >>> transform.implicit_upper_bounds
    (True, True, True)
    >>> transform = ts.IndexTransform(input_shape=[2, 3, 4])
    >>> transform.implicit_upper_bounds
    (False, False, False)
    >>> transform = ts.IndexTransform(input_inclusive_min=[4, 5, 6])
    >>> transform.implicit_upper_bounds
    (True, True, True)
    >>> transform = ts.IndexTransform(
    ...     input_exclusive_max=[4, 5, 6],
    ...     implicit_upper_bounds=[False, True, False])
    >>> transform.implicit_upper_bounds
    (False, True, False)

Group:
  Accessors

)");

  cls.def_property_readonly(
      "output",
      [](const IndexTransform<>& t) -> OutputIndexMapRangeContainer {
        return t.output_index_maps();
      },
      R"(
Output index maps.

Group:
  Accessors

)");

  cls.def(
      "to_json", [](const IndexTransform<>& t) { return ::nlohmann::json(t); },
      R"(
Returns the :json:schema:`JSON representation<IndexTransform>` of the transform.

Example:

   >>> transform = ts.IndexTransform(
   ...     input_inclusive_min=[1, 2, -1],
   ...     implicit_lower_bounds=[1, 0, 0],
   ...     input_shape=[3, 2, 2],
   ...     implicit_upper_bounds=[0, 1, 0],
   ...     input_labels=['x', 'y', 'z'],
   ...     output=[
   ...         ts.OutputIndexMap(offset=7, stride=13, input_dimension=1),
   ...         ts.OutputIndexMap(offset=8),
   ...         ts.OutputIndexMap(
   ...             offset=1,
   ...             stride=-2,
   ...             index_array=[[[1, 2]]],
   ...             index_range=ts.Dim(inclusive_min=-3, exclusive_max=10),
   ...         ),
   ...     ],
   ... )
   >>> transform.to_json()
   {'input_exclusive_max': [4, [4], 1],
    'input_inclusive_min': [[1], 2, -1],
    'input_labels': ['x', 'y', 'z'],
    'output': [{'input_dimension': 1, 'offset': 7, 'stride': 13},
               {'offset': 8},
               {'index_array': [[[1, 2]]], 'offset': 1, 'stride': -2}]}

Group:
  Accessors
)");

  cls.def(
      "__call__",
      [](const IndexTransform<>& self, SequenceParameter<Index> indices) {
        if (static_cast<DimensionIndex>(indices.size()) != self.input_rank()) {
          throw std::invalid_argument(tensorstore::StrCat(
              "input indices vector of length ", indices.size(),
              " cannot be used with index transform with input rank ",
              self.input_rank()));
        }
        Index output_indices[kMaxRank];
        ThrowStatusException(self.TransformIndices(
            indices, span<Index>(output_indices, self.output_rank())));
        return SpanToHomogeneousTuple<Index>(
            span(output_indices, self.output_rank()));
      },
      R"(
Maps an input index vector to an output index vector.

Args:
  indices: Input vector of length :py:obj:`.input_rank`.

Returns:
  Output vector of length :py:obj:`output_rank`.

Examples:

    >>> transform = ts.IndexTransform(2)[ts.d[:].translate_by[1, 2]]
    >>> transform([0, 0])
    (-1, -2)
    >>> transform([1, 2])
    (0, 0)

Group:
  Indexing
)",
      py::arg("indices"));

  cls.def(
      "__repr__",
      [](const IndexTransform<>& t) { return tensorstore::StrCat(t); },
      "Returns the string representation.");

  cls.def(
      "__eq__",
      [](const IndexTransform<>& self, const IndexTransform<>& other) {
        return self == other;
      },
      py::arg("other"));

  cls.def("__copy__", [](const IndexTransform<>& self) { return self; });

  cls.def(
      "__deepcopy__",
      [](const IndexTransform<>& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(
      cls, internal_index_space::IndexTransformNonNullSerializer{});

  cls.attr("__iter__") = py::none();

  DefineIndexTransformOperations(
      &cls,
      /*doc_strings=*/
      {
          /*numpy_indexing=*/{
              /*kDefault*/ {R"(
Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with default index array semantics.

Example:

   >>> transform = ts.IndexTransform(3)
   >>> transform[2, [1, 2, 3], [6, 7, 8]]
   Rank 1 -> 3 index space transform:
     Input domain:
       0: [0, 3)
     Output index maps:
       out[0] = 2
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {1, 2, 3}
       out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {6, 7, 8}

See also:

   - :ref:`python-numpy-style-indexing`
   - py:obj:`IndexTransform.oindex`
   - py:obj:`IndexTransform.vindex`

Group:
  Indexing

Overload:
  indices
)"},
              /*kOindex*/ {R"(
Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`IndexTransform.__getitem__(indices)`, but differs in
that any integer or boolean array indexing terms are applied orthogonally:

Example:

   >>> transform = ts.IndexTransform(3)
   >>> transform.oindex[2, [1, 2, 3], [6, 7, 8]]
   Rank 2 -> 3 index space transform:
     Input domain:
       0: [0, 3)
       1: [0, 3)
     Output index maps:
       out[0] = 2
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{1}, {2}, {3}}
       out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{6, 7, 8}}

See also:

   - :ref:`python-numpy-style-indexing`

Group:
  Indexing

)"},
              /*kVindex*/ {R"(
Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`IndexTransform.__getitem__(indices)`, but differs in
that if :python:`indices` specifies any array indexing terms, the broadcasted
array dimensions are unconditionally added as the first dimensions of the result
domain:

Example:

   >>> transform = ts.IndexTransform(3)
   >>> transform.vindex[2, [1, 2, 3], [6, 7, 8]]
   Rank 1 -> 3 index space transform:
     Input domain:
       0: [0, 3)
     Output index maps:
       out[0] = 2
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {1, 2, 3}
       out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {6, 7, 8}

See also:

   - :ref:`python-numpy-style-indexing`

Group:
  Indexing

)"},
          },
          /*index_transform*/ {R"(
Composes this index transform with another index transform.

The resultant transform maps :python:`x` to :python:`self(transform(x))`.

Examples:

   >>> a = ts.IndexTransform(
   ...     input_rank=1,
   ...     output=[ts.OutputIndexMap(input_dimension=0, offset=5)])
   >>> b = ts.IndexTransform(
   ...     input_rank=1,
   ...     output=[ts.OutputIndexMap(input_dimension=0, offset=3)])
   >>> a[b]
   Rank 1 -> 1 index space transform:
     Input domain:
       0: (-inf*, +inf*)
     Output index maps:
       out[0] = 8 + 1 * in[0]

Group:
  Indexing

Overload:
  transform
)"},
          /*index_domain*/ {R"(
Slices this index transform by another domain.

The result is determined by matching dimensions of :python:`domain` to
dimensions of :python:`self.domain` either by label or by index, according to
one of the following three cases:

.. list-table::
   :widths: auto

   * - :python:`domain` is entirely unlabeled

     - Result is
       :python:`self[ts.d[:][domain.inclusive_min:domain.exclusive_max]`.  It is
       an error if :python:`self.input_rank != domain.rank`.

   * - :python:`self.domain` is entirely unlabeled

     - Result is
       :python:`self[ts.d[:][domain.inclusive_min:domain.exclusive_max].labels[domain.labels]`.
       It is an error if :python:`self.input_rank != domain.rank`.

   * - Both :python:`self.domain` and :python:`domain` have at least one labeled
       dimension.

     - Result is
       :python:`self[ts.d[dims][domain.inclusive_min:domain.exclusive_max]`,
       where the sequence of :python:`domain.rank` dimension identifiers
       :python:`dims` is determined as follows:

       1. If :python:`domain.labels[i]` is specified (i.e. non-empty),
          :python:`dims[i] = self.input_labels.index(domain.labels[i])`.  It is
          an error if no such dimension exists.

       2. Otherwise, ``i`` is the ``j``\ th unlabeled dimension of
          :python:`domain` (left to right), and :python:`dims[i] = k`, where
          ``k`` is the ``j``\ th unlabeled dimension of :python:`self` (left to
          right).  It is an error if no such dimension exists.

       If any dimensions of :python:`domain` are unlabeled, then it is an error
       if :python:`self.input_rank != domain.rank`.  This condition is not
       strictly necessary but serves to avoid a discrepancy in behavior with
       normal :ref:`domain alignment<index-domain-alignment>`.

.. admonition:: Example with all unlabeled dimensions
   :class: example

   >>> a = ts.IndexTransform(input_inclusive_min=[0, 1],
   ...                       input_exclusive_max=[5, 7])
   >>> b = ts.IndexDomain(inclusive_min=[2, 3], exclusive_max=[4, 6])
   >>> transform[domain]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [1, 4)
       1: [2, 5)
       2: [3, 6)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

.. admonition:: Example with fully labeled dimensions
   :class: example

   >>> a = ts.IndexTransform(input_inclusive_min=[0, 1, 2],
   ...                       input_exclusive_max=[5, 7, 8],
   ...                       input_labels=["x", "y", "z"])
   >>> b = ts.IndexDomain(inclusive_min=[2, 3],
   ...                    exclusive_max=[6, 4],
   ...                    labels=["y", "x"])
   >>> transform[domain]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [1, 4)
       1: [2, 5)
       2: [3, 6)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

.. admonition:: Example with mixed labeled and unlabeled dimensions
   :class: example

   >>> a = ts.IndexTransform(input_inclusive_min=[0, 0, 0, 0],
   ...                       input_exclusive_max=[10, 10, 10, 10],
   ...                       input_labels=["x", "", "", "y"])
   >>> b = ts.IndexDomain(inclusive_min=[1, 2, 3, 4],
   ...                    exclusive_max=[6, 7, 8, 9],
   ...                    labels=["y", "", "x", ""])
   >>> a[b]
   Rank 4 -> 4 index space transform:
     Input domain:
       0: [3, 8) "x"
       1: [2, 7)
       2: [4, 9)
       3: [1, 6) "y"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
       out[3] = 0 + 1 * in[3]

Note:

  On :python:`domain`, :ref:`implicit bounds<implicit-bounds>` indicators have
  no effect.

Group:
  Indexing

Overload:
  domain
)"},
          /*dim_expression*/ {R"(
Applies a :ref:`dimension expression<python-dim-expressions>` to this transform.

Example:

   >>> transform = ts.IndexTransform(input_rank=3)
   >>> transform[ts.d[0, 1].label['x', 'y'].translate_by[5]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: (-inf*, +inf*) "y"
       2: (-inf*, +inf*)
     Output index maps:
       out[0] = -5 + 1 * in[0]
       out[1] = -5 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

Group:
  Indexing

Overload:
  expr
)"},
      },
      /*get_transform=*/[](IndexTransform<> self) { return self; },
      /*apply_transform=*/
      [](IndexTransform<> self, IndexTransform<> new_transform) {
        return new_transform;
      });
}

auto MakeDimClass(py::module m) {
  return py::class_<IndexDomainDimension<>>(m, "Dim",
                                            R"(
1-d index interval with optionally-implicit bounds and dimension label.

Represents a contiguous range of integer :ref:`index values<index-space>`.  The
inclusive lower and upper bounds may either be finite values in the closed
interval :math:`[-(2^{62}-2), +(2^{62}-2)]`, or infinite, as indicated by
-/+ :py:obj:`.inf` for the lower and upper bounds, respectively.

The lower and upper bounds may additionally be marked as either
:ref:`explicit or implicit<implicit-bounds>`.

The interval may also have an associated
:ref:`dimension label<dimension-labels>`, which is primarily useful for
specifying the dimensions of an :py:obj:`.IndexDomain`.

Examples:

    >>> ts.Dim('x')
    Dim(label="x")
    >>> ts.Dim(inclusive_min=3, exclusive_max=10, label='x')
    Dim(inclusive_min=3, exclusive_max=10, label="x")

See also:
  :py:obj:`IndexDomain`

Group:
  Indexing
)");
}

void DefineDimAttributes(py::class_<IndexDomainDimension<>>& cls) {
  cls.def(py::init([](std::optional<std::string> label, bool implicit_lower,
                      bool implicit_upper) {
            return IndexDomainDimension<>(
                OptionallyImplicitIndexInterval{IndexInterval(), implicit_lower,
                                                implicit_upper},
                label.value_or(""));
          }),
          R"(
Constructs an unbounded interval ``(-inf, +inf)``.

Args:
  label: :ref:`Dimension label<dimension-labels>`.
  implicit_lower: Indicates whether the lower bound is
    :ref:`implicit<implicit-bounds>`.
  implicit_upper: Indicates whether the upper bound is
    :ref:`implicit<implicit-bounds>`.

Examples:

    >>> x = ts.Dim()
    >>> print(x)
    (-inf*, +inf*)
    >>> x.finite
    False

    >>> x = ts.Dim("x", implicit_upper=False)
    >>> print(x)
    "x": (-inf*, +inf)
    >>> x.finite
    False

Overload:
  unbounded
)",
          py::arg("label") = std::nullopt, py::kw_only(),
          py::arg("implicit_lower") = true, py::arg("implicit_upper") = true);

  cls.def(py::init([](OptionallyImplicitIndex size,
                      std::optional<std::string> label,
                      OptionallyImplicitIndex inclusive_min,
                      bool implicit_lower, std::optional<bool> implicit_upper) {
            Index inclusive_min_value = inclusive_min.value_or(0);
            Index size_value = size.value_or(kInfSize);
            return IndexDomainDimension<>(
                OptionallyImplicitIndexInterval{
                    ValueOrThrow(size_value == kInfSize
                                     ? IndexInterval::HalfOpen(
                                           inclusive_min_value, kInfIndex + 1)
                                     : IndexInterval::Sized(inclusive_min_value,
                                                            size_value)),
                    implicit_lower,
                    implicit_upper.value_or(size.value == kImplicit)},
                label.value_or(""));
          }),
          R"(
Constructs a sized interval ``[inclusive_min, inclusive_min+size)``.

Args:
  size: Size of the interval.
  label: :ref:`Dimension label<dimension-labels>`.
  inclusive_min: Inclusive lower bound.  Defaults to :python:`0`.
  implicit_lower: Indicates whether the lower bound is
    :ref:`implicit<implicit-bounds>`.
  implicit_upper: Indicates whether the upper bound is
    :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
    :python:`size` is specified, otherwise :python:`True`.

Examples:

    >>> x = ts.Dim(10)
    >>> print(x)
    [0, 10)
    >>> print(ts.Dim(inclusive_min=5, size=10))
    [5, 15)

Overload:
  size
)",
          py::arg("size"), py::arg("label") = std::nullopt, py::kw_only(),
          py::arg("inclusive_min") = OptionallyImplicitIndex(),
          py::arg("implicit_lower") = false,
          py::arg("implicit_upper") = std::nullopt);

  cls.def(py::init([](OptionallyImplicitIndex inclusive_min,
                      OptionallyImplicitIndex exclusive_max,
                      std::optional<std::string> label,
                      std::optional<bool> implicit_lower,
                      std::optional<bool> implicit_upper) {
            return IndexDomainDimension<>(
                OptionallyImplicitIndexInterval{
                    ValueOrThrow(IndexInterval::HalfOpen(
                        inclusive_min.value_or(-kInfIndex),
                        exclusive_max.value_or(kInfIndex + 1))),
                    implicit_lower.value_or(inclusive_min.value == kImplicit),
                    implicit_upper.value_or(exclusive_max.value == kImplicit)},
                label.value_or(""));
          }),
          R"(
Constructs a half-open interval ``[inclusive_min, exclusive_max)``.

Args:
  inclusive_min: Inclusive lower bound.
  exclusive_max: Exclusive upper bound.
  label: :ref:`Dimension label<dimension-labels>`.
  implicit_lower: Indicates whether the lower bound is
    :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
    ``inclusive_min`` is specified, otherwise :python:`True`.
  implicit_upper: Indicates whether the upper bound is
    :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
    ``exclusive_max`` is specified, otherwise :python:`True`.

Examples:

    >>> x = ts.Dim(5, 10)
    >>> x
    Dim(inclusive_min=5, exclusive_max=10)
    >>> print(x)
    [5, 10)

Overload:
  exclusive_max
)",
          py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
          py::arg_v("exclusive_max", OptionallyImplicitIndex(), "+inf"),
          py::kw_only(), py::arg("label") = std::nullopt,
          py::arg("implicit_lower") = std::nullopt,
          py::arg("implicit_upper") = std::nullopt);

  cls.def(py::init([](OptionallyImplicitIndex inclusive_min,
                      OptionallyImplicitIndex inclusive_max,
                      std::optional<std::string> label,
                      std::optional<bool> implicit_lower,
                      std::optional<bool> implicit_upper) {
            return IndexDomainDimension<>(
                OptionallyImplicitIndexInterval{
                    ValueOrThrow(IndexInterval::Closed(
                        inclusive_min.value_or(-kInfIndex),
                        inclusive_max.value_or(kInfIndex))),
                    implicit_lower.value_or(inclusive_min.value == kImplicit),
                    implicit_upper.value_or(inclusive_max.value == kImplicit)},
                label.value_or(""));
          }),
          R"(
Constructs a closed interval ``[inclusive_min, inclusive_max]``.

Args:
  inclusive_min: Inclusive lower bound.
  inclusive_max: Inclusive upper bound.
  label: :ref:`Dimension label<dimension-labels>`.
  implicit_lower: Indicates whether the lower bound is
    :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
    ``inclusive_min`` is specified, otherwise :python:`True`.
  implicit_upper: Indicates whether the upper bound is
    :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
    ``exclusive_max`` is specified, otherwise :python:`True`.

Examples:

    >>> x = ts.Dim(inclusive_min=5, inclusive_max=10)
    >>> x
    Dim(inclusive_min=5, exclusive_max=11)
    >>> print(x)
    [5, 11)

Overload:
  inclusive_max
)",
          py::kw_only(),
          py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
          py::arg_v("inclusive_max", OptionallyImplicitIndex(), "+inf"),
          py::arg("label") = std::nullopt,
          py::arg("implicit_lower") = std::nullopt,
          py::arg("implicit_upper") = std::nullopt);

  cls.def(
      "intersect",
      [](const IndexDomainDimension<>& self,
         const IndexDomainDimension<>& b) -> Result<IndexDomainDimension<>> {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto merged_label, MergeDimensionLabels(self.label(), b.label()));

        return IndexDomainDimension<>(
            Intersect(self.optionally_implicit_interval(),
                      b.optionally_implicit_interval()),
            std::string(merged_label));
      },
      R"(
Intersect with another Dim.

The ``implicit`` flag that corresponds to the selected bound is propagated.
The :py:obj:`.label`  field, if non-empty, must match, and will be propagated.

Args:
  other: Object to intersect with.

Example:

    >>> a = ts.Dim(inclusive_min=1, exclusive_max=5, label='x')
    >>> a.intersect(ts.Dim(size=3))
    Dim(inclusive_min=1, exclusive_max=3, label="x")

)",
      py::arg("other"));

  cls.def(
      "hull",
      [](const IndexDomainDimension<>& self,
         const IndexDomainDimension<>& b) -> Result<IndexDomainDimension<>> {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto merged_label, MergeDimensionLabels(self.label(), b.label()));

        return IndexDomainDimension<>(Hull(self.optionally_implicit_interval(),
                                           b.optionally_implicit_interval()),
                                      std::string(merged_label));
      },
      R"(
Hull with another Dim.

The ``implicit`` flag that corresponds to the selected bound is propagated.
The :py:obj:`.label` field, if non-empty, must match, and will be propagated.

Args:
  other: Object to hull with.

Example:

    >>> a = ts.Dim(inclusive_min=1, exclusive_max=5, label='x')
    >>> a.hull(ts.Dim(size=3))
    Dim(inclusive_min=0, exclusive_max=5, label="x")

)",
      py::arg("other"));

  cls.def_property_readonly("inclusive_min", &IndexInterval::inclusive_min,
                            R"(
Inclusive lower bound of the interval.

Equal to :python:`self.exclusive_min + 1`.  If the interval is unbounded below,
equal to the special value of :py:obj:`-inf<tensorstore.inf>`.

Example:

    >>> ts.Dim(5).inclusive_min
    0
    >>> ts.Dim(inclusive_min=5, inclusive_max=10).inclusive_min
    5
    >>> ts.Dim().inclusive_min
    -4611686018427387903

Group:
  Accessors
)");

  cls.def_property_readonly("inclusive_max", &IndexInterval::inclusive_max,
                            R"(
Inclusive upper bound of the interval.

Equal to :python:`self.exclusive_max - 1`.  If the interval is unbounded above,
equal to the special value of :py:obj:`+inf<tensorstore.inf>`.

Example:

    >>> ts.Dim(inclusive_min=5, inclusive_max=10).inclusive_max
    10
    >>> ts.Dim(exclusive_max=5).inclusive_max
    4
    >>> ts.Dim().inclusive_max
    4611686018427387903

Group:
  Accessors
)");

  cls.def_property_readonly("exclusive_min", &IndexInterval::exclusive_min,
                            R"(
Exclusive lower bound of the interval.

Equal to :python:`self.inclusive_min - 1`.  If the interval is unbounded below,
equal to the special value of :py:obj:`-inf-1<tensorstore.inf>`.

Example:

    >>> ts.Dim(inclusive_min=5, inclusive_max=10).exclusive_min
    4
    >>> ts.Dim(5).exclusive_min
    -1
    >>> ts.Dim(exclusive_max=10).exclusive_min
    -4611686018427387904
    >>> ts.Dim().exclusive_min
    -4611686018427387904

Group:
  Accessors
)");

  cls.def_property_readonly("exclusive_max", &IndexInterval::exclusive_max,
                            R"(
Exclusive upper bound of the interval.

Equal to :python:`self.inclusive_max + 1`.  If the interval is unbounded above,
equal to the special value of :py:obj:`+inf+1<tensorstore.inf>`.

Example:

    >>> ts.Dim(inclusive_min=5, inclusive_max=10).exclusive_max
    11
    >>> ts.Dim(exclusive_max=5).exclusive_max
    5
    >>> ts.Dim().exclusive_max
    4611686018427387904

Group:
  Accessors
)");

  cls.def_property_readonly("size", &IndexInterval::size,
                            R"(
Size of the interval.

Equal to :python:`self.exclusive_max - self.inclusive_min`.

Example:

    >>> ts.Dim(5).size
    5
    >>> ts.Dim(inclusive_min=3, inclusive_max=7).size
    5
    >>> ts.Dim().size
    9223372036854775807

Note:

  If the interval is unbounded below or above
  (i.e. :python:`self.finite == False`), this value it not particularly
  meaningful.

Group:
  Accessors
)");

  cls.def_property(
      "implicit_lower",
      [](const IndexDomainDimension<>& x) { return x.implicit_lower(); },
      [](IndexDomainDimension<>& x, bool value) { x.implicit_lower() = value; },
      R"(
Indicates if the lower bound is :ref:`implicit/resizeable<implicit-bounds>`.

Example:

    >>> ts.Dim().implicit_lower
    True
    >>> ts.Dim(5).implicit_lower
    False
    >>> ts.Dim(exclusive_max=5).implicit_lower
    True
    >>> ts.Dim(inclusive_min=1, exclusive_max=5).implicit_lower
    False
    >>> ts.Dim(implicit_lower=False).implicit_lower
    False
    >>> ts.Dim(inclusive_min=5, implicit_lower=True).implicit_lower
    True

Group:
  Accessors
)");

  cls.def_property(
      "implicit_upper",
      [](const IndexDomainDimension<>& x) { return x.implicit_upper(); },
      [](IndexDomainDimension<>& x, bool value) { x.implicit_upper() = value; },
      R"(
Indicates if the upper bound is :ref:`implicit/resizeable<implicit-bounds>`.

Example:

    >>> ts.Dim().implicit_upper
    True
    >>> ts.Dim(5).implicit_upper
    False
    >>> ts.Dim(inclusive_min=5).implicit_upper
    True
    >>> ts.Dim(inclusive_min=1, exclusive_max=5).implicit_upper
    False
    >>> ts.Dim(implicit_upper=False).implicit_upper
    False
    >>> ts.Dim(inclusive_max=5, implicit_upper=True).implicit_upper
    True

Group:
  Accessors
)");

  cls.def_property(
      "label",
      [](const IndexDomainDimension<>& x) { return std::string(x.label()); },
      [](IndexDomainDimension<>& x, const std::string& label) {
        x.label() = label;
      },
      R"(
Dimension label, or the empty string to indicate an unlabeled dimension.

Example:

    >>> ts.Dim().label
    ''
    >>> ts.Dim(label='x').label
    'x'

Group:
  Accessors
)");

  cls.def("__len__", &IndexInterval::size,
          R"(
Size of the interval, equivalent to :py:obj:`.size`.

Group:
  Accessors
)");

  cls.def_property_readonly("empty", &IndexInterval::empty,
                            R"(
Returns `True` if `size` is zero.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "finite", [](const IndexDomainDimension<>& x) { return IsFinite(x); },
      R"(
Indicates if the interval is finite.

Example:

  >>> ts.Dim().finite
  False
  >>> ts.Dim(5).finite
  True
  >>> ts.Dim(exclusive_max=10).finite
  False
  >>> ts.Dim(inclusive_min=10).finite
  False
  >>> ts.Dim(inclusive_min=10, exclusive_max=20).finite
  True

Group:
  Accessors
)");

  cls.def(
      "__contains__",
      [](const IndexDomainDimension<>& x, Index i) { return Contains(x, i); },
      R"(
Checks if the interval contains a given index.

Examples:

    >>> 5 in ts.Dim(inclusive_min=1, exclusive_max=10)
    True
    >>> 5 in ts.Dim()
    True
    >>> 5 in ts.Dim(inclusive_min=6)
    False

Overload:
  index

Group:
  Operations
)",
      py::arg("other"));

  cls.def(
      "__contains__",
      [](const IndexDomainDimension<>& outer,
         const IndexDomainDimension<>& inner) {
        return Contains(outer, inner);
      },
      R"(
Checks if the interval contains another interval.

Examples:

    >>> ts.Dim(inclusive_min=1, exclusive_max=5) in ts.Dim(10)
    True
    >>> ts.Dim(inclusive_min=1, exclusive_max=5) in ts.Dim(4)
    False

Overload:
  dim

Group:
  Operations
)",
      py::arg("inner"));

  cls.def(
      "__iter__",
      [](const IndexDomainDimension<>& self) {
        if (!IsFinite(self)) {
          throw py::value_error("Cannot iterate over infinite interval");
        }
        return py::iter(python_imports.builtins_range_function(
            self.inclusive_min(), self.exclusive_max()));
      },
      R"(
Enables iteration over the indices contained in the interval.

Raises:
    ValueError: If not :py:obj:`.finite`.

Examples:

    >>> list(ts.Dim(inclusive_min=1, exclusive_max=6))
    [1, 2, 3, 4, 5]

Group:
  Operations
)");

  cls.def(
      "__str__",
      [](const IndexDomainDimension<>& x) { return tensorstore::StrCat(x); },
      R"(
Returns the string representation of the interval.

    >>> print(ts.Dim(inclusive_min=5, exclusive_max=10))
    [5, 10)
    >>> print(ts.Dim(exclusive_max=10))
    (-inf*, 10)
    >>> print(ts.Dim(exclusive_max=10, label="x"))
    "x": (-inf*, 10)

)");

  cls.def(
      "__repr__",
      [](const IndexDomainDimension<>& x) {
        std::string repr = "Dim(";
        bool need_comma = false;
        const auto append = [&](auto&&... terms) {
          tensorstore::StrAppend(&repr, need_comma ? ", " : "", terms...);
          need_comma = true;
        };
        if (x.inclusive_min() != -kInfIndex) {
          append("inclusive_min=", x.inclusive_min());
          if (x.implicit_lower()) {
            append("implicit_lower=True");
          }
        } else if (!x.implicit_lower()) {
          append("implicit_lower=False");
        }
        if (x.inclusive_max() != kInfIndex) {
          append("exclusive_max=", x.exclusive_max());
          if (x.implicit_upper()) {
            append("implicit_upper=True");
          }
        } else if (!x.implicit_upper()) {
          append("implicit_upper=False");
        }
        if (!x.label().empty()) {
          append("label=", QuoteString(x.label()));
        }
        repr += ")";
        return repr;
      },
      R"(
Returns the string representation as a Python expression.

    >>> ts.Dim(size=5, label='x', implicit_upper=True)
    Dim(inclusive_min=0, exclusive_max=5, implicit_upper=True, label="x")

)");

  cls.def(
      "__eq__",
      [](const IndexDomainDimension<>& self,
         const IndexDomainDimension<>& other) { return self == other; },
      py::arg("other"),
      R"(
Compares for equality with another interval.

In addition to the bounds, the values of :py:obj:`.label`,
:py:obj:`.implicit_lower`, and :py:obj:`.implicit_upper` are also taken into
account.

    >>> a = ts.Dim(inclusive_min=5, exclusive_max=10)
    >>> b = ts.Dim(inclusive_min=5, inclusive_max=9)
    >>> a == b
    True

Group:
  Operations
)");

  cls.def("__copy__", [](const IndexDomainDimension<>& self) { return self; });
  cls.def(
      "__deepcopy__",
      [](const IndexDomainDimension<>& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(cls);
}

auto MakeOutputIndexMapClass(py::module m) {
  return py::class_<OutputIndexMap>(m, "OutputIndexMap",
                                    R"(
Represents an output index map for an index transform.

See also:
  - :py:obj:`IndexTransform.output`
  - :py:obj:`OutputIndexMaps`
  - :py:obj:`OutputIndexMethod`

Group:
  Indexing
)");
}

void DefineOutputIndexMapAttributes(py::class_<OutputIndexMap>& cls) {
  cls.def(py::init([](Index offset) {
            OutputIndexMap map;
            map.method = OutputIndexMethod::constant;
            map.offset = offset;
            return map;
          }),
          R"(
Constructs a :ref:`constant map<index-transform-constant-map>`.

Example:

    >>> transform = ts.IndexTransform(input_rank=0,
    ...                               output=[ts.OutputIndexMap(offset=5)])
    >>> transform([])
    (5,)

Overload:
  constant
)",
          py::arg("offset") = 0);

  cls.def(py::init([](Index input_dimension, Index offset, Index stride) {
            OutputIndexMap map;
            map.method = OutputIndexMethod::single_input_dimension;
            map.offset = offset;
            map.stride = stride;
            map.input_dimension = input_dimension;
            return map;
          }),
          R"(
Constructs a :ref:`single input dimension map<index-transform-single-input-dimension-map>`.

Example:

    >>> transform = ts.IndexTransform(
    ...     input_rank=1,
    ...     output=[ts.OutputIndexMap(input_dimension=0, offset=5, stride=2)])
    >>> [transform([i]) for i in range(5)]
    [(5,), (7,), (9,), (11,), (13,)]

Overload:
  input_dimension
)",
          py::arg("input_dimension"), py::arg("offset") = Index(0),
          py::arg("stride") = Index(1));

  cls.def(py::init([](SharedArray<const Index> index_array, Index offset,
                      Index stride, const IndexDomainDimension<>& index_range) {
            OutputIndexMap map;
            map.method = OutputIndexMethod::array;
            map.offset = offset;
            map.stride = stride;
            map.index_array = index_array;
            map.index_range = index_range;
            return map;
          }),
          R"(
Constructs an :ref:`index array map<index-transform-array-map>`.

Example:

    >>> transform = ts.IndexTransform(
    ...     input_shape=[5],
    ...     output=[ts.OutputIndexMap(index_array=[2, 3, 5, 7, 11])])
    >>> [transform([i]) for i in range(5)]
    [(2,), (3,), (5,), (7,), (11,)]

Overload:
  index_array
)",
          py::arg("index_array"), py::arg("offset") = Index(0),
          py::arg("stride") = Index(1),
          py::arg("index_range") = IndexDomainDimension<>());

  cls.def_property_readonly(
      "method", [](const OutputIndexMap& self) { return self.method; });

  cls.def_property_readonly(
      "offset", [](const OutputIndexMap& self) { return self.offset; });

  cls.def_property_readonly(
      "stride", [](const OutputIndexMap& self) -> std::optional<Index> {
        if (self.method == OutputIndexMethod::constant) {
          return std::nullopt;
        }
        return self.stride;
      });

  cls.def_property_readonly(
      "input_dimension",
      [](const OutputIndexMap& self) -> std::optional<DimensionIndex> {
        if (self.method != OutputIndexMethod::single_input_dimension) {
          return std::nullopt;
        }
        return self.input_dimension;
      });

  cls.def_property_readonly("index_array",
                            [](const OutputIndexMap& self)
                                -> std::optional<SharedArray<const Index>> {
                              if (self.method != OutputIndexMethod::array) {
                                return std::nullopt;
                              }
                              return self.index_array;
                            });

  cls.def_property_readonly(
      "index_range",
      [](const OutputIndexMap& self) -> std::optional<IndexDomainDimension<>> {
        if (self.method != OutputIndexMethod::array) {
          return std::nullopt;
        }
        IndexDomainDimension<> d;
        d.interval() = self.index_range;
        d.implicit_lower() = false;
        d.implicit_upper() = false;
        return d;
      });

  cls.def("__repr__", &OutputIndexMapToString);

  cls.def(
      "__eq__",
      [](const OutputIndexMap& self, const OutputIndexMap& other) {
        return self == other;
      },
      py::arg("other"));

  cls.def(py::pickle(
      [](const OutputIndexMap& self) {
        switch (self.method) {
          case OutputIndexMethod::constant:
            return py::make_tuple(self.method, self.offset);
          case OutputIndexMethod::single_input_dimension:
            return py::make_tuple(self.method, self.offset, self.stride,
                                  self.input_dimension);
          case OutputIndexMethod::array:
            return py::make_tuple(
                self.method, self.offset, self.stride, self.index_array,
                IndexDomainDimension<>(OptionallyImplicitIndexInterval(
                    self.index_range, false, false)));
        }
        ABSL_UNREACHABLE();  // COV_NF_LINE
      },
      [](py::tuple t) {
        OutputIndexMap map;
        map.method = py::cast<OutputIndexMethod>(t[0]);
        map.offset = py::cast<Index>(t[1]);
        if (map.method != OutputIndexMethod::constant) {
          map.stride = py::cast<Index>(t[2]);
        }
        switch (map.method) {
          case OutputIndexMethod::constant:
            break;
          case OutputIndexMethod::single_input_dimension:
            map.input_dimension = py::cast<DimensionIndex>(t[3]);
            break;
          case OutputIndexMethod::array:
            map.index_array = py::cast<SharedArray<const Index>>(t[3]);
            map.index_range = py::cast<IndexDomainDimension<>>(t[4]);
            break;
          default:
            throw py::value_error("Failed to unpickle OutputIndexMap");
        }
        return map;
      }));
}

auto MakeOutputIndexMapsClass(py::module m) {
  return py::class_<OutputIndexMapRangeContainer>(m, "OutputIndexMaps",
                                                  R"(
View of the output index maps for an index transform.

See also:
  - :py:obj:`IndexTransform.output`
  - :py:obj:`OutputIndexMap`
  - :py:obj:`OutputIndexMethod`

Group:
  Indexing
)");
}

void DefineOutputIndexMapsAttributes(
    py::class_<OutputIndexMapRangeContainer>& cls) {
  cls.def_property_readonly("rank", &OutputIndexMapRangeContainer::size,
                            "Returns the output rank.");

  cls.def("__len__", &OutputIndexMapRangeContainer::size,
          "Returns the output rank.");
  cls.def("__getitem__",
          [](const OutputIndexMapRangeContainer& r,
             PythonDimensionIndex i) -> OutputIndexMap {
            return r[NormalizePythonDimensionIndex(i, r.size())];
          });

  cls.def("__repr__", [](const OutputIndexMapRangeContainer& r) {
    std::string out = "[";
    for (DimensionIndex i = 0; i < r.size(); ++i) {
      if (i != 0) out += ", ";
      out += OutputIndexMapToString(r[i]);
    }
    out += "]";
    return out;
  });

  cls.def("__eq__", [](const OutputIndexMapRangeContainer& r,
                       const SequenceParameter<OutputIndexMap>& other) {
    if (r.size() != static_cast<DimensionIndex>(other.size())) return false;
    for (DimensionIndex i = 0; i < r.size(); ++i) {
      if (OutputIndexMap(r[i]) != other[i]) return false;
    }
    return true;
  });
}

auto MakeOutputIndexMethodClass(py::module m) {
  return py::enum_<OutputIndexMethod>(m, "OutputIndexMethod",
                                      R"(
Indicates the :ref:`output index method<output-index-methods>` of an :py:class:`OutputIndexMap`.

See also:
  - :py:obj:`IndexTransform.output`
  - :py:obj:`OutputIndexMap`
  - :py:obj:`OutputIndexMaps`

Group:
  Indexing

)");
}

void DefineOutputIndexMethodAttributes(py::enum_<OutputIndexMethod>& cls) {
  cls.value("constant", OutputIndexMethod::constant);

  cls.value("single_input_dimension",
            OutputIndexMethod::single_input_dimension);

  cls.value("array", OutputIndexMethod::array);
}

void RegisterIndexSpaceBindings(pybind11::module m, Executor defer) {
  m.attr("inf") = kInfIndex;

  defer([cls = MakeIndexDomainClass(m)]() mutable {
    DefineIndexDomainAttributes(cls);
  });

  defer([cls = MakeIndexTransformClass(m)]() mutable {
    DefineIndexTransformAttributes(cls);
  });

  defer([cls = MakeDimClass(m)]() mutable { DefineDimAttributes(cls); });

  defer([cls = MakeOutputIndexMapClass(m)]() mutable {
    DefineOutputIndexMapAttributes(cls);
  });

  defer([cls = MakeOutputIndexMapsClass(m)]() mutable {
    DefineOutputIndexMapsAttributes(cls);
  });

  defer([cls = MakeOutputIndexMethodClass(m)]() mutable {
    DefineOutputIndexMethodAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterIndexSpaceBindings, /*priority=*/-900);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
