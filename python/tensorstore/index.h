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

#ifndef THIRD_PARTY_PY_TENSORSTORE_INDEX_H_
#define THIRD_PARTY_PY_TENSORSTORE_INDEX_H_

/// \file Defines mappings between Python objects and C++ types intended to
///     represent Index and DimensionIndex values.

#include <string>
#include <variant>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"

namespace tensorstore {
namespace internal_python {

/// Wrapper type used for `Index` parameters to pybind11-exposed functions for
/// which `kImplicit` (represented by `None` in Python) is permitted.
struct OptionallyImplicitIndex {
  Index value = kImplicit;
  Index value_or(Index default_value) const {
    return value == kImplicit ? default_value : value;
  }
  /// Conversion operator, which simplifies construction of an `Index` container
  /// from an `OptionallyImplicitIndex` range.
  constexpr operator Index() const { return value; }
};

/// Converts `kImplicit` -> `"None"`, other values to decimal integer
/// representation.
std::string OptionallyImplicitIndexRepr(Index value);

/// Wrapper type used for `DimensionIndex` parameters to pybind11-exposed
/// functions.
struct PythonDimensionIndex {
  DimensionIndex value;
  /// Conversion operator, which simplifies construction of a `DimensionIndex`
  /// container from a `PythonDimensionIndex` range.
  operator DimensionIndex() const { return value; }
};

/// Represents either an integer index or vector of integer indices.
///
/// We can't use `tensorstore::internal_index_space::IndexVectorOrScalar`, since
/// that merely references a vector, but does not own it.
using IndexVectorOrScalarContainer = std::variant<std::vector<Index>, Index>;

/// Represents either a single integer index or vector of integer indices, where
/// `kImplicit` (represented by `None` in Python) is permitted.
using OptionallyImplicitIndexVectorOrScalarContainer =
    std::variant<std::vector<OptionallyImplicitIndex>, OptionallyImplicitIndex>;

/// Converts `x` to an `IndexVectorOrScalar`, mapping `kImplicit` to
/// `implicit_value`.
IndexVectorOrScalarContainer ToIndexVectorOrScalarContainer(
    const OptionallyImplicitIndexVectorOrScalarContainer& x,
    Index implicit_value = kImplicit);

/// Returns a Python expression representation of an
/// `IndexVectorOrScalarContainer`.
///
/// \param x The scalar/vector to convert.
/// \param implicit If `false`, use the normal decimal representation of each
///     index.  If `true`, use `OptionallyImplicitIndexRepr`.
/// \param subscript If `false`, format vectors as a Python array.
std::string IndexVectorRepr(const IndexVectorOrScalarContainer& x,
                            bool implicit = false, bool subscript = false);

/// Returns an `IndexVectorOrScalar` that references `x`.
internal_index_space::IndexVectorOrScalar ToIndexVectorOrScalar(
    const IndexVectorOrScalarContainer& x);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion of Python objects to `PythonDimensionIndex`
/// parameter types of pybind11-exposed functions.
///
/// For consistency with builtin `__getitem__` and NumPy, this handles any
/// Python type that supports the `__index__` protocol, which is similar to
/// integer conversion but excludes floating-point types.
template <>
struct type_caster<tensorstore::internal_python::PythonDimensionIndex> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::PythonDimensionIndex,
                       _("DimensionIndex"));
  static handle cast(tensorstore::internal_python::PythonDimensionIndex x,
                     return_value_policy /* policy */, handle /* parent */) {
    return int_(x.value).release();
  }
  bool load(handle src, bool convert) {
    value.value = PyNumber_AsSsize_t(src.ptr(), PyExc_IndexError);
    if (value.value == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    return true;
  }
};

/// Defines automatic conversion of Python objects to/from
/// `OptionallyImplicitIndex` (and via the `variant` conversion provided by
/// pybind11, `IndexVectorOrScalarContainer`).  This is used to convert
/// parameters of pybind11-exposed functions.
///
/// `None` maps to `kImplicit`.  Otherwise, the `__index__` protocol is used,
/// which is similar to normal integer conversion but excludes floating-point
/// types.
template <>
struct type_caster<tensorstore::internal_python::OptionallyImplicitIndex> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::OptionallyImplicitIndex,
                       _("Index"));
  static handle cast(tensorstore::internal_python::OptionallyImplicitIndex x,
                     return_value_policy /* policy */, handle /* parent */) {
    if (x.value == tensorstore::kImplicit) return none().release();
    return int_(x.value).release();
  }

  bool load(handle src, bool convert) {
    if (src.is_none()) {
      value.value = tensorstore::kImplicit;
      return true;
    }
    value.value = PyNumber_AsSsize_t(src.ptr(), PyExc_IndexError);
    if (value.value == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_INDEX_H_
