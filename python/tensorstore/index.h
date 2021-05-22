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
#include "tensorstore/index_space/internal/numpy_indexing_spec.h"

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

/// Wrapper around `std::vector<T>` that displays as `Sequence[T]`.
///
/// This is intended as a pybind11 function parameter type, in order to
/// accurately indicate in the signature that any sequence type is supported.
/// In contrast, using `std::vector<T>` directly leads to an annotation of
/// `List[T]` which is correct for return types (conversions to Python), but not
/// correct for parameter types.
template <typename T>
struct SequenceParameter {
  SequenceParameter() = default;
  SequenceParameter(const std::vector<T>& other) : value(other) {}
  SequenceParameter(std::vector<T>&& other) : value(std::move(other)) {}

  auto begin() const { return value.begin(); }
  auto end() const { return value.end(); }

  size_t size() const { return value.size(); }
  const T* data() const { return value.data(); }
  const T& operator[](size_t i) const { return value[i]; }

  std::vector<T> value;
};

/// Wrapper type used for `DimensionIndex` parameters to pybind11-exposed
/// functions.
struct PythonDimensionIndex {
  DimensionIndex value;
  /// Conversion operator, which simplifies construction of a `DimensionIndex`
  /// container from a `PythonDimensionIndex` range.
  operator DimensionIndex() const { return value; }
};

using internal_index_space::IndexVectorOrScalarContainer;

/// Represents either a single integer index or vector of integer indices, where
/// `kImplicit` (represented by `None` in Python) is permitted.
using OptionallyImplicitIndexVectorOrScalarContainer =
    std::variant<SequenceParameter<OptionallyImplicitIndex>,
                 OptionallyImplicitIndex>;

/// Converts `x` to an `IndexVectorOrScalarView`, mapping `kImplicit` to
/// `implicit_value`.
IndexVectorOrScalarContainer ToIndexVectorOrScalarContainer(
    const OptionallyImplicitIndexVectorOrScalarContainer& x,
    Index implicit_value = kImplicit);

using internal::OptionallyImplicitIndexRepr;

/// Returns a Python expression representation of an
/// `IndexVectorOrScalarContainer`.
///
/// \param x The scalar/vector to convert.
/// \param implicit If `false`, use the normal decimal representation of each
///     index.  If `true`, use `OptionallyImplicitIndexRepr`.
/// \param subscript If `false`, format vectors as a Python array.
std::string IndexVectorRepr(const IndexVectorOrScalarContainer& x,
                            bool implicit = false, bool subscript = false);

/// Wrapper around a Python tuple that displays as `Tuple[T, ...]`.
///
/// This is intended as a pybind11 function return type, in order to indicate
/// that the return type is a homogeneous tuple.  For example, this is used as
/// the return type for `IndexDomain.shape`.
template <typename T>
struct HomogeneousTuple {
  pybind11::tuple obj;
};

template <typename T>
HomogeneousTuple<T> SpanToHomogeneousTuple(span<const T> vec) {
  pybind11::tuple t(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    t[i] = pybind11::cast(vec[i]);
  }
  return HomogeneousTuple<T>{std::move(t)};
}

inline std::optional<DimensionIndex> RankOrNone(DimensionIndex rank) {
  if (rank == dynamic_rank) return std::nullopt;
  return rank;
}

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
                       _("int"));
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
                       _("Optional[int]"));
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

/// Defines automatic conversion of Python objects to `SequenceParmaeter`.
///
/// Since `SequenceParameter` is only intended as a function parameter type, we
/// do not define the reverse conversion back to a Python object.
template <typename T>
struct type_caster<tensorstore::internal_python::SequenceParameter<T>> {
  using base_value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::SequenceParameter<T>,
                       _("Sequence[") + base_value_conv::name + _("]"));
  using value_conv = make_caster<std::vector<T>>;

  bool load(handle src, bool convert) {
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) return false;
    value.value = cast_op<std::vector<T>&&>(std::move(inner_caster));
    return true;
  }
};

/// Defines automatic conversion of `HomogeneousTuple<T>` to a Python object.
///
/// Since `HomogeneousTuple` is only intended as a function return type, we do
/// not define the reverse conversion from a Python object.
template <typename T>
struct type_caster<tensorstore::internal_python::HomogeneousTuple<T>> {
  using Value = tensorstore::internal_python::HomogeneousTuple<T>;
  using base_value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(Value,
                       _("Tuple[") + base_value_conv::name + _(", ...]"));
  static handle cast(Value value, return_value_policy policy, handle parent) {
    return value.obj.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_INDEX_H_
