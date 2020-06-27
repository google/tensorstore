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

#ifndef THIRD_PARTY_PY_TENSORSTORE_DIM_EXPRESSION_H_
#define THIRD_PARTY_PY_TENSORSTORE_DIM_EXPRESSION_H_

/// \file Defines the `tensorstore.d` object which supports the
/// `tensorstore.d[...].op0...opN` syntax for specifying a Python "dimension
/// expression".

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "python/tensorstore/index.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Parameter type for pybind11-exposed functions that identifies a dimension by
/// index or label.
///
/// This is used in place of `tensorstore::DimensionIdentifier`, which merely
/// holds a `string_view`, but does not own the label string.
using PythonDimensionIdentifier =
    std::variant<PythonDimensionIndex, std::string>;

/// Converts a `PythonDimensionIdentifier` to a `DimensionIdentifier` that
/// references it.
inline DimensionIdentifier ToDimensionIdentifier(
    const PythonDimensionIdentifier& identifier) {
  if (auto* index = std::get_if<PythonDimensionIndex>(&identifier)) {
    return index->value;
  }
  return std::get<std::string>(identifier);
}

/// Converts a `PythonDimensionIdentifier` to a `DynamicDimSpec` copy.
inline DynamicDimSpec ToDynamicDimSpec(
    const PythonDimensionIdentifier& identifier) {
  if (auto* index = std::get_if<PythonDimensionIndex>(&identifier)) {
    return index->value;
  }
  return std::get<std::string>(identifier);
}

/// Base class for Python representation of a "dimension expression".
///
/// A dimension expression consists of a `DimensionSelection` followed by a
/// sequence of zero or more operations.
///
/// This behaves similarly `tensorstore::DimExpression`.  We can't simply use
/// `tensorstore::DimExpression` because it holds vectors by reference rather
/// than by value, and because it does not do type erasure.
class PythonDimExpressionBase {
 public:
  /// Returns the string representation for `__repr__`.
  virtual std::string repr() const = 0;

  /// Applies the operation to `transform` using the dimension selection
  /// specified by `*dimensions`.
  ///
  /// \param transform The existing transform with which to compose the
  ///     operations represented by this dimension expression.
  /// \param dimensions[in,out] Non-null pointer.  On input, specifies the
  ///     existing dimension selection (corresponding to the domain of
  ///     `transform`).  On output, set to the new dimension selection
  ///     corresponding to the domain of the returned transform.
  virtual Result<IndexTransform<>> Apply(
      IndexTransform<> transform, DimensionIndexBuffer* dimensions) const = 0;

  virtual ~PythonDimExpressionBase() = default;
};

/// Specifies a sequence of existing or new dimensions, and serves as the
/// starting point for a dimension expression.
class DimensionSelection : public PythonDimExpressionBase {
 public:
  DimensionSelection() = default;

  std::string repr() const override;

  /// Sets `*dimensions` to the list of existing dimensions, and returns
  /// `transform` unmodified.
  Result<IndexTransform<>> Apply(
      IndexTransform<> transform,
      DimensionIndexBuffer* dimensions) const override;

  /// Specifies the dimension selection.
  std::vector<DynamicDimSpec> dims;
};

/// Base class for dimension expressions with at least one operation.
class PythonDimExpression : public PythonDimExpressionBase {};

/// Converts a Python object to a dimension selection.
///
/// Supports Python objects that support the `__index__` protocol, unicode
/// strings, `slice` objects, existing `DimensionSelection` objects, and
/// sequences thereof.
bool CastToDimensionSelection(pybind11::handle src, DimensionSelection* out);

void RegisterDimExpressionBindings(pybind11::module m);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from Python objects to `DimensionSelection`
/// parameters.
template <>
struct type_caster<tensorstore::internal_python::DimensionSelection>
    : public type_caster_base<
          tensorstore::internal_python::DimensionSelection> {
  using Base =
      type_caster_base<tensorstore::internal_python::DimensionSelection>;

  bool load(handle src, bool convert) {
    if (Base::load(src, convert)) {
      return true;
    }
    auto obj =
        std::make_unique<tensorstore::internal_python::DimensionSelection>();
    if (tensorstore::internal_python::CastToDimensionSelection(src,
                                                               obj.get())) {
      value = obj.release();
      return true;
    }
    return false;
  }
};

/// Defines automatic conversion between `DimRangeSpec` and Python slice
/// objects.
template <>
struct type_caster<tensorstore::DimRangeSpec> {
  PYBIND11_TYPE_CASTER(tensorstore::DimRangeSpec, _("slice"));

  bool load(handle src, bool convert) {
    if (!PySlice_Check(src.ptr())) return false;
    ssize_t start, stop, step;
    if (PySlice_Unpack(src.ptr(), &start, &stop, &step) != 0) {
      return false;
    }
    auto* slice_obj = reinterpret_cast<PySliceObject*>(src.ptr());
    if (slice_obj->start != Py_None) value.inclusive_start = start;
    if (slice_obj->stop != Py_None) value.exclusive_stop = stop;
    value.step = step;
    return true;
  }

  static handle cast(const tensorstore::DimRangeSpec& x,
                     return_value_policy /* policy */, handle /* parent */) {
    return handle(
        PySlice_New(pybind11::cast(x.inclusive_start).ptr(),
                    pybind11::cast(x.exclusive_stop).ptr(),
                    x.step == 1 ? nullptr : pybind11::cast(x.step).ptr()));
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_DIM_EXPRESSION_H_
