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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "python/tensorstore/index.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/executor.h"
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

/// Appends to `*out` a Python repr of `dims`.
void AppendDimensionSelectionRepr(std::string* out,
                                  span<const DynamicDimSpec> dims);

/// Base class for Python representation of a "dimension expression".
///
/// A dimension expression consists of a `DimensionSelection` followed by a
/// sequence of zero or more operations.
///
/// This behaves similarly `tensorstore::DimExpression`.  We can't simply use
/// `tensorstore::DimExpression` because it holds vectors by reference rather
/// than by value, and because it does not do type erasure.
class PythonDimExpression {
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
  /// \param top_level Indicates whether this expression is the top-level
  ///     (outer-most) expression being applied, i.e. `dimensions` is the
  ///     dimension selection specified directly.  When an operation recursively
  ///     invokes a parent operation, it must specify `top_level=false`.  This
  ///     option is checked by `DimensionSelection::Apply` in order to return an
  ///     error if no operations are specified.
  /// \param domain_only Indicates the output dimensions of `transform` should
  ///     be ignored, and returned transform should have an output rank of 0.
  virtual Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                         DimensionIndexBuffer* dimensions,
                                         bool top_level,
                                         bool domain_only) const = 0;

  virtual ~PythonDimExpression() = default;
};

/// Specifies a sequence of existing or new dimensions, and serves as the
/// starting point for a dimension expression.
class DimensionSelection : public PythonDimExpression {
 public:
  DimensionSelection() = default;

  std::string repr() const override;

  /// Sets `*dimensions` to the list of existing dimensions, and returns
  /// `transform` unmodified.
  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool top_level,
                                 bool domain_only) const override;

  /// Specifies the dimension selection.
  std::vector<DynamicDimSpec> dims;
};

/// Converts a Python object to a dimension selection.
///
/// Supports Python objects that support the `__index__` protocol, unicode
/// strings, `slice` objects, existing `DimensionSelection` objects, and
/// sequences thereof.
bool CastToDimensionSelection(pybind11::handle src, DimensionSelection& out);

/// Wrapper type used to indicate parameters to pybind11-wrapped functions that
/// may be specified either as `tensorstore.d` objects, or anything supported by
/// `CastToDimensionSelection`.
struct DimensionSelectionLike {
  DimensionSelection value;
};

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from Python objects to `DimensionSelectionLike`
/// parameters.
template <>
struct type_caster<tensorstore::internal_python::DimensionSelectionLike> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::DimensionSelectionLike,
                       _("DimSelectionLike"));

  bool load(handle src, bool convert);
  static handle cast(tensorstore::internal_python::DimensionSelectionLike value,
                     return_value_policy policy, handle parent);
};

/// Defines automatic conversion between `DimRangeSpec` and Python slice
/// objects.
template <>
struct type_caster<tensorstore::DimRangeSpec> {
  PYBIND11_TYPE_CASTER(tensorstore::DimRangeSpec, _("slice"));

  bool load(handle src, bool convert);
  static handle cast(const tensorstore::DimRangeSpec& x,
                     return_value_policy /* policy */, handle /* parent */);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_DIM_EXPRESSION_H_
