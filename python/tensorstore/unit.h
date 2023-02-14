// Copyright 2021 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_UNIT_H_
#define THIRD_PARTY_PY_TENSORSTORE_UNIT_H_

/// \file
///
/// Defines `tensorstore.Unit`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal_python {

/// Wrapper type used to indicate parameters that may be specified either as
/// `tensorstore.Unit` objects or `str` or `numbers.Real` or
/// `Tuple[numbers.Real, str]` values.
struct UnitLike {
  Unit value;
};

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from compatible Python objects to
/// `tensorstore::Unit` parameters of pybind11-exposed functions.
template <>
struct type_caster<tensorstore::internal_python::UnitLike> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::UnitLike,
                       _("Union[tensorstore.Unit, str, Real, "
                         "Tuple[Real, str]]"));
  bool load(handle src, bool convert);
  static handle cast(const tensorstore::internal_python::UnitLike& value,
                     return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_UNIT_H_
