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

#ifndef THIRD_PARTY_PY_TENSORSTORE_DOWNSAMPLE_H_
#define THIRD_PARTY_PY_TENSORSTORE_DOWNSAMPLE_H_

/// \file
///
/// Defines `tensorstore.downsample`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "tensorstore/downsample_method.h"

namespace pybind11 {
namespace detail {

/// Defines automatic conversion of `DownsampleMethod` to/from Python string
/// constants.
template <>
struct type_caster<tensorstore::DownsampleMethod> {
  using T = tensorstore::DownsampleMethod;
  PYBIND11_TYPE_CASTER(T, _("DownsampleMethod"));
  bool load(handle src, bool convert);
  static handle cast(tensorstore::DownsampleMethod value,
                     return_value_policy /* policy */, handle /* parent */);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_DOWNSAMPLE_H_
