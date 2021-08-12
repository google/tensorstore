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

#ifndef THIRD_PARTY_PY_TENSORSTORE_SEQUENCE_PARAMETER_H_
#define THIRD_PARTY_PY_TENSORSTORE_SEQUENCE_PARAMETER_H_

#include <cstddef>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace tensorstore {
namespace internal_python {

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

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

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

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_SEQUENCE_PARAMETER_H_
