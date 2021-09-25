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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <cstddef>
#include <vector>

#include "python/tensorstore/type_name_override.h"
#include <pybind11/stl.h>

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

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("Sequence[") +
      pybind11::detail::make_caster<T>::name + pybind11::detail::_("]");
};

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_SEQUENCE_PARAMETER_H_
