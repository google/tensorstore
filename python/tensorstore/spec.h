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

#ifndef THIRD_PARTY_PY_TENSORSTORE_SPEC_H_
#define THIRD_PARTY_PY_TENSORSTORE_SPEC_H_

/// \file
///
/// Defines `tensorstore.Spec`.

#include <string>

#include "pybind11/pybind11.h"
#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_python {

void RegisterSpecBindings(pybind11::module m);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from compatible Python objects to
/// `tensorstore::Spec` parameters of pybind11-exposed functions, via JSON
/// conversion.
template <>
struct type_caster<tensorstore::Spec>
    : public type_caster_base<tensorstore::Spec> {
  using Base = type_caster_base<tensorstore::Spec>;
  bool load(handle src, bool convert);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_SPEC_H_
