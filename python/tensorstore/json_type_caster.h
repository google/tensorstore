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

#ifndef THIRD_PARTY_PY_TENSORSTORE_JSON_TYPE_CASTER_H_
#define THIRD_PARTY_PY_TENSORSTORE_JSON_TYPE_CASTER_H_

/// \file
///
/// Defines conversion between Python objects and `::nlohmann::json`.

#include <nlohmann/json.hpp>
#include "pybind11/pybind11.h"

namespace tensorstore {
namespace internal_python {

/// Converts a JSON value to a Python object representation.
///
/// \returns Non-null object on success, or `nullptr` on failure with a Python
///     error already set.
pybind11::object JsonToPyObject(const ::nlohmann::json& value) noexcept;

/// Returns a JSON representation of a Python object `h`.
///
/// In addition to built-in `int`, `str`, `bool`, `dict`, `list`, `tuple`, and
/// `None` types, NumPy array_like types and array scalars are also supported.
::nlohmann::json PyObjectToJson(pybind11::handle h, int max_depth = 20);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion between Python objects and `::nlohmann::json`.
template <>
struct type_caster< ::nlohmann::json> {
  PYBIND11_TYPE_CASTER(::nlohmann::json, _("json"));

  static handle cast(const ::nlohmann::json& value,
                     return_value_policy /* policy */, handle /* parent */) {
    auto h = tensorstore::internal_python::JsonToPyObject(value).release();
    if (!h) throw error_already_set();
    return h;
  }
  bool load(handle src, bool convert) {
    value = tensorstore::internal_python::PyObjectToJson(src);
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_JSON_TYPE_CASTER_H_
