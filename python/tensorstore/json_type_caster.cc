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

#include "python/tensorstore/json_type_caster.h"

#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace tensorstore {
namespace internal_python {

namespace py = pybind11;

py::object JsonToPyObject(const ::nlohmann::json& value) noexcept {
  using ::nlohmann::json;
  using value_t = json::value_t;
  switch (value.type()) {
    case value_t::object: {
      py::object obj = py::reinterpret_steal<py::object>(PyDict_New());
      if (!obj) return {};
      for (const auto& p : value.get_ref<const json::object_t&>()) {
        auto key_obj = py::reinterpret_steal<py::object>(
            PyUnicode_FromStringAndSize(p.first.data(), p.first.size()));
        if (!key_obj) return {};
        auto value_obj = JsonToPyObject(p.second);
        if (!value_obj) return {};
        if (PyDict_SetItem(obj.ptr(), key_obj.release().ptr(),
                           value_obj.release().ptr()) == -1) {
          return {};
        }
      }
      return obj;
    }
    case value_t::array: {
      const auto& arr = value.get_ref<const json::array_t&>();
      py::object obj =
          py::reinterpret_steal<py::object>(PyList_New(arr.size()));
      if (!obj) return {};
      for (size_t i = 0; i < arr.size(); ++i) {
        auto value_obj = JsonToPyObject(arr[i]);
        if (!value_obj) return {};
        PyList_SET_ITEM(obj.ptr(), i, value_obj.release().ptr());
      }
      return obj;
    }
    case value_t::string: {
      const auto& s = value.get_ref<const std::string&>();
      return py::reinterpret_steal<py::object>(
          PyUnicode_FromStringAndSize(s.data(), s.size()));
    }
    case value_t::boolean:
      return py::reinterpret_borrow<py::object>(value.get<bool>() ? Py_True
                                                                  : Py_False);
    case value_t::number_unsigned:
      return py::reinterpret_steal<py::object>(
          PyLong_FromUnsignedLongLong(value.get<std::uint64_t>()));
    case value_t::number_integer:
      return py::reinterpret_steal<py::object>(
          PyLong_FromLongLong(value.get<std::uint64_t>()));
    case value_t::number_float:
      return py::reinterpret_steal<py::object>(
          PyFloat_FromDouble(value.get<double>()));
    case value_t::null:
    case value_t::discarded:
    default:
      return py::none();
  }
}

namespace {
::nlohmann::json PyObjectToJsonInteger(py::handle h) {
  // Convert all non-negative numbers to `uint64_t`, although for numbers that
  // can also be represented by `int64_t`, using `int64_t` would be equally
  // good.
  if (auto v = PyLong_AsUnsignedLongLong(h.ptr());
      v != static_cast<decltype(v)>(-1) || !PyErr_Occurred()) {
    return static_cast<std::uint64_t>(v);
  }
  PyErr_Clear();
  if (auto v = PyLong_AsLongLong(h.ptr());
      v != static_cast<decltype(v)>(-1) || !PyErr_Occurred()) {
    return static_cast<std::int64_t>(v);
  }
  throw py::error_already_set();
}
}  // namespace

::nlohmann::json PyObjectToJson(py::handle h, int max_depth) {
  if (max_depth <= 0) {
    throw py::value_error("Recursion limit exceeded converting value to JSON");
  }

  if (h.is_none()) return nullptr;
  if (py::isinstance<py::bool_>(h)) return h.cast<bool>();
  if (py::isinstance<py::int_>(h)) return PyObjectToJsonInteger(h);
  if (py::isinstance<py::float_>(h)) {
    double v = PyFloat_AsDouble(h.ptr());
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
  }
  if (py::isinstance<py::str>(h)) {
    return h.cast<std::string>();
  }
  if (py::isinstance<py::tuple>(h) || py::isinstance<py::list>(h)) {
    ::nlohmann::json::array_t arr;
    // Check size on every iteration of loop because size may change during loop
    // due to possible calls back to Python code.
    for (ssize_t i = 0; i < PySequence_Fast_GET_SIZE(h.ptr()); ++i) {
      arr.push_back(PyObjectToJson(py::reinterpret_borrow<py::object>(
                                       PySequence_Fast_GET_ITEM(h.ptr(), i)),
                                   max_depth - 1));
    }
    return arr;
  }
  if (py::isinstance<py::dict>(h)) {
    ::nlohmann::json::object_t obj;
    ssize_t pos = 0;
    PyObject* key;
    PyObject* value;
    while (PyDict_Next(h.ptr(), &pos, &key, &value)) {
      // Create temporary references to ensure objects remain alive during
      // conversion, as calls back to Python code may cause the dict to change.
      py::object key_obj = py::reinterpret_borrow<py::object>(key);
      py::object value_obj = py::reinterpret_borrow<py::object>(value);
      obj.emplace(py::cast<std::string>(py::cast<py::str>(key_obj)),
                  PyObjectToJson(value_obj, max_depth - 1));
    }
    return obj;
  }

  // Handle numpy array.
  if (py::isinstance<py::array>(h)) {
    ::nlohmann::json::array_t arr;
    ssize_t size = PySequence_Size(h.ptr());
    if (size == -1) throw py::error_already_set();
    for (ssize_t i = 0; i < size; ++i) {
      arr.push_back(PyObjectToJson(
          py::reinterpret_steal<py::object>(PySequence_GetItem(h.ptr(), i)),
          max_depth - 1));
    }
    return arr;
  }

  // Check for any integer type.  This must be done after the check for a numpy
  // array, as numpy arrays define an `__int__` type.
  if (PyIndex_Check(h.ptr())) return PyObjectToJsonInteger(h);

  // Check for any floating-point type.
  if (double v = PyFloat_AsDouble(h.ptr()); v != -1 || !PyErr_Occurred()) {
    return v;
  }
  PyErr_Clear();

  return PyObjectToJson(h.attr("to_json")(), max_depth - 1);
}

}  // namespace internal_python
}  // namespace tensorstore
