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

#include "python/tensorstore/data_type.h"

#include <array>
#include <new>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "python/tensorstore/json_type_caster.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorstore/data_type.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

pybind11::dtype GetNumpyDtype(int type_num) {
  using py::detail::npy_api;
  auto& api = npy_api::get();
  if (auto* obj = api.PyArray_DescrFromType_(type_num)) {
    return py::reinterpret_borrow<py::dtype>(obj);
  }
  throw py::error_already_set();
}

DataType GetDataTypeOrThrow(absl::string_view name) {
  auto d = GetDataType(name);
  if (!d.valid()) {
    throw py::value_error(
        StrCat("No TensorStore data type with name: ", QuoteString(name)));
  }
  return d;
}

int GetNumpyTypeNum(DataType data_type) {
  const DataTypeId id = data_type.id();
  if (id == DataTypeId::custom) return -1;
  return kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)];
}

py::dtype GetNumpyDtypeOrThrow(DataType data_type) {
  int type_num = GetNumpyTypeNum(data_type);
  if (type_num != -1) {
    return GetNumpyDtype(type_num);
  }
  throw py::value_error(
      StrCat("No NumPy dtype corresponding to TensorStore data type: ",
             QuoteString(data_type.name())));
}

DataType GetDataType(pybind11::dtype dt) {
  const int type_num = py::detail::array_descriptor_proxy(dt.ptr())->type_num;
  if (type_num < 0 || type_num > NPY_NTYPES_) return DataType();
  const DataTypeId id = kDataTypeIdForNumpyTypeNum[type_num];
  if (id == DataTypeId::custom) return DataType();
  return kDataTypes[static_cast<size_t>(id)];
}

DataType GetDataTypeOrThrow(py::dtype dt) {
  auto data_type = GetDataType(dt);
  if (data_type.valid()) return data_type;
  throw py::value_error(
      StrCat("No TensorStore data type corresponding to NumPy dtype: ",
             py::cast<std::string>(py::repr(dt))));
}

py::object GetTypeObjectOrThrow(DataType data_type) {
  switch (data_type.id()) {
    case DataTypeId::ustring_t:
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(&PyUnicode_Type));
    case DataTypeId::string_t:
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(&PyBytes_Type));
    default:
      break;
  }
  auto numpy_dtype = GetNumpyDtypeOrThrow(data_type);
  return py::reinterpret_borrow<py::object>(
      py::detail::array_descriptor_proxy(numpy_dtype.ptr())->typeobj);
}

void RegisterDataTypeBindings(pybind11::module m) {
  py::class_<DataType> cls_data_type(m, "dtype", R"(
Represents a TensorStore data type.
)");
  cls_data_type
      .def(py::init([](std::string name) { return GetDataTypeOrThrow(name); }),
           "Construct by name.", py::arg("name"))
      .def(py::init([](py::dtype dt) { return GetDataTypeOrThrow(dt); }),
           "Construct from a NumPy dtype.", py::arg("dtype"))
      .def(py::init([](DataType data_type) { return data_type; }),
           "Construct from anything.", py::arg("name"))
      .def_property_readonly(
          "name", [](DataType self) { return std::string(self.name()); })
      .def("__repr__",
           [](DataType self) {
             return StrCat("dtype(", QuoteString(self.name()), ")");
           })
      .def(
          py::pickle([](DataType self) { return std::string(self.name()); },
                     [](std::string name) { return GetDataTypeOrThrow(name); }))
      .def("to_json", [](DataType self) { return std::string(self.name()); })
      .def_property_readonly(
          "numpy_dtype",
          [](DataType self) { return GetNumpyDtypeOrThrow(self); })
      .def("__hash__",
           [](DataType self) {
             absl::Hash<DataType> h;
             return h(self);
           })
      .def_property_readonly("type",
                             [](DataType self) -> py::object {
                               return GetTypeObjectOrThrow(self);
                             })
      .def(
          "__call__",
          [](DataType self, py::object arg) -> py::object {
            if (self.id() == DataTypeId::json_t) {
              return py::cast(PyObjectToJson(arg));
            }
            return GetTypeObjectOrThrow(self)(std::move(arg));
          },
          "Construct a scalar instance of this data type")
      .def(
          "__eq__", [](DataType self, DataType other) { return self == other; },
          py::arg("other"));

  // Like NumPy and Tensorflow, define `tensorstore.<dtype>` constants for each
  // supported data type.
  for (const DataType data_type : kDataTypes) {
    m.attr(std::string(data_type.name()).c_str()) = data_type;
  }
}

bool ConvertToDataType(pybind11::handle src, bool convert, void** value) {
  if (src.is_none()) return false;
  if (src.ptr() == reinterpret_cast<PyObject*>(&PyUnicode_Type)) {
    *value = new DataType(DataTypeOf<ustring_t>());
    return true;
  }
  if (src.ptr() == reinterpret_cast<PyObject*>(&PyBytes_Type)) {
    *value = new DataType(DataTypeOf<string_t>());
    return true;
  }
  PyObject* ptr = nullptr;
  if (!pybind11::detail::npy_api::get().PyArray_DescrConverter_(
          pybind11::reinterpret_borrow<pybind11::object>(src).release().ptr(),
          &ptr) ||
      !ptr) {
    PyErr_Clear();
    return false;
  }
  auto data_type =
      GetDataTypeOrThrow(pybind11::reinterpret_steal<pybind11::dtype>(ptr));
  *value = new DataType(data_type);
  return true;
}

}  // namespace internal_python
}  // namespace tensorstore
