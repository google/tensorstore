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

#include "python/tensorstore/array_type_caster.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "python/tensorstore/data_type.h"
#include "python/tensorstore/json_type_caster.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = pybind11;

namespace {
/// Converts string and json types to Python objects.
struct ConvertToObject {
  py::object operator()(const string_t* x) const noexcept {
    return py::reinterpret_steal<py::object>(
        PyBytes_FromStringAndSize(x->data(), x->size()));
  }

  py::object operator()(const ustring_t* x) const noexcept {
    return py::reinterpret_steal<py::object>(
        PyUnicode_FromStringAndSize(x->utf8.data(), x->utf8.size()));
  }
  py::object operator()(const json_t* x) const noexcept {
    return JsonToPyObject(*x);
  }
};

template <typename T>
constexpr const internal::ElementwiseFunction<2, Status*>*
GetConvertToNumpyObjectArrayFunction() {
  if constexpr (std::is_invocable_v<const ConvertToObject&, const T*>) {
    const auto func = [](const T* from, PyObject** to, Status* status) {
      if (auto obj = ConvertToObject()(from)) {
        Py_XDECREF(std::exchange(*to, obj.release().ptr()));
        return true;
      }
      return false;
    };
    return internal::SimpleElementwiseFunction<decltype(func)(T, PyObject*),
                                               Status*>();
  }
  return nullptr;
}

constexpr const internal::ElementwiseFunction<2, Status*>*
    kConvertDataTypeToNumpyObjectArray[kNumDataTypeIds] = {
#define TENSORSTORE_INTERNAL_DO_CONVERT(T, ...) \
  GetConvertToNumpyObjectArrayFunction<T>(),
        TENSORSTORE_FOR_EACH_DATA_TYPE(TENSORSTORE_INTERNAL_DO_CONVERT)
#undef TENSORSTORE_INTERNAL_DO_CONVERT
};

pybind11::object GetNumpyObjectArrayImpl(SharedArrayView<const void> source) {
  auto& api = npy_api::get();
  ssize_t target_shape_ssize_t[kMaxNumpyRank];
  std::copy(source.shape().begin(), source.shape().end(), target_shape_ssize_t);
  auto array_obj = py::reinterpret_steal<py::array>(api.PyArray_NewFromDescr_(
      api.PyArray_Type_, api.PyArray_DescrFromType_(NPY_OBJECT_),
      static_cast<int>(source.rank()), target_shape_ssize_t,
      /*strides=*/nullptr,
      /*data=*/nullptr,
      /*flags=*/NPY_ARRAY_C_CONTIGUOUS_ | NPY_ARRAY_WRITEABLE_,
      /*obj=*/nullptr));
  if (!array_obj) throw py::error_already_set();
  Index target_strides[kMaxNumpyRank];
  std::copy_n(array_obj.strides(), source.rank(), target_strides);
  auto iterate_result = internal::IterateOverStridedLayouts<2>(
      /*closure=*/{kConvertDataTypeToNumpyObjectArray[static_cast<size_t>(
                       source.data_type().id())],
                   nullptr},
      /*status=*/nullptr,
      /*shape=*/source.shape(),
      {{const_cast<void*>(source.data()),
        static_cast<void*>(py::detail::array_proxy(array_obj.ptr())->data)}},
      {{source.byte_strides().data(), target_strides}},
      /*constraints=*/skip_repeated_elements,
      {{source.data_type().size(), sizeof(PyObject*)}});
  if (!iterate_result.success) throw py::error_already_set();
  return std::move(array_obj);
}

}  // namespace

void AssignArrayLayout(pybind11::array array_obj, DimensionIndex rank,
                       Index* shape, Index* byte_strides) {
  [[maybe_unused]] const int flags =
      py::detail::array_proxy(array_obj.ptr())->flags;
  assert(array_obj.ndim() == rank);
  assert(flags & NPY_ARRAY_ALIGNED_);
  std::copy_n(array_obj.shape(), rank, shape);
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (shape[i] < 0 || shape[i] > kMaxFiniteIndex) {
      throw std::out_of_range(
          StrCat("Array shape[", i, "]=", shape[i], " is not valid"));
    }
  }
  std::copy_n(array_obj.strides(), rank, byte_strides);
}

pybind11::object GetNumpyArrayImpl(SharedArrayView<const void> value,
                                   bool is_const) {
  if (value.rank() > kMaxNumpyRank) {
    throw std::out_of_range(StrCat("Array of rank ", value.rank(),
                                   " (which is greater than ", kMaxNumpyRank,
                                   ") cannot be converted to NumPy array"));
  }
  if (const DataTypeId id = value.data_type().id();
      id != DataTypeId::custom &&
      kConvertDataTypeToNumpyObjectArray[static_cast<size_t>(id)]) {
    return GetNumpyObjectArrayImpl(value);
  }
  auto& api = npy_api::get();
  ssize_t shape[kMaxNumpyRank];
  ssize_t strides[kMaxNumpyRank];
  std::copy_n(value.shape().data(), value.rank(), shape);
  std::copy_n(value.byte_strides().data(), value.rank(), strides);
  int flags = 0;
  if (!is_const) {
    flags |= NPY_ARRAY_WRITEABLE_;
  }
  auto obj = py::reinterpret_steal<py::array>(api.PyArray_NewFromDescr_(
      api.PyArray_Type_,
      GetNumpyDtypeOrThrow(value.data_type()).release().ptr(),
      static_cast<int>(value.rank()), shape, strides,
      const_cast<void*>(value.data()), flags, nullptr));
  if (!obj) throw py::error_already_set();
  using Pointer = std::shared_ptr<const void>;
  api.PyArray_SetBaseObject_(
      obj.ptr(), py::capsule(new Pointer(std::move(value.pointer())),
                             [](void* p) { delete static_cast<Pointer*>(p); })
                     .release()
                     .ptr());
  return std::move(obj);
}

ContiguousLayoutOrder GetContiguousLayoutOrderOrThrow(pybind11::handle obj) {
  Py_UCS4 c;
  if (PyUnicode_Check(obj.ptr())) {
    if (PyUnicode_READY(obj.ptr()) != 0) throw py::error_already_set();
    if (PyUnicode_GET_LENGTH(obj.ptr()) != 1) goto invalid;
    void* data = PyUnicode_DATA(obj.ptr());
    int kind = PyUnicode_KIND(obj.ptr());
    c = PyUnicode_READ(kind, data, 0);
  } else if (PyBytes_Check(obj.ptr())) {
    if (PyBytes_GET_SIZE(obj.ptr()) != 1) goto invalid;
    c = PyBytes_AS_STRING(obj.ptr())[0];
  } else {
    goto invalid;
  }
  switch (c) {
    case 'C':
      return ContiguousLayoutOrder::c;
    case 'F':
      return ContiguousLayoutOrder::fortran;
    default:
      break;
  }
invalid:
  throw py::type_error("`order` must be specified as 'C' or 'F'");
}

}  // namespace internal_python
}  // namespace tensorstore
