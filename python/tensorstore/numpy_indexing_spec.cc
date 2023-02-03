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

#include "python/tensorstore/numpy.h"
// numpy.h must be included first.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/numpy_indexing_spec.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "tensorstore/array.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/numpy_indexing_spec.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

/// Returns `py::cast<T>(handle)`, but throws an exception that maps to a Python
/// `TypeError` exception with a message of `msg` (as is typical for Python
/// APIs), rather than the pybind11-specific `py::cast_error`.
template <typename T>
T CastOrTypeError(py::handle handle, const char* msg) {
  try {
    return py::cast<T>(handle);
  } catch (py::cast_error&) {
    throw py::type_error(msg);
  }
}

}  // namespace

std::string_view GetIndexingModePrefix(NumpyIndexingSpec::Mode mode) {
  switch (mode) {
    case NumpyIndexingSpec::Mode::kDefault:
      return "";
    case NumpyIndexingSpec::Mode::kOindex:
      return ".oindex";
    case NumpyIndexingSpec::Mode::kVindex:
      return ".vindex";
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

NumpyIndexingSpec ParseIndexingSpec(pybind11::handle obj,
                                    NumpyIndexingSpec::Mode mode,
                                    NumpyIndexingSpec::Usage usage) {
  NumpyIndexingSpec spec;
  NumpyIndexingSpec::Builder builder(spec, mode, usage);

  // Process a Python object representing a single indexing term.  This
  // conversion mostly follows the logic in numpy/core/src/multiarray/mapping.c
  // for compatibility with NumPy.
  const auto add_term = [&](py::handle term) -> absl::Status {
    if (term.ptr() == Py_Ellipsis) {
      return builder.AddEllipsis();
    }
    if (term.ptr() == Py_None) {
      return builder.AddNewAxis();
    }
    if (PySlice_Check(term.ptr())) {
      auto* slice_obj = reinterpret_cast<PySliceObject*>(term.ptr());
      const auto get_slice_index = [](py::handle handle) {
        return ToIndexVectorOrScalarContainer(
            CastOrTypeError<OptionallyImplicitIndexVectorOrScalarContainer>(
                handle,
                "slice indices must be integers or None or have an __index__ "
                "method"));
      };
      auto start = get_slice_index(slice_obj->start);
      auto stop = get_slice_index(slice_obj->stop);
      auto step = get_slice_index(slice_obj->step);
      return builder.AddSlice(start, stop, step);
    }
    // Check for an integer index.  Bool scalars are not treated as integer
    // indices; instead, they are treated as rank-0 boolean arrays.
    if (PyLong_CheckExact(term.ptr()) ||
        (!PyBool_Check(term.ptr()) && !PyArray_Check(term.ptr()))) {
      Py_ssize_t x = PyNumber_AsSsize_t(term.ptr(), PyExc_IndexError);
      if (x != -1 || !PyErr_Occurred()) {
        return builder.AddIndex(static_cast<Index>(x));
      }
      PyErr_Clear();
    }

    py::array array_obj;

    // Only remaining cases are index arrays, bool arrays, or invalid values.

    if (!PyArray_Check(term.ptr())) {
      array_obj = py::reinterpret_steal<py::array>(PyArray_FromAny(
          term.ptr(), nullptr, 0, 0, NPY_ARRAY_ALIGNED, nullptr));
      if (!array_obj) throw py::error_already_set();
      if (array_obj.size() == 0) {
        array_obj = py::reinterpret_steal<py::array>(PyArray_FromAny(
            array_obj.ptr(),
            reinterpret_cast<PyArray_Descr*>(
                GetNumpyDtype<Index>().release().ptr()),
            0, 0, NPY_ARRAY_FORCECAST | NPY_ARRAY_ALIGNED, nullptr));
        if (!array_obj) throw py::error_already_set();
      }
    } else {
      array_obj = py::reinterpret_borrow<py::array>(term.ptr());
    }

    auto* array_proxy = py::detail::array_proxy(array_obj.ptr());
    const int type_num =
        py::detail::array_descriptor_proxy(array_proxy->descr)->type_num;
    if (type_num == NPY_BOOL) {
      // Bool array.
      return builder.AddBoolArray(
          UncheckedArrayFromNumpy<bool>(std::move(array_obj)));
    }
    if (type_num >= NPY_BYTE && type_num <= NPY_ULONGLONG) {
      // Integer array.
      array_obj = py::reinterpret_steal<py::array>(
          PyArray_FromAny(array_obj.ptr(),
                          reinterpret_cast<PyArray_Descr*>(
                              GetNumpyDtype<Index>().release().ptr()),
                          0, 0, NPY_ARRAY_ALIGNED, nullptr));
      if (!array_obj) {
        throw py::error_already_set();
      }
      // TODO(jbms): Add mechanism for users to explicitly indicate that an
      // index array can safely be stored by reference rather than copied.  User
      // must ensure that array is not modified.
      return builder.AddIndexArray(MakeCopy(
          UncheckedArrayFromNumpy<Index>(array_obj), skip_repeated_elements));
    }
    // Invalid array data type.
    if (array_obj.ptr() == term.ptr()) {
      // The input was already an array.
      return absl::InvalidArgumentError(
          "Arrays used as indices must be of integer (or boolean) type");
    }
    return absl::InvalidArgumentError(
        "Only integers, slices (`:`), ellipsis (`...`), tensorstore.newaxis "
        "(`None`) and integer or boolean arrays are valid indices");
  };

  if (!PyTuple_Check(obj.ptr())) {
    ThrowStatusException(add_term(obj), StatusExceptionPolicy::kIndexError);
  } else {
    spec.scalar = false;
    py::tuple t = py::reinterpret_borrow<py::tuple>(obj);
    for (size_t i = 0, size = t.size(); i < size; ++i) {
      ThrowStatusException(add_term(t[i]), StatusExceptionPolicy::kIndexError);
    }
  }
  builder.Finalize();
  return spec;
}

SharedArray<bool> GetBoolArrayFromIndices(
    ArrayView<const Index, 2> index_arrays) {
  const DimensionIndex rank = index_arrays.shape()[0];
  Index shape[kMaxRank];
  const Index num_indices = index_arrays.shape()[1];
  for (DimensionIndex j = 0; j < rank; ++j) {
    Index x = 0;
    for (Index i = 0; i < num_indices; ++i) {
      x = std::max(x, index_arrays(j, i));
    }
    shape[j] = x + 1;
  }
  auto bool_array =
      AllocateArray<bool>(span<const Index>(shape, rank), c_order, value_init);
  for (Index i = 0; i < num_indices; ++i) {
    Index offset = 0;
    for (DimensionIndex j = 0; j < rank; ++j) {
      offset += bool_array.byte_strides()[j] * index_arrays(j, i);
    }
    bool_array.byte_strided_pointer()[offset] = true;
  }
  return bool_array;
}

std::string IndexingSpecRepr(const NumpyIndexingSpec& self) {
  std::string r;
  for (size_t i = 0; i < self.terms.size(); ++i) {
    if (i != 0) r += ",";
    const auto& term = self.terms[i];
    if (auto* index = std::get_if<Index>(&term)) {
      tensorstore::StrAppend(&r, *index);
      continue;
    }
    if (auto* s = std::get_if<NumpyIndexingSpec::Slice>(&term)) {
      if (s->start != kImplicit) tensorstore::StrAppend(&r, s->start);
      r += ':';
      if (s->stop != kImplicit) tensorstore::StrAppend(&r, s->stop);
      if (s->step != 1) tensorstore::StrAppend(&r, ":", s->step);
      continue;
    }
    if (std::holds_alternative<NumpyIndexingSpec::NewAxis>(term)) {
      r += "None";
      continue;
    }
    if (std::holds_alternative<NumpyIndexingSpec::Ellipsis>(term)) {
      r += "...";
      continue;
    }
    if (auto* index_array = std::get_if<NumpyIndexingSpec::IndexArray>(&term)) {
      r += py::repr(py::cast(index_array->index_array));
      continue;
    }
    if (auto* bool_array = std::get_if<NumpyIndexingSpec::BoolArray>(&term)) {
      r += py::repr(py::cast(GetBoolArrayFromIndices(
          StaticRankCast<2, unchecked>(bool_array->index_arrays))));
    }
  }
  if (!self.scalar && self.terms.size() == 1) {
    r += ',';
  }
  return r;
}

}  // namespace internal_python
}  // namespace tensorstore
