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

#ifndef THIRD_PARTY_PY_TENSORSTORE_ARRAY_TYPE_CASTER_H_
#define THIRD_PARTY_PY_TENSORSTORE_ARRAY_TYPE_CASTER_H_

/// \file Defines conversion between NumPy arrays and `tensorstore::Array`.

#include <memory>
#include <type_traits>
#include <utility>

#include "absl/meta/type_traits.h"
#include "python/tensorstore/data_type.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorstore/array.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

constexpr DimensionIndex kMaxNumpyRank = 32;

using pybind11::detail::npy_api;
constexpr auto NPY_ARRAY_ALIGNED_ = npy_api::NPY_ARRAY_ALIGNED_;
constexpr auto NPY_ARRAY_WRITEABLE_ = npy_api::NPY_ARRAY_WRITEABLE_;
constexpr auto NPY_ARRAY_C_CONTIGUOUS_ = npy_api::NPY_ARRAY_C_CONTIGUOUS_;
constexpr auto NPY_ARRAY_F_CONTIGUOUS_ = npy_api::NPY_ARRAY_F_CONTIGUOUS_;

/// Copies the shape and byte_strides from `array_obj` to `shape` and
/// `byte_strides`.
///
/// \param array_obj[in] Source NumPy array, must be non-null.
/// \param rank Target array rank.
/// \param shape[out] Non-null pointer to array of length `rank`.
/// \param byte_strides[out] Non-null pointer to array of length `rank`.
/// \dchecks `array_obj` is aligned.
void AssignArrayLayout(pybind11::array array_obj, DimensionIndex rank,
                       Index* shape, Index* byte_strides);

/// `std::shared_ptr`-compatible deleter that holds a reference to a Python
/// object, which it releases when invoked.
struct PythonObjectDeleter {
  pybind11::object obj;
  void operator()(const void*) {
    // Acquire the GIL in order to safely call Python APIs.
    pybind11::gil_scoped_acquire gil;
    // Assign a null handle to release the existing `obj` handle.
    obj = pybind11::reinterpret_steal<pybind11::object>(pybind11::handle());
  }
};

/// Returns an `std::shared_ptr` that refers to the base data pointer of a NumPy
/// array and keeps the NumPy array alive.
template <typename Element>
std::shared_ptr<Element> GetSharedPtrFromNumpyArray(pybind11::array array_obj) {
  auto* element_ptr = static_cast<Element*>(
      static_cast<void*>(pybind11::detail::array_proxy(array_obj.ptr())->data));
  return std::shared_ptr<Element>(element_ptr,
                                  PythonObjectDeleter{std::move(array_obj)});
}

/// Returns a `tensorstore::ElementPointer` that refers to the base data pointer
/// of a NumPy array and keeps the NumPy array alive.
///
/// \tparam Element Element type, must be compatible with the data type of the
///     NumPy array.
template <typename Element>
absl::enable_if_t<!std::is_void<Element>::value, SharedElementPointer<Element>>
GetSharedElementPointerFromNumpyArray(pybind11::array array_obj) {
  assert(GetDataTypeOrThrow(array_obj.dtype()) == DataTypeOf<Element>());
  return GetSharedPtrFromNumpyArray<Element>(std::move(array_obj));
}

template <typename Element>
absl::enable_if_t<std::is_void<Element>::value, SharedElementPointer<Element>>
GetSharedElementPointerFromNumpyArray(pybind11::array array_obj) {
  auto data_type = GetDataTypeOrThrow(array_obj.dtype());
  return SharedElementPointer<Element>(
      GetSharedPtrFromNumpyArray<Element>(std::move(array_obj)), data_type);
}

/// Returns a `SharedArray` that refers to a NumPy array and keeps the NumPy
/// array alive.
///
/// The specified `Element` and `Rank` must match the data type and rank of the
/// NumPy array.
///
/// \tparam Element Element type, or `void` to indicate no constraint.
/// \tparam Rank Rank constraint, or `dynamic_rank` to indicate no constraint.
template <typename Element, DimensionIndex Rank = dynamic_rank>
SharedArray<Element, Rank> UncheckedArrayFromNumpy(pybind11::array array_obj) {
  namespace py = pybind11;
  const DimensionIndex rank = array_obj.ndim();
  SharedArray<Element, Rank> array;
  array.layout().set_rank(StaticRankCast<Rank, unchecked>(rank));
  AssignArrayLayout(array_obj, array.rank(), array.shape().data(),
                    array.byte_strides().data());
  array.element_pointer() =
      GetSharedElementPointerFromNumpyArray<Element>(array_obj);
  return array;
}

/// Converts a Python object to a `SharedArray`.
///
/// \tparam Element Compile-time element type constraint, or `void` to indicate
///     no constraint.
/// \tparam Rank Compile-time rank constraint, or `dynamic_rank` to indicate no
///     constraint.
/// \tparam NoThrow If `true`, return `false` if `src` cannot be converted (for
///     use in pybind11 type caster implementations).  If `false`, throw an
///     exception to indicate an error.
/// \tparam AllowCopy If `true`, permit a new array to be allocated to hold the
///     conversion of `src` to an array.  If `false`, it is an error if `src` is
///     not an existing NumPy array with suitable data type to which `*out` can
///     refer directly.
/// \param src The Python object to convert, may be any `array_like` type as
///     defined by NumPy.
/// \param out[out] Pointer to `SharedArray` to be set to the converted result.
/// \param data_type_constraint If specified, constrains the allowed data types.
///     May result in the array being copied.  If `Element` is not `void`, this
///     has no effect.
/// \param min_rank Specifies the minimum allowed rank, or `dynamic_rank` for no
///     constraint.  If `Rank` is not `dynamic_rank`, this has no effect.
/// \param max_rank Specifies the maximum allowed rank, or `dynamic_rank` for no
///     constraint.  If `Rank` is not `dynamic_rank`, this has no effect.
template <typename Element, DimensionIndex Rank = dynamic_rank,
          bool NoThrow = false, bool AllowCopy = std::is_const_v<Element>>
std::conditional_t<NoThrow, bool, void> ConvertToArray(
    pybind11::handle src, SharedArray<Element, Rank>* out,
    StaticOrDynamicDataTypeOf<Element> data_type_constraint = {},
    StaticOrDynamicRank<Rank> min_rank = GetDefaultRank<Rank>(),
    StaticOrDynamicRank<Rank> max_rank = GetDefaultRank<Rank>()) {
  DataType data_type = data_type_constraint;
  // Determine the NumPy data type corresponding to `data_type_constraint`.
  pybind11::dtype dtype_handle =
      data_type.valid()
          ? GetNumpyDtypeOrThrow(data_type)
          : pybind11::reinterpret_steal<pybind11::dtype>(pybind11::handle());
  int flags = NPY_ARRAY_ALIGNED_;
  if constexpr (!std::is_const_v<Element>) {
    flags |= NPY_ARRAY_WRITEABLE_;
  }
  // Convert `src` to a NumPy array.
  auto obj = pybind11::reinterpret_steal<pybind11::array>(
      npy_api::get().PyArray_FromAny_(
          src.ptr(), dtype_handle.release().ptr(),
          min_rank == dynamic_rank ? 0 : static_cast<DimensionIndex>(min_rank),
          max_rank == dynamic_rank ? 0 : static_cast<DimensionIndex>(max_rank),
          flags, nullptr));
  if (!obj) {
    if constexpr (NoThrow) {
      PyErr_Clear();
      return false;
    }
    throw pybind11::error_already_set();
  }
  if (!AllowCopy && obj.ptr() != src.ptr()) {
    // PyArray_FromAny created a copy, which is not allowed for a non-const
    // Element type.
    if constexpr (NoThrow) {
      return false;
    }
    throw pybind11::value_error(
        "Argument is not a writable array with suitable dtype");
  }
  // PyArray_FromAny does not handle rank constraints of 0, so we need to
  // check them separately.
  if (max_rank == 0 && obj.ndim() != 0) {
    if constexpr (NoThrow) {
      return false;
    }
    throw pybind11::value_error(StrCat(
        "Expected array of rank 0, but received array of rank ", obj.ndim()));
  }
  *out = UncheckedArrayFromNumpy<Element, Rank>(std::move(obj));
  if constexpr (NoThrow) {
    return true;
  }
}

/// Wraps an unvalidated `py::object` but displays as "array_like" in pybind11
/// function signatures.
///
/// This is used as the argument type to pybind11-exposed functions in place of
/// e.g. `SharedArray` when custom conversion is required, e.g. because there
/// are constraints on the rank or data type that cannot be specified at compile
/// time.
struct ArrayArgumentPlaceholder {
  pybind11::object obj;
};

pybind11::object GetNumpyArrayImpl(SharedArrayView<const void> value,
                                   bool is_const);

/// Converts a `SharedArray` to a NumPy array.  For numeric data types, the
/// returned NumPy array refers to the same memory and keeps the underlying data
/// alive.  For string and json data types, this results in a copy.
template <typename T, DimensionIndex Rank, ContainerKind LayoutCKind>
pybind11::object GetNumpyArray(
    const SharedArray<T, Rank, zero_origin, LayoutCKind>& array) {
  return GetNumpyArrayImpl(array, std::is_const_v<T>);
}

/// If `obj` is a unicode or bytes object of length 1 containing the single
/// character 'C' or 'F', returns the corresponding order.
///
/// \throws `py::type_error` if `obj` is not a valid order.
ContiguousLayoutOrder GetContiguousLayoutOrderOrThrow(pybind11::handle obj);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion of Python objects to `ArrayArgumentPlaceholder`
/// function parameters.
template <>
struct type_caster<tensorstore::internal_python::ArrayArgumentPlaceholder> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::ArrayArgumentPlaceholder,
                       _("array_like"));
  bool load(handle src, bool convert) {
    value.obj = reinterpret_borrow<object>(src);
    return true;
  }
};

/// Defines automatic conversion of `tensorstore::SharedArray` to/from Python
/// objects.
template <typename Element, tensorstore::DimensionIndex Rank>
struct type_caster<tensorstore::SharedArray<Element, Rank>> {
  using T = tensorstore::SharedArray<Element, Rank>;
  PYBIND11_TYPE_CASTER(T, _("array_like"));

  bool load(handle src, bool convert) {
    return tensorstore::internal_python::ConvertToArray<Element, Rank,
                                                        /*NoThrow=*/true>(
        src, &value);
  }
  static handle cast(const tensorstore::SharedArray<Element, Rank>& value,
                     return_value_policy /* policy */, handle /* parent */) {
    return tensorstore::internal_python::GetNumpyArray(value).release();
  }
};

/// Defines automatic conversion of `ContiguousLayoutOrder` to/from "C"/"F"
/// Python string constants.
template <>
struct type_caster<tensorstore::ContiguousLayoutOrder> {
  using T = tensorstore::ContiguousLayoutOrder;
  PYBIND11_TYPE_CASTER(T, _("ContiguousLayoutOrder"));
  bool load(handle src, bool convert) {
    value = tensorstore::internal_python::GetContiguousLayoutOrderOrThrow(src);
    return true;
  }
  static handle cast(tensorstore::ContiguousLayoutOrder order,
                     return_value_policy /* policy */, handle /* parent */) {
    return str((order == tensorstore::c_order) ? "C" : "F").release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_ARRAY_TYPE_CASTER_H_
