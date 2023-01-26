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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <memory>
#include <type_traits>
#include <utility>

#include "python/tensorstore/data_type.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/type_name_override.h"
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

/// Returns an `std::shared_ptr` that refers to the base data pointer of a NumPy
/// array and keeps the NumPy array alive.
template <typename Element>
std::shared_ptr<Element> GetSharedPtrFromNumpyArray(pybind11::array array_obj) {
  auto* element_ptr = static_cast<Element*>(
      static_cast<void*>(pybind11::detail::array_proxy(array_obj.ptr())->data));
  return internal_python::PythonObjectOwningSharedPtr(element_ptr,
                                                      std::move(array_obj));
}

/// Returns a `tensorstore::ElementPointer` that refers to the base data pointer
/// of a NumPy array and keeps the NumPy array alive.
///
/// \tparam Element Element type, must be compatible with the data type of the
///     NumPy array.
template <typename Element>
std::enable_if_t<!std::is_void_v<Element>, SharedElementPointer<Element>>
GetSharedElementPointerFromNumpyArray(pybind11::array array_obj) {
  assert(GetDataTypeOrThrow(array_obj.dtype()) == dtype_v<Element>);
  return GetSharedPtrFromNumpyArray<Element>(std::move(array_obj));
}

template <typename Element>
std::enable_if_t<std::is_void_v<Element>, SharedElementPointer<Element>>
GetSharedElementPointerFromNumpyArray(pybind11::array array_obj) {
  auto dtype = GetDataTypeOrThrow(array_obj.dtype());
  return SharedElementPointer<Element>(
      GetSharedPtrFromNumpyArray<Element>(std::move(array_obj)), dtype);
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

/// Implementation of `ConvertToArray`, see documentation below.
bool ConvertToArrayImpl(pybind11::handle src,
                        SharedArray<void, dynamic_rank>* out,
                        DataType data_type_constraint, DimensionIndex min_rank,
                        DimensionIndex max_rank, bool writable, bool no_throw,
                        bool allow_copy);

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
/// \param src The Python object to convert, may be any `ArrayLike` type as
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
    dtype_t<Element> data_type_constraint = {},
    StaticOrDynamicRank<Rank> min_rank = GetDefaultRank<Rank>(),
    StaticOrDynamicRank<Rank> max_rank = GetDefaultRank<Rank>()) {
  SharedArray<void, dynamic_rank> dynamic_out;
  bool result =
      ConvertToArrayImpl(src, &dynamic_out, data_type_constraint, min_rank,
                         max_rank, /*writable=*/std::is_const_v<Element>,
                         /*no_throw=*/NoThrow, /*allow_copy=*/AllowCopy);
  if constexpr (NoThrow) {
    if (!result) return false;
  }
  *out = tensorstore::StaticCast<SharedArray<Element, Rank>, unchecked>(
      std::move(dynamic_out));
  if constexpr (NoThrow) {
    return true;
  }
}

/// Copies a NumPy array to `out`.
///
/// Throws an exception if `src` is not a NumPy array with shape and data type
/// compatible with `out`.
///
/// \param src Handle to source NumPy array.
/// \param out Target array.
void CopyFromNumpyArray(pybind11::handle src, ArrayView<void> out);

/// Wraps an unvalidated `py::object` but displays as "numpy.typing.ArrayLike"
/// in pybind11 function signatures.
///
/// This is used as the argument type to pybind11-exposed functions in place of
/// e.g. `SharedArray` when custom conversion is required, e.g. because there
/// are constraints on the rank or data type that cannot be specified at compile
/// time.
struct ArrayArgumentPlaceholder {
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("numpy.typing.ArrayLike");
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

/// Defines automatic conversion of `tensorstore::SharedArray` to/from Python
/// objects.
template <typename Element, tensorstore::DimensionIndex Rank>
struct type_caster<tensorstore::SharedArray<Element, Rank>> {
  using T = tensorstore::SharedArray<Element, Rank>;
  PYBIND11_TYPE_CASTER(T, _("numpy.typing.ArrayLike"));

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
  PYBIND11_TYPE_CASTER(T, _("Literal['C','F']"));
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
