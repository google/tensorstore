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

#ifndef PYTHON_TENSORSTORE_DATA_TYPE_H_
#define PYTHON_TENSORSTORE_DATA_TYPE_H_

/// \file Defines the `tensorstore.dtype` class (corresponding to
/// `tensorstore::DataType`), the `tensorstore.<dtype>` constants, and automatic
/// conversion from compatible Python objects to `tensorstore::DataType`.

#include "python/tensorstore/numpy.h"
// numpy.h must be included first to ensure the header inclusion order
// constraints are satisfied.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <stddef.h>

#include <array>
#include <complex>
#include <string_view>

#include "tensorstore/data_type.h"

#if defined(NPY_ABI_VERSION) && NPY_ABI_VERSION >= 0x02000000
#define TENSORSTORE_NPY_NTYPES NPY_NTYPES_LEGACY
#else
#define TENSORSTORE_NPY_NTYPES NPY_NTYPES
#endif

namespace tensorstore {
namespace internal_python {

// Mapping from NPY_TYPES enum to tensorstore type.
//
// See the data type details in the numpy docs:
// https://numpy.org/doc/stable/user/basics.types.html#relationship-between-numpy-data-types-and-c-data-types
inline constexpr std::array<DataTypeId, TENSORSTORE_NPY_NTYPES>
    kDataTypeIdForNumpyTypeNum = [] {
      std::array<DataTypeId, TENSORSTORE_NPY_NTYPES> npy_to_id = {};
      for (size_t i = 0; i < TENSORSTORE_NPY_NTYPES; ++i) {
        npy_to_id[i] = DataTypeId::custom;
      }

      // Legacy NPY types
      // NOLINTBEGIN(runtime/int)
      npy_to_id[NPY_BOOL] = DataTypeId::bool_t;
      npy_to_id[NPY_BYTE] = DataTypeIdOf<signed char>;
      npy_to_id[NPY_UBYTE] = DataTypeIdOf<unsigned char>;
      npy_to_id[NPY_SHORT] = DataTypeIdOf<short>;
      npy_to_id[NPY_USHORT] = DataTypeIdOf<unsigned short>;
      npy_to_id[NPY_INT] = DataTypeIdOf<int>;
      npy_to_id[NPY_UINT] = DataTypeIdOf<unsigned int>;
      npy_to_id[NPY_LONG] = DataTypeIdOf<long>;
      npy_to_id[NPY_ULONG] = DataTypeIdOf<unsigned long>;
      npy_to_id[NPY_LONGLONG] = DataTypeIdOf<long long>;
      npy_to_id[NPY_ULONGLONG] = DataTypeIdOf<unsigned long long>;
      npy_to_id[NPY_FLOAT] = DataTypeIdOf<float>;
      npy_to_id[NPY_DOUBLE] = DataTypeIdOf<double>;
      npy_to_id[NPY_LONGDOUBLE] = DataTypeIdOf<long double>;
      npy_to_id[NPY_CFLOAT] = DataTypeIdOf<std::complex<float>>;
      npy_to_id[NPY_CDOUBLE] = DataTypeIdOf<std::complex<double>>;
      npy_to_id[NPY_CLONGDOUBLE] = DataTypeIdOf<std::complex<long double>>;
      // npy_to_id[NPY_OBJECT] = DataTypeId::custom;
      npy_to_id[NPY_STRING] = DataTypeId::char_t;
      // npy_to_id[NPY_UNICODE] = DataTypeId::custom;
      npy_to_id[NPY_VOID] = DataTypeId::byte_t;
      // npy_to_id[NPY_DATETIME] = DataTypeId::custom;
      // npy_to_id[NPY_TIMEDELTA] = DataTypeId::custom;
      npy_to_id[NPY_HALF] = DataTypeId::float16_t;
      // NOLINTEND(runtime/int)

      return npy_to_id;
    }();

// Mapping from tensorstore datatype to NPY_TYPES enum.
constexpr std::array<int, kNumDataTypeIds> kNumpyTypeNumForDataTypeId = [] {
  std::array<int, kNumDataTypeIds> id_to_npy = {};
  for (size_t i = 0; i < kNumDataTypeIds; ++i) {
    id_to_npy[i] = -1;
  }
  for (size_t i = 0; i < TENSORSTORE_NPY_NTYPES; ++i) {
    if (kDataTypeIdForNumpyTypeNum[i] != DataTypeId::custom) {
      id_to_npy[static_cast<size_t>(kDataTypeIdForNumpyTypeNum[i])] = i;
    }
  }

  // Add mapping for `NPY_{U,}LONG` last so that it takes precedence over
  // `NPY_{U,}INT` and `NPY_{U,}LONGLONG` for consistency with how Numpy
  // defines the sized integer types.
  id_to_npy[static_cast<size_t>(kDataTypeIdForNumpyTypeNum[NPY_LONG])] =
      NPY_LONG;
  id_to_npy[static_cast<size_t>(kDataTypeIdForNumpyTypeNum[NPY_ULONG])] =
      NPY_ULONG;

  id_to_npy[static_cast<size_t>(DataTypeId::string_t)] = NPY_OBJECT;
  id_to_npy[static_cast<size_t>(DataTypeId::ustring_t)] = NPY_OBJECT;
  id_to_npy[static_cast<size_t>(DataTypeId::json_t)] = NPY_OBJECT;
  return id_to_npy;
}();

/// Returns `true` if the in-memory representation of `d` is the same in C++ and
/// NumPy.
constexpr inline bool CanDataTypeShareMemoryWithNumpy(DataType d) {
  DataTypeId id = d.id();
  if (id == DataTypeId::custom) return false;
  // NPY_OBJECT and uninitialized mappings (-1) cannot share representation.
  return kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)] != NPY_OBJECT &&
         kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)] != -1;
}

using tensorstore::GetDataType;

/// Returns the DataType if valid.
///
/// \throws `pybind11::value_error` if `name` is not a valid DataType name.
DataType GetDataTypeOrThrow(std::string_view name);

/// Returns the corresponding NumPy dtype.
///
/// \throws `pybind11::value_error` if there is no corresponding NumPy dtype.
pybind11::dtype GetNumpyDtypeOrThrow(DataType dtype);

/// Returns the corresponding NumPy type number, or `-1` if there is no
/// corresponding type number.
int GetNumpyTypeNum(DataType dtype);

/// Returns the NumPy dtype for the specified type number.
///
/// \throws `pybind11::error_already_set` if `type_num` is invalid.
pybind11::dtype GetNumpyDtype(int type_num);

template <typename T>
constexpr int GetNumpyTypeNum() {
  constexpr DataTypeId id = DataTypeIdOf<T>;
  static_assert(id != DataTypeId::custom,
                "Cannot get numpy dtype for non-canonical types");
  constexpr int type_num = kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)];
  static_assert(type_num != -1, "No corresponding numpy type");
  return type_num;
}

template <typename T>
inline pybind11::dtype GetNumpyDtype() {
  return GetNumpyDtype(GetNumpyTypeNum<T>());
}

/// Returns the corresponding DataType, or an invalid DataType if there is no
/// corresponding DataType.
DataType GetDataType(pybind11::dtype dt);

/// Returns the corresponding DataType.
///
/// \throws `pybind11::value_error` if there is no corresponding DataType.
DataType GetDataTypeOrThrow(pybind11::dtype dt);

/// Wrapper type used to indicate parameters that may be specified either as
/// `tensorstore.dtype` objects or any compatible type (such as a numpy data
/// type).
struct DataTypeLike {
  DataType value;
};

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from compatible Python objects to
/// `tensorstore::internal_python::DataTypeLike` parameters of
/// pybind11-exposed functions.
///
/// The `str` and `bytes` Python type constructors map to the `ustring` and
/// `string` types, respectively.
///
/// Otherwise, we rely on NumPy's conversion of Python objects to a NumPy data
/// type
/// (https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html#specifying-and-constructing-data-types)
/// and then convert to a TensorStore data type.
template <>
struct type_caster<tensorstore::internal_python::DataTypeLike> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::DataTypeLike,
                       _("tensorstore.dtype"));
  bool load(handle src, bool convert);
  static handle cast(tensorstore::internal_python::DataTypeLike value,
                     return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // PYTHON_TENSORSTORE_DATA_TYPE_H_
