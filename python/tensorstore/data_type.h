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

#ifndef THIRD_PARTY_PY_TENSORSTORE_DATA_TYPE_H_
#define THIRD_PARTY_PY_TENSORSTORE_DATA_TYPE_H_

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

#include <array>
#include <complex>
#include <string_view>

#include "python/tensorstore/bfloat16.h"
#include "tensorstore/data_type.h"

namespace tensorstore {
namespace internal_python {

inline constexpr std::array<DataTypeId, NPY_NTYPES> kDataTypeIdForNumpyTypeNum =
    [] {
      std::array<DataTypeId, NPY_NTYPES> result = {};
      for (size_t i = 0; i < NPY_NTYPES; ++i) {
        result[i] = DataTypeId::custom;
      }
      result[NPY_BOOL] = DataTypeId::bool_t;
      result[NPY_BYTE] = DataTypeIdOf<signed char>;
      result[NPY_UBYTE] = DataTypeIdOf<unsigned char>;
      result[NPY_SHORT] = DataTypeIdOf<short>;
      result[NPY_USHORT] = DataTypeIdOf<unsigned short>;
      result[NPY_INT] = DataTypeIdOf<int>;
      result[NPY_UINT] = DataTypeIdOf<unsigned int>;
      result[NPY_LONG] = DataTypeIdOf<long>;
      result[NPY_ULONG] = DataTypeIdOf<unsigned long>;
      result[NPY_LONGLONG] = DataTypeIdOf<long long>;
      result[NPY_ULONGLONG] = DataTypeIdOf<unsigned long long>;
      result[NPY_FLOAT] = DataTypeIdOf<float>;
      result[NPY_DOUBLE] = DataTypeIdOf<double>;
      result[NPY_LONGDOUBLE] = DataTypeIdOf<long double>;
      result[NPY_CFLOAT] = DataTypeIdOf<std::complex<float>>;
      result[NPY_CDOUBLE] = DataTypeIdOf<std::complex<double>>;
      result[NPY_CLONGDOUBLE] = DataTypeIdOf<std::complex<long double>>;
      // result[NPY_OBJECT] = DataTypeId::custom;
      result[NPY_STRING] = DataTypeId::char_t;
      // result[NPY_UNICODE] = DataTypeId::custom;
      result[NPY_VOID] = DataTypeId::byte_t;
      // result[NPY_DATETIME] = DataTypeId::custom;
      // result[NPY_TIMEDELTA] = DataTypeId::custom;
      result[NPY_HALF] = DataTypeId::float16_t;
      return result;
    }();

constexpr std::array<int, kNumDataTypeIds> kNumpyTypeNumForDataTypeId = [] {
  std::array<int, kNumDataTypeIds> array = {};
  for (size_t i = 0; i < kNumDataTypeIds; ++i) {
    array[i] = -1;
  }
  const auto AssignMapping = [&array](size_t i) {
    DataTypeId id = kDataTypeIdForNumpyTypeNum[i];
    if (id == DataTypeId::custom) return;
    array[static_cast<size_t>(id)] = i;
  };
  for (size_t i = 0; i < NPY_NTYPES; ++i) {
    AssignMapping(i);
  }
  // Add mapping for `NPY_{U,}LONG` last so that they take precedence over
  // `NPY_{U,}INT` and `NPY_{U,}LONGLONG` for consistency with how Numpy defines
  // the sized integer types.
  AssignMapping(NPY_LONG);
  AssignMapping(NPY_ULONG);
  array[static_cast<size_t>(DataTypeId::string_t)] = NPY_OBJECT;
  array[static_cast<size_t>(DataTypeId::ustring_t)] = NPY_OBJECT;
  array[static_cast<size_t>(DataTypeId::json_t)] = NPY_OBJECT;
  return array;
}();

/// Returns `true` if the in-memory representation of `d` is the same in C++ and
/// NumPy.
constexpr inline bool CanDataTypeShareMemoryWithNumpy(DataType d) {
  DataTypeId id = d.id();
  if (id == DataTypeId::custom) return false;
  return kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)] != NPY_OBJECT;
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

template <>
inline int GetNumpyTypeNum<bfloat16_t>() {
  return Bfloat16NumpyTypeNum();
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
/// `tensorstore::internal_python::DataTypeLike` parameters of pybind11-exposed
/// functions.
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

#endif  // THIRD_PARTY_PY_TENSORSTORE_DATA_TYPE_H_
