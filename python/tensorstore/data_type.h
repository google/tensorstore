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

#include <array>
#include <complex>

#include "absl/strings/string_view.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorstore/data_type.h"

namespace tensorstore {
namespace internal_python {

/// Consistent with how pybind11 works, we don't depend on NumPy headers (which
/// also simplifies the build process).  Therefore, we have to redefine all of
/// the data type constants used by NumPy.
enum NumpyTypeNum {
  NPY_BOOL_ = 0,
  NPY_BYTE_ = 1,
  NPY_UBYTE_ = 2,
  NPY_SHORT_ = 3,
  NPY_USHORT_ = 4,
  NPY_INT_ = 5,
  NPY_UINT_ = 6,
  NPY_LONG_ = 7,
  NPY_ULONG_ = 8,
  NPY_LONGLONG_ = 9,
  NPY_ULONGLONG_ = 10,
  NPY_FLOAT_ = 11,
  NPY_DOUBLE_ = 12,
  NPY_LONGDOUBLE_ = 13,
  NPY_CFLOAT_ = 14,
  NPY_CDOUBLE_ = 15,
  NPY_CLONGDOUBLE_ = 16,
  NPY_OBJECT_ = 17,
  NPY_STRING_ = 18,
  NPY_UNICODE_ = 19,
  NPY_VOID_ = 20,
  NPY_DATETIME_ = 21,
  NPY_TIMEDELTA_ = 22,
  NPY_HALF_ = 23,
  NPY_NTYPES_ = 24,
};

inline constexpr std::array<DataTypeId, NPY_NTYPES_>
    kDataTypeIdForNumpyTypeNum = {{
        DataTypeId::bool_t,                       // NPY_BOOL = 0
        DataTypeIdOf<signed char>,                // NPY_BYTE = 1
        DataTypeIdOf<unsigned char>,              // NPY_UBYTE = 2
        DataTypeIdOf<short>,                      // NPY_SHORT = 3
        DataTypeIdOf<unsigned short>,             // NPY_USHORT = 4
        DataTypeIdOf<int>,                        // NPY_INT = 5
        DataTypeIdOf<unsigned int>,               // NPY_UINT = 6
        DataTypeIdOf<long>,                       // NPY_LONG = 7
        DataTypeIdOf<unsigned long>,              // NPY_ULONG = 8
        DataTypeIdOf<long long>,                  // NPY_LONGLONG = 9
        DataTypeIdOf<unsigned long long>,         // NPY_ULONGLONG = 10
        DataTypeIdOf<float>,                      // NPY_FLOAT = 11,
        DataTypeIdOf<double>,                     // NPY_DOUBLE = 12,
        DataTypeIdOf<long double>,                // NPY_LONGDOUBLE = 13,
        DataTypeIdOf<std::complex<float>>,        // NPY_CFLOAT = 14,
        DataTypeIdOf<std::complex<double>>,       // NPY_CDOUBLE = 15,
        DataTypeIdOf<std::complex<long double>>,  // NPY_CLONGDOUBLE = 16,
        DataTypeId::custom,                       // NPY_OBJECT = 17
        DataTypeId::char_t,                       // NPY_STRING = 18
        DataTypeId::custom,                       // NPY_UNICODE = 19
        DataTypeId::byte_t,                       // NPY_VOID = 20
        DataTypeId::custom,                       // NPY_DATETIME = 21
        DataTypeId::custom,                       // NPY_TIMEDELTA = 22
        DataTypeId::float16_t,                    // NPY_HALF = 23
    }};

constexpr std::array<int, kNumDataTypeIds> GetNumpyTypeNumForDataTypeId() {
  std::array<int, kNumDataTypeIds> array = {};
  for (size_t i = 0; i < kNumDataTypeIds; ++i) {
    array[i] = -1;
  }
  const auto AssignMapping = [&array](NumpyTypeNum i) {
    DataTypeId id = kDataTypeIdForNumpyTypeNum[i];
    if (id == DataTypeId::custom) return;
    array[static_cast<size_t>(id)] = i;
  };
  for (size_t i = 0; i < static_cast<size_t>(NPY_NTYPES_); ++i) {
    AssignMapping(static_cast<NumpyTypeNum>(i));
  }
  // Add mapping for `NPY_{U,}LONG` last so that they take precedence over
  // `NPY_{U,}INT` and `NPY_{U,}LONGLONG` for consistency with how Numpy defines
  // the sized integer types.
  AssignMapping(NPY_LONG_);
  AssignMapping(NPY_ULONG_);
  array[static_cast<size_t>(DataTypeId::string_t)] = NPY_OBJECT_;
  array[static_cast<size_t>(DataTypeId::ustring_t)] = NPY_OBJECT_;
  array[static_cast<size_t>(DataTypeId::json_t)] = NPY_OBJECT_;
  return array;
}

constexpr std::array<int, kNumDataTypeIds> kNumpyTypeNumForDataTypeId =
    GetNumpyTypeNumForDataTypeId();

using tensorstore::GetDataType;

/// Returns the DataType if valid.
///
/// \throws `pybind11::value_error` if `name` is not a valid DataType name.
DataType GetDataTypeOrThrow(absl::string_view name);

/// Returns the corresponding NumPy dtype.
///
/// \throws `pybind11::value_error` if there is no corresponding NumPy dtype.
pybind11::dtype GetNumpyDtypeOrThrow(DataType data_type);

/// Returns the corresponding NumPy type number, or `-1` if there is no
/// corresponding type number.
int GetNumpyTypeNum(DataType data_type);

/// Returns the NumPy dtype for the specified type number.
///
/// \throws `pybind11::error_already_set` if `type_num` is invalid.
pybind11::dtype GetNumpyDtype(int type_num);

template <typename T>
inline pybind11::dtype GetNumpyDtype() {
  constexpr DataTypeId id = DataTypeIdOf<T>;
  static_assert(id != DataTypeId::custom,
                "Cannot get numpy dtype for non-canonical types");
  constexpr int type_num = kNumpyTypeNumForDataTypeId[static_cast<size_t>(id)];
  static_assert(type_num != -1, "No corresponding numpy type");
  return GetNumpyDtype(type_num);
}

/// Returns the corresponding DataType, or an invalid DataType if there is no
/// corresponding DataType.
DataType GetDataType(pybind11::dtype dt);

/// Returns the corresponding DataType.
///
/// \throws `pybind11::value_error` if there is no corresponding DataType.
DataType GetDataTypeOrThrow(pybind11::dtype dt);

/// Implementation of the `DataType` type caster defined below.
///
/// The `str` and `bytes` Python type constructors map to the `ustring` and
/// `string` types, respectively.
///
/// Otherwise, we rely on NumPy's conversion of Python objects to a NumPy data
/// type
/// (https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html#specifying-and-constructing-data-types)
/// and then convert to a TensorStore data type.
///
/// On success, sets `*value` to the address of a new `DataType` object
/// allocated with `operator new` and returns `true`.  On failure, returns
/// `false`.  (This unusual return protocol is for compatibility with pybind11's
/// type caster framework.)
bool ConvertToDataType(pybind11::handle src, bool convert, void** value);

/// Defines the Python types and constants.
void RegisterDataTypeBindings(pybind11::module m);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from compatible Python objects to
/// `tensorstore::DataType` parameters of pybind11-exposed functions.
template <>
struct type_caster<tensorstore::DataType>
    : public type_caster_base<tensorstore::DataType> {
  using Base = type_caster_base<tensorstore::DataType>;
  bool load(handle src, bool convert) {
    if (Base::load(src, convert)) {
      return true;
    }
    return tensorstore::internal_python::ConvertToDataType(src, convert,
                                                           &value);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_DATA_TYPE_H_
