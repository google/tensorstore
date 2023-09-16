// Copyright 2023 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_CUSTOM_INT4_H_
#define THIRD_PARTY_PY_TENSORSTORE_CUSTOM_INT4_H_

// Must be included first
// clang-format off
#include "python/tensorstore/numpy.h" // NOLINT
// clang-format on

#include <cmath>        //NOLINT
#include <complex>      //NOLINT
#include <cstdint>      //NOLINT
#include <limits>       //NOLINT
#include <type_traits>  //NOLINT

#include "absl/strings/str_cat.h"
#include <half.hpp>
#include "python/tensorstore/custom_numpy_dtypes_impl.h"  // NOLINT
#include "python/tensorstore/ufuncs.h"  // NOLINT
#include "tensorstore/util/int4.h"  // NOLINT

// The implementation below is derived from jax-ml/ml_dtypes:
// https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/int4.h

/* Copyright 2023 The ml_dtypes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

namespace tensorstore {
namespace internal_python {

constexpr char kOutOfRange[] = "out of range";

using std::isinf;  // NOLINT
using std::isnan;  // NOLINT

using int4 = tensorstore::Int4Padded;

// TODO(ChromeHearts)
// using uint4 = tensorstore::Int4Padded;

template <typename T>
struct Int4TypeDescriptor {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyObject* type_ptr;

  static PyNumberMethods number_methods;
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_Descr npy_descr;
};

template <typename T>
int Int4TypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* Int4TypeDescriptor<T>::type_ptr = nullptr;

// Representation of a Python custom float object.
template <typename T>
struct PyInt4 {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyInt4.
template <typename T>
bool PyInt4_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

// Extracts the value of a PyInt4 object.
template <typename T>
T PyInt4_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyInt4<T>*>(object)->value;
}

template <typename T>
bool PyInt4_Value(PyObject* arg, T* output) {
  if (PyInt4_Check<T>(arg)) {
    *output = PyInt4_Value_Unchecked<T>(arg);
    return true;
  }
  return false;
}

// Constructs a PyInt4 object from PyInt4<T>::T.
template <typename T>
Safe_PyObjectPtr PyInt4_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyInt4<T>* p = reinterpret_cast<PyInt4<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a reduced float value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToInt4(PyObject* arg, T* output) {
  if (PyInt4_Check<T>(arg)) {
    *output = PyInt4_Value_Unchecked<T>(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    if (isnan(d)) {
      PyErr_SetString(PyExc_ValueError, "cannot convert float NaN to integer");
    }
    if (std::isinf(d)) {
      PyErr_SetString(PyExc_OverflowError,
                      "cannot convert float infinity to integer");
    }
    if (d < static_cast<double>(std::numeric_limits<T>::min()) ||
        d > static_cast<double>(std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError,
                      "out of range value cannot be converted to int4");
    }
    *output = T(d);
    return true;
  }
  if (PyLong_Check(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(l);
    return true;
  }
  if (PyArray_IsScalar(arg, Integer)) {
    int64_t i;
    PyArray_CastScalarToCtype(arg, &i, PyArray_DescrFromType(NPY_INT64));
    if (!(std::numeric_limits<T>::min() <= i &&
          i <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(i);
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    ::half_float::half f;
    PyArray_ScalarAsCtype(arg, &f);
    if (!(std::numeric_limits<T>::min() <= f &&
          f <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(static_cast<::int8_t>(f));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    if (!(std::numeric_limits<T>::min() <= f &&
          f <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(static_cast<::int8_t>(f));
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double d;
    PyArray_ScalarAsCtype(arg, &d);
    if (!(std::numeric_limits<T>::min() <= d &&
          d <= std::numeric_limits<T>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = T(static_cast<::int8_t>(d));
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != Int4TypeDescriptor<T>::npy_type) {
      ref = make_safe(PyArray_Cast(arr, Int4TypeDescriptor<T>::npy_type));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<T*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

// Constructs a new PyInt4.
template <typename T>
PyObject* PyInt4_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "expected number as argument to %s constructor",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (PyInt4_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToInt4<T>(arg, &value)) {
    return PyInt4_FromValue<T>(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      return PyArray_Cast(arr, TypeDescriptor<T>::Dtype());
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse float from string, then cast to T.
    PyObject* f = PyLong_FromUnicodeObject(arg, /*base=*/0);
    if (PyErr_Occurred()) {
      return nullptr;
    }
    if (CastToInt4<T>(f, &value)) {
      return PyInt4_FromValue<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

template <typename T>
PyObject* PyInt4_nb_float(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  return PyFloat_FromDouble(static_cast<double>(static_cast<float>(x)));
}

template <typename T>
PyObject* PyInt4_nb_int(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  long y = static_cast<long>(static_cast<float>(x));  // NOLINT
  return PyLong_FromLong(y);
}

template <typename T>
PyObject* PyInt4_nb_negative(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  return PyInt4_FromValue<T>(-x).release();
}

template <typename T>
PyObject* PyInt4_nb_positive(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  return PyInt4_FromValue<T>(x).release();
}

template <typename T>
PyObject* PyInt4_nb_add(PyObject* a, PyObject* b) {
  T x, y;
  if (PyInt4_Value<T>(a, &x) && PyInt4_Value<T>(b, &y)) {
    return PyInt4_FromValue<T>(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyInt4_nb_subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (PyInt4_Value<T>(a, &x) && PyInt4_Value<T>(b, &y)) {
    return PyInt4_FromValue<T>(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyInt4_nb_multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (PyInt4_Value<T>(a, &x) && PyInt4_Value<T>(b, &y)) {
    return PyInt4_FromValue<T>(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyInt4_nb_remainder(PyObject* a, PyObject* b) {
  T x, y;
  if (PyInt4_Value<T>(a, &x) && PyInt4_Value<T>(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    T v = x % y;
    if (v != 0 && ((v < 0) != (y < 0))) {
      v = v + y;
    }
    return PyInt4_FromValue<T>(v).release();
  }
  return PyArray_Type.tp_as_number->nb_remainder(a, b);
}

template <typename T>
PyObject* PyInt4_nb_floor_divide(PyObject* a, PyObject* b) {
  T x, y;
  if (PyInt4_Value<T>(a, &x) && PyInt4_Value<T>(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    T v = x / y;
    if (((x > 0) != (y > 0)) && x % y != 0) {
      v = v - T(1);
    }
    return PyInt4_FromValue<T>(v).release();
  }
  return PyArray_Type.tp_as_number->nb_floor_divide(a, b);
}

// Python number methods for PyInt4 objects.
template <typename T>
PyNumberMethods Int4TypeDescriptor<T>::number_methods = {
    PyInt4_nb_add<T>,        // nb_add
    PyInt4_nb_subtract<T>,   // nb_subtract
    PyInt4_nb_multiply<T>,   // nb_multiply
    PyInt4_nb_remainder<T>,  // nb_remainder
    nullptr,                 // nb_divmod
    nullptr,                 // nb_power
    PyInt4_nb_negative<T>,   // nb_negative
    PyInt4_nb_positive<T>,   // nb_positive
    nullptr,                 // nb_absolute
    nullptr,                 // nb_nonzero
    nullptr,                 // nb_invert
    nullptr,                 // nb_lshift
    nullptr,                 // nb_rshift
    nullptr,                 // nb_and
    nullptr,                 // nb_xor
    nullptr,                 // nb_or
    PyInt4_nb_int<T>,        // nb_int
    nullptr,                 // reserved
    PyInt4_nb_float<T>,      // nb_float

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    PyInt4_nb_floor_divide<T>,  // nb_floor_divide
    nullptr,                    // nb_true_divide
    nullptr,                    // nb_inplace_floor_divide
    nullptr,                    // nb_inplace_true_divide
    nullptr,                    // nb_index
};

// Implementation of repr() for PyInt4.
template <typename T>
PyObject* PyInt4_Repr(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  std::string s = absl::StrCat(static_cast<::int8_t>(x));
  return PyUnicode_FromString(s.c_str());
}

// Implementation of str() for PyInt4.
template <typename T>
PyObject* PyInt4_Str(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  std::string s = absl::StrCat(static_cast<::int8_t>(x));
  return PyUnicode_FromString(s.c_str());
}

// Hash function for PyInt4.
template <typename T>
Py_hash_t PyInt4_Hash(PyObject* self) {
  T x = PyInt4_Value_Unchecked<T>(self);
  // Hash function for PyInt4. We transform it to 1 <= x <= 17 range.
  // Hash function for PyUInt4. We transform it to 9 <= x <= 25 range.
  return static_cast<Py_hash_t>(static_cast<int8_t>(x) + 9);
}

// Comparisons on PyInt4s.
template <typename T>
PyObject* PyInt4_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!PyInt4_Value<T>(a, &x) || !PyInt4_Value<T>(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "Invalid op type");
      return nullptr;
  }
  return PyBool_FromLong(result);
}

// Numpy support
template <typename T>
PyArray_ArrFuncs Int4TypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_Descr Int4TypeDescriptor<T>::npy_descr = {
    PyObject_HEAD_INIT(nullptr)
    /*typeobj=*/nullptr,  // Filled in later
    /*kind=*/TypeDescriptor<T>::kNpyDescrKind,
    /*type=*/TypeDescriptor<T>::kNpyDescrType,
    /*byteorder=*/TypeDescriptor<T>::kNpyDescrByteorder,
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(T),
    /*alignment=*/alignof(T),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&Int4TypeDescriptor<T>::arr_funcs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyInt4_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyLong_FromLong(static_cast<int>(x));
}

template <typename T>
int NPyInt4_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (!CastToInt4<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
int NPyInt4_Compare(const void* a, const void* b, void* arr) {
  T x;
  memcpy(&x, a, sizeof(T));

  T y;
  memcpy(&y, b, sizeof(T));
  int fy(y);
  int fx(x);
  if (fx < fy) {
    return -1;
  }
  if (fy < fx) {
    return 1;
  }
  return 0;
}

template <typename T>
void NPyInt4_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                       npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (dstride == sizeof(T) && sstride == sizeof(T)) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
    }
  }
}

template <typename T>
void NPyInt4_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(T));
}

template <typename T>
npy_bool NPyInt4_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<T>(0);
}

template <typename T>
int NPyInt4_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  T* const buffer = reinterpret_cast<T*>(buffer_raw);
  const int start(buffer[0]);
  const int delta = static_cast<int>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<T>(start + i * delta);
  }
  return 0;
}

template <typename T>
void NPyInt4_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2, void* op,
                     npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  int acc = 0;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += static_cast<int>(*b1) * static_cast<int>(*b2);
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
int NPyInt4_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

template <typename T>
int NPyInt4_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  // Start with a max_val of NaN, this results in the first iteration preferring
  // bdata[0].
  int max_val = std::numeric_limits<int>::max();
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "max".
    if (!(static_cast<int>(bdata[i]) <= max_val)) {
      max_val = static_cast<int>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

template <typename T>
int NPyInt4_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  int min_val = std::numeric_limits<int>::lowest();
  // Start with a min_val of NaN, this results in the first iteration preferring
  // bdata[0].
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "min".
    if (!(static_cast<int>(bdata[i]) >= min_val)) {
      min_val = static_cast<int>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

template <typename T,
          std::enable_if_t<(std::is_floating_point<T>::value ||
                            std::is_same<T, half_float::half>::value),
                           bool> = true>
int CastToInt(T value) {
  if (isnan(value) || isinf(value) ||
      value < std::numeric_limits<int>::lowest() ||
      value > std::numeric_limits<int>::max()) {
    return 0;
  }
  return static_cast<int>(value);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
int CastToInt(T value) {
  return static_cast<int>(value);
}

int CastToInt(int4 value) { return static_cast<int>(value); }

// TODO(ChromeHearts)
// int CastToInt(uint4 value) { return static_cast<int>(value); }

template <typename T>
int CastToInt(std::complex<T> value) {
  return CastToInt(value.real());
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void IntegerCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<typename TypeDescriptor<To>::T>(
        static_cast<To>(CastToInt(from[i])));
  }
}

// Registers a cast between T (a reduced float) and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterCustomIntCast(int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               IntegerCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&Int4TypeDescriptor<T>::npy_descr, numpy_type,
                               IntegerCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterInt4Casts() {
  if (!RegisterCustomIntCast<T, half_float::half>(NPY_HALF)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned char>(NPY_UINT8)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomIntCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCustomIntCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterCustomIntCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_INT8,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_UINT8,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_INT16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_UINT16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_INT32,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_UINT32,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_INT64,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_UINT64,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_FLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_DOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_LONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CFLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CLONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  // Safe casts to T from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }

  return true;
}

template <typename T>
bool RegisterInt4UFuncs(PyObject* numpy) {
  bool ok = RegisterUFunc<BinaryUFunc<T, T, ufuncs::Add<T>>, T>(numpy, "add") &&
            RegisterUFunc<BinaryUFunc<T, T, ufuncs::Subtract<T>>, T>(
                numpy, "subtract") &&
            RegisterUFunc<BinaryUFunc<T, T, ufuncs::Multiply<T>>, T>(
                numpy, "multiply") &&
            RegisterUFunc<BinaryUFunc<T, T, ufuncs::FloorDivide<T>>, T>(
                numpy, "floor_divide") &&
            RegisterUFunc<BinaryUFunc<T, T, ufuncs::Remainder<T>>, T>(
                numpy, "remainder");

  return ok;
}

template <typename T>
bool RegisterInt4Dtype(PyObject* numpy) {
  Safe_PyObjectPtr name =
      make_safe(PyUnicode_FromString(TypeDescriptor<T>::kTypeName));
  Safe_PyObjectPtr qualname =
      make_safe(PyUnicode_FromString(TypeDescriptor<T>::kTypeName));

  PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  if (!heap_type) {
    return false;
  }
  // Caution: we must not call any functions that might invoke the GC until
  // PyType_Ready() is called. Otherwise the GC might see a half-constructed
  // type object.
  heap_type->ht_name = name.release();
  heap_type->ht_qualname = qualname.release();
  PyTypeObject* type = &heap_type->ht_type;
  type->tp_name = TypeDescriptor<T>::kTypeName;
  type->tp_basicsize = sizeof(PyInt4<T>);
  type->tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  type->tp_base = &PyGenericArrType_Type;
  type->tp_new = PyInt4_tp_new<T>;
  type->tp_repr = PyInt4_Repr<T>;
  type->tp_hash = PyInt4_Hash<T>;
  type->tp_str = PyInt4_Str<T>;
  type->tp_doc = const_cast<char*>(TypeDescriptor<T>::kTpDoc);
  type->tp_richcompare = PyInt4_RichCompare<T>;
  type->tp_as_number = &Int4TypeDescriptor<T>::number_methods;
  if (PyType_Ready(type) < 0) {
    return false;
  }
  TypeDescriptor<T>::type_ptr = reinterpret_cast<PyObject*>(type);

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("tensorstore"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "__module__",
                             module.get()) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs& arr_funcs = Int4TypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyInt4_GetItem<T>;
  arr_funcs.setitem = NPyInt4_SetItem<T>;
  arr_funcs.compare = NPyInt4_Compare<T>;
  arr_funcs.copyswapn = NPyInt4_CopySwapN<T>;
  arr_funcs.copyswap = NPyInt4_CopySwap<T>;
  arr_funcs.nonzero = NPyInt4_NonZero<T>;
  arr_funcs.fill = NPyInt4_Fill<T>;
  arr_funcs.dotfunc = NPyInt4_DotFunc<T>;
  arr_funcs.compare = NPyInt4_CompareFunc<T>;
  arr_funcs.argmax = NPyInt4_ArgMaxFunc<T>;
  arr_funcs.argmin = NPyInt4_ArgMinFunc<T>;

#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
  Py_TYPE(&Int4TypeDescriptor<T>::npy_descr) = &PyArrayDescr_Type;
#else
  Py_SET_TYPE(&Int4TypeDescriptor<T>::npy_descr, &PyArrayDescr_Type);
#endif
  TypeDescriptor<T>::npy_descr.typeobj = type;

  TypeDescriptor<T>::npy_type =
      PyArray_RegisterDataType(&Int4TypeDescriptor<T>::npy_descr);
  if (TypeDescriptor<T>::Dtype() < 0) {
    return false;
  }

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), TypeDescriptor<T>::kTypeName,
                           TypeDescriptor<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(
          TypeDescriptor<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(&Int4TypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterInt4Casts<T>() && RegisterInt4UFuncs<T>(numpy);
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_CUSTOM_INT4_H_
