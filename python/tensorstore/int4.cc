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

#include "python/tensorstore/numpy.h"
// numpy.h must be included first to ensure the header inclusion order
// constraints are satisfied.

#include "python/tensorstore/int4.h"

// Other headers
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "python/tensorstore/bfloat16.h"
#include "python/tensorstore/data_type.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/bfloat16.h"
#include "tensorstore/util/int4.h"
#include "tensorstore/util/str_cat.h"

// This implementation is parallel to bfloat16.

namespace tensorstore {
namespace internal_python {

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_int4 = NPY_NOTYPE;

namespace {

constexpr char kOutOfRange[] = "out of range";
constexpr char kDivideByZero[] = "divide by zero";

// Cheatsheet of remainder (C) vs modulo (Python)
//
// | a   | b   | C /   | C %   | Py //  | Py %   |
// | --- | --- | ----- | ----- | ------ | ------ |
// | +7  | +3  | +2    | +1    | +2     | +1     |
// | +7  | -3  | -2    | +1    | -3     | -2     |
// | -7  | +3  | -2    | -1    | -3     | +2     |
// | -7  | -3  | +2    | -1    | +2     | -1     |
//
// Here we implement modulo from remainder.
std::pair<int, int> divmod(int a, int b) {
  const int quot = a / b;
  const int rem = a % b;
  if ((a >= 0) == (b > 0) || rem == 0) {
    return {quot, rem};
  }
  return {quot - 1, rem + b};
}

// https://bugs.python.org/issue39573  Py_SET_TYPE() added to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
template <typename T>
static inline void Py_SET_TYPE(T* ob, PyTypeObject* type) {
  reinterpret_cast<PyObject*>(ob)->ob_type = type;
}
#endif

using int4 = ::tensorstore::Int4Padded;

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

bool PyLong_CheckNoOverflow(PyObject* object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

// Forward declaration.
extern PyTypeObject int4_type;

// Pointer to the int4 type object we are using. This is either a pointer
// to int4_type, if we choose to register it, or to the int4 type
// registered by another system into NumPy.
PyTypeObject* int4_type_ptr = nullptr;

// Representation of a Python int4 object.
struct PyInt4 {
  PyObject_HEAD;  // Python object header
  int4 value;
};

// Returns true if 'object' is a PyInt4.
bool PyInt4_Check(PyObject* object) {
  return PyObject_IsInstance(object, reinterpret_cast<PyObject*>(&int4_type));
}

// Extracts the value of a PyInt4 object.
int4 PyInt4_Int4(PyObject* object) {
  return reinterpret_cast<PyInt4*>(object)->value;
}

// Constructs a PyInt4 object from a int4.
Safe_PyObjectPtr PyInt4_FromInt4(int4 x) {
  Safe_PyObjectPtr ref = make_safe(int4_type.tp_alloc(&int4_type, 0));
  PyInt4* p = reinterpret_cast<PyInt4*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a int4 value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToInt4(PyObject* arg, int4* output) {
  if (PyInt4_Check(arg)) {
    *output = PyInt4_Int4(arg);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    if (!(std::numeric_limits<int4>::min() <= l &&
          l <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(static_cast<::int8_t>(l));
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    if (!(std::numeric_limits<int4>::min() <= d &&
          d <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(static_cast<::int8_t>(d));
    return true;
  }
  if (PyArray_IsScalar(arg, Integer)) {
    int64_t i;
    PyArray_CastScalarToCtype(arg, &i, PyArray_DescrFromType(NPY_INT64));
    if (!(std::numeric_limits<int4>::min() <= i &&
          i <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(i);
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    tensorstore::dtypes::float16_t f;
    PyArray_ScalarAsCtype(arg, &f);
    if (!(std::numeric_limits<int4>::min() <= f &&
          f <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(static_cast<::int8_t>(f));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    if (!(std::numeric_limits<int4>::min() <= f &&
          f <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(static_cast<::int8_t>(f));
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double d;
    PyArray_ScalarAsCtype(arg, &d);
    if (!(std::numeric_limits<int4>::min() <= d &&
          d <= std::numeric_limits<int4>::max())) {
      PyErr_SetString(PyExc_OverflowError, kOutOfRange);
      return false;
    }
    *output = int4(static_cast<::int8_t>(d));
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_int4) {
      ref = make_safe(PyArray_Cast(arr, npy_int4));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<int4*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

bool SafeCastToInt4(PyObject* arg, int4* output) {
  if (PyInt4_Check(arg)) {
    *output = PyInt4_Int4(arg);
    return true;
  }
  return false;
}

// Converts a PyInt4 into a PyFloat.
PyObject* PyInt4_Float(PyObject* self) {
  int4 x = PyInt4_Int4(self);
  return PyFloat_FromDouble(static_cast<double>(static_cast<::int8_t>(x)));
}

// Converts a PyInt4 into a PyInt.
PyObject* PyInt4_Int(PyObject* self) {
  int4 x = PyInt4_Int4(self);
  long y = static_cast<::int8_t>(x);  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyInt4.
PyObject* PyInt4_Negative(PyObject* self) {
  int4 x = PyInt4_Int4(self);
  return PyInt4_FromInt4(-x).release();
}

// Unary positive sign (no-op).
PyObject* PyInt4_Positive(PyObject* self) {
  int4 x = PyInt4_Int4(self);
  return PyInt4_FromInt4(x).release();
}

PyObject* PyInt4_Add(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    return PyInt4_FromInt4(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

PyObject* PyInt4_Subtract(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    return PyInt4_FromInt4(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

PyObject* PyInt4_Multiply(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    return PyInt4_FromInt4(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

PyObject* PyInt4_Remainder(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, kDivideByZero);
      return nullptr;
    }
    return PyInt4_FromInt4(int4(divmod(x, y).second)).release();
  }
  return PyArray_Type.tp_as_number->nb_remainder(a, b);
}

PyObject* PyInt4_Divmod(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, kDivideByZero);
      return nullptr;
    }
    return PyInt4_FromInt4(int4(divmod(x, y).second)).release();
  }
  return PyArray_Type.tp_as_number->nb_divmod(a, b);
}

PyObject* PyInt4_FloorDivide(PyObject* a, PyObject* b) {
  int4 x, y;
  if (SafeCastToInt4(a, &x) && SafeCastToInt4(b, &y)) {
    if (y == 0) {
      PyErr_SetString(PyExc_ZeroDivisionError, kDivideByZero);
      return nullptr;
    }
    return PyInt4_FromInt4(int4(divmod(x, y).first)).release();
  }
  return PyArray_Type.tp_as_number->nb_floor_divide(a, b);
}

// Python number methods for PyInt4 objects.
PyNumberMethods PyInt4_AsNumber = {
    PyInt4_Add,        // nb_add
    PyInt4_Subtract,   // nb_subtract
    PyInt4_Multiply,   // nb_multiply
    PyInt4_Remainder,  // nb_remainder
    PyInt4_Divmod,     // nb_divmod
    nullptr,           // nb_power
    PyInt4_Negative,   // nb_negative
    PyInt4_Positive,   // nb_positive
    nullptr,           // nb_absolute
    nullptr,           // nb_nonzero
    nullptr,           // nb_invert
    nullptr,           // nb_lshift
    nullptr,           // nb_rshift
    nullptr,           // nb_and
    nullptr,           // nb_xor
    nullptr,           // nb_or
    PyInt4_Int,        // nb_int
    nullptr,           // reserved
    PyInt4_Float,      // nb_float

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

    PyInt4_FloorDivide,  // nb_floor_divide
    nullptr,             // nb_true_divide
    nullptr,             // nb_inplace_floor_divide
    nullptr,             // nb_inplace_true_divide
    nullptr,             // nb_index
};

// Constructs a new PyInt4.
PyObject* PyInt4_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to int4 constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  int4 value;
  if (PyInt4_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToInt4(arg, &value)) {
    return PyInt4_FromInt4(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_int4) {
      return PyArray_Cast(arr, npy_int4);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyInt4s.
PyObject* PyInt4_RichCompare(PyObject* a, PyObject* b, int op) {
  int4 x, y;
  if (!SafeCastToInt4(a, &x) || !SafeCastToInt4(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result = false;
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
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyInt4.
PyObject* PyInt4_Repr(PyObject* self) {
  int4 x = reinterpret_cast<PyInt4*>(self)->value;
  std::string v = tensorstore::StrCat(static_cast<::int8_t>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyInt4.
PyObject* PyInt4_Str(PyObject* self) {
  int4 x = reinterpret_cast<PyInt4*>(self)->value;
  std::string v = tensorstore::StrCat(static_cast<::int8_t>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyInt4. We transform it to 1 <= x <= 17 range.
Py_hash_t PyInt4_Hash(PyObject* self) {
  int4 x = reinterpret_cast<PyInt4*>(self)->value;
  return int8_t{x} + 9;
}

// Python type for PyInt4 objects.
PyTypeObject int4_type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "int4",  // tp_name
    sizeof(PyInt4),                            // tp_basicsize
    0,                                         // tp_itemsize
    nullptr,                                   // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,           // tp_getattr
    nullptr,           // tp_setattr
    nullptr,           // tp_compare / tp_reserved
    PyInt4_Repr,       // tp_repr
    &PyInt4_AsNumber,  // tp_as_number
    nullptr,           // tp_as_sequence
    nullptr,           // tp_as_mapping
    PyInt4_Hash,       // tp_hash
    nullptr,           // tp_call
    PyInt4_Str,        // tp_str
    nullptr,           // tp_getattro
    nullptr,           // tp_setattro
    nullptr,           // tp_as_buffer
                       // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "int4 floating-point values",  // tp_doc
    nullptr,                       // tp_traverse
    nullptr,                       // tp_clear
    PyInt4_RichCompare,            // tp_richcompare
    0,                             // tp_weaklistoffset
    nullptr,                       // tp_iter
    nullptr,                       // tp_iternext
    nullptr,                       // tp_methods
    nullptr,                       // tp_members
    nullptr,                       // tp_getset
    nullptr,                       // tp_base
    nullptr,                       // tp_dict
    nullptr,                       // tp_descr_get
    nullptr,                       // tp_descr_set
    0,                             // tp_dictoffset
    nullptr,                       // tp_init
    nullptr,                       // tp_alloc
    PyInt4_New,                    // tp_new
    nullptr,                       // tp_free
    nullptr,                       // tp_is_gc
    nullptr,                       // tp_bases
    nullptr,                       // tp_mro
    nullptr,                       // tp_cache
    nullptr,                       // tp_subclasses
    nullptr,                       // tp_weaklist
    nullptr,                       // tp_del
    0,                             // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyInt4_ArrFuncs;

PyArray_Descr NPyInt4_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&int4_type),
    // We must register int4 with a kind other than "i", because "<i1" is
    // already there and the even narrower version can only really take "<i0",
    // which stretches the notation too far. Similar to bfloat16, we will use
    // something else (but still unique) to register it.
    // The downside of this is that NumPy scalar promotion does not work with
    // int4 values.
    /*kind=*/'V',
    // TODO(hawkinsp): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'R',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(int4),
    /*alignment=*/alignof(int4),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyInt4_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject* NPyInt4_GetItem(void* data, void* arr) {
  int4 x;
  memcpy(&x, data, sizeof(int4));
  return PyInt4_FromInt4(x).release();
}

int NPyInt4_SetItem(PyObject* item, void* data, void* arr) {
  int4 x;
  if (!CastToInt4(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(int4));
  return 0;
}

int NPyInt4_Compare(const void* a, const void* b, void* arr) {
  int4 x;
  memcpy(&x, a, sizeof(int4));

  int4 y;
  memcpy(&y, b, sizeof(int4));

  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  return 0;
}

// Int4Padded is represented by uint8_t => no swap needed

void NPyInt4_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                       npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (dstride == sizeof(::uint8_t) && sstride == sizeof(::uint8_t)) {
    memcpy(dst, src, n * sizeof(::uint8_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(::uint8_t));
    }
  }
}

void NPyInt4_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint8_t));
}

npy_bool NPyInt4_NonZero(void* data, void* arr) {
  int4 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<int4>(0);
}

int NPyInt4_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  int4* const buffer = reinterpret_cast<int4*>(buffer_raw);
  const int4 start(buffer[0]);
  const int4 delta = static_cast<::int8_t>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<int4>(start + i * delta);
  }
  return 0;
}

void NPyInt4_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2, void* op,
                     npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  ::int8_t acc = 0;
  for (npy_intp i = 0; i < n; ++i) {
    int4* const b1 = reinterpret_cast<int4*>(c1);
    int4* const b2 = reinterpret_cast<int4*>(c2);
    acc += static_cast<::int8_t>(*b1) * static_cast<::int8_t>(*b2);
    c1 += is1;
    c2 += is2;
  }
  int4* out = reinterpret_cast<int4*>(op);
  *out = static_cast<int4>(acc);
}

int NPyInt4_CompareFunc(const void* v1, const void* v2, void* arr) {
  int4 b1 = *reinterpret_cast<const int4*>(v1);
  int4 b2 = *reinterpret_cast<const int4*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

int NPyInt4_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind, void* arr) {
  const int4* bdata = reinterpret_cast<const int4*>(data);
  ::int8_t max_val = std::numeric_limits<::int8_t>::min();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<::int8_t>(bdata[i]) > max_val) {
      max_val = static_cast<::int8_t>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyInt4_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind, void* arr) {
  const int4* bdata = reinterpret_cast<const int4*>(data);
  ::int8_t min_val = std::numeric_limits<::int8_t>::max();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<::int8_t>(bdata[i]) < min_val) {
      min_val = static_cast<::int8_t>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

// NumPy casts

/// C++ representation for the NumPy data type corresponding to `T`.
///
/// This is equal to `T` for all types except `T=bool`.  For `T=bool`, this is
/// equal to `int8_t` to account for the fact that NumPy does not guarantee bool
/// is equal to 0 or 1 (e.g. in the case of `numpy.ndarray.view`), but storing a
/// value that is not `0` or `1` in a C++ `bool` is undefined behavior.
template <typename T>
using NumpyRepType = std::conditional_t<std::is_same_v<T, bool>, int8_t, T>;

template <typename T>
inline T GetReal(T value) {
  return value;
}

template <typename T>
T GetReal(std::complex<T> value) {
  return value.real();
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from = reinterpret_cast<NumpyRepType<From>*>(from_void);
  auto* to = reinterpret_cast<NumpyRepType<To>*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<NumpyRepType<To>>(static_cast<To>(GetReal(from[i])));
  }
}

// Registers a cast between int4 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'.
template <typename T>
bool RegisterInt4Cast(int numpy_type) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, npy_int4, NPyCast<T, int4>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyInt4_Descr, numpy_type, NPyCast<int4, T>) <
      0) {
    return false;
  }
  return true;
}

template <typename Func, typename... T, size_t... Is>
constexpr auto GetUfuncImpl(std::index_sequence<Is...>) {
  return +[](char** args, const npy_intp* dimensions, const npy_intp* steps,
             void* data) {
    char* ptrs[] = {args[Is]...};
    const npy_intp steps_copy[] = {steps[Is]...};
    const size_t n = dimensions[0];
    for (size_t i = 0; i < n; ++i) {
      Func()(*reinterpret_cast<NumpyRepType<T>*>(ptrs[Is])...);
      ((ptrs[Is] += steps_copy[Is]), ...);
    }
  };
}

template <typename Func, typename... T>
constexpr auto GetUfunc() {
  return GetUfuncImpl<Func, T...>(std::index_sequence_for<T...>());
}

template <typename... T, typename Func>
bool RegisterUFunc(PyObject* numpy, const char* name, Func func) {
  static_assert(std::is_empty_v<Func>);
  int types[] = {GetNumpyTypeNum<T>()...};
  // Note: reinterpret_cast to `PyUFuncGenericFunction` is needed because
  // depending on the NumPy version, the const qualification of the parameter
  // types varies.
  auto fn = reinterpret_cast<PyUFuncGenericFunction>(
      GetUfunc<internal::DefaultConstructibleFunction<Func>, T...>());
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (sizeof...(T) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %d", name,
                 ufunc->nargs, static_cast<int>(sizeof...(T)));
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, npy_int4, fn, types, nullptr) < 0) {
    return false;
  }
  return true;
}

template <typename Func, typename... T>
struct SingleOutputAdapter {
  template <typename U>
  inline void operator()(T... inputs, U& output) const {
    output = internal::DefaultConstructibleFunction<Func>()(inputs...);
  }
};

// TODO(hawkinsp): implement spacing

// Initializes the module.
bool Initialize() {
  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // If another module (presumably either TF or JAX) has registered a int4
  // type, use it. We don't want two int4 types if we can avoid it since it
  // leads to confusion if we have two different types with the same name. This
  // assumes that the other module has a sufficiently complete int4
  // implementation. The only known NumPy int4 extension at the time of
  // writing is this one (distributed in TF and JAX).
  // TODO(hawkinsp): distribute the int4 extension as its own pip package,
  // so we can unambiguously refer to a single canonical definition of int4.
  int typenum = PyArray_TypeNumFromName(const_cast<char*>("int4"));
  if (typenum != NPY_NOTYPE) {
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    // The test for an argmax function here is to verify that the
    // int4 implementation is sufficiently new, and, say, not from
    // an older version of TF or JAX.
    if (descr && descr->f && descr->f->argmax) {
      npy_int4 = typenum;
      int4_type_ptr = descr->typeobj;
      return true;
    }
  }

  int4_type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&int4_type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyInt4_ArrFuncs);
  NPyInt4_ArrFuncs.getitem = NPyInt4_GetItem;
  NPyInt4_ArrFuncs.setitem = NPyInt4_SetItem;
  NPyInt4_ArrFuncs.compare = NPyInt4_Compare;
  NPyInt4_ArrFuncs.copyswapn = NPyInt4_CopySwapN;
  NPyInt4_ArrFuncs.copyswap = NPyInt4_CopySwap;
  NPyInt4_ArrFuncs.nonzero = NPyInt4_NonZero;
  NPyInt4_ArrFuncs.fill = NPyInt4_Fill;
  NPyInt4_ArrFuncs.dotfunc = NPyInt4_DotFunc;
  NPyInt4_ArrFuncs.compare = NPyInt4_CompareFunc;
  NPyInt4_ArrFuncs.argmax = NPyInt4_ArgMaxFunc;
  NPyInt4_ArrFuncs.argmin = NPyInt4_ArgMinFunc;

  Py_SET_TYPE(&NPyInt4_Descr, &PyArrayDescr_Type);
  npy_int4 = PyArray_RegisterDataType(&NPyInt4_Descr);
  int4_type_ptr = &int4_type;
  if (npy_int4 < 0) {
    return false;
  }

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy.get(), "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype('int4')` work.
  if (PyDict_SetItemString(typeDict_obj.get(), "int4",
                           reinterpret_cast<PyObject*>(&int4_type)) < 0) {
    return false;
  }

  // Support dtype(int4)
  if (PyDict_SetItemString(int4_type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyInt4_Descr)) < 0) {
    return false;
  }

  // Register casts
  if (!RegisterInt4Cast<::tensorstore::dtypes::float16_t>(NPY_HALF)) {
    return false;
  }
  if (Bfloat16NumpyTypeNum() != NPY_NOTYPE) {
    if (!RegisterInt4Cast<::tensorstore::dtypes::bfloat16_t>(
            Bfloat16NumpyTypeNum())) {
      return false;
    }
  }
  if (!RegisterInt4Cast<float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterInt4Cast<double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterInt4Cast<bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterInt4Cast<uint8_t>(NPY_UINT8)) {
    return false;
  }
  if (!RegisterInt4Cast<int8_t>(NPY_INT8)) {
    return false;
  }
  if (!RegisterInt4Cast<uint16_t>(NPY_UINT16)) {
    return false;
  }
  if (!RegisterInt4Cast<int16_t>(NPY_INT16)) {
    return false;
  }
  if (!RegisterInt4Cast<unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterInt4Cast<int>(NPY_INT)) {
    return false;
  }
  if (!RegisterInt4Cast<unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterInt4Cast<long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterInt4Cast<unsigned long long>(NPY_ULONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterInt4Cast<long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterInt4Cast<uint64_t>(NPY_UINT64)) {
    return false;
  }
  if (!RegisterInt4Cast<int64_t>(NPY_INT64)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterInt4Cast<std::complex<float>>(NPY_COMPLEX64)) {
    return false;
  }
  if (!RegisterInt4Cast<std::complex<double>>(NPY_COMPLEX128)) {
    return false;
  }

  // Safe casts from int4 to other types
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_INT8, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_UINT8, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_INT16, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_UINT16, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_INT32, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_UINT32, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_INT64, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_UINT64, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_FLOAT, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_DOUBLE, NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_COMPLEX64, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyInt4_Descr, NPY_COMPLEX128, NPY_NOSCALAR) <
      0) {
    return false;
  }

  // Safe casts to int4 from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL), npy_int4,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  const auto register_unary = [&](const char* name, auto func) {
    return RegisterUFunc<int4, int4>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), int4>());
  };

  const auto register_unary_predicate = [&](const char* name, auto func) {
    return RegisterUFunc<int4, bool>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), int4>());
  };

  const auto register_binary = [&](const char* name, auto func) {
    return RegisterUFunc<int4, int4, int4>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), int4, int4>());
  };

  const auto register_binary_predicate = [&](const char* name, auto func) {
    return RegisterUFunc<int4, int4, bool>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), int4, int4>());
  };

  constexpr auto remainder_func = [](int4 a, int4 b) {
    return divmod(a, b).second;
  };

#define TENSORSTORE_WRAP_FUNC(NAME) \
  [](auto... args) { return NAME(args...); } /**/

  constexpr auto abs_func = TENSORSTORE_WRAP_FUNC(tensorstore::abs);

  bool ok =
      register_binary("add", std::plus<void>()) &&
      register_binary("subtract", std::minus<void>()) &&
      register_binary("multiply", std::multiplies<void>()) &&
      register_binary("divide", std::divides<void>()) &&
      register_unary("negative", std::negate<void>()) &&
      register_unary("positive", internal::identity()) &&
      register_binary("true_divide", std::divides<void>()) &&
      register_binary("floor_divide",
                      [](int4 a, int4 b) { return divmod(a, b).first; }) &&
      register_binary("remainder", remainder_func) &&
      register_binary("mod", remainder_func) &&
      RegisterUFunc<int4, int4, int4, int4>(
          numpy.get(), "divmod",
          [](int4 a, int4 b, int4& quotient, int4& remainder) {
            std::tie(quotient, remainder) = divmod(a, b);
          }) &&
      register_unary("abs", abs_func) &&
      register_unary("sign",
                     [](int a) -> int {
                       if (a < 0) {
                         return -1;
                       }
                       if (a > 0) {
                         return 1;
                       }
                       return a;
                     }) &&
      // Comparison functions
      register_binary_predicate("equal", std::equal_to<void>()) &&
      register_binary_predicate("not_equal", std::not_equal_to<void>()) &&
      register_binary_predicate("less", std::less<void>()) &&
      register_binary_predicate("greater", std::greater<void>()) &&
      register_binary_predicate("less_equal", std::less_equal<void>()) &&
      register_binary_predicate("greater_equal", std::greater_equal<void>()) &&

      register_binary("maximum",
                      [](int a, int b) { return (a > b) ? a : b; }) &&
      register_binary("minimum",
                      [](int a, int b) { return (a < b) ? a : b; }) &&
      register_binary_predicate("logical_and", std::logical_and<int>()) &&
      register_binary_predicate("logical_or", std::logical_or<int>()) &&
      register_binary_predicate("logical_xor",
                                [](int a, int b) {
                                  return static_cast<bool>(a) ^
                                         static_cast<bool>(b);
                                }) &&
      register_unary_predicate("logical_not", std::logical_not<float>());
#undef TENSORSTORE_WRAP_FUNC

  return ok;
}

}  // namespace

bool RegisterNumpyInt4() {
  if (npy_int4 != NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load int4 module.");
    }
    return false;
  }
  return true;
}

PyObject* Int4Dtype() { return reinterpret_cast<PyObject*>(int4_type_ptr); }

int Int4NumpyType() { return npy_int4; }

}  // namespace internal_python
}  // namespace tensorstore
