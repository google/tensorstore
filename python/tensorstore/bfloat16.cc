// Copyright 2021 The TensorStore Authors
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

#include <type_traits>

#include "python/tensorstore/bfloat16.h"
#include "python/tensorstore/data_type.h"
#include "tensorstore/data_type.h"
#include "tensorstore/util/bfloat16.h"
#include "tensorstore/util/str_cat.h"

// This implementation is based on code from Tensorflow:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/lib/core/bfloat16.cc
//
// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace tensorstore {
namespace internal_python {

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_bfloat16 = NPY_NOTYPE;

namespace {

// https://bugs.python.org/issue39573  Py_SET_TYPE() added to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
template<typename T>
static inline void Py_SET_TYPE(T *ob, PyTypeObject *type) {
    reinterpret_cast<PyObject*>(ob)->ob_type = type;
}
#endif

using bfloat16 = tensorstore::bfloat16_t;

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
extern PyTypeObject bfloat16_type;

// Pointer to the bfloat16 type object we are using. This is either a pointer
// to bfloat16_type, if we choose to register it, or to the bfloat16 type
// registered by another system into NumPy.
PyTypeObject* bfloat16_type_ptr = nullptr;

// Representation of a Python bfloat16 object.
struct PyBfloat16 {
  PyObject_HEAD;  // Python object header
  bfloat16 value;
};

// Returns true if 'object' is a PyBfloat16.
bool PyBfloat16_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&bfloat16_type));
}

// Extracts the value of a PyBfloat16 object.
bfloat16 PyBfloat16_Bfloat16(PyObject* object) {
  return reinterpret_cast<PyBfloat16*>(object)->value;
}

// Constructs a PyBfloat16 object from a bfloat16.
Safe_PyObjectPtr PyBfloat16_FromBfloat16(bfloat16 x) {
  Safe_PyObjectPtr ref = make_safe(bfloat16_type.tp_alloc(&bfloat16_type, 0));
  PyBfloat16* p = reinterpret_cast<PyBfloat16*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a bfloat16 value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToBfloat16(PyObject* arg, bfloat16* output) {
  if (PyBfloat16_Check(arg)) {
    *output = PyBfloat16_Bfloat16(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(hawkinsp): check for overflow
    *output = bfloat16(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(hawkinsp): check for overflow
    *output = bfloat16(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    tensorstore::float16_t f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_bfloat16) {
      ref = make_safe(PyArray_Cast(arr, npy_bfloat16));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<bfloat16*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

bool SafeCastToBfloat16(PyObject* arg, bfloat16* output) {
  if (PyBfloat16_Check(arg)) {
    *output = PyBfloat16_Bfloat16(arg);
    return true;
  }
  return false;
}

// Converts a PyBfloat16 into a PyFloat.
PyObject* PyBfloat16_Float(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyBfloat16 into a PyInt.
PyObject* PyBfloat16_Int(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  long y = static_cast<long>(x);  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyBfloat16.
PyObject* PyBfloat16_Negative(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  return PyBfloat16_FromBfloat16(-x).release();
}

PyObject* PyBfloat16_Add(PyObject* a, PyObject* b) {
  bfloat16 x, y;
  if (SafeCastToBfloat16(a, &x) && SafeCastToBfloat16(b, &y)) {
    return PyBfloat16_FromBfloat16(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

PyObject* PyBfloat16_Subtract(PyObject* a, PyObject* b) {
  bfloat16 x, y;
  if (SafeCastToBfloat16(a, &x) && SafeCastToBfloat16(b, &y)) {
    return PyBfloat16_FromBfloat16(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

PyObject* PyBfloat16_Multiply(PyObject* a, PyObject* b) {
  bfloat16 x, y;
  if (SafeCastToBfloat16(a, &x) && SafeCastToBfloat16(b, &y)) {
    return PyBfloat16_FromBfloat16(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

PyObject* PyBfloat16_TrueDivide(PyObject* a, PyObject* b) {
  bfloat16 x, y;
  if (SafeCastToBfloat16(a, &x) && SafeCastToBfloat16(b, &y)) {
    return PyBfloat16_FromBfloat16(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Python number methods for PyBfloat16 objects.
PyNumberMethods PyBfloat16_AsNumber = {
    PyBfloat16_Add,       // nb_add
    PyBfloat16_Subtract,  // nb_subtract
    PyBfloat16_Multiply,  // nb_multiply
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyBfloat16_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
    PyBfloat16_Int,       // nb_int
    nullptr,              // reserved
    PyBfloat16_Float,     // nb_float

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

    nullptr,                // nb_floor_divide
    PyBfloat16_TrueDivide,  // nb_true_divide
    nullptr,                // nb_inplace_floor_divide
    nullptr,                // nb_inplace_true_divide
    nullptr,                // nb_index
};

// Constructs a new PyBfloat16.
PyObject* PyBfloat16_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to bfloat16 constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  bfloat16 value;
  if (PyBfloat16_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToBfloat16(arg, &value)) {
    return PyBfloat16_FromBfloat16(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_bfloat16) {
      return PyArray_Cast(arr, npy_bfloat16);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyBfloat16s.
PyObject* PyBfloat16_RichCompare(PyObject* a, PyObject* b, int op) {
  bfloat16 x, y;
  if (!SafeCastToBfloat16(a, &x) || !SafeCastToBfloat16(b, &y)) {
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

// Implementation of repr() for PyBfloat16.
PyObject* PyBfloat16_Repr(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  std::string v = tensorstore::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyBfloat16.
PyObject* PyBfloat16_Str(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  std::string v = tensorstore::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyBfloat16. We use the identity function, which is a weak
// hash function.
Py_hash_t PyBfloat16_Hash(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  return internal::bit_cast<uint16_t>(x);
}

// Python type for PyBfloat16 objects.
PyTypeObject bfloat16_type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "bfloat16",  // tp_name
    sizeof(PyBfloat16),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyBfloat16_Repr,       // tp_repr
    &PyBfloat16_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyBfloat16_Hash,       // tp_hash
    nullptr,               // tp_call
    PyBfloat16_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
    nullptr,               // tp_as_buffer
                           // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "bfloat16 floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyBfloat16_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    nullptr,                           // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyBfloat16_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyBfloat16_ArrFuncs;

PyArray_Descr NPyBfloat16_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&bfloat16_type),
    // We must register bfloat16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != bfloat16.
    // The downside of this is that NumPy scalar promotion does not work with
    // bfloat16 values.
    /*kind=*/'V',
    // TODO(hawkinsp): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(bfloat16),
    /*alignment=*/alignof(bfloat16),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyBfloat16_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject* NPyBfloat16_GetItem(void* data, void* arr) {
  bfloat16 x;
  memcpy(&x, data, sizeof(bfloat16));
  return PyBfloat16_FromBfloat16(x).release();
}

int NPyBfloat16_SetItem(PyObject* item, void* data, void* arr) {
  bfloat16 x;
  if (!CastToBfloat16(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(bfloat16));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

int NPyBfloat16_Compare(const void* a, const void* b, void* arr) {
  bfloat16 x;
  memcpy(&x, a, sizeof(bfloat16));

  bfloat16 y;
  memcpy(&y, b, sizeof(bfloat16));

  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  // NaNs sort to the end.
  if (!isnan(x) && isnan(y)) {
    return -1;
  }
  if (isnan(x) && !isnan(y)) {
    return 1;
  }
  return 0;
}

void NPyBfloat16_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyBfloat16_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyBfloat16_NonZero(void* data, void* arr) {
  bfloat16 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<bfloat16>(0);
}

int NPyBfloat16_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  bfloat16* const buffer = reinterpret_cast<bfloat16*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<bfloat16>(start + i * delta);
  }
  return 0;
}

void NPyBfloat16_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                         void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    bfloat16* const b1 = reinterpret_cast<bfloat16*>(c1);
    bfloat16* const b2 = reinterpret_cast<bfloat16*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  bfloat16* out = reinterpret_cast<bfloat16*>(op);
  *out = static_cast<bfloat16>(acc);
}

int NPyBfloat16_CompareFunc(const void* v1, const void* v2, void* arr) {
  bfloat16 b1 = *reinterpret_cast<const bfloat16*>(v1);
  bfloat16 b2 = *reinterpret_cast<const bfloat16*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

int NPyBfloat16_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                           void* arr) {
  const bfloat16* bdata = reinterpret_cast<const bfloat16*>(data);
  float max_val = -std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) > max_val) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyBfloat16_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                           void* arr) {
  const bfloat16* bdata = reinterpret_cast<const bfloat16*>(data);
  float min_val = std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) < min_val) {
      min_val = static_cast<float>(bdata[i]);
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

// Registers a cast between bfloat16 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'.
template <typename T>
bool RegisterBfloat16Cast(int numpy_type) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, npy_bfloat16, NPyCast<T, bfloat16>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyBfloat16_Descr, numpy_type,
                               NPyCast<bfloat16, T>) < 0) {
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
  if (PyUFunc_RegisterLoopForType(ufunc, npy_bfloat16, fn, types, nullptr) <
      0) {
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

std::pair<float, float> divmod(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

constexpr float PI = 3.14159265358979323846f;

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

  // If another module (presumably either TF or JAX) has registered a bfloat16
  // type, use it. We don't want two bfloat16 types if we can avoid it since it
  // leads to confusion if we have two different types with the same name. This
  // assumes that the other module has a sufficiently complete bfloat16
  // implementation. The only known NumPy bfloat16 extension at the time of
  // writing is this one (distributed in TF and JAX).
  // TODO(hawkinsp): distribute the bfloat16 extension as its own pip package,
  // so we can unambiguously refer to a single canonical definition of bfloat16.
  int typenum = PyArray_TypeNumFromName(const_cast<char*>("bfloat16"));
  if (typenum != NPY_NOTYPE) {
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    // The test for an argmax function here is to verify that the
    // bfloat16 implementation is sufficiently new, and, say, not from
    // an older version of TF or JAX.
    if (descr && descr->f && descr->f->argmax) {
      npy_bfloat16 = typenum;
      bfloat16_type_ptr = descr->typeobj;
      return true;
    }
  }

  bfloat16_type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&bfloat16_type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyBfloat16_ArrFuncs);
  NPyBfloat16_ArrFuncs.getitem = NPyBfloat16_GetItem;
  NPyBfloat16_ArrFuncs.setitem = NPyBfloat16_SetItem;
  NPyBfloat16_ArrFuncs.compare = NPyBfloat16_Compare;
  NPyBfloat16_ArrFuncs.copyswapn = NPyBfloat16_CopySwapN;
  NPyBfloat16_ArrFuncs.copyswap = NPyBfloat16_CopySwap;
  NPyBfloat16_ArrFuncs.nonzero = NPyBfloat16_NonZero;
  NPyBfloat16_ArrFuncs.fill = NPyBfloat16_Fill;
  NPyBfloat16_ArrFuncs.dotfunc = NPyBfloat16_DotFunc;
  NPyBfloat16_ArrFuncs.compare = NPyBfloat16_CompareFunc;
  NPyBfloat16_ArrFuncs.argmax = NPyBfloat16_ArgMaxFunc;
  NPyBfloat16_ArrFuncs.argmin = NPyBfloat16_ArgMinFunc;

  Py_SET_TYPE(&NPyBfloat16_Descr, &PyArrayDescr_Type);
  npy_bfloat16 = PyArray_RegisterDataType(&NPyBfloat16_Descr);
  bfloat16_type_ptr = &bfloat16_type;
  if (npy_bfloat16 < 0) {
    return false;
  }

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy.get(), "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype('bfloat16')` work.
  if (PyDict_SetItemString(typeDict_obj.get(), "bfloat16",
                           reinterpret_cast<PyObject*>(&bfloat16_type)) < 0) {
    return false;
  }

  // Support dtype(bfloat16)
  if (PyDict_SetItemString(bfloat16_type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyBfloat16_Descr)) <
      0) {
    return false;
  }

  // Register casts
  if (!RegisterBfloat16Cast<float16_t>(NPY_HALF)) {
    return false;
  }

  if (!RegisterBfloat16Cast<float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterBfloat16Cast<double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterBfloat16Cast<bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterBfloat16Cast<uint8_t>(NPY_UINT8)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int8_t>(NPY_INT8)) {
    return false;
  }
  if (!RegisterBfloat16Cast<uint16_t>(NPY_UINT16)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int16_t>(NPY_INT16)) {
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int>(NPY_INT)) {
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterBfloat16Cast<long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterBfloat16Cast<unsigned long long>(NPY_ULONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterBfloat16Cast<long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterBfloat16Cast<uint64_t>(NPY_UINT64)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int64_t>(NPY_INT64)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterBfloat16Cast<std::complex<float>>(NPY_COMPLEX64)) {
    return false;
  }
  if (!RegisterBfloat16Cast<std::complex<double>>(NPY_COMPLEX128)) {
    return false;
  }

  // Safe casts from bfloat16 to other types
  if (PyArray_RegisterCanCast(&NPyBfloat16_Descr, NPY_FLOAT, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyBfloat16_Descr, NPY_DOUBLE, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyBfloat16_Descr, NPY_COMPLEX64, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyBfloat16_Descr, NPY_COMPLEX128,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  // Safe casts to bfloat16 from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL), npy_bfloat16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_UINT8), npy_bfloat16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_INT8), npy_bfloat16,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  const auto register_unary = [&](const char* name, auto func) {
    return RegisterUFunc<bfloat16, bfloat16>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), bfloat16>());
  };

  const auto register_unary_predicate = [&](const char* name, auto func) {
    return RegisterUFunc<bfloat16, bool>(
        numpy.get(), name, SingleOutputAdapter<decltype(func), bfloat16>());
  };

  const auto register_binary = [&](const char* name, auto func) {
    return RegisterUFunc<bfloat16, bfloat16, bfloat16>(
        numpy.get(), name,
        SingleOutputAdapter<decltype(func), bfloat16, bfloat16>());
  };

  const auto register_binary_predicate = [&](const char* name, auto func) {
    return RegisterUFunc<bfloat16, bfloat16, bool>(
        numpy.get(), name,
        SingleOutputAdapter<decltype(func), bfloat16, bfloat16>());
  };

  constexpr auto remainder_func = [](bfloat16 a, bfloat16 b) {
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
      register_binary("logaddexp",
                      [](bfloat16 bx, bfloat16 by) {
                        float x = static_cast<float>(bx);
                        float y = static_cast<float>(by);
                        if (x == y) {
                          // Handles infinities of the same sign.
                          return bfloat16(x + std::log(2.0f));
                        }
                        float out = std::numeric_limits<float>::quiet_NaN();
                        if (x > y) {
                          out = x + std::log1p(std::exp(y - x));
                        } else if (x < y) {
                          out = y + std::log1p(std::exp(x - y));
                        }
                        return bfloat16(out);
                      }) &&
      register_unary("negative", std::negate<void>()) &&
      register_unary("positive", internal::identity()) &&
      register_binary("true_divide", std::divides<void>()) &&
      register_binary(
          "floor_divide",
          [](bfloat16 a, bfloat16 b) { return divmod(a, b).first; }) &&
      register_binary("power", TENSORSTORE_WRAP_FUNC(pow)) &&
      register_binary("remainder", remainder_func) &&
      register_binary("mod", remainder_func) &&
      register_binary("fmod", TENSORSTORE_WRAP_FUNC(fmod)) &&
      RegisterUFunc<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>(
          numpy.get(), "divmod",
          [](bfloat16 a, bfloat16 b, bfloat16& quotient, bfloat16& remainder) {
            std::tie(quotient, remainder) = divmod(a, b);
          }) &&
      register_unary("abs", abs_func) && register_unary("fabs", abs_func) &&
      register_unary("rint", TENSORSTORE_WRAP_FUNC(rint)) &&
      register_unary("sign",
                     [](float a) -> float {
                       if (a < 0) {
                         return -1.0;
                       }
                       if (a > 0) {
                         return 1.0;
                       }
                       return a;
                     }) &&
      register_binary("heaviside",
                      [](bfloat16 bx, bfloat16 h0) {
                        float x = static_cast<float>(bx);
                        if (std::isnan(x)) {
                          return bx;
                        }
                        if (x < 0) {
                          return bfloat16(0.0f);
                        }
                        if (x > 0) {
                          return bfloat16(1.0f);
                        }
                        return h0;  // x == 0
                      }) &&
      register_unary("conjugate", internal::identity()) &&
      register_unary("exp", TENSORSTORE_WRAP_FUNC(tensorstore::exp)) &&
      register_unary("exp2", TENSORSTORE_WRAP_FUNC(tensorstore::exp2)) &&
      register_unary("expm1", TENSORSTORE_WRAP_FUNC(tensorstore::expm1)) &&
      register_unary("log", TENSORSTORE_WRAP_FUNC(tensorstore::log)) &&
      register_unary("log2", TENSORSTORE_WRAP_FUNC(tensorstore::log2)) &&
      register_unary("log10", TENSORSTORE_WRAP_FUNC(tensorstore::log10)) &&
      register_unary("log1p", TENSORSTORE_WRAP_FUNC(tensorstore::log1p)) &&
      register_unary("sqrt", TENSORSTORE_WRAP_FUNC(tensorstore::sqrt)) &&
      register_unary("square", [](bfloat16 a) { return a * a; }) &&
      register_unary("cbrt", TENSORSTORE_WRAP_FUNC(std::cbrt)) &&
      register_unary("reciprocal", [](bfloat16 a) { return 1.0f / a; }) &&

      // Trigonometric functions

      register_unary("sin", TENSORSTORE_WRAP_FUNC(sin)) &&
      register_unary("cos", TENSORSTORE_WRAP_FUNC(cos)) &&
      register_unary("tan", TENSORSTORE_WRAP_FUNC(tan)) &&
      register_unary("arcsin", TENSORSTORE_WRAP_FUNC(asin)) &&
      register_unary("arccos", TENSORSTORE_WRAP_FUNC(acos)) &&
      register_unary("arctan", TENSORSTORE_WRAP_FUNC(atan)) &&
      register_binary("arctan2", TENSORSTORE_WRAP_FUNC(std::atan2)) &&
      register_binary("hypot", TENSORSTORE_WRAP_FUNC(std::hypot)) &&
      register_unary("sinh", TENSORSTORE_WRAP_FUNC(sinh)) &&
      register_unary("cosh", TENSORSTORE_WRAP_FUNC(cosh)) &&
      register_unary("tanh", TENSORSTORE_WRAP_FUNC(tanh)) &&
      register_unary("arcsinh", TENSORSTORE_WRAP_FUNC(asinh)) &&
      register_unary("arccosh", TENSORSTORE_WRAP_FUNC(acosh)) &&
      register_unary("arctanh", TENSORSTORE_WRAP_FUNC(atanh)) &&
      register_unary("deg2rad",
                     [](bfloat16 a) {
                       static constexpr float radians_per_degree = PI / 180.0f;
                       return a * radians_per_degree;
                     }) &&
      register_unary("rad2deg",
                     [](bfloat16 a) {
                       static constexpr float degrees_per_radian = 180.0f / PI;
                       return a * degrees_per_radian;
                     }) &&

      // Comparison functions
      register_binary_predicate("equal", std::equal_to<void>()) &&
      register_binary_predicate("not_equal", std::not_equal_to<void>()) &&
      register_binary_predicate("less", std::less<void>()) &&
      register_binary_predicate("greater", std::greater<void>()) &&
      register_binary_predicate("less_equal", std::less_equal<void>()) &&
      register_binary_predicate("greater_equal", std::greater_equal<void>()) &&

      register_binary(
          "maximum",
          [](float a, float b) { return (std::isnan(a) || a > b) ? a : b; }) &&
      register_binary(
          "minimum",
          [](float a, float b) { return (std::isnan(a) || a < b) ? a : b; }) &&
      register_binary("fmax", TENSORSTORE_WRAP_FUNC(std::fmax)) &&
      register_binary("fmin", TENSORSTORE_WRAP_FUNC(std::fmin)) &&
      register_binary_predicate("logical_and", std::logical_and<float>()) &&
      register_binary_predicate("logical_or", std::logical_or<float>()) &&
      register_binary_predicate("logical_xor",
                                [](float a, float b) {
                                  return static_cast<bool>(a) ^
                                         static_cast<bool>(b);
                                }) &&
      register_unary_predicate("logical_not", std::logical_not<float>()) &&

      // Floating point functions
      register_unary_predicate("isfinite", TENSORSTORE_WRAP_FUNC(isfinite)) &&
      register_unary_predicate("isinf", TENSORSTORE_WRAP_FUNC(isinf)) &&
      register_unary_predicate("isnan", TENSORSTORE_WRAP_FUNC(isnan)) &&
      register_unary_predicate("signbit",
                               TENSORSTORE_WRAP_FUNC(tensorstore::signbit)) &&
      register_binary("copysign", TENSORSTORE_WRAP_FUNC(std::copysign)) &&
      RegisterUFunc<bfloat16, bfloat16, bfloat16>(
          numpy.get(), "modf",
          [](bfloat16 a, bfloat16& fraction, bfloat16& integral) {
            float integral_float;
            fraction = std::modf(static_cast<float>(a), &integral_float);
            integral = integral_float;
          }) &&
      RegisterUFunc<bfloat16, int, bfloat16>(
          numpy.get(), "ldexp",
          [](bfloat16 a, int exponent, bfloat16& b) {
            b = std::ldexp(a, exponent);
          }) &&
      RegisterUFunc<bfloat16, bfloat16, int>(
          numpy.get(), "frexp",
          [](bfloat16 a, bfloat16& fraction, int& exponent) {
            fraction = std::frexp(static_cast<float>(a), &exponent);
          }) &&
      register_unary("floor", TENSORSTORE_WRAP_FUNC(tensorstore::floor)) &&
      register_unary("ceil", TENSORSTORE_WRAP_FUNC(tensorstore::ceil)) &&
      register_unary("trunc", TENSORSTORE_WRAP_FUNC(tensorstore::trunc)) &&
      register_binary("nextafter",
                      TENSORSTORE_WRAP_FUNC(tensorstore::nextafter));
#undef TENSORSTORE_WRAP_FUNC

  return ok;
}

}  // namespace

bool RegisterNumpyBfloat16() {
  if (npy_bfloat16 != NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load bfloat16 module.");
    }
    return false;
  }
  return true;
}

PyObject* Bfloat16Dtype() {
  return reinterpret_cast<PyObject*>(bfloat16_type_ptr);
}

int Bfloat16NumpyType() { return npy_bfloat16; }

}  // namespace internal_python
}  // namespace tensorstore
