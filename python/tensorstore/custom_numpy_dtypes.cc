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

// Enable cmath defines on Windows
#define _USE_MATH_DEFINES

// Must be included first
// clang-format off
#include "python/tensorstore/numpy.h" //NOLINT
// clang-format on

#include <array>  // NOLINT
#include <cmath>  // NOLINT
#include <limits>  // NOLINT
#include <locale>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
// #include <Python.h>

#include "python/tensorstore/custom_float.h"
#include "python/tensorstore/custom_int4.h"
#include "python/tensorstore/custom_numpy_dtypes.h"
#include "python/tensorstore/custom_numpy_dtypes_impl.h"
#include "tensorstore/util/bfloat16.h"
#include "tensorstore/util/float8.h"
#include "tensorstore/util/int4.h"

// The implementation below is derived from jax-ml/ml_dtypes:
// https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/_src/dtypes.h

/* Copyright 2017 The ml_dtypes Authors.

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

namespace {

// Performs a NumPy array cast from type 'From' to 'To' via float.
template <typename From, typename To>
void FloatPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
                 void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<float>(from[i]));
  }
}

template <typename Type1, typename Type2>
bool RegisterTwoWayCustomCast() {
  int nptype1 = tensorstore::internal_python::TypeDescriptor<Type1>::npy_type;
  int nptype2 = tensorstore::internal_python::TypeDescriptor<Type2>::npy_type;
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, FloatPyCast<Type1, Type2>) <
      0) {
    return false;
  }
  PyArray_Descr* descr2 = PyArray_DescrFromType(nptype2);
  if (PyArray_RegisterCastFunc(descr2, nptype1, FloatPyCast<Type2, Type1>) <
      0) {
    return false;
  }
  return true;
}

}  // namespace

namespace tensorstore {
namespace internal_python {

using bfloat16 = tensorstore::BFloat16;
using float8_e4m3fn = tensorstore::Float8e4m3fn;
using float8_e4m3fnuz = tensorstore::Float8e4m3fnuz;
using float8_e4m3b11fnuz = tensorstore::Float8e4m3b11fnuz;
using float8_e5m2 = tensorstore::Float8e5m2;
using float8_e5m2fnuz = tensorstore::Float8e5m2fnuz;
using int4 = ::tensorstore::Int4Padded;
// TODO(ChromeHearts)
// using uint4 = ::tensorstore::Int4Padded;

template <>
struct TypeDescriptor<bfloat16> : CustomFloatType<bfloat16> {
  typedef bfloat16 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bfloat16";
  static constexpr const char* kQualifiedTypeName = "tensorstore.bfloat16";
  static constexpr const char* kTpDoc = "bfloat16 floating-point values";
  // We must register bfloat16 with a kind other than "f", because numpy
  // considers two types with the same kind and size to be equal, but
  // float16 != bfloat16.
  // The downside of this is that NumPy scalar promotion does not work with
  // bfloat16 values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'E';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3b11fnuz>
    : CustomFloatType<float8_e4m3b11fnuz> {
  typedef float8_e4m3b11fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3b11fnuz";
  static constexpr const char* kQualifiedTypeName =
      "tensorstore.float8_e4m3b11fnuz";
  static constexpr const char* kTpDoc =
      "float8_e4m3b11fnuz floating-point values";
  // We must register float8_e4m3b11fnuz with a kind other than "f", because
  // numpy considers two types with the same kind and size to be equal, and we
  // expect multiple 1 byte floating point types.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8_e4m3b11fnuz values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'L';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fn> : CustomFloatType<float8_e4m3fn> {
  typedef float8_e4m3fn T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fn";
  static constexpr const char* kQualifiedTypeName = "tensorstore.float8_e4m3fn";
  static constexpr const char* kTpDoc = "float8_e4m3fn floating-point values";
  // We must register float8_e4m3fn with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.  Using 'V' to mirror bfloat16 vs float16.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '4';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e4m3fnuz> : CustomFloatType<float8_e4m3fnuz> {
  typedef float8_e4m3fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e4m3fnuz";
  static constexpr const char* kQualifiedTypeName =
      "tensorstore.float8_e4m3fnuz";
  static constexpr const char* kTpDoc = "float8_e4m3fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'G';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2> : CustomFloatType<float8_e5m2> {
  typedef float8_e5m2 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2";
  static constexpr const char* kQualifiedTypeName = "tensorstore.float8_e5m2";
  static constexpr const char* kTpDoc = "float8_e5m2 floating-point values";
  // Treating e5m2 as the natural "float" type since it is IEEE-754 compliant.
  static constexpr char kNpyDescrKind = 'f';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = '5';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e5m2fnuz> : CustomFloatType<float8_e5m2fnuz> {
  typedef float8_e5m2fnuz T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e5m2fnuz";
  static constexpr const char* kQualifiedTypeName =
      "tensorstore.float8_e5m2fnuz";
  static constexpr const char* kTpDoc = "float8_e5m2fnuz floating-point values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'C';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int4> : Int4TypeDescriptor<int4> {
  typedef int4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int4";
  static constexpr const char* kQualifiedTypeName = "tensorstore.int4";
  static constexpr const char* kTpDoc = "int4 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'a';
  static constexpr char kNpyDescrByteorder = '=';
};

// TODO(ChromeHearts) UInt4
// template <>
// struct TypeDescriptor<uint4> : Int4TypeDescriptor<uint4> {
//   typedef uint4 T;
//   static constexpr bool is_floating = false;
//   static constexpr bool is_integral = true;
//   static constexpr const char* kTypeName = "uint4";
//   static constexpr const char* kQualifiedTypeName = "tensorstore.uint4";
//   static constexpr const char* kTpDoc = "uint4 integer values";
//   static constexpr char kNpyDescrKind = 'V';
//   // TODO(ChromeHearts): there doesn't seem to be a way of guaranteeing a type
//   // character is unique.
//   static constexpr char kNpyDescrType = 'A';
//   static constexpr char kNpyDescrByteorder = '=';
// };

int Int4NumpyTypeNum() { return TypeDescriptor<int4>::npy_type; }
int BFloat16NumpyTypeNum() { return TypeDescriptor<bfloat16>::npy_type; }
int Float8E4m3fnNumpyTypeNum() {
  return TypeDescriptor<float8_e4m3fn>::npy_type;
}
int Float8E4m3fnuzNumpyTypeNum() {
  return TypeDescriptor<float8_e4m3fnuz>::npy_type;
}
int Float8E4m3b11fnuzNumpyTypeNum() {
  return TypeDescriptor<float8_e4m3b11fnuz>::npy_type;
}
int Float8E5m2NumpyTypeNum() { return TypeDescriptor<float8_e5m2>::npy_type; }
int Float8E5m2fnuzNumpyTypeNum() {
  return TypeDescriptor<float8_e5m2fnuz>::npy_type;
}

// Casts between TYPE1 and TYPE2. Only perform the cast if
// both TYPE1 and TYPE2 haven't been previously registered, presumably by a
// different library. In this case, we assume the cast has also already been
// registered, and registering it again can cause segfaults due to accessing
// an uninitialized type descriptor in this library.
#define TENSORSTORE_INTERNAL_REGISTER_CAST(TYPE1, TYPE2)            \
  if (!TYPE1##_already_registered && !TYPE2##_already_registered && \
      !RegisterTwoWayCustomCast<TYPE1, TYPE2>()) {                  \
    return false;                                                   \
  }

// Initializes the module.
bool RegisterCustomNumpyDtypes() {
  InitializeNumpy();

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  bool bfloat16_already_registered;
  if (!RegisterFloatDtype<bfloat16>(numpy.get(),
                                    &bfloat16_already_registered)) {
    return false;
  }
  bool float8_e4m3b11fnuz_already_registered;
  if (!RegisterFloatDtype<float8_e4m3b11fnuz>(
          numpy.get(), &float8_e4m3b11fnuz_already_registered)) {
    return false;
  }
  bool float8_e4m3fn_already_registered;
  if (!RegisterFloatDtype<float8_e4m3fn>(numpy.get(),
                                         &float8_e4m3fn_already_registered)) {
    return false;
  }
  bool float8_e4m3fnuz_already_registered;
  if (!RegisterFloatDtype<float8_e4m3fnuz>(
          numpy.get(), &float8_e4m3fnuz_already_registered)) {
    return false;
  }
  bool float8_e5m2_already_registered;
  if (!RegisterFloatDtype<float8_e5m2>(numpy.get(),
                                       &float8_e5m2_already_registered)) {
    return false;
  }
  bool float8_e5m2fnuz_already_registered;
  if (!RegisterFloatDtype<float8_e5m2fnuz>(
          numpy.get(), &float8_e5m2fnuz_already_registered)) {
    return false;
  }

  if (!RegisterInt4Dtype<int4>(numpy.get())) {
    return false;
  }

  // TODO(ChromeHearts) UInt4
  // if (!RegisterInt4Dtype<uint4>(numpy.get())) {
  //   return false;
  // }

  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3b11fnuz, bfloat16)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fnuz, float8_e5m2fnuz)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fn, float8_e5m2)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3b11fnuz, float8_e4m3fn)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3b11fnuz, float8_e5m2)
  TENSORSTORE_INTERNAL_REGISTER_CAST(bfloat16, float8_e4m3fn)
  TENSORSTORE_INTERNAL_REGISTER_CAST(bfloat16, float8_e5m2)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fnuz, bfloat16)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e5m2fnuz, bfloat16)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fnuz, float8_e4m3b11fnuz)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e5m2fnuz, float8_e4m3b11fnuz)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fnuz, float8_e4m3fn)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e5m2fnuz, float8_e4m3fn)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e4m3fnuz, float8_e5m2)
  TENSORSTORE_INTERNAL_REGISTER_CAST(float8_e5m2fnuz, float8_e5m2)

  return true;
}

}  // namespace internal_python
}  // namespace tensorstore
