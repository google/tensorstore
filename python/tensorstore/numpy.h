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

#ifndef THIRD_PARTY_PY_TENSORSTORE_NUMPY_H_
#define THIRD_PARTY_PY_TENSORSTORE_NUMPY_H_

/// \file
///
/// Glue header for use by TensorStore in accessing the NumPy C API.

// NumPy uses a custom dynamic linking mechanism: instead of directly exposing
// symbols to be resolved by the usual system dynamic linker, the NumPy headers
// only define structs, constants, and a few static functions like
// `_import_array` used for initialization.
//
// To access the NumPy API functions (e.g. the PyArray_* functions), the static
// _import_array function uses the CPython API to import a Python module and
// then obtains a pointer to a custom NumPy symbol table data structure via a
// PyCapsule stored as an attribute of the module.  This symbol table is stored
// in a global variable.  The NumPy headers actually define all of the NumPy API
// functions as macros that call the real API function by way of that global
// variable.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.  We actually only need the normal Python
// C API, not pybind11, but we rely on pybind11 to define appropriate macros to
// avoid build failures when including `<Python.h>`.

#ifdef PyArray_Type
#error "Numpy cannot be included before numpy.h."
#endif

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL _tensorstore_numpy_array_api
#define PY_UFUNC_UNIQUE_SYMBOL _tensorstore_numpy_ufunc_api

// This is defined only in `numpy.cc` in order to ensure there is a single
// definition of the symbol table global variables.
#ifndef TENSORSTORE_INTERNAL_PYTHON_IMPORT_NUMPY_API
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#undef NO_IMPORT_ARRAY
#undef NO_IMPORT_UFUNC

namespace tensorstore {
namespace internal_python {

/// Called from the module initialization function in `tensorstore.cc` to
/// initialize the NumPy API.  This should not be called anywhere else.
///
/// \returns `true` on success, or `false` if a Python exception has been set.
bool InitializeNumpy();

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_NUMPY_H_
