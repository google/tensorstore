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

#ifndef THIRD_PARTY_PY_TENSORSTORE_INT4_H_
#define THIRD_PARTY_PY_TENSORSTORE_INT4_H_

/// \file
///
/// NumPy int4 dtype definition

// NumPy does not natively support int4, but allows new data types to be
// defined, This implementation is based on Tensorflow, and uses the same
// approach to conditionally define a int4 NumPy data type: if there is not
// existing data type registered with the name "int4", we define one and
// register it.  Otherwise, we just use the already-registered data type.
//
// Thus, whichever of TensorStore, Tensorflow, JAX (or any other library that
// may define a int4 data type) is imported first is the one to actually
// define the data type, and the subsequently imported libraries interoperate
// with it.  This does mean that there can be slight differences in behavior
// depending on the import order, but hopefully those differences won't be
// important.

#include "python/tensorstore/numpy.h"
// We actually just required `Python.h`, but all inclusions of that header are
// done via `numpy.h` to ensure the header order constraints are satisfied.

namespace tensorstore {
namespace internal_python {

// This implementation is parallel to bfloat16.

/// Register the int4 numpy type if one has not already been registered.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \returns `true` on success (including if another library has already
///     registered a "int4" dtype), or `false` if a Python exception has
///     been set.
bool RegisterNumpyInt4();

/// Returns a pointer to the int4 dtype object (registered either by this
/// library or another library).
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \pre `RegisterNumpyInt4()` must have previously returned successfully.
PyObject* Int4Dtype();

// Do not access directly.
extern int npy_int4;

/// Returns the id number of the int4 numpy type.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \pre `RegisterNumpyInt4()` must have previously returned successfully.
inline int Int4NumpyTypeNum() { return npy_int4; }

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_INT4_H_
