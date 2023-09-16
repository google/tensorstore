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

#ifndef THIRD_PARTY_PY_TENSORSTORE_CUSTOM_NUMPY_DTYPES_H_
#define THIRD_PARTY_PY_TENSORSTORE_CUSTOM_NUMPY_DTYPES_H_

/// \file
///
/// NumPy custom dtype definition

// NumPy  allows new data types to be  defined, This implementation is based on
// Tensorflow, and uses the same approach to conditionally define a bfloat16,
// float8 variants, int4 NumPy data type: if there is not existing data
// type registered with the name "int4", we define one and register it.
// Otherwise, we just use the already-registered data type.
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

/// Register all the custom numpy type if they have not already been registered.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \returns `true` on success (including if another library has already
///     registered a "int4" dtype), or `false` if a Python exception has
///     been set.
bool RegisterCustomNumpyDtypes();

/// Returns the id number of the custom dtype's numpy type.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \pre `RegisterNumpyInt4()` must have previously returned successfully.
int Int4NumpyTypeNum();
int BFloat16NumpyTypeNum();
int Float8E4m3fnNumpyTypeNum();
int Float8E4m3fnuzNumpyTypeNum();
int Float8E4m3b11fnuzNumpyTypeNum();
int Float8E5m2NumpyTypeNum();
int Float8E5m2fnuzNumpyTypeNum();

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_INT4_H_
