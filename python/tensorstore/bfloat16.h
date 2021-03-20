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

#ifndef THIRD_PARTY_PY_TENSORSTORE_BFLOAT16_H_
#define THIRD_PARTY_PY_TENSORSTORE_BFLOAT16_H_

/// \file
///
/// NumPy bfloat16 dtype definition

// NumPy does not natively support bfloat16, but allows new data types to be
// defined, This implementation is based on Tensorflow, and uses the same
// approach to conditionally define a bfloat16 NumPy data type: if there is not
// existing data type registered with the name "bfloat16", we define one and
// register it.  Otherwise, we just use the already-registered data type.
//
// Thus, whichever of TensorStore, Tensorflow, JAX (or any other library that
// may define a bfloat16 data type) is imported first is the one to actually
// define the data type, and the subsequently imported libraries interoperate
// with it.  This does mean that there can be slight differences in behavior
// depending on the import order: for example, Tensorflow's bfloat16
// implementation flushes subnormal values to zero when converting from
// `float32`.  But hopefully those differences won't be important.

#include "python/tensorstore/numpy.h"
// We actually just required `Python.h`, but all inclusions of that header are
// done via `numpy.h` to ensure the header order constraints are satisfied.

namespace tensorstore {
namespace internal_python {

// This implementation is based on code from Tensorflow:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/lib/core/bfloat16.h
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

/// Register the bfloat16 numpy type if one has not already been registered.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \returns `true` on success (including if another library has already
///     registered a "bfloat16" dtype), or `false` if a Python exception has
///     been set.
bool RegisterNumpyBfloat16();

/// Returns a pointer to the bfloat16 dtype object (registered either by this
/// library or another library).
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \pre `RegisterNumpyBfloat16()` must have previously returned successfully.
PyObject* Bfloat16Dtype();

// Do not access directly.
extern int npy_bfloat16;

/// Returns the id number of the bfloat16 numpy type.
///
/// \pre The Python Global Interpreter Lock (GIL) must be owned by the calling
///     thread.
/// \pre `RegisterNumpyBfloat16()` must have previously returned successfully.
inline int Bfloat16NumpyTypeNum() { return npy_bfloat16; }

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_BFLOAT16_H_
