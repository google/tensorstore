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

// This file defines the global symbol table variables used by the NumPy
// headers.
#define TENSORSTORE_INTERNAL_PYTHON_IMPORT_NUMPY_API

#include "python/tensorstore/numpy.h"

namespace tensorstore {
namespace internal_python {

bool InitializeNumpy() {
  import_array1(false);
  import_umath1(false);
  return true;
}

}  // namespace internal_python
}  // namespace tensorstore
