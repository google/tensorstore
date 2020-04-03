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

#ifndef TENSORSTORE_INTERNAL_EXCEPTION_MACROS_H_
#define TENSORSTORE_INTERNAL_EXCEPTION_MACROS_H_

#include "absl/base/macros.h"
#include "tensorstore/util/assert_macros.h"

#if ABSL_HAVE_EXCEPTIONS
#define TENSORSTORE_THROW_BAD_ALLOC throw std::bad_alloc()
#else
#define TENSORSTORE_THROW_BAD_ALLOC TENSORSTORE_CHECK(false && "Out of memory.")
#endif

#endif  // TENSORSTORE_INTERNAL_EXCEPTION_MACROS_H_
