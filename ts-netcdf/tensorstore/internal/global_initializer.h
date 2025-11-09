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

#ifndef TENSORSTORE_INTERNAL_GLOBAL_INITIALIZER_H_
#define TENSORSTORE_INTERNAL_GLOBAL_INITIALIZER_H_

#include "tensorstore/internal/preprocessor/cat.h"

/// Defines a global initialization block to be run before `main()`, sequenced
/// in the same way as dynamic initialization of global variables
/// (https://en.cppreference.com/w/cpp/language/initialization#Dynamic_initialization).
///
/// Usage:
///
///     TENSORSTORE_GLOBAL_INITIALIZER {
///       // dynamic initialization code
///     }
///
/// This must only be used at global/namespace scope, and should not be used in
/// a header file.
///
/// Within a single translation unit, multiple initializer blocks are guaranteed
/// to run in the order they appear in the source code.  Across multiple
/// translation units, there is not guarantee on order.
///
/// Since this macro uses `__LINE__` to generate a unique identifier, it must
/// not be used more than once per line (not likely to occur except if used
/// within another macro).
#define TENSORSTORE_GLOBAL_INITIALIZER                            \
  namespace {                                                     \
  const struct TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__) {       \
    TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__)                    \
    ();                                                           \
  } TENSORSTORE_PP_CAT(tensorstore_global_init, __LINE__);        \
  }                                                               \
  TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__)::TENSORSTORE_PP_CAT( \
      TsGlobalInit, __LINE__)() /**/

#endif  // TENSORSTORE_INTERNAL_GLOBAL_INITIALIZER_H_
