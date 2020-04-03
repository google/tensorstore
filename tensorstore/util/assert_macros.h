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

#ifndef TENSORSTORE_INTERNAL_ASSERT_MACROS_H_
#define TENSORSTORE_INTERNAL_ASSERT_MACROS_H_

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "tensorstore/internal/log_message.h"
#include "tensorstore/internal/source_location.h"

/// Defines a TENSORSTORE_UNREACHABLE macro that indicates to the compiler that
/// control flow cannot reach the point where the macro is used.
#if defined(__clang__) || defined(__GNUC__)
#define TENSORSTORE_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define TENSORSTORE_UNREACHABLE __assume(false)
#else
#define TENSORSTORE_UNREACHABLE \
  do {                          \
  } while (false)
#endif

/// If the argument evaluates to `false`, logs an error message and terminates.
///
/// This macro should be called with a single C++ expression.  We use a variadic
/// macro to allow calls like TENSORSTORE_CHECK(foo<1,2>()).
#define TENSORSTORE_CHECK(...)                                                \
  do {                                                                        \
    if (ABSL_PREDICT_FALSE(!(__VA_ARGS__))) {                                 \
      ::tensorstore::internal::LogMessageFatal("CHECK failed: " #__VA_ARGS__, \
                                               TENSORSTORE_LOC);              \
    }                                                                         \
  } while (false)

#endif  // TENSORSTORE_INTERNAL_ASSERT_MACROS_H_
