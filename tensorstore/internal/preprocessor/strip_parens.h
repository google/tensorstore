// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_PREPROCESSOR_STRIP_PARENS_H_
#define TENSORSTORE_INTERNAL_PREPROCESSOR_STRIP_PARENS_H_

// Strips a single set of outer parentheses from `X`, if they are present.
//
// This is useful for passing types containing commas (like template arguments
// `Counter<int64_t, std::string>`) to preprocessor macros.
//
// If `X` is parenthesized, e.g., `(Foo)`, it evaluates to `Foo`.
// If `X` is not parenthesized, e.g., `Foo`, it evaluates to `Foo`.
//
// How it works:
// If `X` is `(Value)`:
//   TENSORSTORE_STRIP_PARENS((Value))
//   -> TENSORSTORE_ESC(_TENSORSTORE_ISH (Value))
//   -> TENSORSTORE_ESC(_TENSORSTORE_ISH Value)
//      (invokes function-like macro _TENSORSTORE_ISH)
//   -> TENSORSTORE_ESC_(_TENSORSTORE_ISH Value)
//   -> TENSORSTORE_VAN##_TENSORSTORE_ISH Value
//      (pastes to TENSORSTORE_VAN_TENSORSTORE_ISH)
//   -> TENSORSTORE_VAN_TENSORSTORE_ISH Value
//   -> Value
//
// If `X` is `Value`:
//   TENSORSTORE_STRIP_PARENS(Value)
//   -> TENSORSTORE_ESC(_TENSORSTORE_ISH Value)
//   -> TENSORSTORE_ESC_(_TENSORSTORE_ISH Value)
//      (_TENSORSTORE_ISH is not expanded since it is not invoked)
//   -> TENSORSTORE_VAN##_TENSORSTORE_ISH Value
//   -> TENSORSTORE_VAN_TENSORSTORE_ISH Value
//   -> Value
#define TENSORSTORE_STRIP_PARENS(X) TENSORSTORE_ESC(_TENSORSTORE_ISH X)
#define _TENSORSTORE_ISH(...) _TENSORSTORE_ISH __VA_ARGS__
#define TENSORSTORE_ESC(...) TENSORSTORE_ESC_(__VA_ARGS__)
#define TENSORSTORE_ESC_(...) TENSORSTORE_VAN##__VA_ARGS__
#define TENSORSTORE_VAN_TENSORSTORE_ISH

#endif  // TENSORSTORE_INTERNAL_PREPROCESSOR_STRIP_PARENS_H_
