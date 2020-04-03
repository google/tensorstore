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

#ifndef TENSORSTORE_UTIL_DEFAULT_ITERATION_RESULT_H_
#define TENSORSTORE_UTIL_DEFAULT_ITERATION_RESULT_H_

namespace tensorstore {

/// Metafunction that defines the default value returned by elementwise
/// iteration functions, given the return type of the elementwise function, when
/// there are no elements to iterate over.
///
/// By default this returns `T()`, but is specialized for `bool` to return a
/// default value of `true`.  Users may specialize this template for their own
/// types.
template <typename T>
struct DefaultIterationResult {
  static constexpr T value() { return T(); }
};

/// Specialization of `DefaultIterationResult` for `bool`.  A value of `false`
/// indicates that iteration stopped early (due to false return from the
/// elementwise function); therefore, the natural default is `true` to avoid
/// ambiguity.
template <>
struct DefaultIterationResult<bool> {
  static constexpr bool value() { return true; }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_DEFAULT_ITERATION_RESULT_H_
