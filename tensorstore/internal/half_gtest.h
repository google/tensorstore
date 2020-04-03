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

#ifndef TENSORSTORE_INTERNAL_HALF_GTEST_H_
#define TENSORSTORE_INTERNAL_HALF_GTEST_H_

#include <iosfwd>

#include <half.hpp>

namespace half_float {

/// Workaround for Google Test bug.
///
/// Without this, attempts by Google Test to print values of type
/// `half_float::half` result in an ambiguous call to `operator<<`.
inline void PrintTo(half const& h, std::ostream* os) { *os << h; }

}  // namespace half_float

#endif  // TENSORSTORE_INTERNAL_HALF_GTEST_H_
