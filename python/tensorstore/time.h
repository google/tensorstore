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

#ifndef THIRD_PARTY_PY_TENSORSTORE_TIME_H_
#define THIRD_PARTY_PY_TENSORSTORE_TIME_H_

#include "absl/time/time.h"

namespace tensorstore {
namespace internal_python {

/// Converts to a "Python timestamp", i.e. floating-point second since Unix
/// epoch.
///
/// Positive and negative infinity correspond to absl::Time::InfinitePast() and
/// absl::Time::InfiniteFuture(), respectively.
///
/// We do not use the Python `datetime.datetime` class since it uses a
/// split-time representation, uses time zones, and generally is more oriented
/// towards date calculations than representing precise time instants.
double ToPythonTimestamp(const absl::Time& time);

/// Inverse of `ToPythonTimestamp`.
absl::Time FromPythonTimestamp(double t);

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_TIME_H_
