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

#include "absl/time/time.h"

#include <limits>

#include "python/tensorstore/time.h"

namespace tensorstore {
namespace internal_python {

double ToPythonTimestamp(const absl::Time& time) {
  if (time == absl::InfinitePast()) {
    return -std::numeric_limits<double>::infinity();
  }
  if (time == absl::InfiniteFuture()) {
    return std::numeric_limits<double>::infinity();
  }
  return absl::ToDoubleSeconds(time - absl::UnixEpoch());
}

absl::Time FromPythonTimestamp(double t) {
  if (std::isfinite(t)) {
    return absl::UnixEpoch() + absl::Seconds(t);
  }
  if (t == -std::numeric_limits<double>::infinity()) {
    return absl::InfinitePast();
  }
  return absl::InfiniteFuture();
}

}  // namespace internal_python
}  // namespace tensorstore
