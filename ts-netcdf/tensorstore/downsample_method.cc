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

#include "tensorstore/downsample_method.h"

#include <ostream>

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, DownsampleMethod method) {
  switch (method) {
    case DownsampleMethod::kStride:
      return os << "stride";
    case DownsampleMethod::kMean:
      return os << "mean";
    case DownsampleMethod::kMin:
      return os << "min";
    case DownsampleMethod::kMax:
      return os << "max";
    case DownsampleMethod::kMedian:
      return os << "median";
    case DownsampleMethod::kMode:
      return os << "mode";
    default:
      return os << "<invalid downsamping mode>";
  }
}

}  // namespace tensorstore
