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

#include "tensorstore/index_space/output_index_method.h"

#include <ostream>

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, OutputIndexMethod method) {
  switch (method) {
    case OutputIndexMethod::constant:
      return os << "constant";
    case OutputIndexMethod::single_input_dimension:
      return os << "single_input_dimension";
    case OutputIndexMethod::array:
      return os << "array";
    default:
      return os << "<unknown>";
  }
}

}  // namespace tensorstore
