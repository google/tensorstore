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

#include "tensorstore/kvstore/read_result.h"

#include <ostream>

#include "absl/strings/str_format.h"

namespace tensorstore {
namespace kvstore {

std::ostream& operator<<(std::ostream& os, ReadResult::State state) {
  return os << absl::StreamFormat("%v", state);
}

std::ostream& operator<<(std::ostream& os, const ReadResult& x) {
  return os << absl::StreamFormat("%v", x);
}
}  // namespace kvstore
}  // namespace tensorstore
