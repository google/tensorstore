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
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace kvstore {

std::ostream& operator<<(std::ostream& os, ReadResult::State state) {
  switch (state) {
    case ReadResult::kUnspecified:
      os << "<unspecified>";
      break;
    case ReadResult::kMissing:
      os << "<missing>";
      break;
    case ReadResult::kValue:
      os << "<value>";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ReadResult& x) {
  std::string value;
  switch (x.state) {
    case ReadResult::kUnspecified:
      value = "<unspecified>";
      break;
    case ReadResult::kMissing:
      value = "<missing>";
      break;
    case ReadResult::kValue:
      value = tensorstore::QuoteString(absl::Cord(x.value).Flatten());
      break;
  }
  return os << "{value=" << value << ", stamp=" << x.stamp << "}";
}
}  // namespace kvstore
}  // namespace tensorstore
