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

#include "tensorstore/util/unit.h"

#include <ostream>
#include <string>

#include "absl/strings/ascii.h"
#include "re2/re2.h"

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, const Unit& unit) {
  if (unit.base_unit.empty()) {
    return os << unit.multiplier;
  } else {
    if (unit.multiplier != 1) {
      os << unit.multiplier << ' ';
    }
    return os << unit.base_unit;
  }
}

bool operator==(const Unit& a, const Unit& b) {
  return a.multiplier == b.multiplier && a.base_unit == b.base_unit;
}

Unit::Unit(std::string_view unit) {
  static LazyRE2 kNumberPattern = {
      "([-+]?(?:\\.[0-9]+|[0-9]+(?:\\.[0-9]*)?)(?:[eE][-+]?\\d+)?)\\s*"};
  while (!unit.empty() && absl::ascii_isspace(unit.front())) {
    unit.remove_prefix(1);
  }
  while (!unit.empty() && absl::ascii_isspace(unit.back())) {
    unit.remove_suffix(1);
  }
  RE2::Consume(&unit, *kNumberPattern, &multiplier);
  base_unit = unit;
}

}  // namespace tensorstore
