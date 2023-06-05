// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/array_storage_statistics.h"

#include <ostream>

namespace tensorstore {

bool operator==(const ArrayStorageStatistics& a,
                const ArrayStorageStatistics& b) {
  return a.mask == b.mask && a.not_stored == b.not_stored &&
         a.fully_stored == b.fully_stored;
}

std::ostream& operator<<(std::ostream& os, const ArrayStorageStatistics& a) {
  os << "{not_stored=";
  if (a.mask & ArrayStorageStatistics::query_not_stored) {
    os << a.not_stored;
  } else {
    os << "<unknown>";
  }
  os << ", fully_stored=";
  if (a.mask & ArrayStorageStatistics::query_fully_stored) {
    os << a.fully_stored;
  } else {
    os << "<unknown>";
  }
  os << "}";
  return os;
}

}  // namespace tensorstore
