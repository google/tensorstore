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

#ifndef TENSORSTORE_UTIL_UTF8_STRING_H_
#define TENSORSTORE_UTIL_UTF8_STRING_H_

#include <ostream>
#include <string>

namespace tensorstore {

/// Wrapper around `std::string` to indicate a UTF-8 encoded string.
struct Utf8String {
  std::string utf8;
  friend bool operator<(const Utf8String& a, const Utf8String& b) {
    return a.utf8 < b.utf8;
  }
  friend bool operator<=(const Utf8String& a, const Utf8String& b) {
    return a.utf8 <= b.utf8;
  }
  friend bool operator>(const Utf8String& a, const Utf8String& b) {
    return a.utf8 > b.utf8;
  }
  friend bool operator>=(const Utf8String& a, const Utf8String& b) {
    return a.utf8 >= b.utf8;
  }
  friend bool operator==(const Utf8String& a, const Utf8String& b) {
    return a.utf8 == b.utf8;
  }
  friend bool operator!=(const Utf8String& a, const Utf8String& b) {
    return a.utf8 != b.utf8;
  }
  friend std::ostream& operator<<(std::ostream& os, const Utf8String& s) {
    return os << s.utf8;
  }
};

static_assert(sizeof(Utf8String) == sizeof(std::string), "");

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_UTF8_STRING_H_
