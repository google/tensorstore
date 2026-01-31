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

#ifndef TENSORSTORE_UTIL_QUOTE_STRING_H_
#define TENSORSTORE_UTIL_QUOTE_STRING_H_

#include <stddef.h>

#include <ostream>
#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorstore {
namespace internal {
std::ostream& PrintQuotedString(std::ostream& os, std::string_view v);
}  // namespace internal

// Returns a representation using C++ string literal syntax (including opening
// and closing quotes) of the contents of `s`.
//
// Example:
//     EXPECT_EQ("\"hello\\nworld\"",
//               absl::StrFormat("%v", QuoteString("hello\nworld")));
//
struct QuoteString {
  std::string_view s;
  explicit QuoteString(std::string_view sv ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : s(sv) {}

  std::string ToString() const {
    return absl::StrCat("\"", absl::CHexEscape(s), "\"");
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, QuoteString v) {
    sink.Append("\"");
    sink.Append(absl::CHexEscape(v.s));
    sink.Append("\"");
  }
  friend std::ostream& operator<<(std::ostream& os, QuoteString v) {
    return internal::PrintQuotedString(os, v.s);
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_QUOTE_STRING_H_
