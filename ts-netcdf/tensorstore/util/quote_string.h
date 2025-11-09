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

#include <string>
#include <string_view>

namespace tensorstore {

/// Returns a representation using C++ string literal syntax (including opening
/// and closing quotes) of the contents of `s`.
///
/// Example:
///
///     EXPECT_EQ("\"hello\\nworld\"", QuoteString("hello\nworld"));
std::string QuoteString(std::string_view s);

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_QUOTE_STRING_H_
