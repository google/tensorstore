// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_TSCLI_LIB_GLOB_TO_REGEX_H_
#define TENSORSTORE_TSCLI_LIB_GLOB_TO_REGEX_H_

#include <string>
#include <string_view>

namespace tensorstore {
namespace cli {

// Convert a glob pattern to a RE2 regular expression.
// The supported glob syntax is:
//  / to separate path segments
//  * to match zero or more characters in a path segment
//  ? to match on one character in a path segment
//  ** to match any number of path segments, including none
//  [] to declare a range of characters to match
//  [!...] to negate a range of characters to match
std::string GlobToRegex(std::string_view glob);

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_GLOB_TO_REGEX_H_
