// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_AWS_STRING_VIEW_H_
#define TENSORSTORE_INTERNAL_AWS_STRING_VIEW_H_

#include <stdint.h>

#include <string_view>

#include <aws/common/byte_buf.h>
#include <aws/common/string.h>

namespace tensorstore {
namespace internal_aws {

/// Converts aws_byte_cursor to std::string_view
inline std::string_view AwsByteCursorToStringView(const aws_byte_cursor &c) {
  return std::string_view(reinterpret_cast<const char *>(c.ptr), c.len);
}
inline std::string_view AwsByteCursorToStringView(const aws_byte_cursor *c) {
  return c ? std::string_view(reinterpret_cast<const char *>(c->ptr), c->len)
           : std::string_view();
}

/// Converts aws_string to std::string_view
inline std::string_view AwsStringToStringView(const aws_string &s) {
  return std::string_view(reinterpret_cast<const char *>(s.bytes), s.len);
}

/// Converts std::string_view to aws_byte_cursor
inline aws_byte_cursor StringViewToAwsByteCursor(std::string_view s) {
  aws_byte_cursor c;
  c.len = s.size();
  c.ptr = reinterpret_cast<uint8_t *>(const_cast<char *>(s.data()));
  return c;
}

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_AWS_STRING_VIEW_H_
