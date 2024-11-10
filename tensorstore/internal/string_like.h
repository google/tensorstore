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

#ifndef TENSORSTORE_INTERNAL_STRING_LIKE_H_
#define TENSORSTORE_INTERNAL_STRING_LIKE_H_

#include <stddef.h>

#include <cassert>
#include <string>
#include <string_view>

#include "absl/base/optimization.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Bool-valued metafunction equal to `true` iff `T` is `std::string`,
/// `std::string_view`, or `const char *`.
template <typename T>
constexpr inline bool IsStringLike = false;

template <>
constexpr inline bool IsStringLike<std::string_view> = true;

template <>
constexpr inline bool IsStringLike<std::string> = true;

template <>
constexpr inline bool IsStringLike<const char*> = true;

/// Holds a span of `std::string`, `std::string_view`, or `const char *`.
class StringLikeSpan {
 public:
  StringLikeSpan() = default;

  StringLikeSpan(tensorstore::span<const char* const> c_strings)
      : c_strings_(c_strings.data()), size_and_tag_(c_strings.size() << 2) {}

  StringLikeSpan(tensorstore::span<const std::string> strings)
      : strings_(strings.data()), size_and_tag_((strings.size() << 2) | 1) {}

  StringLikeSpan(tensorstore::span<const std::string_view> string_views)
      : string_views_(string_views.data()),
        size_and_tag_((string_views.size() << 2) | 2) {}

  std::string_view operator[](ptrdiff_t i) const {
    assert(i >= 0 && i < size());
    switch (size_and_tag_ & 3) {
      case 0:
        return c_strings_[i];
      case 1:
        return strings_[i];
      case 2:
        return string_views_[i];
      default:
        ABSL_UNREACHABLE();  // COV_NF_LINE
    }
  }

  ptrdiff_t size() const { return size_and_tag_ >> 2; }

 private:
  union {
    const char* const* c_strings_;
    const std::string* strings_;
    const std::string_view* string_views_;
  };
  ptrdiff_t size_and_tag_ = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_STRING_LIKE_H_
