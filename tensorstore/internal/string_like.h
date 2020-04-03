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

#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Bool-valued metafunction equal to `true` iff `T` is `std::string`,
/// `absl::string_view`, or `const char *`.
template <typename T>
struct IsStringLike : public std::false_type {};

template <>
struct IsStringLike<absl::string_view> : public std::true_type {};

template <>
struct IsStringLike<std::string> : public std::true_type {};

template <>
struct IsStringLike<const char*> : public std::true_type {};

/// Holds a span of `std::string`, `absl::string_view`, or `const char *`.
class StringLikeSpan {
 public:
  StringLikeSpan() = default;

  StringLikeSpan(span<const char* const> c_strings)
      : c_strings_(c_strings.data()), size_and_tag_(c_strings.size() << 2) {}

  StringLikeSpan(span<const std::string> strings)
      : strings_(strings.data()), size_and_tag_((strings.size() << 2) | 1) {}

  StringLikeSpan(span<const absl::string_view> string_views)
      : string_views_(string_views.data()),
        size_and_tag_((string_views.size() << 2) | 2) {}

  absl::string_view operator[](std::ptrdiff_t i) const {
    ABSL_ASSERT(i >= 0 && i < size());
    switch (size_and_tag_ & 3) {
      case 0:
        return c_strings_[i];
      case 1:
        return strings_[i];
      case 2:
        return string_views_[i];
      default:
        assert(false);
        TENSORSTORE_UNREACHABLE;
    }
  }

  std::ptrdiff_t size() const { return size_and_tag_ >> 2; }

 private:
  union {
    const char* const* c_strings_;
    const std::string* strings_;
    const absl::string_view* string_views_;
  };
  std::ptrdiff_t size_and_tag_ = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_STRING_LIKE_H_
