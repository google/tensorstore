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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_RATIONAL_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_RATIONAL_H_

#include <stddef.h>

#include <string>
#include <string_view>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/rational.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_json_binding {

namespace rational_binder {
struct RationalBinder {
  template <typename Options, typename T>
  absl::Status operator()(std::true_type is_loading, const Options& options,
                          Rational<T>* obj, ::nlohmann::json* j) const {
    if (j->is_array()) {
      T values[2];
      span<T, 2> values_span(values);
      TENSORSTORE_RETURN_IF_ERROR(
          FixedSizeArray()(is_loading, options, &values_span, j));
      *obj = Rational<T>(values[0], values[1]);
      return absl::OkStatus();
    } else if (auto* s = j->get_ptr<const std::string*>()) {
      std::string_view sv = *s;
      size_t slash_index = sv.find('/');
      T numerator;
      T denominator;
      if (slash_index == std::string_view::npos) {
        denominator = 1;
        if (!absl::SimpleAtoi(sv, &numerator)) {
          return internal_json::ExpectedError(
              *j, "number or rational number `a/b`");
        }
      } else {
        if (!absl::SimpleAtoi(sv.substr(0, slash_index), &numerator) ||
            !absl::SimpleAtoi(sv.substr(slash_index + 1), &denominator)) {
          return internal_json::ExpectedError(*j, "rational number `a/b`");
        }
      }
      *obj = Rational<T>(numerator, denominator);
      return absl::OkStatus();
    }
    T value;
    TENSORSTORE_RETURN_IF_ERROR(
        DefaultBinder<>(is_loading, options, &value, j));
    *obj = value;
    return absl::OkStatus();
  }

  template <typename Options, typename T>
  absl::Status operator()(std::false_type is_loading, const Options& options,
                          const Rational<T>* obj, ::nlohmann::json* j) const {
    if (obj->denominator() == static_cast<T>(1)) {
      T num = obj->numerator();
      return DefaultBinder<>(is_loading, options, &num, j);
    }
    *j = absl::StrFormat("%d/%d", obj->numerator(), obj->denominator());
    return absl::OkStatus();
  }
};

}  // namespace rational_binder

template <typename T>
constexpr inline auto DefaultBinder<Rational<T>> =
    rational_binder::RationalBinder{};

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_RATIONAL_H_
