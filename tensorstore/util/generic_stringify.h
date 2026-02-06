// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_UTIL_GENERIC_STRINGIFY_H_
#define TENSORSTORE_UTIL_GENERIC_STRINGIFY_H_

#include <stdint.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/has_absl_stringify.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/meta/requires.h"
#include "tensorstore/internal/meta/type_traits.h"

namespace tensorstore {
namespace internal_stringify {

template <typename T>
constexpr bool generic_stringify_false = false;

template <typename Sink, typename T>
void GenericStringifyImpl(Sink& sink, const T& v);

template <typename Sink, typename ResultLike>
void GenericStringifyResult(Sink& sink, const ResultLike& v) {
  sink.Append("<");
  if (v.ok()) {
    sink.Append("OK: ");
    GenericStringifyImpl(sink, *v);
  } else {
    GenericStringifyImpl(sink, v.status());
  }
  sink.Append(">");
}

template <typename Sink, class... Ts>
void GenericStringifyTuple(Sink& sink, const std::tuple<Ts...>& tuple) {
  absl::string_view sep = "";
  const auto print_one = [&](const auto& v) {
    sink.Append(sep);
    GenericStringifyImpl(sink, v);
    sep = ", ";
  };
  sink.Append("{");
  std::apply([&](const auto&... v) { (print_one(v), ...); }, tuple);
  sink.Append("}");
}

template <typename Sink, typename T, typename U>
void GenericStringifyPair(Sink& sink, const std::pair<T, U>& p) {
  sink.Append("{");
  GenericStringifyImpl(sink, p.first);
  sink.Append(", ");
  GenericStringifyImpl(sink, p.second);
  sink.Append("}");
}

template <typename Sink, typename T>
void GenericStringifyOptional(Sink& sink, const std::optional<T>& v) {
  if (v.has_value()) {
    sink.Append("<");
    GenericStringifyImpl(sink, *v);
    sink.Append(">");
  } else {
    GenericStringifyImpl(sink, std::nullopt);
  }
}

template <typename Sink, typename Iterator>
void GenericStringifyContainer(Sink& sink, Iterator begin, Iterator end) {
  sink.Append("{");
  if (begin != end) {
    GenericStringifyImpl(sink, *begin++);
  }
  for (; begin != end; ++begin) {
    sink.Append(", ");
    GenericStringifyImpl(sink, *begin);
  }
  sink.Append("}");
}

// Converts arbitrary input values to a type supported by `absl::StrCat`.
template <typename Sink, typename T>
void GenericStringifyImpl(Sink& sink, const T& v) {
  if constexpr (absl::HasAbslStringify<T>::value) {
    AbslStringify(sink, v);
  } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
    sink.Append("null");
  } else if constexpr (std::is_same_v<T, std::nullopt_t>) {
    sink.Append("null");
  } else if constexpr (std::is_same_v<T, std::string_view> ||
                       std::is_same_v<T, std::string> ||
                       std::is_same_v<T, char*> ||
                       std::is_same_v<T, const char*>) {
    sink.Append(std::string_view(v));
  } else if constexpr (std::is_same_v<T, char> ||
                       std::is_same_v<T, signed char> ||
                       std::is_same_v<T, unsigned char>) {
    absl::Format(&sink, "%c", v);
  } else if constexpr (std::is_same_v<T, bool>) {
    sink.Append(v ? "true" : "false");
  } else if constexpr (std::is_enum_v<T>) {
    // Non-streamable enum
    using I = typename std::underlying_type<T>::type;
    absl::Format(&sink, "%d", static_cast<I>(v));
  } else if constexpr (std::is_floating_point_v<T>) {
    absl::Format(&sink, "%f", v);
  } else if constexpr (std::is_integral_v<T>) {
    absl::Format(&sink, "%d", v);
  } else if constexpr (
      internal_meta::Requires<const T>(
          [&](auto&& w)
              -> decltype((
                  GenericStringifyOptional)(std::declval<absl::FormatSink&>(),
                                            w)) {})) {
    GenericStringifyOptional(sink, v);
  } else if constexpr (
      internal_meta::Requires<const T>(
          [&](auto&& w)
              -> decltype((
                  GenericStringifyTuple)(std::declval<absl::FormatSink&>(),
                                         w)) {})) {
    // For tuples, use `{ elem0, ..., elemN }`.
    GenericStringifyTuple(sink, v);
  } else if constexpr (
      internal_meta::Requires<const T>(
          [&](auto&& w)
              -> decltype((
                  GenericStringifyPair)(std::declval<absl::FormatSink&>(), w)) {
          })) {
    // For pairs, use `{ first, second }`.
    GenericStringifyPair(sink, v);
  } else if constexpr (
      internal_meta::Requires<const T>(
          [&](auto&& w)
              -> decltype((
                  GenericStringifyContainer)(std::declval<absl::FormatSink&>(),
                                             w.cbegin(), w.cend())) {})) {
    // For containers, use `{ elem0, ..., elemN }`.
    GenericStringifyContainer(sink, v.cbegin(), v.cend());
  } else if constexpr (internal_meta::Requires<const T>(
                           [&](auto&& w) -> decltype(w.ok(), w.status(), *w) {
                           })) {
    GenericStringifyResult(sink, v);
  } else if constexpr (internal::IsOstreamable<T>) {
    absl::Format(&sink, "%s", absl::FormatStreamed(v));
  } else {
    // Workaround for http://wg21.link/p2593.
    static_assert(generic_stringify_false<T>, "Type is not stringifiable");
  }
}

// Wrapper for T which implements AbslStringify.
template <typename T>
struct GenericStringify {
  T v;

  explicit GenericStringify(T v) : v(v) {}

  template <typename Sink>
  friend void AbslStringify(Sink& sink, GenericStringify self) {
    GenericStringifyImpl(sink, self.v);
  }
};

struct GenericStringifyNiebloid {
  template <typename T>
  GenericStringify<const T&> operator()(const T& v) const {
    static_assert(!absl::HasAbslStringify<T>::value,
                  "Type already supports AbslStringify");
    return GenericStringify<const T&>{v};
  }
};

}  // namespace internal_stringify

/// Allows passing arbitrary C++ value references to `absl::StrFormat`,
/// `absl::StrCat`, and similar string formatting functions.
///
/// Example::
///
///   std::vector<int> v = {1, 2, 3};
///   absl::StrFormat("%sv", GenericStringify(v));
///
/// \ingroup string-utilities
inline constexpr internal_stringify::GenericStringifyNiebloid
    GenericStringify{};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_GENERIC_STRINGIFY_H_
