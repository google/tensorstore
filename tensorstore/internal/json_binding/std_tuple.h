// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STD_TUPLE_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STD_TUPLE_H_

#include <stddef.h>

#include <array>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

inline absl::Status MaybeAnnotateTupleElementError(absl::Status status,
                                                   std::size_t i,
                                                   bool is_loading) {
  return status.ok()
             ? status
             : MaybeAnnotateStatus(
                   status, tensorstore::StrCat(
                               "Error ", is_loading ? "parsing" : "converting",
                               " value at position ", i));
}

template <bool IsLoading>
Result<::nlohmann::json::array_t*> EnsureJsonTupleRepresentationImpl(
    std::integral_constant<bool, IsLoading> is_loading, ::nlohmann::json* j,
    size_t n) {
  if constexpr (is_loading) {
    auto* array_ptr = j->get_ptr<::nlohmann::json::array_t*>();
    if (!array_ptr) return internal_json::ExpectedError(*j, "array");
    TENSORSTORE_RETURN_IF_ERROR(
        internal_json::JsonValidateArrayLength(array_ptr->size(), n));
    return array_ptr;
  } else {
    *j = ::nlohmann::json::array_t(n);
    return j->get_ptr<::nlohmann::json::array_t*>();
  }
}

template <size_t... Is, typename... ElementBinder>
constexpr auto TupleJsonBinderImpl(std::index_sequence<Is...>,
                                   ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    if (absl::Status status;
        (((status = element_binder(is_loading, options, &std::get<Is>(*obj),
                                   &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}

template <size_t... Is>
constexpr auto TupleDefaultJsonBinderImpl(std::index_sequence<Is...>) {
  return [](auto is_loading, const auto& options, auto* obj,
            ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    using std::get;
    if (absl::Status status;
        (((status = DefaultBinder<>(is_loading, options, &get<Is>(*obj),
                                    &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}

template <size_t... Is, typename... ElementBinder>
constexpr auto HeterogeneousArrayJsonBinderImpl(
    std::index_sequence<Is...>, ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    if (absl::Status status;
        (((status = element_binder(is_loading, options, obj, &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}

/// JSON binder for converting between JSON arrays and types that support the
/// `get<I>` tuple interface, such as `std::pair` and `std::tuple`.
///
/// This overload requires that the binders for each element are specified
/// explicitly.  To use `DefaultBinder<>` for all elements, use the
/// zero-argument overload below.
///
/// \param element_binder The binders for each element of the tuple.  The size
///     of the tuple (and length of the JSON array) must match
///     `sizeof...(ElementBinder)`.
template <typename... ElementBinder>
constexpr auto Tuple(ElementBinder... element_binder) {
  return TupleJsonBinderImpl(std::index_sequence_for<ElementBinder...>{},
                             std::move(element_binder)...);
}

/// Same as above, but can be uses `DefaultBinder<>` for all elements.
constexpr auto Tuple() {
  return [](auto is_loading, const auto& options, auto* obj, auto* j) {
    constexpr size_t N =
        std::tuple_size_v<internal::remove_cvref_t<decltype(*obj)>>;
    return TupleDefaultJsonBinderImpl(std::make_index_sequence<N>{})(
        is_loading, options, obj, j);
  };
}

/// JSON binder for converting between JSON arrays, possibly containing
/// heterogeneous element types, and a non-array C++ representation.
///
/// Note that each element binder is invoked with the corresponding member of
/// the JSON array but the entire C++ object.  Therefore, each element binder
/// should typically make use of `Projection` or `GetterSetter` or similar.
///
/// For example:
///
///     struct Foo {
///       std::string a;
///       double b;
///     };
///
///     const auto binder = jb::HeterogeneousArray(jb::Projection<&Foo::a>(),
///                                                jb::Projection<&Foo::b>());
///
/// \param element_binder The binders for each element of the JSON array.
template <typename... ElementBinder>
constexpr auto HeterogeneousArray(ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) {
    TENSORSTORE_ASSIGN_OR_RETURN(::nlohmann::json::array_t * array_ptr,
                                 EnsureJsonTupleRepresentationImpl(
                                     is_loading, j, sizeof...(ElementBinder)));
    absl::Status status;
    size_t i = 0;
    [[maybe_unused]] bool ok =
        (((status =
               element_binder(is_loading, options, obj, &(*array_ptr)[i++]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, i - 1, is_loading)),
           false)) &&
         ...);
    return status;
  };
}

template <typename... T>
constexpr inline auto DefaultBinder<std::tuple<T...>> = Tuple();

template <typename T, typename U>
constexpr inline auto DefaultBinder<std::pair<T, U>> = Tuple();

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_STD_TUPLE_H_
