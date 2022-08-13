// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STD_VARIANT_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STD_VARIANT_H_

#include <variant>

#include "absl/status/status.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_json_binding {

absl::Status GetVariantErrorStatus(span<const absl::Status> status_values);

template <size_t... Is, typename Options, typename Obj, typename Json,
          typename... ValueBinder>
absl::Status VariantBinderImpl(std::index_sequence<Is...>,
                               std::true_type is_loading,
                               const Options& options, Obj* obj, Json* json,
                               ValueBinder&&... value_binder) {
  absl::Status status_values[sizeof...(ValueBinder)];
  if (((status_values[Is] = value_binder(is_loading, options,
                                         &obj->template emplace<Is>(), json))
           .ok() ||
       ...)) {
    return absl::OkStatus();
  }
  return GetVariantErrorStatus(status_values);
}

template <size_t... Is, typename Options, typename Obj, typename Json,
          typename... ValueBinder>
absl::Status VariantBinderImpl(std::index_sequence<Is...>,
                               std::false_type is_loading,
                               const Options& options, Obj* obj, Json* json,
                               ValueBinder&&... value_binder) {
  absl::Status status;
  size_t index = obj->index();
  if (((index == Is &&
        (status = value_binder(is_loading, options, &std::get<Is>(*obj), json))
            .ok()) ||
       ...)) {
    return absl::OkStatus();
  }
  return status;
}

template <size_t... Is, typename Options, typename Obj, typename Json>
absl::Status VariantDefaultBinderImpl(std::index_sequence<Is...>,
                                      std::true_type is_loading,
                                      const Options& options, Obj* obj,
                                      Json* json) {
  absl::Status status_values[std::variant_size_v<Obj>];
  if (((status_values[Is] = DefaultBinder<>(is_loading, options,
                                            &obj->template emplace<Is>(), json))
           .ok() ||
       ...)) {
    return absl::OkStatus();
  }
  return GetVariantErrorStatus(status_values);
}

template <size_t... Is, typename Options, typename Obj, typename Json>
absl::Status VariantDefaultBinderImpl(std::index_sequence<Is...>,
                                      std::false_type is_loading,
                                      const Options& options, Obj* obj,
                                      Json* json) {
  absl::Status status;
  size_t index = obj->index();
  if (((index == Is && (status = DefaultBinder<>(is_loading, options,
                                                 &std::get<Is>(*obj), json))
                           .ok()) ||
       ...)) {
    return absl::OkStatus();
  }
  return status;
}

/// Returns a `Binder` for `std::variant`.
///
/// The size of the variant must match `sizeof...(value_binder)`.
///
/// When loading, the first successful `value_binder` is used.
///
/// When saving, the corresponding `value_binder` is used.
///
/// Note: Because loading requires sequentially trying each binder, it is not
/// particularly efficient.  In cases where greater efficiency is needed, a
/// custom binder should be implemented.
///
/// \param value_binder `Binder` for each type in the variant.
template <typename... ValueBinder>
constexpr auto Variant(ValueBinder... value_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    static_assert(
        sizeof...(value_binder) ==
            std::variant_size_v<internal::remove_cvref_t<decltype(*obj)>>,
        "value_binder pack must have the same size as the variant");
    return VariantBinderImpl(
        std::make_index_sequence<sizeof...(value_binder)>{}, is_loading,
        options, obj, j, value_binder...);
  };
}

/// Same as above, but uses `DefaultBinder<>` for all types.
constexpr auto Variant() {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    return VariantDefaultBinderImpl(
        std::make_index_sequence<
            std::variant_size_v<internal::remove_cvref_t<decltype(*obj)>>>{},
        is_loading, options, obj, j);
  };
}

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace variant_binder {
constexpr inline auto VariantBinder = [](auto is_loading, const auto& options,
                                         auto* obj, ::nlohmann::json* j) {
  return Variant()(is_loading, options, obj, j);
};
}  // namespace variant_binder
using variant_binder::VariantBinder;

/// Registers `VariantBinder` as the default binder for `std::variant`.
template <typename... T>
inline constexpr auto& DefaultBinder<std::variant<T...>> = VariantBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_STD_VARIANT_H_
