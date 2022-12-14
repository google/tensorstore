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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STD_ARRAY_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STD_ARRAY_H_

#include <stddef.h>

#include <array>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

/// Implementation details for jb::Array and similar.
template <bool kDiscardEmpty, typename GetSize, typename SetSize,
          typename GetElement, typename ElementBinder>
struct ArrayBinderImpl {
  GetSize get_size;
  SetSize set_size;
  GetElement get_element;
  ElementBinder element_binder;

  template <typename Loading, typename Options, typename Obj>
  absl::Status operator()(Loading is_loading, const Options& options, Obj* obj,
                          ::nlohmann::json* j) const {
    ::nlohmann::json::array_t* j_arr;
    if constexpr (is_loading) {
      if constexpr (kDiscardEmpty) {
        if (j->is_discarded()) return absl::OkStatus();
      }
      j_arr = j->get_ptr<::nlohmann::json::array_t*>();
      if (!j_arr) {
        return internal_json::ExpectedError(*j, "array");
      }
      const size_t size = j_arr->size();
      TENSORSTORE_RETURN_IF_ERROR(
          internal::InvokeForStatus(set_size, *obj, size));
    } else {
      const auto size = get_size(*obj);
      if constexpr (kDiscardEmpty) {
        if (size == 0) {
          *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
          return absl::OkStatus();
        }
      }
      *j = ::nlohmann::json::array_t(size);
      j_arr = j->get_ptr<::nlohmann::json::array_t*>();
    }
    for (size_t i = 0, size = j_arr->size(); i < size; ++i) {
      auto&& element = get_element(*obj, i);
      TENSORSTORE_RETURN_IF_ERROR(
          element_binder(is_loading, options, &element, &(*j_arr)[i]),
          MaybeAnnotateStatus(
              _, tensorstore::StrCat("Error ",
                                     is_loading ? "parsing" : "converting",
                                     " value at position ", i)));
    }
    return absl::OkStatus();
  }
};

/// Bind a JSON array to a homogeneous array-like type `Container` with type
/// `T`.
///
/// \param get_size Function with signature `std::size_t (const Container&)`
///     that returns the size of the container.
/// \param set_size Function with signature
///     `absl::Status (Container&, std::size_t size)` that resizes the
///     container to the specified size, or returns an error.
/// \param get_element Function with overloaded signatures
///     `T& (Container&, std::size_t i)` and
///     `const T& (const Container&, std::size_t i)` that returns a reference
///     to the `i`th element of the container.
/// \param element_binder JSON binder for `T` to use for each element of the
///     array.
template <typename GetSize, typename SetSize, typename GetElement,
          typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto Array(GetSize get_size, SetSize set_size, GetElement get_element,
                     ElementBinder element_binder = DefaultBinder<>) {
  return ArrayBinderImpl<false, GetSize, SetSize, GetElement, ElementBinder>{
      std::move(get_size), std::move(set_size), std::move(get_element),
      std::move(element_binder)};
}

/// Binds a JSON array like jb::Array, but when the array is size is zero,
/// then the value is discarded.
template <typename GetSize, typename SetSize, typename GetElement,
          typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto OptionalArray(GetSize get_size, SetSize set_size,
                             GetElement get_element,
                             ElementBinder element_binder = DefaultBinder<>) {
  return ArrayBinderImpl<true, GetSize, SetSize, GetElement, ElementBinder>{
      std::move(get_size), std::move(set_size), std::move(get_element),
      std::move(element_binder)};
}

/// Binds a JSON array to a container-like type (e.g. `std::vector`) that
/// supports `size`, `resize`, and `operator[]`.
template <typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto Array(ElementBinder element_binder = DefaultBinder<>) {
  return internal_json_binding::Array(
      [](auto& c) { return c.size(); },
      [](auto& c, std::size_t size) { c.resize(size); },
      [](auto& c, std::size_t i) -> decltype(auto) { return c[i]; },
      element_binder);
}

/// Binds a JSON array like jb::Array, but when the array is size is zero,
/// then the value is discarded.
template <typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto OptionalArray(ElementBinder element_binder = DefaultBinder<>) {
  return internal_json_binding::OptionalArray(
      [](auto& c) { return c.size(); },
      [](auto& c, std::size_t size) { c.resize(size); },
      [](auto& c, std::size_t i) -> decltype(auto) { return c[i]; },
      element_binder);
}

/// Binds a JSON array to a fixed-size array-like type (e.g. `std::array`)
/// that supports `std::size` and `operator[]`.
template <typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto FixedSizeArray(ElementBinder element_binder = DefaultBinder<>) {
  return internal_json_binding::Array(
      [](auto& c) { return std::size(c); },
      [](auto& c, std::size_t new_size) {
        return internal_json::JsonValidateArrayLength(new_size, std::size(c));
      },
      [](auto& c, std::size_t i) -> decltype(auto) { return c[i]; },
      element_binder);
}

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace array_binder {
inline constexpr auto ArrayBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) -> absl::Status {
  return internal_json_binding::Array()(is_loading, options, obj, j);
};
}  // namespace array_binder

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace fixed_size_array_binder {
inline constexpr auto FixedSizeArrayBinder = [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) -> absl::Status {
  return internal_json_binding::FixedSizeArray()(is_loading, options, obj, j);
};
}  // namespace fixed_size_array_binder
using array_binder::ArrayBinder;
using fixed_size_array_binder::FixedSizeArrayBinder;

template <typename T, typename Allocator>
constexpr inline auto DefaultBinder<std::vector<T, Allocator>> = ArrayBinder;

template <typename T, std::size_t N>
constexpr inline auto DefaultBinder<std::array<T, N>> = FixedSizeArrayBinder;

/// Use `FixedSizeArrayBinder` as default binder for `tensorstore::span`.
///
/// Note that the user is responsible for ensuring that the `span` refers to
/// valid memory both when loading and saving.
template <typename T, std::ptrdiff_t Extent>
constexpr inline auto DefaultBinder<tensorstore::span<T, Extent>> =
    FixedSizeArrayBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_STD_ARRAY_H_
