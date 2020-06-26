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

#ifndef TENSORSTORE_INTERNAL_JSON_H_
#define TENSORSTORE_INTERNAL_JSON_H_

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/meta/type_traits.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/function_view.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

/// Returns `true` if `a` and `b` are equal.
///
/// Unlike `operator==`, two `discarded` values are considered equal.
bool JsonSame(const ::nlohmann::json& a, const ::nlohmann::json& b);

/// GetTypeName returns the expected json field type name for error messages.
inline constexpr const char* GetTypeName(
    internal::type_identity<std::int64_t>) {
  return "64-bit signed integer";
}
inline constexpr const char* GetTypeName(
    internal::type_identity<std::uint64_t>) {
  return "64-bit unsigned integer";
}
inline constexpr const char* GetTypeName(internal::type_identity<double>) {
  return "64-bit floating-point number";
}
inline constexpr const char* GetTypeName(internal::type_identity<std::string>) {
  return "string";
}
inline constexpr const char* GetTypeName(internal::type_identity<bool>) {
  return "boolean";
}

/// Retuns an error message for a json value with the expected type.
Status ExpectedError(const ::nlohmann::json& j, absl::string_view type_name);

/// Returns an error message indicating that json field validation failed.
Status ValidationError(const ::nlohmann::json& j, absl::string_view type_name);

/// When passed an error status for parsing JSON, returns a status annotated
/// with the member name.
Status MaybeAnnotateMemberError(const Status& status,
                                absl::string_view member_name);

/// When passed an error status for converting to JSON, returns a status
/// annotated with the member name.
Status MaybeAnnotateMemberConvertError(const Status& status,
                                       absl::string_view member_name);

Status MaybeAnnotateArrayElementError(const Status& status, std::size_t i,
                                      bool is_loading);

inline ::nlohmann::json::object_t* GetObject(::nlohmann::json* j) {
  return j->template get_ptr<::nlohmann::json::object_t*>();
}

}  // namespace internal_json
namespace internal {

// ParseJson wraps the ::nlohmann::json::parse calls to avoid throwing
// exceptions.
::nlohmann::json ParseJson(absl::string_view str);

template <typename... Args>
std::enable_if_t<(sizeof...(Args) > 1), ::nlohmann::json> ParseJson(
    Args&&... args) {
  return ::nlohmann::json::parse({args...}, nullptr, false);
}

/// Return a json object as the C++ type, if conversion is possible.
/// Will attempt to parse strings and perform exact-numeric conversions
/// between the json data value and the c++ type.
///
/// For <bool>, accepted string formats are:  "true", "false"
/// For <double>, accepted string formats are those supported by std::strtod.
/// For <int64_t, uint64_t>, accepted string formats are base-10 numbers,
///   e.g. (-+)?[0-9]+, with the restriction that the value must fit into a
///   64-bit integer and uint64_t cannot include a negative prefix.
///
/// \param json The value to parse.
/// \param strict If `true`, string conversions and double -> integer
///     conversions are not permitted.
template <typename T>
absl::optional<T> JsonValueAs(const ::nlohmann::json& j, bool strict = false) {
  static_assert(!std::is_same<T, T>::value, "Target type not supported.");
}

template <>
absl::optional<bool> JsonValueAs<bool>(const ::nlohmann::json& j, bool strict);

template <>
absl::optional<int64_t> JsonValueAs<int64_t>(const ::nlohmann::json& j,
                                             bool strict);

template <>
absl::optional<uint64_t> JsonValueAs<uint64_t>(const ::nlohmann::json& j,
                                               bool strict);

template <>
absl::optional<double> JsonValueAs<double>(const ::nlohmann::json& j,
                                           bool strict);

template <>
absl::optional<std::string> JsonValueAs<std::string>(const ::nlohmann::json& j,
                                                     bool strict);

/// JsonRequireValueAs attempts to convert and assign the json object to the
/// result value.
///
/// \param j The JSON value to parse
/// \param result A pointer to an assigned value (may be nullptr)
/// \param is_valid A function or lambda taking T, returning true when
///     valid.
/// \param strict If `true`, string conversions and double -> integer
///     conversions are not permitted.
template <typename T, typename ValidateFn>
std::enable_if_t<!std::is_same<ValidateFn, bool>::value, Status>
JsonRequireValueAs(const ::nlohmann::json& j, T* result, ValidateFn is_valid,
                   bool strict = false) {
  auto value = internal::JsonValueAs<T>(j, strict);
  if (!value) {
    return internal_json::ExpectedError(
        j, internal_json::GetTypeName(type_identity<T>{}));
  }
  // NOTE: improve is_valid; allow bool or status returns.
  if (!is_valid(*value)) {
    return internal_json::ValidationError(
        j, internal_json::GetTypeName(type_identity<T>{}));
  }
  if (result != nullptr) {
    *result = std::move(*value);
  }
  return absl::OkStatus();
}

/// As above, omitting the validation function.
template <typename T>
Status JsonRequireValueAs(const ::nlohmann::json& j, T* result,
                          bool strict = false) {
  return JsonRequireValueAs(
      j, result, [](const T&) { return true; }, strict);
}
}  // namespace internal

namespace internal_json {
/// Implementation of `JsonRequireInteger` defined below.
///
/// Only defined for `T` equal to `std::int64_t` or `std::uint64_t`.
///
/// Defined as a class to simplify explicit instantiation.
template <typename T>
struct JsonRequireIntegerImpl {
  static Status Execute(const ::nlohmann::json& json, T* result, bool strict,
                        T min_value, T max_value);
};
}  // namespace internal_json

namespace internal {

/// Attempts to convert `json` to an integer in the range
/// `[min_value, max_value]`.
///
/// \tparam T A built-in signed or unsigned integer type.
/// \param json The JSON value.
/// \param result[out] Non-null pointer to location where result is stored on
///     success.
/// \param strict If `true`, conversions from string and double are not
///     permitted.
/// \param min_value The minimum allowed value (inclusive).
/// \param max_value The maximum allowed value (inclusive).
/// \returns `Status()` if successful.
/// \error `absl::StatusCode::kInvalidArgument` on failure.
template <typename T>
Status JsonRequireInteger(
    const ::nlohmann::json& json, T* result, bool strict = false,
    type_identity_t<T> min_value = std::numeric_limits<T>::min(),
    type_identity_t<T> max_value = std::numeric_limits<T>::max()) {
  static_assert(std::is_signed<T>::value || std::is_unsigned<T>::value,
                "T must be an integer type.");
  using U = absl::conditional_t<std::is_signed<T>::value, std::int64_t,
                                std::uint64_t>;
  U temp;
  auto status = internal_json::JsonRequireIntegerImpl<U>::Execute(
      json, &temp, strict, min_value, max_value);
  if (status.ok()) *result = temp;
  return status;
}

/// Parses a JSON array.
///
/// \param j The JSON value to parse.
/// \param size_callback Callback invoked with the array size before parsing any
///     elements.  Parsing stops if it returns an error.
/// \param element_callback Callback invoked for each array element after
///     `size_callback` has been invoked.  Parsing stops if it returns an error.
/// \returns `Status()` on success, or otherwise the first error returned by
///     `size_callback` or `element_callback`.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not an array.
Status JsonParseArray(
    const ::nlohmann::json& j,
    FunctionView<Status(std::ptrdiff_t size)> size_callback,
    FunctionView<Status(const ::nlohmann::json& value, std::ptrdiff_t index)>
        element_callback);

/// Validates that `parsed_size` matches `expected_size`.
///
/// If the sizes don't match, returns a `Status` with an informative error
/// message.
///
/// This function is particularly useful to call from a `size_callback` passed
/// to `JsonParseArray`.
///
/// \param parsed_size Parsed size of array.
/// \param expected_size Expected size of array.
/// \returns `Status()` if `parsed_size == expected_size`.
/// \error `absl::StatusCode::kInvalidArgument` if `parsed_size !=
///     expected_size`.
Status JsonValidateArrayLength(std::ptrdiff_t parsed_size,
                               std::ptrdiff_t expected_size);

/// Validates that `j` only contains the `allowed_members`.
///
/// If other member fields exist, returns an error Status,
/// otherwise returns an ok status.
Status JsonValidateObjectMembers(const ::nlohmann::json& j,
                                 span<const absl::string_view> allowed_members);

Status JsonValidateObjectMembers(const ::nlohmann::json::object_t& j,
                                 span<const absl::string_view> allowed_members);

inline Status JsonValidateObjectMembers(
    const ::nlohmann::json& j,
    std::initializer_list<absl::string_view> allowed_members) {
  return JsonValidateObjectMembers(
      j, span(std::begin(allowed_members), std::end(allowed_members)));
}

inline Status JsonValidateObjectMembers(
    const ::nlohmann::json::object_t& j,
    std::initializer_list<absl::string_view> allowed_members) {
  return JsonValidateObjectMembers(
      j, span(std::begin(allowed_members), std::end(allowed_members)));
}

/// Json object field type conversion helpers.
/// Invokes the `handler` function for the specified field object.
///
/// Example:
///
/// JsonRequireObjectMember(j, "member_name", [](const ::nlohmann::json& j) {
///    return JsonRequireValueAs(j, &field);
/// });
///
/// \param j The JSON object
/// \param member_name The name of the JSON field.
/// \param handler The handler function
/// \error 'absl::StatusCode::kInvalidArgument' if `j` is not an object.
/// \error 'absl::StatusCode::kInvalidArgument' if `member_name` is not found
///     (only `JsonRequireObjectMember`).
Status JsonRequireObjectMember(
    const ::nlohmann::json& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handler);

Status JsonRequireObjectMember(
    const ::nlohmann::json::object_t& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handler);

Status JsonHandleObjectMember(
    const ::nlohmann::json& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handler);

Status JsonHandleObjectMember(
    const ::nlohmann::json::object_t& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handler);

/// Removes the specified member from `*j_obj` if it is present.
///
/// \returns The extracted member if present, or
///     `::nlohmann::json::value_t::discarded` if not present.
::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   absl::string_view name);

/// Returns an error indicating that all members of `j_obj` are unexpected.
Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj);

/// Framework for bidirectional binding between C++ types and JSON values.
///
/// Example:
///
///     struct Foo {
///       int x;
///       std::string y;
///       std::optional<int> z;
///     };
///
///     namespace jb = tensorstore::internal::json_binding;
///
///     constexpr auto FooBinder() {
///       return jb::Object(
///           jb::Member("x", jb::Projection(&Foo::x)),
///           jb::Member("y",
///                      jb::Projection(&Foo::y, jb::DefaultValue([](auto *y) {
///                        *y = "default";
///                      }))),
///           jb::Member("z", jb::Projection(&Foo::z)));
///     }
///
///     EXPECT_EQ(::nlohmann::json({{"x", 3}}),
///               jb::ToJson(Foo{3, "default", std::nullopt}, FooBinder(),
///                          tensorstore::IncludeDefaults{false}));
///
///     auto value = jb::FromJson<Foo>({{"x", 3}, {"y", "value"}, {"z", 10}},
///                                    FooBinder()).value();
///     EXPECT_EQ(3, value.x);
///     EXPECT_EQ("value", value.y);
///     EXPECT_EQ(10, value.z);
///
/// `Binder` concept:
/// -----------------
///
/// A type satisfies the `Binder<T>` concept if it is a function object
/// compatible with the following two signatures:
///
///     Status (std::true_type is_loading, const LoadOptions& options,
///             T *obj, ::nlohmann::json* j)
///
///     Status (std::false_type is_loading, const SaveOptions& options,
///           const T *obj, ::nlohmann::json* j)
///
/// where `LoadOptions` and `SaveOptions` are arbitrary options classes.
///
/// The first signature is used for converting from a JSON value.  The function
/// must update `*obj` to reflect the converted value of `*j`, and may consume
/// `*j`.
///
/// The second signature is used for converting to a JSON value.  The function
/// must update `*j` to reflect the value of `*obj`.
///
/// In both cases the returned `Status` value indicates whether the conversion
/// was successful.
///
/// For simple cases, as in the example above, a suitable binder may be composed
/// from other binders using the various functions in this namespace.  For
/// custom behavior, a `Binder` is typically defined concisely as a polymorphic
/// lambda, where the constexpr value `is_loading` may be queried to distinguish
/// between the save and load paths if necessary:
///
///     auto binder = [](auto is_loading, const auto& options, auto *obj,
///                      ::nlohmann::json* j) -> Status {
///       if constexpr (is_loading) {
///         // Handle loading...
///       } else {
///         // Handle saving...
///       }
///     };
namespace json_binding {
// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace require_value_as_binder {
constexpr inline auto JsonRequireValueAsBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> Status {
  if constexpr (is_loading) {
    return JsonRequireValueAs(*j, obj, /*strict=*/true);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  // namespace require_value_as_binder
using require_value_as_binder::JsonRequireValueAsBinder;

template <>
constexpr inline auto DefaultBinder<bool> = JsonRequireValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::int64_t> = JsonRequireValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::string> = JsonRequireValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::uint64_t> = JsonRequireValueAsBinder;
template <>
constexpr inline auto DefaultBinder<double> = JsonRequireValueAsBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace float_binder {
constexpr inline auto FloatBinder = [](auto is_loading, const auto& options,
                                       auto* obj,
                                       ::nlohmann::json* j) -> Status {
  if constexpr (is_loading) {
    double x;
    auto status = JsonRequireValueAs(*j, &x, /*strict=*/true);
    if (status.ok()) *obj = x;
    return status;
  } else {
    *j = static_cast<double>(*obj);
    return absl::OkStatus();
  }
};
}  // namespace float_binder
using float_binder::FloatBinder;

template <typename T>
constexpr inline auto
    DefaultBinder<T, std::enable_if_t<std::is_floating_point_v<T>>> =
        FloatBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace copy_binder {
constexpr inline auto CopyJsonBinder = [](auto is_loading, const auto& options,
                                          auto* obj,
                                          ::nlohmann::json* j) -> Status {
  if constexpr (is_loading) {
    *obj = std::move(*j);
  } else {
    *j = *obj;
  }
  return absl::OkStatus();
};
}  // namespace copy_binder
using copy_binder::CopyJsonBinder;

template <>
constexpr inline auto DefaultBinder<::nlohmann::json> = CopyJsonBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace object_binder {
constexpr inline auto CopyJsonObjectBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
  if constexpr (is_loading) {
    if constexpr (std::is_same_v<decltype(j), ::nlohmann::json::object_t*>) {
      *obj = std::move(*j);
    } else {
      if (auto* j_obj = internal_json::GetObject(j)) {
        *obj = std::move(*j_obj);
      } else {
        return internal_json::ExpectedError(*j, "object");
      }
    }
  } else {
    *j = *obj;
  }
  return absl::OkStatus();
};
}  // namespace object_binder
using object_binder::CopyJsonObjectBinder;

template <>
constexpr inline auto DefaultBinder<::nlohmann::json::object_t> =
    CopyJsonObjectBinder;

/// Binder adapter that projects the parsed representation.
///
/// Commonly this is used with `Member`, in order to bind a data member of the
/// parsed representation to a JSON object member.
///
/// Example:
///
///     struct Foo {
///       int x;
///     };
///
///     const auto FooBinder = jb::Object(
///         jb::Member("x", jb::Projection(&Foo::x)));
///
/// The similar `GetterSetter` adapter handles the case of separate
/// getter/setter functions rather than a single reference-returning projection
/// function.
///
/// \param projection Invocable with signature `T& (Obj& obj)`, where `T` is the
///     projected value type, and `Obj` is the (possibly const-qualified) parsed
///     representation type.  This may be a pointer to a data member or nullary
///     method of `Obj`.
/// \param binder Binder to apply to the projected value obtained from
///     `projection`.
template <typename Proj, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Projection(Proj projection, Binder binder = DefaultBinder<>) {
  return
      [projection = std::move(projection), binder = std::move(binder)](
          auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        // Use `&&` rather than `&` in case the projection returns an object
        // with reference semantics (such as `span`) rather than an actual
        // reference.
        auto&& projected = std::invoke(projection, *obj);
        return binder(is_loading, options, &projected, j);
      };
}

/// Binder adapter that projects the parsed representation using gettter/setter
/// functions.
///
/// Commonly this is used with `Member`, in order to bind a
/// getter/setter-accessed member of the parsed representation to a JSON object
/// member.
///
/// Example:
///
///     struct Foo {
///       int x;
///       int get_x() const { return x; }
///       void set_x(int value) { this->x = value; }
///     };
///
///     const auto FooBinder = jb::Object(
///         jb::Member("x", jb::GetterSetter(&Foo::get_x,
///                                          &Foo::set_x)));
///
/// \tparam T The projected value type to use when parsing.  If not specified
///     (or `void`), the type is inferred from the return type of `get`.
/// \param get Invocable function that, when called with a reference to the
///     parent object, returns the corresponding projected value (possibly by
///     value).
/// \param set Invocable function that, when called with a reference to the
///     parent object and the new projected value, sets the corresponding
///     projected value.  May return either `void` (if infallible) or `Status`.
/// \param binder Optional.  Binder to apply to the projected value type.  If
///     not specified, the default binder for the projected value type is used.
template <typename T = void, typename Get, typename Set,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto GetterSetter(Get get, Set set, Binder binder = DefaultBinder<>) {
  return [get = std::move(get), set = std::move(set),
          binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> Status {
    if constexpr (is_loading) {
      using Projected =
          std::conditional_t<std::is_void_v<T>,
                             std::invoke_result_t<Get, decltype(*obj)>, T>;
      Projected projected;
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, &projected, j));
      return tensorstore::InvokeForStatus(set, *obj, std::move(projected));
    } else {
      auto&& projected = std::invoke(get, *obj);
      return binder(is_loading, options, &projected, j);
    }
  };
}

/// Composes multiple binders into a single binder.
///
/// This returns a binder that simply invokes each specified binder in forward
/// order when loading, and reverse order when saving.
template <typename... Binder>
constexpr auto Sequence(Binder... binder) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    Status status;
    if constexpr (sizeof...(Binder) != 0) {
      constexpr std::size_t N = sizeof...(binder);
      // Type-erase `binder` types in order to permit reverse iteration without
      // more complicated metaprogramming.
      using BinderInvoker =
          Status (*)(const void* binder, decltype(is_loading),
                     decltype(options), decltype(obj), decltype(j));
      const BinderInvoker binder_invokers[N] = {
          +[](const void* binder_ptr, decltype(is_loading) is_loading,
              decltype(options) options, decltype(obj) obj,
              decltype(j) j) -> Status {
            return (*static_cast<const decltype(binder)*>(binder_ptr))(
                is_loading, options, obj, j);
          }...};
      const void* const binder_ptrs[N] = {&binder...};
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t binder_i = is_loading ? i : N - 1 - i;
        status = binder_invokers[binder_i](binder_ptrs[binder_i], is_loading,
                                           options, obj, j);
        if (!status.ok()) break;
      }
    }
    return status;
  };
}

/// Returns a `Binder` for JSON objects.
///
/// When loading, verifies that the input JSON value is an object, calls each
/// `member_binder` in order, then verifies that there are no remaining members.
///
/// When saving, constructs an empty JSON object, then calls each
/// `member_binder` in reverse order with a pointer to the JSON object.
///
/// Example:
///
///     namespace jb = tensorstore::internal::json_binding;
///     auto binder = jb::Object(jb::Member("x", jb::Projection(&Foo::x)),
///                              jb::Member("y", jb::Projection(&Foo::y)));
///
/// \param member_binder An object binder, which is the same as a `Binder`
///     except that it is called with a second argument of
///     `::nlohmann::json::object_t*` instead of `::nlohmann::json*`.  Typically
///     these are obtained by calling `Member`.
template <typename... MemberBinder>
constexpr auto Object(MemberBinder... member_binder) {
  return
      [=](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        ::nlohmann::json::object_t* j_obj;
        if constexpr (is_loading) {
          if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
            j_obj = internal_json::GetObject(j);
            if (!j_obj) {
              return internal_json::ExpectedError(*j, "object");
            }
          } else {
            j_obj = j;
          }
        } else {
          if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
            *j = ::nlohmann::json::object_t();
            j_obj = internal_json::GetObject(j);
          } else {
            j_obj = j;
            j_obj->clear();
          }
        }
        TENSORSTORE_RETURN_IF_ERROR(json_binding::Sequence(member_binder...)(
            is_loading, options, obj, j_obj));
        if constexpr (is_loading) {
          if (!j_obj->empty()) {
            return JsonExtraMembersError(*j_obj);
          }
        }
        return absl::OkStatus();
      };
}

/// Returns an object binder (for use with `Object`) that saves/loads a specific
/// named member.
///
/// When loading, this removes the member from the JSON object if it is present.
/// If the member is not present, `binder` is called with a JSON value of type
/// `::nlohmann::json::value_t::discarded`.
///
/// When saving, the member is added unless `binder` generates a discarded JSON
/// value.
///
/// Typically `Member` is composed with `Projection`, in order to bind a JSON
/// object member to a C++ object data member:
///
///     jb::Member("name", jb::Projection(&Obj::name))
///
/// or equivalently:
///
///     jb::Projection(&Obj::name, jb::Member("name"))
///
/// \param name The member name, must be explicitly convertible to
///     `std::string`.  The name is captured by value in the returned binder
///     using the same `MemberName` type in which it is passed (not converted to
///     `std::string`); therefore, be careful when passing a `const char*` or
///     `absl::string_view` that the referenced string outlives the returned
///     binder.
/// \param binder Optional.  Binder to use for the member.  If not specified,
///     the default binder for object is used.
template <typename MemberName, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Member(MemberName name, Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j_obj) -> Status {
    if constexpr (is_loading) {
      ::nlohmann::json j_member = JsonExtractMember(j_obj, name);
      return internal_json::MaybeAnnotateMemberError(
          binder(is_loading, options, obj, &j_member), name);
    } else {
      ::nlohmann::json j_member(::nlohmann::json::value_t::discarded);
      TENSORSTORE_RETURN_IF_ERROR(
          binder(is_loading, options, obj, &j_member),
          internal_json::MaybeAnnotateMemberConvertError(_, name));
      if (!j_member.is_discarded()) {
        j_obj->emplace(name, std::move(j_member));
      }
      return absl::OkStatus();
    }
  };
}

/// Returns an object binder (for use with `Object`) that ignores any extra
/// members.
///
/// When loading, this removes all remaining members.
///
/// When saving, this does nothing.
///
/// Example:
///
///     namespace jb = tensorstore::internal::json_binding;
///     auto binder = jb::Object(jb::Member("x", &Foo::x),
///                              jb::Member("y", &Foo::y),
///                              jb::IgnoreExtraMembers);
constexpr auto IgnoreExtraMembers =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json::object_t* j_obj) -> Status {
  if constexpr (is_loading) {
    j_obj->clear();
  }
  return absl::OkStatus();
};

/// Returns a `Binder` for use with `Member` that performs default value
/// handling.
///
/// When loading, in case of a discarded JSON value (i.e. missing JSON object
/// member), the `get_default` function is called to obtain the converted value.
///
/// When saving, if `IncludeDefaults` is set to `false` and the resultant JSON
/// representation is equal to the JSON representation of the default value, the
/// JSON value is set to discarded.
///
/// Example:
///
///     namespace jb = tensorstore::internal::json_binding;
///     auto binder = jb::Object(
///         jb::Member("x", &Foo::x, DefaultValue([](auto *x) { *x = 10; },
///                                               jb::DefaultBinder<int>));
///
/// Example where the default value depends on other members of the object:
///
///     auto binder = [](auto is_loading, const auto& options, auto *obj,
///     ::nlohmann::json *j) {
///       return jb::Object(
///           jb::Member("x", &Foo::x),
///           jb::Member("y", &Foo::y,
///                      DefaultValue([obj](auto *y) { *y = obj->x; })))
///           (is_loading, options, obj, j);
///    };
///
/// \tparam DisallowIncludeDefaults If `true`, the `IncludeDefaults` option is
///     ignored.
/// \param get_default Function with signature `void (T *obj)` or
///     `Status (T *obj)` called with a pointer to the object.  Must assign the
///     default value to `*obj` and return `Status()` or `void`, or return an
///     error `Status`.
/// \param binder The `Binder` to use if the JSON value is not discarded.
template <bool DisallowIncludeDefaults = false, typename GetDefault,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultValue(GetDefault get_default,
                            Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(obj)>>;
    if constexpr (is_loading) {
      if (j->is_discarded()) {
        return tensorstore::InvokeForStatus(get_default, obj);
      }
      return binder(is_loading, options, obj, j);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
      if constexpr (!DisallowIncludeDefaults) {
        IncludeDefaults include_defaults(options);
        if (include_defaults.include_defaults()) {
          return absl::OkStatus();
        }
      }
      T default_obj;
      // If `get_default` fails, just use original value.
      ::nlohmann::json default_j;
      if (tensorstore::InvokeForStatus(get_default, &default_obj).ok() &&
          binder(is_loading, options, &default_obj, &default_j).ok() &&
          internal_json::JsonSame(default_j, *j)) {
        // Successfully obtained default value matches original value.
        *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
      }
      return absl::OkStatus();
    }
  };
}

/// Same as `DefaultValue` above, except that the default value is obtained via
/// value initialization rather than via a specified `get_default` function.
template <bool DisallowIncludeDefaults = false,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultInitializedValue(Binder binder = DefaultBinder<>) {
  return json_binding::DefaultValue<DisallowIncludeDefaults>(
      [](auto* obj) { *obj = remove_cvref_t<decltype(*obj)>{}; },
      std::move(binder));
}

/// Returns a `Binder` for `std::optional<T>`.
///
/// When loading, if the JSON value is equal to `nullopt_to_json()`, sets the
/// optional object to `std::nullopt`.  Otherwise, constructs the object and
/// calls `value_binder`.
///
/// When saving, if the optional object is equal to `std::nullopt`, sets the
/// JSON value to `nullopt_to_json()`.  Otherwise, calls `value_binder`.
///
/// \param value_binder `Binder` for `T` to use in the case the value is
///     present.
/// \param nullopt_to_json Function with signature `::nlohmann::json ()` that
///     returns the JSON representation for `std::nullopt`.
template <typename ValueBinder, typename NulloptToJson>
constexpr auto Optional(ValueBinder value_binder,
                        NulloptToJson nullopt_to_json) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    if constexpr (is_loading) {
      ::nlohmann::json nullopt_json = nullopt_to_json();
      if (internal_json::JsonSame(*j, nullopt_json)) {
        *obj = std::nullopt;
        return absl::OkStatus();
      } else {
        return value_binder(is_loading, options, &obj->emplace(), j);
      }
    } else {
      if (obj->has_value()) {
        return value_binder(is_loading, options, &**obj, j);
      } else {
        *j = nullopt_to_json();
        return absl::OkStatus();
      }
    }
  };
}

/// Same as above, but converts `std::optional` to
/// ::nlohmann::json::value_t::discarded` (for use with `Member`).
template <typename ValueBinder = decltype(DefaultBinder<>)>
constexpr auto Optional(ValueBinder value_binder = DefaultBinder<>) {
  return Optional(value_binder, [] {
    return ::nlohmann::json(::nlohmann::json::value_t::discarded);
  });
}

template <typename Value, typename JsonValue,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto MapValue(Value value, JsonValue json_value,
                        Binder binder = DefaultBinder<>) {
  return
      [=](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        if constexpr (is_loading) {
          if (internal_json::JsonSame(*j, json_value)) {
            *obj = value;
            return absl::OkStatus();
          }
        } else {
          if (*obj == value) {
            *j = json_value;
            return absl::OkStatus();
          }
        }
        return binder(is_loading, options, obj, j);
      };
}

template <typename EnumValue, typename JsonValue, std::size_t N>
constexpr auto Enum(const std::pair<EnumValue, JsonValue> (&values)[N]) {
  return
      [=](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        for (const auto& p : values) {
          if constexpr (is_loading) {
            if (internal_json::JsonSame(p.second, *j)) {
              *obj = p.first;
              return absl::OkStatus();
            }
          } else {
            if (p.first == *obj) {
              *j = p.second;
              return absl::OkStatus();
            }
          }
        }
        if constexpr (is_loading) {
          return internal_json::ExpectedError(
              *j, StrCat("one of",
                         absl::StrJoin(
                             values, ", ", [](std::string* out, const auto& p) {
                               *out += ::nlohmann::json(p.second).dump();
                             })));
        } else {
          TENSORSTORE_UNREACHABLE;
        }
      };
}

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace optional_binder {
constexpr inline auto OptionalBinder = [](auto is_loading, const auto& options,
                                          auto* obj, ::nlohmann::json* j) {
  return Optional()(is_loading, options, obj, j);
};
}  // namespace optional_binder
using optional_binder::OptionalBinder;

/// Registers `Optional` as the default binder for `std::optional`.
template <typename T>
inline constexpr auto& DefaultBinder<std::optional<T>> = OptionalBinder;

/// Returns a Binder for integers.
template <typename T>
constexpr auto Integer(T min = std::numeric_limits<T>::min(),
                       T max = std::numeric_limits<T>::max()) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    if constexpr (is_loading) {
      return JsonRequireInteger(*j, obj, /*strict=*/true, min, max);
    } else {
      *j = *obj;
      return absl::OkStatus();
    }
  };
}

template <typename T>
constexpr inline auto
    DefaultBinder<T, std::enable_if_t<std::is_integral_v<T>>> = Integer<T>();

template <typename T, typename TransformedValueBinder,
          typename OriginalValueBinder = decltype(DefaultBinder<>)>
constexpr auto Compose(
    TransformedValueBinder transformed_value_binder,
    OriginalValueBinder original_value_binder = DefaultBinder<>) {
  return
      [=](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        T value;
        if constexpr (is_loading) {
          TENSORSTORE_RETURN_IF_ERROR(
              original_value_binder(is_loading, options, &value, j));
          return transformed_value_binder(is_loading, options, obj, &value);
        } else {
          TENSORSTORE_RETURN_IF_ERROR(
              transformed_value_binder(is_loading, options, obj, &value));
          return original_value_binder(is_loading, options, &value, j);
        }
      };
}

template <typename GetBinder>
constexpr auto Dependent(GetBinder get_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto*... j) -> Status {
    return get_binder(is_loading, options, obj, j...)(is_loading, options, obj,
                                                      j...);
  };
}

template <typename Validator, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Validate(Validator validator, Binder binder = DefaultBinder<>) {
  return
      [=](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
        if constexpr (is_loading) {
          TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
          return validator(options, obj);
        } else {
          return binder(is_loading, options, obj, j);
        }
      };
}

template <typename Initializer>
constexpr auto Initialize(Initializer initializer) {
  return [=](auto is_loading, const auto& options, [[maybe_unused]] auto* obj,
             auto* j) -> Status {
    if constexpr (is_loading) {
      return tensorstore::InvokeForStatus(initializer, obj);
    } else {
      return absl::OkStatus();
    }
  };
}

/// Matches a constant JSON value (doesn't parse anything).
///
/// When loading from JSON, this validates that the JSON value matches the
/// expected value.  When saving to JSON, stores the expected JSON value.
///
/// The parsed object may be of any type and is ignored.
///
/// Example usage:
///
///     jb::Constant([] { return "expected"; })
///
/// \param get_value Nullary function that returns the expected
///     `::nlohmann::json` value.
template <typename GetValue>
constexpr auto Constant(GetValue get_value) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> Status {
    if constexpr (is_loading) {
      const auto& value = get_value();
      if (!internal_json::JsonSame(*j, value)) {
        return internal_json::ExpectedError(*j, ::nlohmann::json(value).dump());
      }
    } else {
      *j = get_value();
    }
    return absl::OkStatus();
  };
}

/// Bind a JSON array to a homogeneous array-like type `Container` with type
/// `T`.
///
/// \param get_size Function with signature `std::size_t (const Container&)`
///     that returns the size of the container.
/// \param set_size Function with signature
///     `absl::Status (Container&, std::size_t size)` that resizes the container
///     to the specified size, or returns an error.
/// \param get_element Function with overloaded signatures
///     `T& (Container&, std::size_t i)` and
///     `const T& (const Container&, std::size_t i)` that returns a reference to
///     the `i`th element of the container.
/// \param element_binder JSON binder for `T` to use for each element of the
///     array.
template <typename GetSize, typename SetSize, typename GetElement,
          typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto Array(GetSize get_size, SetSize set_size, GetElement get_element,
                     ElementBinder element_binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    ::nlohmann::json::array_t* j_arr;
    if constexpr (is_loading) {
      j_arr = j->get_ptr<::nlohmann::json::array_t*>();
      if (!j_arr) {
        return internal_json::ExpectedError(*j, "array");
      }
      const size_t size = j_arr->size();
      TENSORSTORE_RETURN_IF_ERROR(
          tensorstore::InvokeForStatus(set_size, *obj, size));
    } else {
      *j = ::nlohmann::json::array_t();
      j_arr = j->get_ptr<::nlohmann::json::array_t*>();
      j_arr->resize(get_size(*obj));
    }
    for (size_t i = 0, size = j_arr->size(); i < size; ++i) {
      auto&& element = get_element(*obj, i);
      TENSORSTORE_RETURN_IF_ERROR(
          element_binder(is_loading, options, &element, &(*j_arr)[i]),
          internal_json::MaybeAnnotateArrayElementError(_, i, is_loading));
    }
    return absl::OkStatus();
  };
}

/// Binds a JSON array to a container-like type (e.g. `std::vector`) that
/// supports `size`, `resize`, and `operator[]`.
template <typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto Array(ElementBinder element_binder = DefaultBinder<>) {
  return json_binding::Array(
      [](auto& c) { return c.size(); },
      [](auto& c, std::size_t size) { c.resize(size); },
      [](auto& c, std::size_t i) -> decltype(auto) { return c[i]; },
      element_binder);
}

/// Binds a JSON array to a fixed-size array-like type (e.g. `std::array`) that
/// supports `std::size` and `operator[]`.
template <typename ElementBinder = decltype(DefaultBinder<>)>
constexpr auto FixedSizeArray(ElementBinder element_binder = DefaultBinder<>) {
  return json_binding::Array(
      [](auto& c) { return std::size(c); },
      [](auto& c, std::size_t new_size) {
        return internal::JsonValidateArrayLength(new_size, std::size(c));
      },
      [](auto& c, std::size_t i) -> decltype(auto) { return c[i]; },
      element_binder);
}

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace array_binder {
inline constexpr auto ArrayBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) -> Status {
  return json_binding::Array()(is_loading, options, obj, j);
};
}  // namespace array_binder

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace fixed_size_array_binder {
inline constexpr auto FixedSizeArrayBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
  return json_binding::FixedSizeArray()(is_loading, options, obj, j);
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

/// Type-erased `Binder` type.
template <typename T, typename LoadOptions, typename SaveOptions,
          typename JsonValue = ::nlohmann::json, typename... ExtraValue>
using AnyBinder = Poly<0, /*Copyable=*/true,  //
                       Status(std::true_type, const LoadOptions&, T* obj,
                              JsonValue* j, ExtraValue*...) const,
                       Status(std::false_type, const SaveOptions&, const T* obj,
                              JsonValue* j, ExtraValue*...) const>;

}  // namespace json_binding
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_H_
