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

/// \file
///
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
///     namespace jb = tensorstore::internal_json_binding;
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
///     absl::Status (std::true_type is_loading, const LoadOptions& options,
///             T *obj, ::nlohmann::json* j)
///
///     absl::Status (std::false_type is_loading, const SaveOptions& options,
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
/// In both cases the returned `absl::Status` value indicates whether the
/// conversion was successful.
///
/// For simple cases, as in the example above, a suitable binder may be composed
/// from other binders using the various functions in this namespace.  For
/// custom behavior, a `Binder` is typically defined concisely as a polymorphic
/// lambda, where the constexpr value `is_loading` may be queried to distinguish
/// between the save and load paths if necessary:
///
///     auto binder = [](auto is_loading, const auto& options, auto *obj,
///                      ::nlohmann::json* j) -> absl::Status {
///       if constexpr (is_loading) {
///         // Handle loading...
///       } else {
///         // Handle saving...
///       }
///     };

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_JSON_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_JSON_H_

#include <functional>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

/// Binder that always returns absl::OkStatus.
namespace empty_binder {
constexpr inline auto EmptyBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) -> absl::Status {
  return absl::OkStatus();
};
}
using empty_binder::EmptyBinder;

/// Binders for natively supported types. These binders come in
/// loose and strict flavors. The Loose... binders are permissive and
/// will accept json types that can be parsed as the target type
/// (e.g. "3" as int), where the default binders are less permissive.
///
/// The strict binders are installed as DefaultBinder<> specializations.
///
/// Example:
///     namespace jb = tensorstore::internal_json_binding;
///
///     ::nlohmann::json j("3")
///     int x;
///     LooseValueAsBinder(is_loading, options, &x, j)
///     ValueAsBinder(is_loading, options, &x, j)
///
namespace loose_value_as_binder {
// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
constexpr inline auto LooseValueAsBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(*j, obj, /*strict=*/false);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  // namespace loose_value_as_binder
using loose_value_as_binder::LooseValueAsBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace value_as_binder {
constexpr inline auto ValueAsBinder = [](auto is_loading, const auto& options,
                                         auto* obj,
                                         ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(*j, obj, /*strict=*/true);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  // namespace value_as_binder
using value_as_binder::ValueAsBinder;

template <>
constexpr inline auto DefaultBinder<bool> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::int64_t> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::string> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::uint64_t> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<double> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::nullptr_t> = ValueAsBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace loose_float_binder {
constexpr inline auto LooseFloatBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    double x;
    auto status = internal_json::JsonRequireValueAs(*j, &x, /*strict=*/false);
    if (status.ok()) *obj = x;
    return status;
  } else {
    *j = static_cast<double>(*obj);
    return absl::OkStatus();
  }
};
}  // namespace loose_float_binder
using loose_float_binder::LooseFloatBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace float_binder {
constexpr inline auto FloatBinder = [](auto is_loading, const auto& options,
                                       auto* obj,
                                       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    double x;
    auto status = internal_json::JsonRequireValueAs(*j, &x, /*strict=*/true);
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

/// Returns a Binder for integers.
template <typename T>
constexpr auto LooseInteger(T min = std::numeric_limits<T>::min(),
                            T max = std::numeric_limits<T>::max()) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      return internal_json::JsonRequireInteger(*j, obj, /*strict=*/false, min,
                                               max);
    } else {
      *j = *obj;
      return absl::OkStatus();
    }
  };
}

/// Returns a Binder for integers.
template <typename T>
constexpr auto Integer(T min = std::numeric_limits<T>::min(),
                       T max = std::numeric_limits<T>::max()) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      return internal_json::JsonRequireInteger(*j, obj, /*strict=*/true, min,
                                               max);
    } else {
      *j = *obj;
      return absl::OkStatus();
    }
  };
}

template <typename T>
constexpr inline auto
    DefaultBinder<T, std::enable_if_t<std::is_integral_v<T>>> = Integer<T>();

// Binder that requires
//
//

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace non_empty_string_binder {
constexpr inline auto NonEmptyStringBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(
        *j, obj, [](const std::string& value) { return !value.empty(); },
        /*strict=*/true);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  // namespace non_empty_string_binder
using non_empty_string_binder::NonEmptyStringBinder;

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace copy_binder {
constexpr inline auto CopyJsonBinder = [](auto is_loading, const auto& options,
                                          auto* obj,
                                          ::nlohmann::json* j) -> absl::Status {
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
constexpr inline auto CopyJsonObjectBinder = [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) -> absl::Status {
  if constexpr (is_loading) {
    if constexpr (std::is_same_v<decltype(j), ::nlohmann::json::object_t*>) {
      *obj = std::move(*j);
    } else {
      if (auto* j_obj = j->template get_ptr<::nlohmann::json::object_t*>()) {
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
             auto* j) -> absl::Status {
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

/// When loading, invokes the provided lambda the current object after the
/// subsequent binder has been invoked.
///
/// This is useful for checking single-value constraints.
///
/// Example:
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(
///                       jb::Member("x",
///                           jb::Projection(&Foo::x,
///                             jb::Validate([](const auto &options,
///                                             auto *obj) {
///                               assert(is_prime(*obj));
///                             }))));
///
template <typename Validator, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Validate(Validator validator, Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
      return internal::InvokeForStatus(validator, options, obj);
    } else {
      return binder(is_loading, options, obj, j);
    }
  };
}

/// When loading, invokes the provided lambda with the current object.
/// This is useful for checking constraints that depend on multiple members.
///
/// Example:
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(
///                       jb::Member("x", jb::Projection(&Foo::x)),
///                       jb::Member("y", jb::Projection(&Foo::y)),
///                       jb::Initialize([](Foo* f) { assert(f->x > f->y);
///                       }));
///
template <typename Initializer>
constexpr auto Initialize(Initializer initializer) {
  return [=](auto is_loading, const auto& options, [[maybe_unused]] auto* obj,
             auto*) -> absl::Status {
    if constexpr (is_loading) {
      return internal::InvokeForStatus(initializer, obj);
    } else {
      return absl::OkStatus();
    }
  };
}

/// Binder adapter that projects the parsed representation to a function pointer
/// or member pointer, specified as a template parameter.
///
/// This is commonly used with `Member`, in order to bind a data member of the
/// parsed representation to a JSON object member.
///
/// Example:
///
///     struct Foo {
///       int x;
///     };
///
///     const auto FooBinder = jb::Object(
///         jb::Member("x", jb::Projection<&Foo::x>()));
///
/// \tparam Proj Invocable with signature `T& (Obj& obj)`, where `T` is the
///     projected value type, and `Obj` is the (possibly const-qualified) parsed
///     representation type.
/// \param binder Binder to apply to the projected value obtained from `Proj`.
template <auto Proj, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Projection(Binder binder = DefaultBinder<>) {
  return [binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    // Use `&&` rather than `&` in case the projection returns an object
    // with reference semantics (such as `span`) rather than an actual
    // reference.
    auto&& projected = std::invoke(Proj, *obj);
    return binder(is_loading, options, &projected, j);
  };
}

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
  return [projection = std::move(projection), binder = std::move(binder)](
             auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
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
///     projected value.  May return either `void` (if infallible) or
///     `absl::Status`.
/// \param binder Optional.  Binder to apply to the projected value type.  If
///     not specified, the default binder for the projected value type is used.
template <typename T = void, typename Get, typename Set,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto GetterSetter(Get get, Set set, Binder binder = DefaultBinder<>) {
  return [get = std::move(get), set = std::move(set),
          binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    if constexpr (is_loading) {
      using Projected = std::conditional_t<
          std::is_void_v<T>,
          internal::remove_cvref_t<std::invoke_result_t<Get, decltype(*obj)>>,
          T>;
      Projected projected;
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, &projected, j));
      return internal::InvokeForStatus(set, *obj, std::move(projected));
    } else {
      auto&& projected = std::invoke(get, *obj);
      return binder(is_loading, options, &projected, j);
    }
  };
}

// Binder parameterized by distinct load and save objects.
// Invokes LoadBinder when loading and SaveBinder when saving.
template <typename LoadBinder = decltype(EmptyBinder),
          typename SaveBinder = decltype(EmptyBinder)>
constexpr auto LoadSave(LoadBinder load_binder = EmptyBinder,
                        SaveBinder save_binder = EmptyBinder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      return load_binder(is_loading, options, obj, j);
    } else {
      return save_binder(is_loading, options, obj, j);
    }
  };
}

/// Policy for `DefaultValue` and related binders that determines when default
/// values are included in the JSON.
enum IncludeDefaultsPolicy {
  /// Include defaults according to the `IncludeDefaults` option.
  kMaybeIncludeDefaults,
  /// Never include defaults (ignores `IncludeDefaults` option).
  kNeverIncludeDefaults,
  /// Always include defaults (ignores `IncludeDefaults` option).
  kAlwaysIncludeDefaults,
};

/// Returns a `Binder` for use with `Member` that performs default value
/// handling.
///
/// When loading, in case of a discarded JSON value (i.e. missing JSON object
/// member), the `get_default` function is called to obtain the converted
/// value.
///
/// When saving, according to the `Policy` and the `IncludeDefaults` option, if
/// the resultant JSON representation is equal to the JSON representation of the
/// default value, the JSON value is set to discarded.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
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
/// \tparam Policy Specifies the conditions under which default values are
///     included in the JSON.
/// \param get_default Function with signature `void (T *obj)` or
///     `absl::Status (T *obj)` called with a pointer to the object.  Must
///     assign the default value to `*obj` and return `absl::Status()` or
///     `void`, or return an error `absl::Status`.
/// \param binder The `Binder` to use if the JSON value is not discarded.
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename GetDefault, typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultValue(GetDefault get_default,
                            Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(obj)>>;
    if constexpr (is_loading) {
      if (j->is_discarded()) {
        return internal::InvokeForStatus(get_default, obj);
      }
      return binder(is_loading, options, obj, j);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
      if constexpr (Policy == kAlwaysIncludeDefaults) {
        return absl::OkStatus();
      }
      if constexpr (Policy == kMaybeIncludeDefaults) {
        IncludeDefaults include_defaults(options);
        if (include_defaults.include_defaults()) {
          return absl::OkStatus();
        }
      }
      T default_obj;
      // If `get_default` fails, just use original value.
      ::nlohmann::json default_j;
      if (internal::InvokeForStatus(get_default, &default_obj).ok() &&
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
template <IncludeDefaultsPolicy DefaultsPolicy = kMaybeIncludeDefaults,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultInitializedValue(Binder binder = DefaultBinder<>) {
  return internal_json_binding::DefaultValue<DefaultsPolicy>(
      [](auto* obj) { *obj = internal::remove_cvref_t<decltype(*obj)>{}; },
      std::move(binder));
}

/// Returns a `Binder` for use with `Member` that performs default value
/// handling based on a predicate.
///
/// When loading, in case of a discarded JSON value (i.e. missing JSON object
/// member), the `get_default` function is called to obtain the converted value.
///
/// When saving, according to the `Policy`, the `IncludeDefaults` option, and
/// the result of the `is_default` predicate, the JSON value is set to
/// discarded.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(
///         jb::Member("x", &Foo::x, DefaultPredicate(
///                                     [](auto *x) { *x = 10; },
///                                     [](auto *x) { return *x == 10; },
///                                     jb::DefaultBinder<int>));
///
/// \tparam Policy Specifies the conditions under which default values are
///     included in the JSON.
/// \param get_default Function with signature `void (T* obj)` or
///     `absl::Status (T *obj)` called with a pointer to the object.  Must
///     assign the default value to `*obj` and return `absl::Status()` or
///     `void`, or return an error `absl::Status`.
/// \param is_default Function with signature `bool (const T* obj)` called with
///     a pointer to the object to determine whether the object is equal to the
///     default value.
/// \param binder The `Binder` to use if the JSON value is not discarded.
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename GetDefault, typename IsDefault,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultPredicate(GetDefault get_default, IsDefault is_default,
                                Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      if (j->is_discarded()) {
        return internal::InvokeForStatus(get_default, obj);
      }
      return binder(is_loading, options, obj, j);
    } else {
      bool include_defaults_value = Policy == kAlwaysIncludeDefaults;
      if constexpr (Policy == kMaybeIncludeDefaults) {
        IncludeDefaults include_defaults(options);
        include_defaults_value = include_defaults.include_defaults();
      }
      if (!include_defaults_value && is_default(obj)) {
        *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
        return absl::OkStatus();
      }
      return binder(is_loading, options, obj, j);
    }
  };
}

/// Same as `DefaultPredicate` above, except that the default value is obtained
/// via value initialization rather than via a specified `get_default` function.
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename IsDefault, typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultInitializedPredicate(IsDefault is_default,
                                           Binder binder = DefaultBinder<>) {
  return internal_json_binding::DefaultPredicate<Policy>(
      [](auto* obj) { *obj = internal::remove_cvref_t<decltype(*obj)>{}; },
      std::move(is_default), std::move(binder));
}

template <typename T, typename TransformedValueBinder,
          typename OriginalValueBinder = decltype(DefaultBinder<>)>
constexpr auto Compose(
    TransformedValueBinder transformed_value_binder,
    OriginalValueBinder original_value_binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
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
             auto*... j) -> absl::Status {
    return get_binder(is_loading, options, obj, j...)(is_loading, options, obj,
                                                      j...);
  };
}

/// Implementation details for jb::Sequence.
namespace sequence_impl {

template <typename Loading, typename Options, typename Obj, typename J,
          typename... Binder>
inline absl::Status invoke_reverse(Loading is_loading, Options& options,
                                   Obj* obj, J* j, Binder... binder) {
  // Use operator=, which folds from right-to-left to invoke in reverse order.
  absl::Status s;
  std::true_type right_to_left;
  right_to_left =
      (((s.ok() ? (void)(s = binder(is_loading, options, obj, j)) : (void)0),
        right_to_left) = ... = right_to_left);
  return s;
}

template <typename Loading, typename Options, typename Obj, typename J,
          typename... Binder>
inline absl::Status invoke_forward(Loading is_loading, Options& options,
                                   Obj* obj, J* j, Binder... binder) {
  // Use operator&& which folds from left-to-right to invoke in forward order.
  absl::Status s;
  [[maybe_unused]] bool ok =
      (((s = binder(is_loading, options, obj, j)).ok()) && ...);
  return s;
}

}  // namespace sequence_impl

/// Composes multiple binders into a single binder.
///
/// This returns a binder that simply invokes each specified binder in forward
/// order when loading, and reverse order when saving.
///
/// Example:
///   jb::Sequence(jb::Initialize([](auto* obj) { obj->x = 1; },
///                jb::Projection(&T::y));
///
template <typename... Binder>
constexpr auto Sequence(Binder... binder) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    if constexpr (is_loading) {
      return sequence_impl::invoke_forward(is_loading, options, obj, j,
                                           binder...);
    } else {
      /// Like the forward fold expression, however it grabs the reverse
      /// binder from the tuple using the reverse index sequence.
      return sequence_impl::invoke_reverse(is_loading, options, obj, j,
                                           binder...);
    }
  };
}

/// Returns a `Binder` for JSON objects.
///
/// When loading, verifies that the input JSON value is an object, calls each
/// `member_binder` in order, then verifies that there are no remaining
/// members.
///
/// When saving, constructs an empty JSON object, then calls each
/// `member_binder` in reverse order with a pointer to the JSON object.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(jb::Member("x", jb::Projection(&Foo::x)),
///                              jb::Member("y", jb::Projection(&Foo::y)));
///
/// \param member_binder An object binder, which is the same as a `Binder`
///     except that it is called with a second argument of
///     `::nlohmann::json::object_t*` instead of `::nlohmann::json*`.
///     Typically these are obtained by calling `Member`.
template <typename... MemberBinder>
constexpr auto Object(MemberBinder... member_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    ::nlohmann::json::object_t* j_obj;
    if constexpr (is_loading) {
      if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
        j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
        if (!j_obj) {
          return internal_json::ExpectedError(*j, "object");
        }
      } else {
        j_obj = j;
      }
      TENSORSTORE_RETURN_IF_ERROR(sequence_impl::invoke_forward(
          is_loading, options, obj, j_obj, member_binder...));
      // If any members remain in j_obj after this, error.
      if (!j_obj->empty()) {
        return internal_json::JsonExtraMembersError(*j_obj);
      }
      return absl::OkStatus();
    } else {
      if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
        *j = ::nlohmann::json::object_t();
        j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
      } else {
        j_obj = j;
        j_obj->clear();
      }
      return sequence_impl::invoke_reverse(is_loading, options, obj, j_obj,
                                           member_binder...);
    }
  };
}

/// Implementation details for jb::Member and jb::OptionalMember.
template <bool kDropDiscarded, typename MemberName, typename Binder>
struct MemberBinderImpl {
  MemberName name;
  Binder binder;
  template <typename Options, typename Obj>
  absl::Status operator()(std::true_type is_loading, const Options& options,
                          Obj* obj, ::nlohmann::json::object_t* j_obj) const {
    ::nlohmann::json j_member = internal_json::JsonExtractMember(j_obj, name);
    if constexpr (kDropDiscarded) {
      if (j_member.is_discarded()) return absl::OkStatus();
    }
    auto status = binder(is_loading, options, obj, &j_member);
    return status.ok()
               ? status
               : MaybeAnnotateStatus(
                     status, tensorstore::StrCat("Error parsing object member ",
                                                 QuoteString(name)));
  }
  template <typename Options, typename Obj>
  absl::Status operator()(std::false_type is_loading, const Options& options,
                          Obj* obj, ::nlohmann::json::object_t* j_obj) const {
    ::nlohmann::json j_member(::nlohmann::json::value_t::discarded);
    TENSORSTORE_RETURN_IF_ERROR(
        binder(is_loading, options, obj, &j_member),

        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error converting object member ",
                                   QuoteString(name))));
    if (!j_member.is_discarded()) {
      j_obj->emplace(name, std::move(j_member));
    }
    return absl::OkStatus();
  }
};

/// Returns an object binder (for use with `Object`) that saves/loads a
/// specific named member.
///
/// When loading, this removes the member from the JSON object if it is
/// present. If the member is not present, `binder` is called with a JSON
/// value of type
/// `::nlohmann::json::value_t::discarded`.
///
/// When saving, the member is added unless `binder` generates a discarded
/// JSON value.
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
///     using the same `MemberName` type in which it is passed (not converted
///     to `std::string`); therefore, be careful when passing a `const char*`
///     or `std::string_view` that the referenced string outlives the returned
///     binder.
/// \param binder Optional.  Binder to use for the member.  If not specified,
///     the default binder for object is used.
template <typename MemberName, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Member(MemberName name, Binder binder = DefaultBinder<>) {
  return MemberBinderImpl<false, MemberName, Binder>{std::move(name),
                                                     std::move(binder)};
}

/// As above, however when loading, if the member is not present, then `binder`
/// is not called. This is used, for example, when loading an optional json
/// member into a c++ type which is not wrapped in std::optional<>.
///
///     jb::OptionalMember("name", jb::Projection(&Obj::name))
///
/// The above is similar to this:
///     jb::Member("name", jb::Projection(&Obj::name,
///     DefaultInitializedValue()))
///
/// However it allows chaining dependent logic which is only invoked when
/// the member actually exists:
///
///     jb::OptionalMember("name", jb::Sequence(
///                                jb::Initialize([](auto* obj) {...},
///                                jb::Projection(&Obj::name)))
///
template <typename MemberName, typename Binder = decltype(DefaultBinder<>)>
constexpr auto OptionalMember(MemberName name,
                              Binder binder = DefaultBinder<>) {
  return MemberBinderImpl<true, MemberName, Binder>{std::move(name),
                                                    std::move(binder)};
}

/// Used in conjunction with jb::Object, returns an error when more than one of
/// the indicated names is present in the object when loading.
///
/// When saving, this does nothing.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(jb::AtMostOne("a", "b", "c"));
///
template <typename... MemberName>
constexpr auto AtMostOne(MemberName... names) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    if constexpr (is_loading) {
      const auto has_member = [&](auto name) {
        return j->find(name) == j->end() ? 0 : 1;
      };
      if ((has_member(names) + ...) > 1) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "At most one of ",
            absl::StrJoin({QuoteString(std::string_view(names))...}, ", "),
            " members is allowed"));
      }
    }
    return absl::OkStatus();
  };
}

/// Used in conjunction with jb::Object, returns an error when fewer than one of
/// the indicated names is present in the object when loading.
///
/// When saving, this does nothing.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(jb::AtLeastOne("a", "b", "c"));
///
template <typename... MemberName>
constexpr auto AtLeastOne(MemberName... names) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    if constexpr (is_loading) {
      const auto has_member = [&](auto name) {
        return j->find(name) == j->end() ? 0 : 1;
      };
      if ((has_member(names) + ...) == 0) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "At least one of ",
            absl::StrJoin(
                std::make_tuple(QuoteString(std::string_view(names))...), ", "),
            " members must be specified"));
      }
    }
    return absl::OkStatus();
  };
}

/// Returns an object binder (for use with `Object`) that clears any extra
/// members.
///
/// When loading, this removes all remaining members.
///
/// When saving, this does nothing.
///
/// Example:
///
///     namespace jb = tensorstore::internal_json_binding;
///     auto binder = jb::Object(jb::Member("x", &Foo::x),
///                              jb::Member("y", &Foo::y),
///                              jb::DiscardExtraMembers);
namespace discard_extra_members_binder {
// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
constexpr inline auto DiscardExtraMembers =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json::object_t* j_obj) -> absl::Status {
  if constexpr (is_loading) {
    j_obj->clear();
  }
  return absl::OkStatus();
};
}  // namespace discard_extra_members_binder
using discard_extra_members_binder::DiscardExtraMembers;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_JSON_H_
