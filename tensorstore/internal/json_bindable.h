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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDABLE_H_
#define TENSORSTORE_INTERNAL_JSON_BINDABLE_H_

#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {
namespace json_binding {

struct NoOptions {
  constexpr NoOptions() = default;
  template <typename T>
  constexpr NoOptions(const T&) {}
};

/// Specifies the default `Binder` for a given unqualified object type.
///
/// To define the default binder for a given type `T`, this variable template
/// may be specialized.  Alternatively, for a class type `T`, a static data
/// member `default_json_binder`, typically with a `StaticBinder` type, may be
/// defined within `T`.
///
/// The special `DefaultBinder<>` (i.e. `DefaultBinder<void>`) value forwards to
/// `DefaultBinder<T>` when called with an object of type `T`.
template <typename T = void, typename SFINAE = void>
constexpr inline auto DefaultBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
  return T::default_json_binder(is_loading, options, obj, j);
};

/// Function object that forwards to the default binder implementation.
template <>
constexpr inline auto DefaultBinder<void> = [](auto is_loading,
                                               const auto& options, auto* obj,
                                               ::nlohmann::json* j) -> Status {
  using T = std::remove_cv_t<std::remove_pointer_t<decltype(obj)>>;
  return DefaultBinder<T>(is_loading, options, obj, j);
};

/// Converts an object to JSON using the specified binder.
template <typename T, typename Binder = decltype(DefaultBinder<>),
          typename Options = IncludeDefaults>
Result<::nlohmann::json> ToJson(const T& obj, Binder binder = DefaultBinder<>,
                                const Options& options = IncludeDefaults{
                                    true}) {
  ::nlohmann::json value(::nlohmann::json::value_t::discarded);
  TENSORSTORE_RETURN_IF_ERROR(binder(std::false_type{}, options, &obj, &value));
  return value;
}

/// Converts an object from its JSON representation using the specified binder.
template <typename T, typename Binder = decltype(DefaultBinder<>),
          typename Options = NoOptions>
Result<T> FromJson(::nlohmann::json j, Binder binder = DefaultBinder<>,
                   const Options& options = NoOptions{}) {
  T obj;
  if (auto status = binder(std::true_type{}, options, &obj, &j); !status.ok()) {
    return status;
  }
  return obj;
}

/// Type-erasure stateless JSON binder wrapper.
///
/// This is like `AnyBinder`, except that it merely stores a load and save
/// function pointer and is constexpr compatible.
template <typename T, typename FromJsonOptionsType = NoOptions,
          typename ToJsonOptionsType = IncludeDefaults,
          typename JsonValueType = ::nlohmann::json, typename... ExtraValue>
class StaticBinder {
 public:
  using Value = T;
  using FromJsonOptions = FromJsonOptionsType;
  using ToJsonOptions = ToJsonOptionsType;
  using JsonValue = JsonValueType;

  template <typename Binder>
  explicit StaticBinder(Binder binder) : load_(binder), save_(binder) {}

  Status operator()(std::true_type is_loading, const FromJsonOptions& options,
                    T* obj, JsonValue* j, ExtraValue*... extra) const {
    return load_(is_loading, options, obj, j, extra...);
  }

  Status operator()(std::false_type is_loading, const ToJsonOptions& options,
                    const T* obj, JsonValue* j, ExtraValue*... extra) const {
    return save_(is_loading, options, obj, j, extra...);
  }

 private:
  using LoadFunction = Status (*)(std::true_type,
                                  const FromJsonOptions& options, T* obj,
                                  JsonValue* j, ExtraValue*... extra);
  using SaveFunction = Status (*)(std::false_type, const ToJsonOptions& options,
                                  const T* obj, JsonValue* j,
                                  ExtraValue*... extra);

  LoadFunction load_;
  SaveFunction save_;
};

/// Declares a JSON binder in a header that may be defined in a separate source
/// file.
///
/// Example (at namespace scope):
///
///     TENSORSTORE_DECLARE_JSON_BINDER(foo, X, FromJsonOptions,
///                                            ToJsonOptions);
///
/// Defines a JSON binder for `X` named `foo`.  The object `foo` is a stateless
/// constexpr function object.
///
/// In a separate source file, this binder is defined as follows (at namespace
/// scope):
///
///     TENSORSTORE_DEFINE_JSON_BINDER(foo, jb::Object(...));
///
/// To intrusively make a class support JSON binding via
/// `json_binding::DefaultBinder<>`, use
/// `TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER` instead of this macro.
///
#define TENSORSTORE_DECLARE_JSON_BINDER(name, ...)                     \
  TENSORSTORE_INTERNAL_DECLARE_JSON_BINDER_IMPL(name##_JsonBinderImpl, \
                                                __VA_ARGS__)           \
  inline constexpr name##_JsonBinderImpl name = {};                    \
  /**/

#define TENSORSTORE_DEFINE_JSON_BINDER(name, ...)                     \
  TENSORSTORE_INTERNAL_DEFINE_JSON_BINDER_IMPL(name##_JsonBinderImpl, \
                                               __VA_ARGS__)
/**/

/// Declares that a class `X` supports JSON binding via
/// `json_binding::DefaultBinder<>` but requires the binder to be defined in a
/// separate source file.
///
/// Specifically, this macro declares a `default_json_binder` member and also
/// defines the following convenience methods, just like the `JsonBindable`
/// template:
///
///     Result<::nlohmann::json> ToJson(const ToJsonOptions = ToJsonOptions{});
///     static Result<X> FromJson(::nlohmann::json j,
///                               const FromJsonOptions = FromJsonOptions{});
///     explicit operator ::nlohmann::json() const;
///
/// Example usage:
///
///     class X {
///      public:
///       TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(X, XFromJsonOptions,
///                                                XToJsonOptions)
///       // ...
///     };
///
/// In the implementation source file, define the JSON binder for `X` as follows
/// (at namespace scope):
///
///     namespace jb = tensorstore::internal::json_binding;
///     TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(X, jb::Object(...));
///
#define TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(name, ...)            \
  TENSORSTORE_INTERNAL_DECLARE_JSON_BINDER_IMPL(JsonBinderImpl, name, \
                                                __VA_ARGS__)          \
  static inline constexpr JsonBinderImpl default_json_binder = {};    \
  tensorstore::Result<::nlohmann::json> ToJson(                       \
      const JsonBinderImpl::JsonBinderToJsonOptions& options =        \
          JsonBinderImpl::JsonBinderToJsonOptions{}) const {          \
    return tensorstore::internal::json_binding::ToJson(               \
        *this, default_json_binder, options);                         \
  }                                                                   \
  static tensorstore::Result<name> FromJson(                          \
      ::nlohmann::json j,                                             \
      const JsonBinderImpl::JsonBinderFromJsonOptions& options =      \
          JsonBinderImpl::JsonBinderFromJsonOptions{}) {              \
    return tensorstore::internal::json_binding::FromJson<name>(       \
        std::move(j), default_json_binder, options);                  \
  }                                                                   \
  explicit operator ::nlohmann::json() const {                        \
    return this->ToJson().value();                                    \
  }                                                                   \
  /**/

#define TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(name, ...)            \
  TENSORSTORE_INTERNAL_DEFINE_JSON_BINDER_IMPL(name::JsonBinderImpl, \
                                               __VA_ARGS__)          \
  /**/

// Note: We use the uglier aliases `JsonBinderFromJsonOptions` and
// `JsonBinderToJsonOptions` rather than the shorter `FromJsonOptions` and
// `ToJsonOptions` in case `__VA_ARGS__` references `FromJsonOptions` or
// `ToJsonOptions` from the outer scope, as GCC gives an error about changing
// the meaning of a name within a scope.
#define TENSORSTORE_INTERNAL_DECLARE_JSON_BINDER_IMPL(name, ...)              \
  class name {                                                                \
    using StaticBinderType =                                                  \
        ::tensorstore::internal::json_binding::StaticBinder<__VA_ARGS__>;     \
    using Value = typename StaticBinderType::Value;                           \
    using JsonValue = typename StaticBinderType::JsonValue;                   \
                                                                              \
   public:                                                                    \
    using JsonBinderFromJsonOptions =                                         \
        typename StaticBinderType::FromJsonOptions;                           \
    using JsonBinderToJsonOptions = typename StaticBinderType::ToJsonOptions; \
    absl::Status operator()(std::true_type is_loading,                        \
                            const JsonBinderFromJsonOptions& options,         \
                            Value* value, JsonValue* j) const {               \
      return this->Do(is_loading, options, value, j);                         \
    }                                                                         \
    absl::Status operator()(std::false_type is_loading,                       \
                            const JsonBinderToJsonOptions& options,           \
                            const Value* value, JsonValue* j) const {         \
      return this->Do(is_loading, options, value, j);                         \
    }                                                                         \
                                                                              \
   private:                                                                   \
    static absl::Status Do(std::true_type is_loading,                         \
                           const JsonBinderFromJsonOptions& options,          \
                           Value* value, JsonValue* j);                       \
    static absl::Status Do(std::false_type is_loading,                        \
                           const JsonBinderToJsonOptions& options,            \
                           const Value* value, JsonValue* j);                 \
  };                                                                          \
  /**/

#define TENSORSTORE_INTERNAL_DEFINE_JSON_BINDER_IMPL(name, ...)   \
  absl::Status name::Do(std::true_type is_loading,                \
                        const JsonBinderFromJsonOptions& options, \
                        Value* value, JsonValue* j) {             \
    return (__VA_ARGS__)(is_loading, options, value, j);          \
  }                                                               \
  absl::Status name::Do(std::false_type is_loading,               \
                        const JsonBinderToJsonOptions& options,   \
                        const Value* value, JsonValue* j) {       \
    return (__VA_ARGS__)(is_loading, options, value, j);          \
  }                                                               \
  /**/

/// CRTP base class for defining JSON-bindable types.
///
/// It is not necessary to inherit from this type for compatibility with the
/// JSON binding mechanism and `DefaultBinder`, but doing so provides the
/// benefit of the `ToJson` and `FromJson` methods as well as explicit
/// conversion to `::nlohmann::json`.
///
/// \tparam Derived Derived class type that inherits from this class and is
///     compatible with `DefaultBinder`, either by defining a
///     `default_json_binder` static member or by specializing `DefaultBinder`
///     directly.  Defining `default_json_binder` is likely to be most
///     convenient.
/// \tparam FromJsonOptionsType The options type for converting from JSON.
/// \tparam ToJsonOptionsType The options type for converting to JSON.
template <typename Derived, typename FromJsonOptionsType = NoOptions,
          typename ToJsonOptionsType = IncludeDefaults>
class JsonBindable {
 public:
  using ToJsonOptions = ToJsonOptionsType;
  using FromJsonOptions = FromJsonOptionsType;

  /// Type of `json_binding::StaticBinder` which may optionally be used to
  /// define JSON bindings for this class in a separately-compiled source file
  /// rather than in the header.
  ///
  /// To use, add a static member to the derived class definition:
  ///
  ///     static const StaticBinder default_json_binder;
  ///
  /// Then, in the source file, define the `default_json_binder` member as:
  ///
  ///     const Derived::StaticBinder Derived::default_json_binder(
  ///         [](auto is_loading, const auto& options, auto *obj, auto *j) {
  ///           // ...
  ///         });
  ///
  /// \remark Even if the binder is just composed from an existing binder like
  ///     `Object`, for compatibility with `StaticBinder` it is still necessary
  ///     to write it as a capture-less lambda.
  ///
  /// \remark If separate compilation is not needed, `default_json_binder` can
  ///     be defined as a static constexpr member directly in the class
  ///     definition without the use of `StaticBinder`.  For example:
  ///
  ///         static constexpr auto default_json_binder =
  ///             json_binding::Object(/*...*/);
  using StaticBinder =
      json_binding::StaticBinder<Derived, FromJsonOptions, ToJsonOptions>;

  Result<::nlohmann::json> ToJson(
      const ToJsonOptions& options = ToJsonOptions{}) const {
    return json_binding::ToJson(static_cast<const Derived&>(*this),
                                DefaultBinder<>, options);
  }

  static Result<Derived> FromJson(
      ::nlohmann::json j, const FromJsonOptions& options = FromJsonOptions{}) {
    return json_binding::FromJson<Derived>(std::move(j), DefaultBinder<>,
                                           options);
  }

  explicit operator ::nlohmann::json() const { return this->ToJson().value(); }
};

}  // namespace json_binding
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDABLE_H_
