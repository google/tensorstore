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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_BINDABLE_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_BINDABLE_H_

#include <type_traits>
#include <variant>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_json_binding {

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
constexpr inline auto DefaultBinder = [](auto is_loading, const auto& options,
                                         auto* obj, auto* j) -> absl::Status {
  return T::default_json_binder(is_loading, options, obj, j);
};

/// Function object that forwards to the default binder implementation.
template <>
constexpr inline auto DefaultBinder<void> = [](auto is_loading,
                                               const auto& options, auto* obj,
                                               auto* j) -> absl::Status {
  using T = std::remove_cv_t<std::remove_pointer_t<decltype(obj)>>;
  return DefaultBinder<T>(is_loading, options, obj, j);
};

/// Converts an object to JSON using the specified binder.
template <typename JsonValue = ::nlohmann::json, typename T,
          typename Binder = decltype(DefaultBinder<>),
          typename Options = IncludeDefaults>
Result<JsonValue> ToJson(const T& obj, Binder binder = DefaultBinder<>,
                         const Options& options = Options{}) {
  JsonValue value([] {
    if constexpr (std::is_same_v<JsonValue, ::nlohmann::json>) {
      return ::nlohmann::json::value_t::discarded;
    } else {
      return JsonValue();
    }
  }());
  TENSORSTORE_RETURN_IF_ERROR(binder(std::false_type{}, options, &obj, &value));
  return value;
}

/// Converts an object from its JSON representation using the specified binder.
template <typename T, typename JsonValue = ::nlohmann::json,
          typename Binder = decltype(DefaultBinder<>),
          typename Options = NoOptions>
Result<T> FromJson(JsonValue j, Binder binder = DefaultBinder<>,
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

 private:
  using LoadFunction = absl::Status (*)(std::true_type,
                                        const FromJsonOptions& options, T* obj,
                                        JsonValue* j, ExtraValue*... extra);
  using SaveFunction = absl::Status (*)(std::false_type,
                                        const ToJsonOptions& options,
                                        const T* obj, JsonValue* j,
                                        ExtraValue*... extra);

 public:
  template <typename Binder, typename = std::enable_if_t<
                                 (std::is_convertible_v<Binder, LoadFunction> &&
                                  std::is_convertible_v<Binder, SaveFunction>)>>
  StaticBinder(Binder binder) : load_(binder), save_(binder) {}

  absl::Status operator()(std::true_type is_loading,
                          const FromJsonOptions& options, T* obj, JsonValue* j,
                          ExtraValue*... extra) const {
    return load_(is_loading, options, obj, j, extra...);
  }

  absl::Status operator()(std::false_type is_loading,
                          const ToJsonOptions& options, const T* obj,
                          JsonValue* j, ExtraValue*... extra) const {
    return save_(is_loading, options, obj, j, extra...);
  }

 private:
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
/// `internal_json_binding::DefaultBinder<>`, use
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
/// `internal_json_binding::DefaultBinder<>` but requires the binder to be
/// defined in a separate source file.
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
///     namespace jb = tensorstore::internal_json_binding;
///     TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(X, jb::Object(...));
///
#define TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(name, ...)               \
  TENSORSTORE_INTERNAL_DECLARE_JSON_BINDER_IMPL(JsonBinderImpl, name,    \
                                                __VA_ARGS__)             \
  static inline constexpr JsonBinderImpl default_json_binder = {};       \
  tensorstore::Result<JsonBinderImpl::JsonValue> ToJson(                 \
      const JsonBinderImpl::JsonBinderToJsonOptions& options =           \
          JsonBinderImpl::JsonBinderToJsonOptions{}) const {             \
    return tensorstore::internal_json_binding::ToJson<                   \
        JsonBinderImpl::JsonValue>(*this, default_json_binder, options); \
  }                                                                      \
  static tensorstore::Result<name> FromJson(                             \
      JsonBinderImpl::JsonValue j,                                       \
      const JsonBinderImpl::JsonBinderFromJsonOptions& options =         \
          JsonBinderImpl::JsonBinderFromJsonOptions{}) {                 \
    return tensorstore::internal_json_binding::FromJson<name>(           \
        std::move(j), default_json_binder, options);                     \
  }                                                                      \
  explicit operator ::nlohmann::json() const {                           \
    return this->ToJson().value();                                       \
  }                                                                      \
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
        ::tensorstore::internal_json_binding::StaticBinder<__VA_ARGS__>;      \
    using Value = typename StaticBinderType::Value;                           \
                                                                              \
   public:                                                                    \
    using JsonValue = typename StaticBinderType::JsonValue;                   \
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

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_BINDABLE_H_
