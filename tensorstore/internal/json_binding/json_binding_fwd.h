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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_JSON_BINDING_FWD_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_JSON_BINDING_FWD_H_

#include <type_traits>

#include <nlohmann/json_fwd.hpp>
#include "tensorstore/json_serialization_options_base.h"

namespace tensorstore {
namespace internal_json_binding {

/// Helper type used by `TENSORSTORE_DECLARE_JSON_BINDER_METHODS` for "parsing"
/// the macro varargs and handling default arguments.
template <typename T, typename FromJsonOptionsType = NoOptions,
          typename ToJsonOptionsType = IncludeDefaults,
          typename JsonValueType = ::nlohmann::json>
struct JsonBindingMethodTypeHelper {
  using Value = T;
  using FromJsonOptions = FromJsonOptionsType;
  using ToJsonOptions = ToJsonOptionsType;
  using JsonValue = JsonValueType;
};

/// Declares that an arbitrary class may be used as a JSON binder type.
///
/// This allows the members of the class to be used as parameters of the binder,
/// without requiring the JSON binder to be defined inline.
///
/// The macro should be invoked with up to 4 types as arguments:
///
/// - ValueType (required)
/// - FromJsonOptionsType = NoOptions
/// - ToJsonOptionsType = IncludeDefaults
/// - JsonValueType = ::nlohmann::json
///
/// This macro handles the arguments specially such that commas within <>
/// brackets are not a problem.
///
/// Example:
///
///     struct Foo {
///       int value;
///     };
///
///     struct FooBinder {
///       int default_value = 10;
///
///       TENSORSTORE_DECLARE_JSON_BINDER_METHODS(Foo)
///     };
///
///     namespace jb = tensorstore::internal_json_binding;
///     TENSORSTORE_DEFINE_JSON_BINDER_METHODS(
///         FooBinder,
///         jb::Object(
///             jb::Member("value", jb::Projection<&Foo::value>(
///                 jb::DefaultValue([&](int *v) { *v = default_value; })))))
///
#define TENSORSTORE_DECLARE_JSON_BINDER_METHODS(...)                          \
  using BindingHelperType =                                                   \
      ::tensorstore::internal_json_binding::JsonBindingMethodTypeHelper<      \
          __VA_ARGS__>;                                                       \
  using Value = typename BindingHelperType::Value;                            \
  using JsonValue = typename BindingHelperType::JsonValue;                    \
  using JsonBinderFromJsonOptions =                                           \
      typename BindingHelperType::FromJsonOptions;                            \
  using JsonBinderToJsonOptions = typename BindingHelperType::ToJsonOptions;  \
  absl::Status operator()(std::true_type is_loading,                          \
                          const JsonBinderFromJsonOptions& options,           \
                          Value* value, JsonValue* j) const {                 \
    return this->DoLoadSaveJson(is_loading, options, value, j);               \
  }                                                                           \
  absl::Status operator()(std::false_type is_loading,                         \
                          const JsonBinderToJsonOptions& options,             \
                          const Value* value, JsonValue* j) const {           \
    return this->DoLoadSaveJson(is_loading, options, value, j);               \
  }                                                                           \
  template <bool IsLoading>                                                   \
  absl::Status DoLoadSaveJson(                                                \
      std::integral_constant<bool, IsLoading> is_loading,                     \
      const std::conditional_t<IsLoading, JsonBinderFromJsonOptions,          \
                               JsonBinderToJsonOptions>& options,             \
      std::conditional_t<IsLoading, Value, const Value>* value, JsonValue* j) \
      const;                                                                  \
  /**/

/// Defines the JSON binder methods declared by
/// `TENSORSTORE_DECLARE_JSON_BINDER_METHODS`.  See the documentation of that
/// macro for details.
#define TENSORSTORE_DEFINE_JSON_BINDER_METHODS(NAME, ...)                     \
  template <bool IsLoading>                                                   \
  absl::Status NAME::DoLoadSaveJson(                                          \
      std::integral_constant<bool, IsLoading> json_binder_is_loading,         \
      const std::conditional_t<IsLoading, JsonBinderFromJsonOptions,          \
                               JsonBinderToJsonOptions>& json_binder_options, \
      std::conditional_t<IsLoading, Value, const Value>* json_binder_value,   \
      JsonValue* json_binder_j) const {                                       \
    return (__VA_ARGS__)(json_binder_is_loading, json_binder_options,         \
                         json_binder_value, json_binder_j);                   \
  }                                                                           \
  template absl::Status NAME::DoLoadSaveJson<true>(                           \
      std::true_type, const JsonBinderFromJsonOptions&, Value*, JsonValue*)   \
      const;                                                                  \
  template absl::Status NAME::DoLoadSaveJson<false>(                          \
      std::false_type, const JsonBinderToJsonOptions&, const Value*,          \
      JsonValue*) const;                                                      \
  /**/

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_JSON_BINDING_FWD_H_
