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

#ifndef TENSORSTORE_KVSTORE_SPEC_H_
#define TENSORSTORE_KVSTORE_SPEC_H_

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/option.h"

namespace tensorstore {
namespace kvstore {

/// Options for mutating `Spec` objects that are handled directly by drivers.
struct DriverSpecOptions {
  bool minimal_spec = false;

  template <typename T>
  constexpr static bool IsOption = false;

  void Set(MinimalSpec value) { minimal_spec = value.minimal_spec(); }
};

template <>
constexpr inline bool DriverSpecOptions::IsOption<MinimalSpec> = true;

/// Options that may be specified for modifying an existing `Spec`.
///
/// Refer to the documentation of `Spec::Set` for details.
///
/// \relates Spec
struct SpecConvertOptions : public DriverSpecOptions {
  /// Specifies the context binding mode.
  ContextBindingMode context_binding_mode = ContextBindingMode::unspecified;

  /// Specifies the context to bind.
  Context context;

  /// Indicates which option types are supported.
  ///
  /// Supported types include:
  ///
  /// - `Context`
  /// - `ContextBindingMode`
  template <typename T>
  constexpr static bool IsOption = DriverSpecOptions::IsOption<T>;

  using DriverSpecOptions::Set;

  void Set(Context value) { context = std::move(value); }
  void Set(ContextBindingMode value) {
    if (value > context_binding_mode) context_binding_mode = value;
  }
};

template <>
constexpr inline bool SpecConvertOptions::IsOption<Context> = true;

template <>
constexpr inline bool SpecConvertOptions::IsOption<ContextBindingMode> = true;

class DriverSpec;
void intrusive_ptr_increment(const DriverSpec* p);
void intrusive_ptr_decrement(const DriverSpec* p);

/// Smart pointer that represents a driver-specific kvstore spec.
///
/// \relates Spec
class DriverSpecPtr : public internal::IntrusivePtr<const DriverSpec> {
  using Base = internal::IntrusivePtr<const DriverSpec>;

 public:
  using Base::Base;

  /// Binds any unbound context resources using the specified context.  Any
  /// already-bound context resources remain unmodified.
  ///
  /// If an error occurs, some context resources may remain unbound.
  absl::Status BindContext(const Context& context);

  /// Unbinds any bound context resources, replacing them with context
  /// resource specs that may be used to recreate the context resources.  Any
  /// already-unbound context resources remain unmodified.
  void UnbindContext() { return UnbindContext({}); }

  void UnbindContext(const internal::ContextSpecBuilder& context_builder);

  /// Replaces any context resources with a default context resource spec.
  void StripContext();

  /// Indicates the context binding state of the spec.
  ContextBindingState context_binding_state() const;

  /// Mutates this spec according to the specified options.
  ///
  /// Options may be specified in any order and are identified by their type.
  /// Supported option types are:
  ///
  /// - ContextBindingMode: Defaults to `ContextBindingMode::retain`, which
  ///   does nothing.  Specifying `ContextBindingMode::unbind` is equivalent
  ///   to calling `UnbindContext`.  Specifying `ContextBindingMode::strip` is
  ///   equivalent to calling `StripContext`.
  ///
  /// - Context: If a non-null context is specified, any unbound context
  ///   resources are resolved using the specified context, equivalent to
  ///   calling `BindContext`.  If not specified, unbound context resources
  ///   remain unmodified.  If `ContextBindingMode::unbind` or
  ///   `ContextBindingMode::strip` is specified along with this option, the
  ///   unbind or strip operation is performed before re-binding with the
  ///   specified context.
  ///
  /// If an error occurs, the spec may be left in a partially modified state.
  ///
  /// \param option Any option type supported by `SpecConvertOptions`.
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecConvertOptions, Option...>,
                   absl::Status>
  Set(Option&&... option) {
    SpecConvertOptions options;
    (options.Set(option), ...);
    return Set(std::move(options));
  }
  absl::Status Set(DriverSpecOptions&& options);
  absl::Status Set(SpecConvertOptions&& options);

  // For compatibility with `tensorstore::internal::EncodeCacheKey`.
  friend void EncodeCacheKeyAdl(std::string* out, const DriverSpecPtr& ptr);
};

class Driver;
void intrusive_ptr_increment(Driver* p);
void intrusive_ptr_decrement(Driver* p);

/// Handle to an open kvstore driver.
///
/// \relates KvStore
using DriverPtr = internal::IntrusivePtr<Driver>;

// For compatibility with `tensorstore::internal::EncodeCacheKey`.
void EncodeCacheKeyAdl(std::string* out, const DriverPtr& ptr);

}  // namespace kvstore

namespace internal_kvstore {
template <typename Derived, typename DerivedSpec, typename Parent>
class RegisteredDriver;
template <typename Derived, typename SpecDataT, typename Parent>
class RegisteredDriverSpec;
}  // namespace internal_kvstore

namespace kvstore {

/// Combines a driver-specific kvstore spec with a string `path` that serves as
/// a key prefix.
///
/// \ingroup kvstore
class Spec {
 public:
  /// Constructs an invalid (null) spec.
  ///
  /// \id default
  Spec() = default;

  /// Constructs from a driver with empty path.
  ///
  /// \id driver
  Spec(DriverSpecPtr driver) : driver(std::move(driver)) {}

  /// Constructs a path from the specified driver and key prefix.
  ///
  /// \id driver, path
  explicit Spec(DriverSpecPtr driver, std::string path)
      : driver(std::move(driver)), path(std::move(path)) {}

  /// Appends `suffix` to the `path`.
  ///
  /// There is no special treatment of '/'.
  void AppendSuffix(std::string_view suffix) { path += suffix; }

  /// Joins a '/'-separated path component to the end `path`.
  void AppendPathComponent(std::string_view component) {
    internal::AppendPathComponent(path, component);
  }

  /// Returns `true` if this is a valid (non-null) spec.
  bool valid() const { return static_cast<bool>(driver); }

  /// Driver spec.
  DriverSpecPtr driver;

  /// Path within the `driver`.
  std::string path;

  // ApplyMembers support.
  static constexpr auto ApplyMembers = [](auto& x, auto f) {
    return f(x.driver, x.path);
  };

  /// Binds any unbound context resources using the specified context.  Any
  /// already-bound context resources remain unmodified.
  ///
  /// If an error occurs, some context resources may remain unbound.
  absl::Status BindContext(const Context& context);

  /// Unbinds any bound context resources, replacing them with context
  /// resource specs that may be used to recreate the context resources.
  ///
  /// Any already-unbound context resources remain unmodified.
  void UnbindContext() { UnbindContext({}); }

  void UnbindContext(const internal::ContextSpecBuilder& context_builder);

  /// Replaces any context resources with a default context resource spec.
  void StripContext();

  /// Returns the context binding state of the spec.
  ContextBindingState context_binding_state() const {
    return driver.context_binding_state();
  }

  /// Mutates this spec according to the specified options.
  ///
  /// Options may be specified in any order and are identified by their type.
  /// Supported option types are:
  ///
  /// - ContextBindingMode: Defaults to `ContextBindingMode::retain`, which
  ///   does nothing.  Specifying `ContextBindingMode::unbind` is equivalent
  ///   to calling `UnbindContext`.  Specifying `ContextBindingMode::strip` is
  ///   equivalent to calling `StripContext`.
  ///
  /// - Context: If a non-null context is specified, any unbound context
  ///   resources are resolved using the specified context, equivalent to
  ///   calling `BindContext`.  If not specified, unbound context resources
  ///   remain unmodified.  If `ContextBindingMode::unbind` or
  ///   `ContextBindingMode::strip` is specified along with this option, the
  ///   unbind or strip operation is performed before re-binding with the
  ///   specified context.
  ///
  /// If an error occurs, the spec may be left in a partially modified state.
  ///
  /// \param option Any option type supported by `SpecConvertOptions`.
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecConvertOptions, Option...>,
                   absl::Status>
  Set(Option&&... option) {
    SpecConvertOptions options;
    (options.Set(option), ...);
    return Set(std::move(options));
  }
  absl::Status Set(SpecConvertOptions&& options);

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, JsonSerializationOptions,
                                          JsonSerializationOptions)

  /// Parses a `kvstore::Spec` from its URL representation.
  static Result<Spec> FromUrl(std::string_view url);

  /// Returns the URL representation of this spec, if supported.
  Result<std::string> ToUrl() const;
};

}  // namespace kvstore

namespace internal {

template <typename, typename>
struct ContextBindingTraits;

/// Make `DriverSpecPtr` compatible with `ContextBindingTraits`.
template <>
struct ContextBindingTraits<kvstore::DriverSpecPtr, /*SFINAE=*/void> {
  using Spec = kvstore::DriverSpecPtr;
  static absl::Status Bind(Spec& spec, const Context& context) {
    if (!spec) return absl::OkStatus();
    return spec.BindContext(context);
  }
  static void Unbind(Spec& spec, const ContextSpecBuilder& builder) {
    spec.UnbindContext(builder);
  }
  static void Strip(Spec& spec) { spec.StripContext(); }
};
}  // namespace internal

namespace internal_json_binding {
/// JSON binder that converts between
/// `{"kvstore": {...}, "path": "path/within/kvstore"}` and a
/// `kvstore::Spec`.
///
/// When loading, if the additional deprecated "path" member is specified, its
/// value is combined via `AppendPathComponent` with any path specified within
/// the "kvstore".  When saving, the additional "path" is not specified.
TENSORSTORE_DECLARE_JSON_BINDER(KvStoreSpecAndPathJsonBinder, kvstore::Spec,
                                JsonSerializationOptions,
                                JsonSerializationOptions,
                                ::nlohmann::json::object_t)
}  // namespace internal_json_binding

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::kvstore::Spec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Spec)

#endif  // TENSORSTORE_KVSTORE_SPEC_H_
