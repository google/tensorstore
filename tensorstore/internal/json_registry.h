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

#ifndef TENSORSTORE_INTERNAL_JSON_REGISTRY_H_
#define TENSORSTORE_INTERNAL_JSON_REGISTRY_H_

/// Framework for converting between objects of a polymorphic class type and
/// their JSON representation.  The type is specified by a string identifier,
/// and a registry is used to map a given string identifier to a given class
/// type and JSON binder.
///
/// Class types compatible with this framework are held by reference-counted
/// pointer and should be logically immutable.
///
/// See `json_registry_test.cc` for example usage.

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_registry_fwd.h"
#include "tensorstore/internal/json_registry_impl.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal {

/// Registry of JSON binders.
///
/// \tparam Base Base class type for the object types to be registered.  Must be
///     compatible with `IntrusivePtr`.
/// \tparam LoadOptions Load options supported by the registered JSON binders.
///     Must inherit from `AllowUnregistered`.
/// \tparam SaveOptions Save options supported by the registered JSON binders.
/// \tparam Unregistered The type used in the case that
///     `AllowUnregistered{true}` is specified and the string identifier is
///     invalid.  Must inherit from `Base`.  By default, `Base` is used, in
///     which case any virtual methods of that type should have a default
///     implementation that accounts for this case.
template <typename Base, typename LoadOptions, typename SaveOptions,
          typename UnregisteredBase>
class JsonRegistry {
  using BasePtr = IntrusivePtr<Base>;
  static_assert(std::has_virtual_destructor_v<Base>);

 public:
  JsonRegistry() { impl_.Initialize<Base, UnregisteredBase>(); }

  /// Returns an `IntrusivePtr<Base>` binder for a JSON string specifying a
  /// registered id.
  ///
  /// When parsing JSON, this resets the `IntrusivePtr<Base>` to a newly
  /// allocated object of the derived type registered for id, or to
  /// `UnregisteredBase` if the id is not registered and
  /// `AllowUnregistered{true}`.  If `AllowUnregistered{false}` is specified
  /// (the default), an unregistered id results in a error.
  ///
  /// When converting from JSON, this simply sets the JSON value to the string
  /// identifier; this binder must not be used with a null pointer.
  ///
  /// Note that this binder does not call the type-specific binder corresponding
  /// to the id.  That must be done separately using the
  /// `RegisteredObjectBinder`. These steps are kept separate because in some
  /// cases it is useful to sequence other bindings after `KeyBinder` but before
  /// `RegisteredObjectBinder`, e.g. in order to initialize common fields of the
  /// allocated object that may be needed by the type-specific binders.
  ///
  /// Example usage:
  ///
  ///     jb::Object(jb::Member("id", GetRegistry().KeyBinder()),
  ///                Registry::RegisteredObjectBinder())
  ///
  auto KeyBinder() { return KeyBinderImpl{impl_}; }

  /// Forwards to the registered type-specific object binder.
  ///
  /// If the target `IntrusivePtr<Base>` is `nullptr`, this binder is a no-op.
  ///
  /// If the object representation correspond to an unregistered id (e.g. due to
  /// a prior call to `KeyBinder` with `AllowUnregistered{true}` specified),
  /// then this merely copies the JSON object representation as is without any
  /// further parsing.
  ///
  /// Normally the type-specific object binders using `json_binding::Object`,
  /// and therefore this should be the last binder specified within an outer
  /// `json_binding::Object`.  That ensures that all other members are already
  /// handled when parsing from JSON, and that no existing members will be
  /// cleared when converting to JSON.
  constexpr auto RegisteredObjectBinder() {
    return RegisteredObjectBinderImpl{impl_};
  }

  /// Convenience wrapper that combines `Member` with `Binder()`.
  ///
  /// This returns a JSON object binder where a member with the specified
  /// `member_name` indicates the string identifier within the registry and the
  /// remaining object members map to the corresponding registered binder.
  ///
  /// Example usage:
  ///
  ///     jb::Object(registry.MemberBinder("id"))
  ///
  /// The above expression evaluates to a JSON object binder where the `"id"`
  /// member specifies the string identifier within the registry and the
  /// remaining members are handled by the registered binder.  When loading, the
  /// appropriate binder is found based on the specified `"id"` member.  When
  /// saving, the `"id"` member is set to the string identifier used when
  /// loading, and the remaining object members are set using the binder used
  /// when loading.
  template <typename MemberName>
  auto MemberBinder(MemberName member_name) {
    namespace jb = tensorstore::internal::json_binding;
    return jb::Sequence(jb::Member(member_name, this->KeyBinder()),
                        RegisteredObjectBinder());
  }

  /// Registers a JSON binder for a given class type `T` and string identifier.
  ///
  /// \tparam T The class type to register.  Must inherit from `Base`.  Must not
  ///     previously have been registered.
  /// \param id The string identifier.  Must not previously have been
  ///     registered.
  /// \param binder JSON object binder for `T` compatible with `LoadOptions` and
  ///     `SaveOptions`.
  template <typename T, typename Binder>
  void Register(absl::string_view id, Binder binder) {
    static_assert(std::is_base_of_v<Base, T>);
    auto entry =
        std::make_unique<internal_json_registry::JsonRegistryImpl::Entry>();
    entry->id = std::string(id);
    entry->type = &typeid(T);
    entry->allocate =
        +[](void* obj) { static_cast<BasePtr*>(obj)->reset(new T); };
    entry->binder = [binder](
                        auto is_loading, const void* options, auto* obj,
                        ::nlohmann::json::object_t* j_obj) -> absl::Status {
      using Options = std::conditional_t<decltype(is_loading)::value,
                                         LoadOptions, SaveOptions>;
      using Obj = std::conditional_t<decltype(is_loading)::value, T, const T>;
      return binder(is_loading, *static_cast<const Options*>(options),
                    static_cast<Obj*>(static_cast<const BasePtr*>(obj)->get()),
                    j_obj);
    };
    impl_.Register(std::move(entry));
  }

 private:
  // These two classes are defined here rather than within `KeyBinder` and
  // `RegisteredObjectBinder` to work around a clang-cl error.
  struct KeyBinderImpl {
    const internal_json_registry::JsonRegistryImpl& impl;
    absl::Status operator()(std::true_type is_loading,
                            const LoadOptions& options, BasePtr* obj,
                            ::nlohmann::json* j) const {
      return impl.LoadKey(options.allow_unregistered(), obj, j);
    }
    absl::Status operator()(std::false_type is_loading,
                            const SaveOptions& options, const BasePtr* obj,
                            ::nlohmann::json* j) const {
      return impl.SaveKey(typeid(*obj->get()), obj, j);
    }
  };

  struct RegisteredObjectBinderImpl {
    const internal_json_registry::JsonRegistryImpl& impl;
    absl::Status operator()(std::true_type is_loading,
                            const LoadOptions& options, BasePtr* obj,
                            ::nlohmann::json::object_t* j_obj) const {
      if (!*obj) return absl::OkStatus();
      return impl.LoadRegisteredObject(typeid(*obj->get()), &options, obj,
                                       j_obj);
    }

    absl::Status operator()(std::false_type is_loading,
                            const SaveOptions& options, const BasePtr* obj,
                            ::nlohmann::json::object_t* j_obj) const {
      if (!*obj) return absl::OkStatus();
      return impl.SaveRegisteredObject(typeid(*obj->get()), &options, obj,
                                       j_obj);
    }
  };

  internal_json_registry::JsonRegistryImpl impl_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_REGISTRY_H_
