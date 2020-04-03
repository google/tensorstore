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

#ifndef TENSORSTORE_INTERNAL_JSON_REGISTRY_IMPL_H_
#define TENSORSTORE_INTERNAL_JSON_REGISTRY_IMPL_H_

/// Implementation details for `json_registry.h`.

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal_json_registry {

/// Returns an `absl::StatusCode::kInvalidArgument` error specifying that `id`
/// is not registered.
absl::Status GetJsonUnregisteredError(absl::string_view id);

struct JsonUnregisteredData {
  /// Unregistered identifier.
  std::string id;

  /// JSON representation of unregistered object.
  ::nlohmann::json::object_t obj;
};

/// Derived class that represents an object with an unregistered identifier,
/// obtained from a `JsonRegistry` using `AllowUnregistered{true}`.
template <typename Base>
class JsonUnregistered : public Base {
 public:
  JsonUnregisteredData unregistered_data_;
};

/// Type-erased implementation details of `JsonRegistry`.
///
/// The following types are implicitly associated with each instance of this
/// class:
///
///   - `Base`: base class type with virtual destructor.
///
///   - `BasePtr`: reference-counted smart pointer type that holds an object of
///     type `Base`.
///
///   - `LoadOptions`: Options used when parsing from JSON.
///
///   - `SaveOptions`: Options used when converting to JSON.
///
///   - `UnregisteredBase`: Class type that inherits from `Base` (possibly equal
///     to `Base`) used to represent an unregistered object type.
class JsonRegistryImpl {
 public:
  /// Represents a registered object type, `T`, which must inherit from `Base`.
  struct Entry {
    /// String identifier used in the JSON representation.
    std::string id;

    /// Object type.
    const std::type_info* type;

    using Allocate = void (*)(void* obj);

    /// Resets `obj`, which must be a pointer to `BasePtr`, to hold a newly
    /// allocated pointer to `T`.
    Allocate allocate;

    /// JSON binder, where `obj` is a non-null pointer to `BasePtr` holding a
    /// non-null pointer to `T`, and `options` is a pointer to `LoadOptions` or
    /// `SaveOptions` depending on the type of `is_loading`.
    internal::Poly<
        0, /*Copyable=*/false,
        absl::Status(std::true_type is_loading, const void* options,
                     const void* obj, ::nlohmann::json::object_t*) const,
        absl::Status(std::false_type is_loading, const void* options,
                     const void* obj, ::nlohmann::json::object_t*) const>
        binder;
  };

  /// Must be called immediately after construction to initialize this object.
  template <typename Base, typename UnregisteredBase>
  void Initialize() {
    using BasePtr = internal::IntrusivePtr<Base>;
    using Unregistered =
        internal_json_registry::JsonUnregistered<UnregisteredBase>;
    static_assert(std::is_base_of_v<Base, UnregisteredBase>);
    unregistered_type_ = &typeid(Unregistered);
    allocate_unregistered_ = +[](void* obj) -> JsonUnregisteredData* {
      auto* x = new Unregistered;
      static_cast<BasePtr*>(obj)->reset(x);
      return &x->unregistered_data_;
    };
    get_unregistered_ = +[](const void* obj) -> JsonUnregisteredData* {
      return &(
          static_cast<Unregistered*>(static_cast<const BasePtr*>(obj)->get())
              ->unregistered_data_);
    };
  }

  /// Registers an object type.
  ///
  /// Logs a fatal error if the type or id is already registered.
  void Register(std::unique_ptr<Entry> entry);

  /// Initializes a `BasePtr` from a JSON representation of the object
  /// identifier.
  ///
  /// \param allow_unregistered Specifies whether to allow unregistered object
  ///     types.
  /// \param obj Non-null pointer to `BasePtr`.
  /// \param j Non-null JSON object assumed to specify the string object
  ///     identifier.
  absl::Status LoadKey(bool allow_unregistered, void* obj,
                       ::nlohmann::json* j) const;

  /// Converts a `BasePtr` to a JSON representation of the object identifier.
  ///
  /// \param obj Non-null pointer to `BasePtr`.
  /// \param j Non-null JSON object to hold the string identifier.
  absl::Status SaveKey(std::type_index type, const void* obj,
                       ::nlohmann::json* j) const;

  /// Parses the JSON representation of a registered object using the associated
  /// binder.
  ///
  /// \param type Type of `*obj`.
  /// \param options Non-null pointer to `LoadOptions`.
  /// \param obj Non-null pointer to `BasePtr` holding a non-null pointer.
  /// \param j_obj Non-null JSON object.
  absl::Status LoadRegisteredObject(std::type_index type, const void* options,
                                    const void* obj,
                                    ::nlohmann::json::object_t* j_obj) const;

  /// Converts a registered object to its JSON representation using the
  /// associated binder.
  ///
  /// \param type Type of `*obj`.
  /// \param options Non-null pointer to `SaveOptions`.
  /// \param obj Non-null pointer to `BasePtr` holding a non-null pointer.
  /// \param j_obj Non-null JSON object.
  absl::Status SaveRegisteredObject(std::type_index type, const void* options,
                                    const void* obj,
                                    ::nlohmann::json::object_t* j_obj) const;

 private:
  struct EntryIdKey : public absl::string_view {
    EntryIdKey(const std::string& key) : absl::string_view(key) {}
    EntryIdKey(absl::string_view key) : absl::string_view(key) {}
    EntryIdKey(const std::unique_ptr<Entry>& p) : absl::string_view(p->id) {}
  };
  struct EntryIdHash : public absl::Hash<EntryIdKey> {
    using is_transparent = void;
  };
  struct EntryIdEqualTo : public std::equal_to<EntryIdKey> {
    using is_transparent = void;
  };

  struct EntryTypeKey : public std::type_index {
    EntryTypeKey(std::type_index type) : std::type_index(type) {}
    EntryTypeKey(const Entry* p) : std::type_index(*p->type) {}
    template <typename H>
    friend H AbslHashValue(H h, EntryTypeKey key) {
      return H::combine(std::move(h), key.hash_code());
    }
  };
  struct EntryTypeHash : public absl::Hash<EntryTypeKey> {
    using is_transparent = void;
  };
  struct EntryTypeEqualTo : public std::equal_to<EntryTypeKey> {
    using is_transparent = void;
  };

  mutable absl::Mutex mutex_;
  // Allows lookup of entries by string identifier rather than pointer identity.
  absl::flat_hash_set<std::unique_ptr<Entry>, EntryIdHash, EntryIdEqualTo>
      entries_ ABSL_GUARDED_BY(mutex_);
  // Allows lookup of entries by `std::type_index` rather than pointer identity.
  absl::flat_hash_set<Entry*, EntryTypeHash, EntryTypeEqualTo> entries_by_type_
      ABSL_GUARDED_BY(mutex_);
  using AllocateUnregistered = JsonUnregisteredData* (*)(void* obj);

  /// Function that resets `obj`, which must be a non-null pointer to `BasePtr`,
  /// to a newly allocated object of a type `JsonUnregistered<Base>`.
  AllocateUnregistered allocate_unregistered_;

  /// Set to `&typeid(JsonUnregistered<Base>)`.
  const std::type_info* unregistered_type_;

  using GetUnregistered = JsonUnregisteredData* (*)(const void* obj);

  /// Function that returns the non-null stored `JsonUnregisteredData` from
  /// `obj`, which must be a non-null pointer to `BasePtr` that holds a non-null
  /// pointer to `JsonUnregistered<Base>`.
  GetUnregistered get_unregistered_;
};

}  // namespace internal_json_registry
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_REGISTRY_IMPL_H_
