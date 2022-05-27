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

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/heterogeneous_container.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal_json_registry {

/// Returns an `absl::StatusCode::kInvalidArgument` error specifying that `id`
/// is not registered.
absl::Status GetJsonUnregisteredError(std::string_view id);

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
class JsonRegistryImpl {
 public:
  /// Represents a registered object type, `T`, which must inherit from `Base`.
  struct Entry {
    /// String identifier used in the JSON representation.
    std::string id;

    /// Object type.
    const std::type_info* type;

    std::type_index type_index() const { return *type; }

    using Allocate = void (*)(void* obj);

    /// Resets `obj`, which must be a pointer to `BasePtr`, to hold a newly
    /// allocated pointer to `T`.
    Allocate allocate;

    /// JSON binder, where `obj` is a non-null pointer to `BasePtr` holding a
    /// non-null pointer to `T`, and `options` is a pointer to `LoadOptions` or
    /// `SaveOptions` depending on the type of `is_loading`.
    poly::Poly<0, /*Copyable=*/false,
               absl::Status(std::true_type is_loading, const void* options,
                            const void* obj, ::nlohmann::json::object_t*) const,
               absl::Status(std::false_type is_loading, const void* options,
                            const void* obj, ::nlohmann::json::object_t*) const>
        binder;
  };

  /// Registers an object type.
  ///
  /// Logs a fatal error if the type or id is already registered.
  void Register(std::unique_ptr<Entry> entry);

  /// Initializes a `BasePtr` from a JSON representation of the object
  /// identifier.
  ///
  /// \param obj Non-null pointer to `BasePtr`.
  /// \param j Non-null JSON object assumed to specify the string object
  ///     identifier.
  absl::Status LoadKey(void* obj, ::nlohmann::json* j) const;

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
  mutable absl::Mutex mutex_;
  // Allows lookup of entries by string identifier rather than pointer identity.
  internal::HeterogeneousHashSet<std::unique_ptr<Entry>, std::string_view,
                                 &Entry::id>
      entries_ ABSL_GUARDED_BY(mutex_);
  // Allows lookup of entries by `std::type_index` rather than pointer identity.
  internal::HeterogeneousHashSet<const Entry*, std::type_index,
                                 &Entry::type_index>
      entries_by_type_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal_json_registry
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_REGISTRY_IMPL_H_
