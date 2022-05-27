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

#ifndef TENSORSTORE_DRIVER_REGISTRY_H_
#define TENSORSTORE_DRIVER_REGISTRY_H_

/// \file Interface for defining and registering a TensorStore driver that
///     supports a JSON representation.
///
/// To define a TensorStore driver, create a `Derived` class that inherits from
/// the CRTP base `RegisteredDriverSpec<Derived>`, and define a global constant
/// of type `DriverRegistration<Derived>` to register it.
///

#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/registry.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace internal {

using DriverRegistry = JsonRegistry<DriverSpec, JsonSerializationOptions,
                                    JsonSerializationOptions, DriverSpecPtr>;

/// Returns the global driver registry.
DriverRegistry& GetDriverRegistry();

/// CRTP base class for `DriverSpec` implementations that support a JSON
/// representation or serialization.
///
/// The `Derived` class must override all of the virtual methods of `DriverSpec`
/// except for the ones defined below, which are provided automatically.
///
/// In addition:
///
/// - It must define a `constexpr static const char id[]` member.
///
/// - A valid specialization of
///   `tensorstore::garbage_collection::GarbageCollection<Derived>` must exist.
///   If there are no members that require garbage collection support,
///   `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED` may be used.
///   Otherwise, this may be accomplished either by defining a specialization
///   manually, or defining an `ApplyMembers` static member.
///
/// - A valid specialization of
///   `tensorstore::serialization::Serializer<Derived>` must exist.  In most
///   cases this is done implicitly by defining an `ApplyMembers` static member.
///
/// - A valid specialization of
///   `tensorstore::internal::ContextBindingTraits<Derived>` must exist.  In
///   most cases this is done implicitly by defining an `ApplyMembers` static
///   member.
///
/// - It must also support `internal_json_binding::DefaultBinder`, which can
///   most easily be accomplished by defining a `default_json_binder` static
///   member.
template <typename Derived, typename Parent>
class RegisteredDriverSpec : public Parent {
  static_assert(std::is_base_of_v<DriverSpec, Parent>);

  DriverSpec::Ptr Clone() const override {
    return DriverSpec::Ptr(new Derived(static_cast<const Derived&>(*this)));
  }

  absl::Status BindContext(const Context& context) override {
    return ContextBindingTraits<Derived>::Bind(static_cast<Derived&>(*this),
                                               context);
  }

  void UnbindContext(const ContextSpecBuilder& context_builder) override {
    ContextBindingTraits<Derived>::Unbind(static_cast<Derived&>(*this),
                                          context_builder);
  }

  void StripContext() override {
    ContextBindingTraits<Derived>::Strip(static_cast<Derived&>(*this));
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    garbage_collection::GarbageCollectionVisit(
        visitor, static_cast<const Derived&>(*this));
  }

  std::string_view GetId() const override { return Derived::id; }
};

/// CRTP base class for `Driver` implementations.
///
/// - A valid specialization of
///   `tensorstore::garbage_collection::GarbageCollection<Derived>` must exist.
///   If there are no members that require garbage collection support,
///   `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED` may be used.
///   Otherwise, this may be accomplished either by defining a specialization
///   manually, or defining an `ApplyMembers` static member.
///
/// \tparam Derived The derived driver type.
/// \tparam Parent The super class, must equal or be derived from
///     `internal::Driver` (for example, may be `ChunkCacheDriver`).
template <typename Derived, typename Parent>
class RegisteredDriver : public Parent {
 public:
  using Parent::Parent;

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    garbage_collection::GarbageCollectionVisit(
        visitor, *static_cast<const Derived*>(this));
  }

 private:
  template <typename>
  friend class DriverRegistration;

  template <typename>
  friend class SerializationOnlyDriverRegistration;
};

/// Registers a driver implementation.
///
/// Example usage:
///
///     class MyDriverSpec : public RegisteredDriverSpec<MyDriverSpec> {
///       // ...
///     };
///
///     const DriverRegistration<MyDriverSpec> registration;
///
template <typename Spec>
class DriverRegistration {
 public:
  DriverRegistration() {
    GetDriverRegistry().Register<Spec>(Spec::id,
                                       internal_json_binding::DefaultBinder<>);
    serialization::Register<DriverSpecPtr, Spec>();
  }
};

template <typename Spec>
class SerializationOnlyDriverRegistration {
 public:
  SerializationOnlyDriverRegistration() {
    serialization::Register<DriverSpecPtr, Spec>();
  }
};

}  // namespace internal

extern template serialization::Registry&
serialization::GetRegistry<internal::DriverSpecPtr>();

}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_REGISTRY_H_
