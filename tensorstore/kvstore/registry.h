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

#ifndef TENSORSTORE_KVSTORE_REGISTRY_H_
#define TENSORSTORE_KVSTORE_REGISTRY_H_

/// \file Interface for defining and registering a KeyValueStore driver that
///     supports a JSON representation.
///
/// To define a KeyValueStore driver, create a `Derived` class that inherits
/// from the CRTP base `internal_kvstore::RegisteredDriverSpec<Derived>`, and
/// define a global constant of type
/// `internal_kvstore::DriverRegistration<Derived>` to register it.
///
/// Refer to `memory/memory_key_value_store.cc` for an example.

#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/serialization/registry.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace internal_kvstore {

using kvstore::Driver;
using kvstore::DriverPtr;
using kvstore::DriverSpec;
using kvstore::DriverSpecPtr;

template <typename Derived>
class DriverOpenState;

struct DriverFromJsonOptions : public JsonSerializationOptions {
  const std::string& path;
};

using DriverRegistry =
    internal::JsonRegistry<DriverSpec, DriverFromJsonOptions,
                           JsonSerializationOptions,
                           internal::IntrusivePtr<const DriverSpec>>;

/// Returns the global KeyValueStore driver registry.
///
/// This should not be called directly by code outside this module.
DriverRegistry& GetDriverRegistry();

/// CRTP base class for KeyValueStore `DriverSpec` implementations that support
/// a JSON representation.
///
/// The `Derived` class must override all of the virtual methods of
/// `kvstore::DriverSpec`, except for the members noted as being defined
/// automatically by `RegisteredDriverSpec`, and in addition must define the
/// following members:
///
/// - The `id` member specifies the string driver identifier:
///
///     static constexpr char id[] = "my_driver_id";
///
/// - The cache key (as computed by `EncodeCacheKey`) must encode the `SpecData`
///   representation as a cache key.  It will only be called after binding
///   context resources.  When this is defined implicitly via `ApplyMembers`,
///   the `internal::CacheKeyExcludes` wrapper may be used to indicate members
///   that should be excluded from the cache key.  Members that only affect
///   creation but not opening should normally be excluded.
///
/// - The `Derived` class must be default constructible.
///
/// Refer to `memory/memory_key_value_store.cc` for an example driver
/// implementation.
///
/// \tparam Derived Derived class inheriting from this CRTP base.
/// \tparam SpecDataT Structure type that includes as members the parameters and
///     resources necessary to create/open the driver.
///
///     It must be compatible with `ContextBindingTraits`, serialization,
///     garbage collection, and `EncodeCacheKey`.  While it is possible to
///     individually define a `ContextBindingTraits` implementation, a
///     `Serializer` specialization, and an `EncodeCacheKeyAdl` function, in
///     most cases all three can be defined automatically simply by defining an
///     `ApplyMembers` method.
///
///     It must also support JSON binding, which can most easily be achieved by
///     defining a `default_json_binder` static data member.
///
///       struct MySpecData {
///         // Example members:
///         int mem1;
///         Context::Resource<SomeResource> mem2;
///         int open_option;
///
///         // For compatibility with `ContextBindingTraits`.
///         constexpr static auto ApplyMembers = [](auto& x, auto f) {
///           return f(f.mem1, f.mem2, internal::CacheKeyExcludes{open_option});
///         };
///
///         constexpr static auto default_json_binder = jb::Object(
///             jb::Member("mem1",
///                        jb::Projection<&MySpecData::mem1>()),
///             jb::Member("mem2",
///                        jb::Projection<&MySpecData::mem2>()),
///             jb::Member("open_option",
///                        jb::Projection<&MySpecData::open_option>()));
///       };
template <typename Derived, typename SpecDataT, typename Parent = DriverSpec>
class RegisteredDriverSpec : public Parent {
 public:
  using SpecData = SpecDataT;

  static_assert(std::is_base_of_v<DriverSpec, Parent>);

  std::string_view driver_id() const override { return Derived::id; }

  DriverSpecPtr Clone() const final {
    return DriverSpecPtr(new Derived(static_cast<const Derived&>(*this)));
  }

  absl::Status BindContext(const Context& context) override {
    return internal::ContextBindingTraits<SpecData>::Bind(data_, context);
  }

  void UnbindContext(
      const internal::ContextSpecBuilder& context_builder) override {
    internal::ContextBindingTraits<SpecData>::Unbind(data_, context_builder);
  }

  void StripContext() override {
    internal::ContextBindingTraits<SpecData>::Strip(data_);
  }

  void EncodeCacheKey(std::string* out) const override {
    EncodeCacheKeyImpl(out, data_);
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    garbage_collection::GarbageCollectionVisit(visitor, data_);
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.context_spec_, x.data_);
  };

  /// Encodes the cache key from the context-bound `SpecData` representation.
  ///
  /// This is used by `RegisteredKeyValueStore::EncodeCacheKey` below and by
  /// `RegisteredKeyValueStoreBoundSpec::EncodeCacheKey`.
  ///
  /// The `SpecData` template parameter is always equal to
  /// `typename Derived::SpecData`, but is specified as a template parameter
  /// because `Derived` is incomplete when this class template is instantiated.
  static void EncodeCacheKeyImpl(std::string* out, const SpecData& data) {
    internal::EncodeCacheKey(out, typeid(Derived), data);
  }

  /// Data members required by `Derived` spec class.
  SpecData data_;
};

/// CRTP base class for KeyValueStore `Driver` implementations.
///
/// The `Derived` class must define the following members:
///
/// - The `GetBoundSpecData` method must set `spec` to the context-bound
///   representation of the JSON specification of the driver.
///
///     absl::Status GetBoundSpecData(SpecData& spec) const;
///
/// - A valid specialization of
///   `tensorstore::garbage_collection::GarbageCollection<Derived>` must exist.
///   If there are no members that require garbage collection support,
///   `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED` may be used.
///   Otherwise, this may be accomplished either by defining a specialization
///   manually, or defining an `ApplyMembers` static member.
template <typename Derived, typename DerivedSpec, typename Parent = Driver>
class RegisteredDriver : public Parent {
 public:
  using SpecData = typename DerivedSpec::SpecData;

  void EncodeCacheKey(std::string* out) const override {
    // Generates a cache key by obtaining the `SpecData` representation,
    // then computing the cache key from that.
    SpecData bound_spec_data;
    if (auto status = static_cast<const Derived*>(this)->GetBoundSpecData(
            bound_spec_data);
        !status.ok()) {
      // Could not obtain bound spec data.  Just use the default implementation
      // that encodes the exact object identity.
      return Driver::EncodeCacheKey(out);
    }
    DerivedSpec::EncodeCacheKeyImpl(out, bound_spec_data);
  }

  Result<DriverSpecPtr> GetBoundSpec() const override {
    auto spec = internal::MakeIntrusivePtr<DerivedSpec>();
    spec->context_binding_state_ = ContextBindingState::bound;
    TENSORSTORE_RETURN_IF_ERROR(
        static_cast<const Derived*>(this)->GetBoundSpecData(spec->data_));
    return spec;
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    garbage_collection::GarbageCollectionVisit(
        visitor, *static_cast<const Derived*>(this));
  }
};

/// Registers a KeyValueStore driver implementation.
///
/// Example usage:
///
///     class MyDriverSpec
///         : public internal_kvstore::RegisteredDriverSpec<MyDriverSpec,
///                                                         MyDriverSpecData> {
///       constexpr static const char id[] = "my_driver";
///       // ...
///     };
///
///     const internal_kvstore::DriverRegistration<MyDriverSpec> registration;
template <typename Derived>
class DriverRegistration {
 public:
  DriverRegistration() {
    GetDriverRegistry().Register<Derived>(
        Derived::id, internal_json_binding::Projection<&Derived::data_>());
    serialization::Register<internal::IntrusivePtr<const DriverSpec>,
                            Derived>();
  }
};

}  // namespace internal_kvstore

extern template serialization::Registry&
serialization::GetRegistry<internal::IntrusivePtr<const kvstore::DriverSpec>>();

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_REGISTRY_H_
