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
/// from the CRTP base `RegisteredKeyValueStore<Derived>`, and define a global
/// constant of type `KeyValueStoreDriverRegistration<Derived>` to register it.
///
/// Refer to `memory/memory_key_value_store.cc` for an example.

#include "tensorstore/internal/json_registry.h"
#include "tensorstore/kvstore/key_value_store.h"

namespace tensorstore {
namespace internal {

template <typename Derived>
class KeyValueStoreOpenState;

using KeyValueStoreDriverRegistry =
    JsonRegistry<KeyValueStoreSpec, KeyValueStoreSpec::FromJsonOptions,
                 KeyValueStoreSpec::ToJsonOptions>;

/// Returns the global KeyValueStore driver registry.
///
/// This should not be called directly by code outside this module.
KeyValueStoreDriverRegistry& GetKeyValueStoreDriverRegistry();

/// CRTP base class for KeyValueStore implementations that support a JSON
/// representation.
///
/// The `Derived` class must override all of the virtual methods of
/// `KeyValueStore`, except for the members noted as being defined automatically
/// by `RegisteredKeyValueStore`, and in addition must define the following
/// members:
///
/// - The `id` member specifies the string driver identifier:
///
///     static constexpr char id[] = "my_driver_id";
///
/// - The `SpecT` class template includes as members the parameters and
///   resources necessary to create/open the driver.  Depending on the
///   `MaybeBound` argument, which is either `ContextUnbound` or `ContextBound`,
///   it specifies either the context-unbound or context-bound state of the
///   parameters/resources.  The dependence on `MaybeBound` permits
///   `Context::ResourceSpec` objects (as used for the JSON representation) to
///   be converted automatically to/from `Context::Resource` objects (as used by
///   the driver implementation).
///
///   It must define an `ApplyMembers` method for compatibility with
///   `ContextBindingTraits` (refer to `tensorstore/internal/context_binding.h`
///   for details).
///
///   Members of `SpecT` should be referenced in the `json_binder` and
///   `EncodeCacheKey` implementation, as noted below.
///
///     template <template <typename> class MaybeBound>
///     struct SpecT {
///        // Example members:
///       int mem1;
///       MaybeBound<Context::ResourceSpec<SomeResource>> mem2;
///
///       // For compatibility with `ContextBindingTraits`.
///       constexpr static auto ApplyMembers = [](auto& x, auto f) {
///         return f(x.mem1, f.mem2);
///       };
///     };
///
/// - The `json_binder` member must be a JSON object binder for
///   `SpecT<ContextUnbound>`.  This should handle converting each member of
///   `SpecT` to/from the JSON representation.
///
///     constexpr static auto json_binder = jb::Object(
///         jb::Member("mem1", jb::Projection(&SpecT<ContextUnbound>::mem1)),
///         jb::Member("mem2", jb::Projection(&SpecT<ContextUnbound>::mem2)));
///
/// - The static `EncodeCacheKey` method encodes the `SpecT<ContextBound>`
///   representation as a cache key.  Typically this just calls
///   `tensorstore::internal::EncodeCacheKey` with the members that are relevant
///   to caching.  Members that only affect creation but not opening should
///   normally be skipped.
///
///     static void EncodeCacheKey(std::string *out,
///                                const SpecT<ContextBound>& data) {
///       tensorstore::internal::EncodeCacheKey(out, data.mem1, data.mem2);
///     }
///
/// - The static `ConvertSpec` method should apply any modifications requested
///   in `options` in place to `*spec`.
///
///     static Status ConvertSpec(
///         SpecT<ContextUnbound>* spec,
///         const KeyValueStore::SpecRequestOptions& options);
///
/// - The static `Open` method is called to initiate opening the driver.  This
///   is called by `KeyValueStore::BoundSpec::Open`.  Note that
///   `KeyValueStoreOpenState` is a CRTP class template parameterized by the
///   `Derived` driver type.
///
///     static void Open(internal::KeyValueStoreOpenState<Derived> state) {
///       // Access the `SpecT<ContextBound>` representation as `state.spec()`.
///       // Access the newly allocated `Derived` object as `state.driver()`.
///       // Report errors via `state.SetError`.
///     }
///
/// - The `GetBoundSpecData` method must set `*spec` to the context-bound
///   representation of the JSON specification of the driver.
///
///     Status GetBoundSpecData(SpecT<ContextBound>* spec) const;
///
/// - The `Derived` class must be default constructible.  Any required
///   initialization should be performed in the `Open` method.
///
/// Refer to `memory/memory_key_value_store.cc` for an example driver
/// implementation.
template <typename Derived, typename Parent = KeyValueStore>
class RegisteredKeyValueStore : public Parent {
 private:
  /// Encodes the cache key from the `BoundSpecData` representation.
  ///
  /// This is used by `RegisteredKeyValueStore::EncodeCacheKey` below and by
  /// `RegisteredKeyValueStoreBoundSpec::EncodeCacheKey`.
  ///
  /// The `BoundSpecData` template parameter is always equal to
  /// `Derived::SpecT<ContextBound>`, but is specified as a template parameter
  /// because `Derived` is incomplete when this class template is instantiated.
  template <typename BoundSpecData>
  static void EncodeCacheKeyImpl(std::string* out, const BoundSpecData& data) {
    internal::EncodeCacheKey(out, typeid(Derived));
    Derived::EncodeCacheKey(out, data);
  }

 public:
  void EncodeCacheKey(std::string* out) const override {
    using BoundSpecData = typename Derived::template SpecT<ContextBound>;
    // Generates a cache key by obtaining the `BoundSpecData` representation,
    // then computing the cache key from that.
    BoundSpecData bound_spec_data;
    if (auto status = static_cast<const Derived*>(this)->GetBoundSpecData(
            &bound_spec_data);
        !status.ok()) {
      // Could not obtain bound spec data.  Just use the default implementation
      // that encodes the exact object identity.
      return KeyValueStore::EncodeCacheKey(out);
    }
    EncodeCacheKeyImpl(out, bound_spec_data);
  }

  Result<KeyValueStoreSpec::Ptr> spec(
      const ContextSpecBuilder& context_builder) const override {
    using Spec = RegisteredKeyValueStoreSpec<Derived>;
    using SpecData = typename Derived::template SpecT<ContextUnbound>;
    using BoundSpecData = typename Derived::template SpecT<ContextBound>;
    // 1. Obtain the `BoundSpecData` representation of the `KeyValueStore.
    BoundSpecData bound_spec_data;
    TENSORSTORE_RETURN_IF_ERROR(
        static_cast<const Derived*>(this)->GetBoundSpecData(&bound_spec_data));
    // 2. "Unbind" to convert the `BoundSpecData` representation to the
    // `SpecData` representation.
    IntrusivePtr<Spec> spec(new Spec);
    // Since `KeyValueStoreSpec` can also specify context resource spec
    // overrides, construct a child `ContextSpecBuilder`.  Currently, if a
    // parent `context_builder` is specified, all shared context resources will
    // be specified in the parent `Context::Spec`.  Otherwise, all shared
    // context resources are specified in the `Context::Spec` embedded in the
    // `KeyValueStoreSpec`.
    auto child_builder = internal::ContextSpecBuilder::Make(context_builder);
    spec->context_spec_ = child_builder.spec();
    ContextBindingTraits<SpecData>::Unbind(&spec->data_, &bound_spec_data,
                                           child_builder);
    return spec;
  }

  Result<KeyValueStoreSpec::BoundPtr> GetBoundSpec() const override {
    using BoundSpec = RegisteredKeyValueStoreBoundSpec<Derived>;
    IntrusivePtr<BoundSpec> spec(new BoundSpec);
    TENSORSTORE_RETURN_IF_ERROR(
        static_cast<const Derived*>(this)->GetBoundSpecData(&spec->data_));
    return spec;
  }

 private:
  template <typename>
  friend class KeyValueStoreDriverRegistration;
  template <typename>
  friend class RegisteredKeyValueStoreSpec;
  template <typename>
  friend class RegisteredKeyValueStoreBoundSpec;
};

/// Parameter type for the static `Open` method that driver types inherited from
/// `RegisteredKeyValueStore` must implement.  For asynchronous open
/// implementations, this type may be copied and the copy retained until the
/// operation completes.
template <typename Derived>
class KeyValueStoreOpenState {
  template <typename, typename>
  friend class RegisteredKeyValueStore;

  template <typename>
  friend class RegisteredKeyValueStoreBoundSpec;

 public:
  using BoundSpecData = typename Derived::template SpecT<ContextBound>;

  /// Returns the promise that must be marked ready to indicate the open has
  /// completed.  The result is initialized in a success state with a copy of
  /// `driver()`, such that when the last reference to the `promise` is
  /// released, the promise is marked ready and the open is considered to have
  /// completed successfully.  The result should only be changed to indicate an
  /// error.
  const Promise<KeyValueStore::Ptr>& promise() const { return promise_; }

  /// Sets an error on the promise, indicating that the open failed.
  void SetError(Status status) { promise_.SetResult(std::move(status)); }

  /// Returns a reference to the `Driver` being opened.
  Derived& driver() const { return *driver_; }

  /// Returns a reference to the bound spec.
  const BoundSpecData& spec() const { return bound_spec_->data_; }

 private:
  KeyValueStore::PtrT<Derived> driver_;
  Promise<KeyValueStore::Ptr> promise_;
  IntrusivePtr<const RegisteredKeyValueStoreBoundSpec<Derived>> bound_spec_;
};

template <typename Derived>
class RegisteredKeyValueStoreBoundSpec : public KeyValueStoreSpec::Bound {
  friend class KeyValueStoreOpenState<Derived>;
  using SpecData = typename Derived::template SpecT<ContextUnbound>;
  using BoundSpecData = typename Derived::template SpecT<ContextBound>;

 public:
  KeyValueStoreSpec::Ptr Unbind(
      const internal::ContextSpecBuilder& context_builder) const override {
    IntrusivePtr<RegisteredKeyValueStoreSpec<Derived>> spec(
        new RegisteredKeyValueStoreSpec<Derived>);
    auto child_builder = internal::ContextSpecBuilder::Make(context_builder);
    spec->context_spec_ = child_builder.spec();
    ContextBindingTraits<SpecData>::Unbind(&spec->data_, &data_, child_builder);
    return spec;
  }

  void EncodeCacheKey(std::string* out) const override {
    Derived::RegisteredKeyValueStore::EncodeCacheKeyImpl(out, data_);
  }

 private:
  Future<KeyValueStore::Ptr> DoOpen() const override {
    KeyValueStoreOpenState<Derived> open_state;
    open_state.bound_spec_.reset(this);
    open_state.driver_.reset(new Derived);
    auto [promise, future] =
        PromiseFuturePair<KeyValueStore::Ptr>::Make(open_state.driver_);
    open_state.promise_ = std::move(promise);
    Derived::Open(std::move(open_state));
    return future;
  }

  friend class RegisteredKeyValueStoreSpec<Derived>;
  template <typename, typename>
  friend class RegisteredKeyValueStore;
  BoundSpecData data_;
};

template <typename Derived>
class RegisteredKeyValueStoreSpec : public KeyValueStoreSpec {
  using BoundSpec = RegisteredKeyValueStoreBoundSpec<Derived>;
  using SpecData = typename Derived::template SpecT<ContextUnbound>;

 public:
  Result<KeyValueStoreSpec::BoundPtr> Bind(
      const Context& context) const override {
    Context child_context(context_spec_, context);
    IntrusivePtr<BoundSpec> bound_spec(new BoundSpec);
    TENSORSTORE_RETURN_IF_ERROR(ContextBindingTraits<SpecData>::Bind(
        &data_, &bound_spec->data_, child_context));
    return bound_spec;
  }

  Result<KeyValueStoreSpec::Ptr> Convert(
      const KeyValueStoreSpec::RequestOptions& options) const override {
    IntrusivePtr<RegisteredKeyValueStoreSpec> new_spec(
        new RegisteredKeyValueStoreSpec);
    new_spec->data_ = data_;
    new_spec->context_spec_ = context_spec_;
    TENSORSTORE_RETURN_IF_ERROR(
        Derived::ConvertSpec(&new_spec->data_, options));
    return new_spec;
  }

  SpecData data_;
};

/// Registers a KeyValueStore driver implementation.
///
/// Example usage:
///
///     class MyDriver : public RegisteredKeyValueStore<MyDriver> {
///       // ...
///     };
///
///     const KeyValueStoreDriverRegistration<MyDriver> registration;
///
template <typename Derived>
class KeyValueStoreDriverRegistration {
 public:
  KeyValueStoreDriverRegistration() {
    GetKeyValueStoreDriverRegistry()
        .Register<RegisteredKeyValueStoreSpec<Derived>>(
            Derived::id, json_binding::Projection(
                             &RegisteredKeyValueStoreSpec<Derived>::data_,
                             Derived::json_binder));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_REGISTRY_H_
