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

#ifndef TENSORSTORE_CONTEXT_RESOURCE_PROVIDER_H_
#define TENSORSTORE_CONTEXT_RESOURCE_PROVIDER_H_

/// \file Interfaces for defining context resource providers.
///
/// To define a new context resource provider, you must define both a `Provider`
/// class (that satisfies the `ContextResourceProviderConcept` concept) and a
/// corresponding `Traits` class (that satisfies the
/// `ContextResourceTraitsConcept<Provider>` concept).  The `Provider` specifies
/// the public interface of the resource, while the `Traits` class specifies the
/// implementation.
///
/// For resource providers that are defined in a header file and are not local
/// to a particular source file, it is recommended to define the `Provider` type
/// in the header file, and to define the separate corresponding `Traits` type
/// in the source file that registers the provider.  For resource providers that
/// are local to a particular source file, it is simpler for the `Provider` type
/// to also serve as the `Traits` type.

#include "tensorstore/context.h"
#include "tensorstore/internal/json_binding/bindable.h"

namespace tensorstore {
namespace internal {

template <typename Spec>
using AnyContextResourceJsonBinder =
    internal_json_binding::StaticBinder<Spec, Context::FromJsonOptions,
                                        Context::ToJsonOptions>;

/// Base class for defining context resources.
///
/// To define a new context resource type, create a class that publicly inherits
/// from `ContextResourceTraits` as described by the
/// `ContextResourceTraitsConcept`.
template <typename Provider_>
class ContextResourceTraits {
 public:
  using Provider = Provider_;
  using ToJsonOptions = Context::ToJsonOptions;
  using FromJsonOptions = Context::FromJsonOptions;

  template <typename Resource>
  static void AcquireContextReference(Resource& obj) {}
  template <typename Resource>
  static void ReleaseContextReference(Resource& obj) {}
  template <typename Spec>
  static void UnbindContext(Spec& spec,
                            const internal::ContextSpecBuilder& builder) {}
};

/// Registers a context resource type.
///
/// \tparam Traits A model of `ContextResourceTraitsConcept`.
template <typename Traits>
class ContextResourceRegistration {
 public:
  template <typename... U>
  ContextResourceRegistration(U&&... arg) {
    using Impl = internal_context::ResourceProviderImpl<Traits>;
    internal_context::RegisterContextResourceProvider(
        std::make_unique<Impl>(std::forward<U>(arg)...));
  }
};

/// Requirements for`ContextResourceProviderConcept`:
class ContextResourceProviderConcept {
  /// Required. Specifies the resource identifier.  Must be unique within the
  /// program.
  static constexpr char id[] = "...";

  /// Specifies the resource type that may be accessed via const reference from
  /// a `Context::Resource<Provider>` object.
  ///
  /// Users of this resource can access the constructed `Resource` object by
  /// const reference using a `Context::Resource` handle.
  struct Resource {
    /// ...
    ///
    /// If this resource type depends on other context resources, the
    /// `Resource` class should include a `Context::Resource` member for
    /// each dependency.
  };
};

/// Requirements for `ContextResourceTraitsConcept`:
///
/// A `Traits` type satisfying the `ContextResourceTraitsConcept<Provider>`
/// concept must inherit publicly from `ContextResourceTraits<Provider>` and
/// define the members documented below.
///
/// Additionally, the `Traits` type must be registered by constructing a
/// `ContextResourceRegistration` object, usually at global/namespace scope to
/// ensure registration happens as part of static initialization.
///
/// The traits type does not have to be copyable or movable.  A single instance
/// of the traits type is constructed when it is registered, and it may have
/// non-static data members (e.g. for a shared value used by default).  In the
/// simple case, the traits type is an empty class and all of the required
/// methods are static.
template <typename Provider>
class ContextResourceTraitsConcept : public ContextResourceTraits<Provider> {
 public:
  /// Required. Specifies the type used for the validated representation of a
  /// resource specification created from a JSON object.
  ///
  /// The `Spec` type is not exposed publicly; it is only for internal use by
  /// the context resource provider.
  struct Spec {
    /// ...
    ///
    /// If this resource type depends on other resources, the `Spec` class
    /// should include a `Context::Resource` member for each dependency.
  };

  /// Required.  Returns the default `Spec`.
  Spec Default() const;

  /// Required.  Returns a JSON binder (see
  /// `tensorstore/internal/json_binding/json_binding.h`) for `Spec`.
  AnyContextResourceJsonBinder<Spec> JsonBinder() const;

  /// Creates a resource from a `Spec` object.
  ///
  /// If this resource depends on other context resources, they may be
  /// obtained using `context`.
  Result<typename Provider::Resource> Create(
      const Spec& spec, ContextResourceCreationContext context) const;

  /// Converts a `Resource` object back to a `Spec`.
  ///
  /// This is essentially the inverse of `Create`.
  ///
  /// If `resource` depends on other context resources, they may be converted
  /// back to resource spec objects using `builder`.
  Spec GetSpec(const typename Provider::Resource& resource,
               const ContextSpecBuilder& builder) const;

  /// Ensures all nested context resources are unbound, by calling
  /// `Context::Resource<T>::UnbindContext(builder)`.
  ///
  /// If there are no nested context resources, this need not be defined.
  void UnbindContext(Spec& spec, const ContextSpecBuilder& builder) const;

  /// Optional.  Increments the strong reference count associated with
  ///
  /// A `Resource` object is assumed to represent a weak reference by default.
  /// This method acquires an additional strong reference, if supported.  If
  /// this resource type does not distinguish between strong and weak
  /// references, this method need not be defined, in which case the default
  /// do-nothing implementation in the `ContextResourceProvider` base class is
  /// used.
  ///
  /// Resources contained within a `Context` object are held by strong
  /// reference; resources not contained within a `Context` object are held by
  /// weak reference.
  ///
  /// For example, a strong `CachePool` reference ensures that recently-used
  /// caches are retained even if they are not referenced.  Resources that
  /// behave like caches may need to distinguish between strong and weak
  /// references to prevent reference cycles.
  void AcquireContextReference(typename Provider::Resource& value) const;

  /// Optional.  Decrements the strong reference count.
  ///
  /// If this resource type does not distinguish between strong and weak
  /// references, this method need not be defined, in which case the default
  /// do-nothing implementation in the `ContextResourceProvider` base class is
  /// used.
  void ReleaseContextReference(typename Provider::Resource& value) const;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_CONTEXT_RESOURCE_PROVIDER_H_
