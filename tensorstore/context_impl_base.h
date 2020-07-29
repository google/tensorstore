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

#ifndef TENSORSTORE_CONTEXT_IMPL_BASE_H_
#define TENSORSTORE_CONTEXT_IMPL_BASE_H_

/// \file
///
/// Context implementation details and forward declarations required by
/// `context.h`.

#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {
class ContextResourceCreationContext;
class ContextSpecBuilder;
}  // namespace internal
namespace internal_context {

/// Provides access to implementation details of `Context`.  For internal use
/// only.
class Access {
 public:
  template <typename T>
  static auto impl(T&& x) -> decltype((static_cast<T&&>(x).impl_)) {
    return static_cast<T&&>(x).impl_;
  }
};

/// Immutable representation of `Context::Spec` held via reference-counted
/// pointer.  Holds a hash table of pointers to `ContextResourceSpecImplBase`
/// objects that were obtained from a context JSON specification.
class ContextSpecImpl;

class ContextResourceImplBase;

class ContextResourceSpecImplBase;

/// Representation of `Context` held via reference-counted pointer.  Logically
/// immutable.  References a `ContextSpecImpl` and holds a hash table of
/// pointers to `ContextResourceContainer` objects, which represent
/// lazily-constructed resources corresponding to `ContextImpl`
class ContextImpl;

/// Manages the lazy construction of a context resource.
class ContextResourceContainer;

/// Representation of a `ContextSpecBuilder` held via reference-counted pointer.
class BuilderImpl;

void intrusive_ptr_increment(BuilderImpl* p);
void intrusive_ptr_decrement(BuilderImpl* p);

void intrusive_ptr_increment(ContextSpecImpl* p);
void intrusive_ptr_decrement(ContextSpecImpl* p);

void intrusive_ptr_increment(ContextImpl* p);
void intrusive_ptr_decrement(ContextImpl* p);

/// IntrusivePtr traits used by `ContextResourceImplPtr`.
struct ContextResourceImplWeakPtrTraits {
  template <typename>
  using pointer = ContextResourceImplBase*;
  static void increment(ContextResourceImplBase* p);
  static void decrement(ContextResourceImplBase* p);
};

struct ContextResourceImplStrongPtrTraits {
  template <typename>
  using pointer = ContextResourceImplBase*;
  static void increment(ContextResourceImplBase* p);
  static void decrement(ContextResourceImplBase* p);
};

using BuilderImplPtr = internal::IntrusivePtr<BuilderImpl>;
using ContextSpecImplPtr = internal::IntrusivePtr<ContextSpecImpl>;
using ContextImplPtr = internal::IntrusivePtr<ContextImpl>;

using ContextResourceSpecImplPtr =
    internal::IntrusivePtr<ContextResourceSpecImplBase>;

using ContextResourceImplWeakPtr =
    internal::IntrusivePtr<ContextResourceImplBase,
                           ContextResourceImplWeakPtrTraits>;

using ContextResourceImplStrongPtr =
    internal::IntrusivePtr<ContextResourceImplBase,
                           ContextResourceImplStrongPtrTraits>;

/// Abstract base class for representing a registered context resource provider.
///
/// For a given `Traits` type, `ContextResourceProviderImpl<Traits>` is the
/// derived class that implements this interface.
class ContextResourceProviderImplBase {
 public:
  absl::string_view id_;
  virtual ContextResourceSpecImplPtr Default() const = 0;
  virtual Result<ContextResourceSpecImplPtr> FromJson(
      const ::nlohmann::json& j, ContextFromJsonOptions options) const = 0;
  virtual ContextResourceSpecImplPtr GetSpec(
      const ContextResourceImplBase* resource,
      const internal::ContextSpecBuilder& spec_builder) const = 0;
  virtual void AcquireContextReference(
      ContextResourceImplBase* resource) const = 0;
  virtual void ReleaseContextReference(
      ContextResourceImplBase* resource) const = 0;
  virtual ~ContextResourceProviderImplBase();
};

/// Representation of a `Context::ResourceSpec` held via reference-counted
/// pointer.  Also used to represent a resource specification contained in a
/// `Context::Spec`.
///
/// For each resource provider type, the template machinery in the derived class
/// `ContextResourceProviderImpl<Traits>::SpecImpl` is the derived class that
/// holds the actual `Traits::Spec` object.
class ContextResourceSpecImplBase
    : public internal::AtomicReferenceCount<ContextResourceSpecImplBase> {
 public:
  virtual Result<ContextResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context) = 0;

  virtual Result<::nlohmann::json> ToJson(ContextToJsonOptions options) = 0;

  virtual ~ContextResourceSpecImplBase();

  /// Specifies the key associated with this resource spec.  Must be empty if,
  /// and only if, the resource spec is specified inline outside a context spec.
  /// If `key_` is not empty, no `Context::Spec` objects may point to this
  /// resource spec.
  std::string key_;

  /// Provider associated with this spec.
  const ContextResourceProviderImplBase* provider_;

  bool is_default_ = false;
};

/// Representation of `Context::Resource` held via reference-counted pointer.
class ContextResourceImplBase
    : public internal::AtomicReferenceCount<ContextResourceImplBase> {
 public:
  ContextResourceImplBase(ContextResourceSpecImplPtr spec = {})
      : spec_(std::move(spec)) {}
  virtual ~ContextResourceImplBase();
  ContextResourceSpecImplPtr spec_;
};

/// For each resource provider type, `ContextResourceImpl<Provider>` is the
/// derived class of `ContextResourceImplBase` that holds the actual
/// `Provider::Resource` object.
///
/// This is parameterized by the `Provider` type rather than the `Traits` type
/// because it is needed by `Context::Resource`, which is also parameterized by
/// the `Provider` type rather than the `Traits` type.  The `Traits` type for a
/// given `Provider` type may only be defined in a separate source file that
/// actually registers the provider; code that merely uses the resource provider
/// does not need it.
template <typename Provider>
class ContextResourceImpl : public ContextResourceImplBase {
 public:
  using Resource = typename Provider::Resource;
  template <typename... Arg>
  ContextResourceImpl(ContextResourceSpecImplPtr spec, Arg&&... arg)
      : ContextResourceImplBase(std::move(spec)),
        value_(std::forward<Arg>(arg)...) {}

  Resource value_;
};

/// Implements the `ContextResourceProviderImplBase` type for a given context
/// resource provider traits type.
template <typename Traits>
class ContextResourceProviderImpl : public ContextResourceProviderImplBase {
 public:
  using Provider = typename Traits::Provider;
  template <typename... U>
  ContextResourceProviderImpl(U&&... arg) : traits_(std::forward<U>(arg)...) {
    id_ = Provider::id;
  }
  using Spec = typename Traits::Spec;
  using Resource = typename Provider::Resource;
  using ResourceImpl = ContextResourceImpl<Provider>;
  class SpecImpl : public ContextResourceSpecImplBase {
   public:
    template <typename... U>
    SpecImpl(U&&... arg) : value_(std::forward<U>(arg)...) {}

    Result<ContextResourceImplStrongPtr> CreateResource(
        const internal::ContextResourceCreationContext& creation_context)
        override {
      auto result = static_cast<const ContextResourceProviderImpl*>(provider_)
                        ->traits_.Create(value_, creation_context);
      if (!result) {
        return std::move(result).status();
      }
      return ContextResourceImplStrongPtr(new ResourceImpl(
          ContextResourceSpecImplPtr(this), std::move(*result)));
    }

    Result<::nlohmann::json> ToJson(ContextToJsonOptions options) override {
      return internal::json_binding::ToJson(
          value_,
          static_cast<const ContextResourceProviderImpl*>(provider_)
              ->traits_.JsonBinder(),
          options);
    }

    Spec value_;
  };

  ContextResourceSpecImplPtr Default() const override {
    return ContextResourceSpecImplPtr(new SpecImpl(traits_.Default()));
  }

  Result<ContextResourceSpecImplPtr> FromJson(
      const ::nlohmann::json& j,
      ContextFromJsonOptions options) const override {
    auto result = internal::json_binding::FromJson<Spec>(
        j, traits_.JsonBinder(), options);
    if (!result) return std::move(result).status();
    return ContextResourceSpecImplPtr(new SpecImpl(std::move(*result)));
  }

  void AcquireContextReference(
      ContextResourceImplBase* resource) const override {
    traits_.AcquireContextReference(
        &(static_cast<ResourceImpl*>(resource)->value_));
  }
  void ReleaseContextReference(
      ContextResourceImplBase* resource) const override {
    traits_.ReleaseContextReference(
        &(static_cast<ResourceImpl*>(resource)->value_));
  }

  ContextResourceSpecImplPtr GetSpec(
      const ContextResourceImplBase* resource,
      const internal::ContextSpecBuilder& spec_builder) const override {
    ContextResourceSpecImplPtr spec_impl(new SpecImpl(traits_.GetSpec(
        static_cast<const ResourceImpl*>(resource)->value_, spec_builder)));
    spec_impl->provider_ = resource->spec_->provider_;
    spec_impl->is_default_ = resource->spec_->is_default_;
    spec_impl->key_ = resource->spec_->key_;
    return spec_impl;
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Traits traits_;
};

void RegisterContextResourceProvider(
    std::unique_ptr<const ContextResourceProviderImplBase> provider);

Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJson(
    absl::string_view provider_id, const ::nlohmann::json& j,
    ContextFromJsonOptions options);

ContextResourceSpecImplPtr DefaultContextResourceSpec(
    absl::string_view provider_id);

Result<ContextResourceImplWeakPtr> GetResource(
    ContextImpl* context, ContextResourceSpecImplBase* spec,
    ContextResourceContainer* trigger);

ContextResourceSpecImplPtr AddResource(
    const internal::ContextSpecBuilder& builder,
    ContextResourceImplBase* resource);

}  // namespace internal_context
}  // namespace tensorstore

#endif  // TENSORSTORE_CONTEXT_IMPL_BASE_H_
