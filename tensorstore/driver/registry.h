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
/// the CRTP base `RegisteredDriver<Derived>`, and define a global constant of
/// type `DriverRegistration<Derived>` to register it.
///

#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal {

class UnregisteredDriverSpec;

using DriverRegistry =
    JsonRegistry<DriverSpec, Context::FromJsonOptions, Context::ToJsonOptions,
                 UnregisteredDriverSpec>;

/// Returns the global driver registry.
DriverRegistry& GetDriverRegistry();

template <typename Derived, typename Parent>
class RegisteredDriver;

template <typename T>
class RegisteredDriverOpener;

/// CRTP base class for `Driver` implementations that support a JSON
/// representation.
///
/// The `Derived` class must override all of the virtual methods of `Driver`,
/// except for `GetSpec` and `GetBoundSpec`, which are defined automatically.
/// In addition it must define the following members:
///
/// - The `SpecT` class template must inherit from `internal::DriverConstraints`
///   and includes as members the parameters and resources necessary to
///   create/open the driver.  Depending on the `MaybeBound` argument, which is
///   either `ContextUnbound` or `ContextBound`, it specifies either the
///   context-unbound or context-bound state of the parameters/resources.  The
///   dependence on `MaybeBound` permits `Context::ResourceSpec` objects (as
///   used for the JSON representation) to be converted automatically to/from
///   `Context::Resource` objects (as used by the driver implementation).
///
///   It must define an `ApplyMembers` method for compatibility with
///   `ContextBindingTraits` (refer to `tensorstore/internal/context_binding.h`
///   for details).
///
///   Members of `SpecT` should be referenced in the `json_binder`
///   implementation, as noted below.
///
///     template <template <typename> class MaybeBound>
///     struct SpecT : public internal::DriverConstraints {
///       // Example members:
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
/// - The static `ConvertSpec` method should apply any modifications requested
///   in `options` in place to `*spec`.
///
///     static absl::Status ConvertSpec(
///         SpecT<ContextUnbound>* spec, const SpecRequestOptions& options);
///
/// - The static `Open` method is called to initiate opening the driver.  This
///   is called by `DriverSpec::Bound::Open`.  Note that
///   `RegisteredDriverOpener` is a CRTP class template parameterized by the
///   `Derived` driver type, and serves as a reference-counted smart pointer to
///   the `SpecT<ContextBound>`.
///
///     static Future<internal::Driver::ReadWriteHandle> Open(
///         internal::OpenTransactionPtr transaction,
///         internal::RegisteredDriverOpener<Derived> spec,
///         ReadWriteMode read_write_mode) {
///       // Access the `SpecT<ContextBound>` representation as `*spec`.
///       auto [promise, future] = PromiseFuturePair<Derived>::Make();
///       // ...
///       return future;
///     }
///
/// - The `GetBoundSpecData` method must set `*spec` to the context-bound
///   representation of the JSON specification of the TensorStore defined by
///   this driver and the specified `transform`.  The returned
///   `IndexTransform<>` must have the same domain as `transform`, but the
///   returned `IndexTransform<>` may or may not equal `transform`.  For
///   example, the returned `transform` may be composed with another invertible
///   transform, or `*spec` may somehow incorporate part or all of the
///   transform.
///
///     Result<IndexTransform<>> GetBoundSpecData(
///         SpecT<ContextBound>* spec,
///         IndexTransformView<> transform) const;
///
/// \tparam Derived The derived driver type.
/// \tparam Parent The super class, must equal or be derived from
///     `internal::Driver` (for example, may be `ChunkCacheDriver`).
template <typename Derived, typename Parent>
class RegisteredDriver : public Parent {
 public:
  using Parent::Parent;

  Result<TransformedDriverSpec<>> GetSpec(
      internal::OpenTransactionPtr transaction, IndexTransformView<> transform,
      const SpecRequestOptions& options,
      const ContextSpecBuilder& context_builder) override {
    using SpecData = typename Derived::template SpecT<ContextUnbound>;
    using BoundSpecData = typename Derived::template SpecT<ContextBound>;
    // 1. Obtain the `BoundSpecData` representation of the `Driver`.
    BoundSpecData bound_spec_data;
    TransformedDriverSpec<> transformed_spec;
    TENSORSTORE_ASSIGN_OR_RETURN(
        transformed_spec.transform_spec,
        static_cast<Derived*>(this)->GetBoundSpecData(
            std::move(transaction), &bound_spec_data, transform));
    // 2. "Unbind" to convert the `BoundSpecData` representation to the
    // `SpecData` representation.
    IntrusivePtr<DriverSpecImpl> spec(new DriverSpecImpl);
    // Since `DriverSpec` can also specify context resource spec overrides,
    // construct a child `ContextSpecBuilder`.  Currently, if a parent
    // `context_builder` is specified, all shared context resources will be
    // specified in the parent `Context::Spec`.  Otherwise, all shared context
    // resources are specified in the `Context::Spec` embedded in the
    // `DriverSpec`.
    auto child_builder = internal::ContextSpecBuilder::Make(context_builder);
    spec->context_spec_ = child_builder.spec();
    ContextBindingTraits<SpecData>::Unbind(&spec->data_, &bound_spec_data,
                                           child_builder);
    // 3. Convert the `SpecData` using `options`.
    TENSORSTORE_RETURN_IF_ERROR(Derived::ConvertSpec(&spec->data_, options));
    transformed_spec.driver_spec = std::move(spec);
    return transformed_spec;
  }

  Result<TransformedDriverSpec<ContextBound>> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    IntrusivePtr<typename DriverSpecImpl::Bound> bound_spec(
        new typename DriverSpecImpl::Bound);
    TransformedDriverSpec<ContextBound> transformed_spec;
    TENSORSTORE_ASSIGN_OR_RETURN(
        transformed_spec.transform_spec,
        static_cast<Derived*>(this)->GetBoundSpecData(
            std::move(transaction), &bound_spec->data_, transform));
    transformed_spec.driver_spec = std::move(bound_spec);
    return transformed_spec;
  }

 private:
  class DriverSpecImpl : public internal::DriverSpec {
    using SpecData = typename Derived::template SpecT<ContextUnbound>;
    using BoundSpecData = typename Derived::template SpecT<ContextBound>;
    static_assert(std::is_base_of_v<internal::DriverConstraints, SpecData>);
    static_assert(
        std::is_base_of_v<internal::DriverConstraints, BoundSpecData>);

   public:
    Result<DriverSpec::Ptr> Convert(
        const SpecRequestOptions& options) override {
      IntrusivePtr<DriverSpecImpl> new_spec(new DriverSpecImpl);
      new_spec->data_ = data_;
      new_spec->context_spec_ = context_spec_;
      TENSORSTORE_RETURN_IF_ERROR(
          Derived::ConvertSpec(&new_spec->data_, options));
      return new_spec;
    }

    Result<Driver::BoundSpec::Ptr> Bind(Context context) const override {
      IntrusivePtr<Bound> bound_spec(new Bound);
      Context child_context(context_spec_, context);
      TENSORSTORE_RETURN_IF_ERROR(ContextBindingTraits<SpecData>::Bind(
          &data_, &bound_spec->data_, child_context));
      return bound_spec;
    }

    DriverConstraints& constraints() override { return data_; }

    class Bound : public internal::DriverSpec::Bound {
     public:
      Future<Driver::ReadWriteHandle> Open(
          OpenTransactionPtr transaction,
          ReadWriteMode read_write_mode) const override {
        RegisteredDriverOpener<BoundSpecData> data_ptr;
        data_ptr.owner_.reset(this);
        data_ptr.ptr_ = &data_;
        return tensorstore::MapFutureError(
            InlineExecutor{},
            [](const Status& status) {
              return tensorstore::MaybeAnnotateStatus(
                  status,
                  tensorstore::StrCat("Error opening ",
                                      tensorstore::QuoteString(Derived::id),
                                      " driver"));
            },
            Derived::Open(std::move(transaction), std::move(data_ptr),
                          read_write_mode));
      }

      DriverSpecPtr Unbind(
          const ContextSpecBuilder& context_builder) const override {
        auto child_builder =
            internal::ContextSpecBuilder::Make(context_builder);
        IntrusivePtr<DriverSpecImpl> spec(new DriverSpecImpl);
        spec->context_spec_ = child_builder.spec();
        ContextBindingTraits<SpecData>::Unbind(&spec->data_, &data_,
                                               child_builder);
        return spec;
      }
      const BoundSpecData& data() const { return data_; }

      BoundSpecData data_;
    };

    SpecData data_;
  };

  template <typename>
  friend class DriverRegistration;
};

/// Smart pointer to `const T` owned by a `DriverSpec::Bound`.
///
/// This serves as a parameter to the `Open` method that `RegisteredDriver`
/// implementations must define and provides access to the bound spec data.
template <typename T>
class RegisteredDriverOpener {
 public:
  using element_type = const T;

  RegisteredDriverOpener() = default;

  template <typename U, typename = std::enable_if_t<
                            std::is_convertible_v<const U*, const T*>>>
  RegisteredDriverOpener(RegisteredDriverOpener<U> other)
      : owner_(std::move(other.owner_)), ptr_(other.ptr_) {}

  const T* get() const { return ptr_; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

  explicit operator bool() const { return static_cast<bool>(ptr_); }

 private:
  template <typename>
  friend class RegisteredDriverOpener;

  template <typename, typename>
  friend class RegisteredDriver;

  internal::DriverSpec::Bound::Ptr owner_;
  const T* ptr_;
};

/// Registers a driver implementation.
///
/// Example usage:
///
///     class MyDriver : public RegisteredDriver<MyDriver> {
///       // ...
///     };
///
///     const DriverRegistration<MyDriver> registration;
///
template <typename Derived>
class DriverRegistration {
 public:
  DriverRegistration() {
    using Spec = typename Derived::DriverSpecImpl;
    GetDriverRegistry().Register<Spec>(
        Derived::id,
        json_binding::Projection(&Spec::data_, Derived::json_binder));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_REGISTRY_H_
