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
#include <string_view>
#include <type_traits>

#include "tensorstore/context.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal {

using DriverRegistry = JsonRegistry<DriverSpec, JsonSerializationOptions,
                                    JsonSerializationOptions, DriverSpecPtr>;

/// Returns the global driver registry.
DriverRegistry& GetDriverRegistry();

template <typename Derived, typename Parent>
class RegisteredDriver;

template <typename T>
class RegisteredDriverOpener;

/// Methods of `RegisteredDriver` that don't depend on template parameters.
///
/// These serve as default implementations that may be overridden by derived
/// classes.
class RegisteredDriverBase {
 public:
  static Result<IndexDomain<>> SpecGetDomain(const DriverSpecCommonData& spec);

  static Result<ChunkLayout> SpecGetChunkLayout(
      const DriverSpecCommonData& spec);

  static Result<CodecSpec::Ptr> SpecGetCodec(const DriverSpecCommonData& spec);

  static Result<SharedArray<const void>> SpecGetFillValue(
      const DriverSpecCommonData& spec, IndexTransformView<> transform);
};

/// CRTP base class for `Driver` implementations that support a JSON
/// representation.
///
/// The `Derived` class must override all of the virtual methods of `Driver`,
/// except for `GetSpec` and `GetBoundSpec`, which are defined automatically.
/// In addition it must define the following members:
///
/// - The `SpecData` class must inherit from `internal::DriverSpecCommonData`
///   and includes as members the parameters and resources necessary to
///   create/open the driver.
///
///   It must define an `ApplyMembers` method for compatibility with
///   `ContextBindingTraits` (refer to `tensorstore/internal/context_binding.h`
///   for details).
///
///   Members of `SpecData` should be referenced in the `json_binder`
///   implementation, as noted below.
///
///     struct SpecData : public internal::DriverSpecCommonData {
///       // Example members:
///       int mem1;
///       Context::Resource<SomeResource> mem2;
///
///       // For compatibility with `ContextBindingTraits`.
///       constexpr static auto ApplyMembers = [](auto& x, auto f) {
///         return f(internal::BaseCast<internal::DriverSpecCommonData>(x),
///                  x.mem1, f.mem2);
///       };
///     };
///
/// - The `json_binder` member must be a JSON object binder for `SpecData`.
///   This should handle converting each member of `SpecData` to/from the JSON
///   representation.
///
///     constexpr static auto json_binder = jb::Object(
///         jb::Member("mem1", jb::Projection(&SpecData::mem1)),
///         jb::Member("mem2", jb::Projection(&SpecData::mem2)));
///
/// - The static `ApplyOptions` method should apply any modifications requested
///   in `options` in place to `spec`.
///
///     static absl::Status ApplyOptions(
///         SpecData& spec, SpecConvertOptions&& options);
///
/// - The static `GetEffectiveSchema` method should return the effective schema.
///
///     static Result<Schema> GetEffectiveSchema(
///         const SpecData &spec);
///
/// - The static `Open` method is called to initiate opening the driver.  This
///   is called by `DriverSpec::Open`.  Note that `RegisteredDriverOpener` is a
///   CRTP class template parameterized by the `Derived` driver type, and serves
///   as a reference-counted smart pointer to the `SpecData`.
///
///     static Future<internal::Driver::Handle> Open(
///         internal::OpenTransactionPtr transaction,
///         internal::RegisteredDriverOpener<Derived> spec,
///         ReadWriteMode read_write_mode) {
///       // Access the `SpecData` representation as `*spec`.
///       auto [promise, future] = PromiseFuturePair<Derived>::Make();
///       // ...
///       return future;
///     }
///
/// - The `GetBoundSpecData` method must set `spec` to the context-bound
///   representation of the JSON specification of the TensorStore defined by
///   this driver and the specified `transform`.  The returned
///   `IndexTransform<>` must have the same domain as `transform`, but the
///   returned `IndexTransform<>` may or may not equal `transform`.  For
///   example, the returned `transform` may be composed with another invertible
///   transform, or `*spec` may somehow incorporate part or all of the
///   transform.
///
///     Result<IndexTransform<>> GetBoundSpecData(
///         internal::OpenTransactionPtr transaction,
///         SpecData& spec,
///         IndexTransformView<> transform) const;
///
/// \tparam Derived The derived driver type.
/// \tparam Parent The super class, must equal or be derived from
///     `internal::Driver` (for example, may be `ChunkCacheDriver`).
template <typename Derived, typename Parent>
class RegisteredDriver : public Parent, RegisteredDriverBase {
  class DriverSpecImpl;

 public:
  using Parent::Parent;

  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    IntrusivePtr<DriverSpecImpl> spec(new DriverSpecImpl);
    internal_context::Access::context_binding_state(*spec) =
        ContextBindingState::bound;
    TransformedDriverSpec transformed_spec;
    TENSORSTORE_ASSIGN_OR_RETURN(
        transformed_spec.transform,
        static_cast<Derived*>(this)->GetBoundSpecData(std::move(transaction),
                                                      spec->data_, transform));
    transformed_spec.driver_spec = std::move(spec);
    return transformed_spec;
  }

  /// Builder class that may be used to construct a DriverSpec directly for the
  /// derived driver without parsing JSON.
  ///
  /// Example usage:
  ///
  ///     auto driver_spec_builder = Derived::DriverSpecBuilder::Make();
  ///     driver_spec_builder->some_property = value;
  ///     DriverSpec::Ptr driver_spec = std::move(driver_spec_builder).Build();
  class DriverSpecBuilder {
   public:
    using SpecData = typename Derived::SpecData;

    static DriverSpecBuilder Make() {
      DriverSpecBuilder builder;
      builder.impl_.reset(new DriverSpecImpl);
      return builder;
    }

    DriverSpecBuilder() = default;
    DriverSpecBuilder(DriverSpecBuilder&&) = default;
    DriverSpecBuilder(const DriverSpecBuilder&) = delete;
    DriverSpecBuilder& operator=(const DriverSpecBuilder&) = delete;
    DriverSpecBuilder& operator=(DriverSpecBuilder&&) = default;

    ContextBindingState& context_binding_state() {
      return internal_context::Access::context_binding_state(*impl_);
    }

    SpecData& operator*() { return impl_->data_; }
    SpecData* operator->() { return &impl_->data_; }
    DriverSpec::Ptr Build() && { return std::move(impl_); }

   private:
    IntrusivePtr<DriverSpecImpl> impl_;
  };

 private:
  class DriverSpecImpl : public internal::DriverSpec {
    using SpecData = typename Derived::SpecData;
    static_assert(std::is_base_of_v<internal::DriverSpecCommonData, SpecData>);

   public:
    DriverSpec::Ptr Clone() const override {
      IntrusivePtr<DriverSpecImpl> new_spec(new DriverSpecImpl);
      new_spec->data_ = data_;
      internal_context::Access::context_spec(*new_spec) =
          internal_context::Access::context_spec(*this);
      return new_spec;
    }

    absl::Status ApplyOptions(SpecOptions&& options) override {
      assert(use_count() == 1);
      return Derived::ApplyOptions(data_, std::move(options));
    }

    Result<IndexDomain<>> GetDomain() const override {
      return Derived::SpecGetDomain(data_);
    }

    Result<ChunkLayout> GetChunkLayout() const override {
      return Derived::SpecGetChunkLayout(data_);
    }

    Result<CodecSpec::Ptr> GetCodec() const override {
      return Derived::SpecGetCodec(data_);
    }

    Result<SharedArray<const void>> GetFillValue(
        IndexTransformView<> transform) const override {
      return Derived::SpecGetFillValue(data_, transform);
    }

    absl::Status BindContext(const Context& context) override {
      return ContextBindingTraits<SpecData>::Bind(data_, context);
    }

    void UnbindContext(const ContextSpecBuilder& context_builder) override {
      ContextBindingTraits<SpecData>::Unbind(data_, context_builder);
    }

    void StripContext() override {
      ContextBindingTraits<SpecData>::Strip(data_);
    }

    Future<Driver::Handle> Open(OpenTransactionPtr transaction,
                                ReadWriteMode read_write_mode) const override {
      RegisteredDriverOpener<SpecData> data_ptr;
      data_ptr.owner_.reset(this);
      data_ptr.ptr_ = &data_;
      return tensorstore::MapFutureError(
          InlineExecutor{},
          [](const Status& status) {
            return tensorstore::MaybeAnnotateStatus(
                status, tensorstore::StrCat(
                            "Error opening ",
                            tensorstore::QuoteString(Derived::id), " driver"));
          },
          Derived::Open(std::move(transaction), std::move(data_ptr),
                        read_write_mode));
    }

    SpecData& data() override { return data_; }

    SpecData data_;
  };

  template <typename>
  friend class DriverRegistration;
};

/// Smart pointer to `const T` owned by a `DriverSpec`.
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

  internal::DriverSpec::Ptr owner_;
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
        internal_json_binding::Projection(&Spec::data_, Derived::json_binder));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_REGISTRY_H_
