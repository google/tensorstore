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

#ifndef TENSORSTORE_DRIVER_DRIVER_SPEC_H_
#define TENSORSTORE_DRIVER_DRIVER_SPEC_H_

#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

class DriverSpec;

using DriverSpecPtr = IntrusivePtr<const DriverSpec>;

/// Abstract base class representing a TensorStore driver specification, for
/// creating a `Driver` from a JSON representation.
///
/// A `DriverSpec` object specifies:
///
/// - The driver id (as a string, implicitly);
///
/// - Any driver-specific options, such as any necessary `KeyValueStore::Spec`
///   objects, relevant paths, and `Context::Resource` objects for any necessary
///   concurrency pools or caches.
///
/// - A `Context::Spec` with context resource specifications that may be
///   referenced by driver-specific context resource specifications; these
///   context resource specifications override any resources provided by the
///   `Context` object used to bind/open the driver.
///
/// For each `Derived` driver implementation that supports a JSON
/// representation, `internal::RegisteredDriverSpec<Derived>` defined in
/// `registry.h` serves as the corresponding `DriverSpec` implementation.
class DriverSpec : public internal::AtomicReferenceCount<DriverSpec> {
 public:
  /// DriverSpec objects are logically immutable and always managed by
  /// reference-counted smart pointer.
  using Ptr = IntrusivePtr<const DriverSpec>;

  template <typename T>
  using PtrT = IntrusivePtr<T>;

  template <typename T, typename... U>
  static PtrT<T> Make(U&&... u) {
    return PtrT<T>(new T(std::forward<U>(u)...));
  }

  virtual ~DriverSpec();

  /// Returns a copy.  This is used prior to calling `ApplyOptions` for
  /// copy-on-write behavior.
  virtual Ptr Clone() const = 0;

  /// Modifies this `DriverSpec` according to `options`.  This must only be
  /// called if `use_count() == 1`.
  virtual absl::Status ApplyOptions(SpecOptions&& options) = 0;

  /// Indicates the binding state of the spec.
  ContextBindingState context_binding_state() const {
    return context_binding_state_;
  }

  /// Resolves any `Context` resources.
  ///
  /// \pre `use_count() == 1`.
  virtual absl::Status BindContext(const Context& context) = 0;

  /// Converts any bound context resources to unbound resource specs.
  ///
  /// \pre `use_count() == 1`.
  /// \param context_builder Optional.  Specifies a parent context spec builder,
  ///     if the returned `DriverSpec` is to be used in conjunction with a
  ///     parent context.  If specified, all required shared context resources
  ///     are recorded in the specified builder.  If not specified, required
  ///     shared context resources are recorded in the `Context::Spec` owned by
  ///     the returned `DriverSpec`.
  virtual void UnbindContext(const ContextSpecBuilder& context_builder) = 0;

  /// Converts any context resources to default context resource specs.
  ///
  /// \pre `use_count() == 1`.
  /// \param context_builder Optional.  Specifies a parent context spec builder,
  ///     if the returned `DriverSpec` is to be used in conjunction with a
  ///     parent context.  If specified, all required shared context resources
  ///     are recorded in the specified builder.  If not specified, required
  ///     shared context resources are recorded in the `Context::Spec` owned by
  ///     the returned `DriverSpec`.
  virtual void StripContext() = 0;

  /// Opens the driver.
  ///
  /// In the resultant `DriverHandle`, the `transform` specifies any "intrinsic"
  /// transform implicit in the specification.  It will be composed with the
  /// `IndexTransform` specified in the `TransformedDriverSpec`.
  ///
  /// If this is a multiscale spec, this opens the base resolution.
  ///
  /// \param transaction The transaction to use for opening, or `nullptr` to not
  ///     use a transaction.  If specified, the same transaction should be
  ///     returned in the `DriverHandle`.
  /// \param read_write_mode Required mode, or `ReadWriteMode::dynamic` to
  ///     determine the allowed modes.
  virtual Future<DriverHandle> Open(OpenTransactionPtr transaction,
                                    ReadWriteMode read_write_mode) const = 0;

  /// Returns the effective domain, or a null domain if unknown.
  ///
  /// By default, returns `schema.domain()`.
  virtual Result<IndexDomain<>> GetDomain() const;

  /// Returns the effective chunk layout.
  ///
  /// By default, returns `schema.chunk_layout()`.
  virtual Result<ChunkLayout> GetChunkLayout() const;

  /// Returns the effective codec spec.
  ///
  /// By default, returns `schema.codec()`.
  virtual Result<CodecSpec> GetCodec() const;

  /// Returns the effective fill value.
  ///
  /// By default, returns `schema.fill_value()`.
  virtual Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const;

  /// Returns the effective dimension units.
  ///
  /// By default, returns `schema.dimension_units()`.
  virtual Result<DimensionUnitsVector> GetDimensionUnits() const;

  /// Returns the associated KeyValueStore path spec, or an invalid (null) path
  /// spec if there is none.
  ///
  /// By default returns a null spec.
  virtual kvstore::Spec GetKvstore() const;

  virtual void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const = 0;

  virtual std::string_view GetId() const = 0;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    // Exclude `context_binding_state_` because it is handled specially.
    return f(x.schema, x.context_spec_);
  };

  Schema schema;

  /// Specifies any context resource overrides.
  Context::Spec context_spec_;

  ContextBindingState context_binding_state_ = ContextBindingState::unknown;
};

template <>
struct ContextBindingTraits<DriverSpec>
    : public NoOpContextBindingTraits<DriverSpec> {};

absl::Status DriverSpecBindContext(DriverSpecPtr& spec, const Context& context);
void DriverSpecUnbindContext(DriverSpecPtr& spec,
                             const ContextSpecBuilder& context_builder = {});
void DriverSpecStripContext(DriverSpecPtr& spec);

/// For compatibility with `ContextBindingTraits`.  `DriverSpec::Ptr` is the
/// context-unbound type corresponding to the context-bound type
/// `DriverSpec::Bound::Ptr`.
template <>
struct ContextBindingTraits<DriverSpecPtr> {
  using Spec = DriverSpecPtr;
  static absl::Status Bind(Spec& spec, const Context& context) {
    return DriverSpecBindContext(spec, context);
  }
  static void Unbind(Spec& spec, const ContextSpecBuilder& builder) {
    return DriverSpecUnbindContext(spec, builder);
  }
  static void Strip(Spec& spec) { return DriverSpecStripContext(spec); }
};

/// Pairs a `DriverSpec` with an `IndexTransform`.
///
/// This is the underlying representation of the public `tensorstore::Spec`
/// class.
///
/// If `transform.valid()`, `transform.output_rank()` must equal
/// `driver_spec->schema().rank()`.
struct TransformedDriverSpec {
  bool valid() const { return static_cast<bool>(driver_spec); }

  DriverSpecPtr driver_spec;
  IndexTransform<> transform;

  // Forwarding method required by NestedContextJsonBinder
  ContextBindingState context_binding_state() const {
    return driver_spec ? driver_spec->context_binding_state()
                       : ContextBindingState::unknown;
  }

  // Forwarding method required by NestedContextJsonBinder
  void UnbindContext(const ContextSpecBuilder& context_builder = {}) {
    DriverSpecUnbindContext(driver_spec, context_builder);
  }

  void StripContext() { DriverSpecStripContext(driver_spec); }

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.driver_spec, x.transform);
  };
};

absl::Status ApplyOptions(DriverSpec::Ptr& spec, SpecOptions&& options);
absl::Status TransformAndApplyOptions(TransformedDriverSpec& spec,
                                      SpecOptions&& options);

Result<IndexDomain<>> GetEffectiveDomain(const TransformedDriverSpec& spec);

Result<ChunkLayout> GetEffectiveChunkLayout(const TransformedDriverSpec& spec);

Result<SharedArray<const void>> GetEffectiveFillValue(
    const TransformedDriverSpec& spec);

Result<CodecSpec> GetEffectiveCodec(const TransformedDriverSpec& spec);

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const TransformedDriverSpec& spec);

Result<Schema> GetEffectiveSchema(const TransformedDriverSpec& spec);

DimensionIndex GetRank(const TransformedDriverSpec& spec);

/// JSON binder for TensorStore specification.
TENSORSTORE_DECLARE_JSON_BINDER(TransformedDriverSpecJsonBinder,
                                TransformedDriverSpec, JsonSerializationOptions,
                                JsonSerializationOptions, ::nlohmann::json)

struct TransformedDriverSpecNonNullSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const TransformedDriverSpec& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   TransformedDriverSpec& value);
};

}  // namespace internal
namespace internal_json_binding {
template <>
inline constexpr auto DefaultBinder<internal::TransformedDriverSpec> =
    internal::TransformedDriverSpecJsonBinder;
}  // namespace internal_json_binding

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::DriverSpecPtr)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::TransformedDriverSpec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::DriverSpecPtr)

#endif  // TENSORSTORE_DRIVER_DRIVER_SPEC_H_
