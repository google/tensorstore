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

#ifndef TENSORSTORE_SPEC_H_
#define TENSORSTORE_SPEC_H_

#include <iosfwd>
#include <type_traits>

#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/spec_impl.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies the parameters necessary to open or create a `TensorStore`.
///
/// Includes the driver identifier, driver parameters, optionally an
/// `IndexTransform` and `Schema`.
///
/// \ingroup core
class Spec {
 public:
  /// Constructs an invalid specification.
  ///
  /// \id default
  Spec() = default;

  /// Returns `true` if this is a valid spec.
  bool valid() const { return static_cast<bool>(impl_.driver_spec); }

  /// Returns the data type.
  ///
  /// If the data type is unknown, returns the invalid data type.
  DataType dtype() const {
    return impl_.driver_spec ? impl_.driver_spec->schema.dtype() : DataType();
  }

  /// Returns the rank of the TensorStore, or `dynamic_rank` if unknown.
  DimensionIndex rank() const { return internal::GetRank(impl_); }

  /// Returns the effective schema, which includes all known information,
  /// propagated to the transformed space.
  Result<Schema> schema() const;

  /// Returns the effective domain, based on the schema constraints as well as
  /// any driver-specific constraints.  If the domain cannot be determined,
  /// returns a null index domain.
  Result<IndexDomain<>> domain() const;

  /// Returns the effective chunk layout, which includes all known information,
  /// propagated to the transformed space.
  Result<ChunkLayout> chunk_layout() const;

  /// Returns the effective codec, propagated to the transformed space.
  Result<CodecSpec> codec() const;

  /// Returns the effective fill value, propagated to the transformed space.
  Result<SharedArray<const void>> fill_value() const;

  /// Returns the effective dimension units, propagated to the transformed
  /// space.
  Result<DimensionUnitsVector> dimension_units() const;

  /// Returns the associated key-value store used as the underlying storage.  If
  /// unspecified or not applicable, returns a null (invalid) spec.
  tensorstore::kvstore::Spec kvstore() const;

  /// Returns the transform applied on top of the driver.
  const IndexTransform<>& transform() const { return impl_.transform; }

  /// Applies the specified options in place.
  ///
  /// Supported options include:
  ///
  ///   - Schema options
  ///   - OpenMode
  ///   - RecheckCachedData
  ///   - RecheckCachedMetadata
  ///   - ContextBindingMode
  ///   - Context
  ///   - kvstore::Spec
  ///
  /// If an error occurs, the spec may be in a partially modified state.
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecConvertOptions, Option...>,
                   absl::Status>
  Set(Option&&... option) {
    TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(SpecConvertOptions, options,
                                                  option)
    return this->Set(std::move(options));
  }
  absl::Status Set(SpecConvertOptions&& options);

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, JsonSerializationOptions,
                                          JsonSerializationOptions)

  /// Applies a function that operates on an IndexTransform to a Spec.
  ///
  /// This definition allows DimExpression objects to be applied to Spec
  /// objects.
  ///
  /// \returns The transformed `Spec` on success.
  /// \error `absl::StatusCode::kInvalidArgument` if `spec.transform()` is not
  ///     valid.
  /// \id expr
  template <typename Expr>
  friend internal::FirstType<
      std::enable_if_t<!IsIndexTransform<internal::remove_cvref_t<Expr>>,
                       Result<Spec>>,
      decltype(ApplyIndexTransform(std::declval<Expr>(),
                                   std::declval<IndexTransform<>>()))>
  ApplyIndexTransform(Expr&& expr, Spec spec) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto transform,
                                 spec.GetTransformForIndexingOperation());
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec.impl_.transform,
        ApplyIndexTransform(std::forward<Expr>(expr), std::move(transform)));
    return spec;
  }

  /// Applies an index transform to a Spec.
  ///
  /// \param transform Index transform to apply.  If
  ///     `transform.valid() == false`, this is a no-op.  Otherwise,
  ///     `transform.output_rank()` must be compatible with `spec.rank()`.
  /// \param spec The spec to transform.
  /// \returns New Spec, with rank equal to `transform.input_rank()` (or
  ///     unchanged if `transform.valid() == false`.
  /// \id transform
  friend Result<Spec> ApplyIndexTransform(IndexTransform<> transform,
                                          Spec spec);

  /// Binds any unbound context resources using the specified context.  Any
  /// already-bound context resources remain unmodified.
  ///
  /// If an error occurs, some context resources may remain unbound.
  absl::Status BindContext(const Context& context);

  /// Unbinds any bound context resources, replacing them with context resource
  /// specs that may be used to recreate the context resources.  Any
  /// already-unbound context resources remain unmodified.
  void UnbindContext() { UnbindContext({}); }

  void UnbindContext(const internal::ContextSpecBuilder& context_builder);

  /// Replaces any context resources with a default context resource spec.
  void StripContext();

  /// Indicates the context binding state of the spec.
  ContextBindingState context_binding_state() const;

  friend std::ostream& operator<<(std::ostream& os, const Spec& spec);

  /// Compares for equality via JSON representation.
  friend bool operator==(const Spec& a, const Spec& b);

  friend bool operator!=(const Spec& a, const Spec& b) { return !(a == b); }

  template <typename Func>
  friend PipelineResultType<Spec, Func> operator|(Spec spec, Func&& func) {
    return std::forward<Func>(func)(std::move(spec));
  }

  /// Returns a transform that may be used for apply a DimExpression.
  ///
  /// If `transform().valid()`, returns `transform()`.
  ///
  /// If `rank() != dynamic_rank`, returns `IdentityTransform(rank())`.
  ///
  /// Otherwise, returns an error.
  Result<IndexTransform<>> GetTransformForIndexingOperation() const;

 private:
  friend class internal_spec::SpecAccess;

  internal::TransformedDriverSpec impl_;
};

namespace internal {
/// Implementation of `TensorStore::spec`.
///
/// Refer to that method documentation for details.
Result<Spec> GetSpec(const DriverHandle& handle, SpecRequestOptions&& options);

struct SpecNonNullSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const Spec& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   Spec& value);
};

}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::Spec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(tensorstore::Spec)

#endif  // TENSORSTORE_SPEC_H_
