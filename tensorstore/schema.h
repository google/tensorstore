// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_SCHEMA_H_
#define TENSORSTORE_SCHEMA_H_

#include <iosfwd>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {

/// Collection of constraints for a TensorStore schema.
///
/// When opening an existing TensorStore, specifies constraints that must be
/// satisfied by the existing schema for the operation to succeed.
///
/// When creating a new TensorStore, the constraints are used in conjunction
/// with any driver-dependent defaults and additional driver-specific
/// constraints included in the spec to determine the new schema.
///
/// For interoperability with `tensorstore::Open` and other interfaces accepting
/// a variable-length list of strongly-typed options, there is a unique wrapper
/// type for each constraint.
///
/// Constraints are set by calling `Schema::Set`.  Constraints are retrieved
/// using either the named accessor methods, like `Schema::dtype()`, or in
/// generic code using the explicit conversion operators, e.g.
/// ``static_cast<DataType>(constraints)``.
///
/// \ingroup core
class Schema {
 public:
  Schema() = default;

  /// Specifies the rank (`dynamic_rank` indicates unspecified).
  ///
  /// The rank, if specified, is always a hard constraint.
  tensorstore::RankConstraint rank() const {
    return tensorstore::RankConstraint{rank_};
  }
  explicit operator tensorstore::RankConstraint() const { return rank(); }
  absl::Status Set(tensorstore::RankConstraint rank);
  using RankConstraint = tensorstore::RankConstraint;

  /// Specifies the data type.
  ///
  /// The data type, if specified, is always a hard constraint.
  DataType dtype() const { return dtype_; }
  explicit operator DataType() const { return dtype(); }
  absl::Status Set(DataType value);

  /// Overrides the data type.
  ///
  /// \post `dtype() == value`
  absl::Status Override(DataType value);

  /// Specifies the domain.
  ///
  /// The domain, if specified, is always a hard constraint.  If an additional
  /// domain is specified when an existing domain has been set, the two domains
  /// are merged according to `MergeIndexDomains` (and it is an error if they
  /// are incompatible).
  IndexDomain<> domain() const;
  explicit operator IndexDomain<>() const { return domain(); }
  absl::Status Set(IndexDomain<> value);

  /// Overrides the domain.
  ///
  /// \post `domain() == value`
  /// \id IndexDomain
  absl::Status Override(IndexDomain<> value);

  /// Strongly-typed alias of `span<const Index>` for representing a shape
  /// constraint.
  struct Shape : public span<const Index> {
   public:
    explicit Shape(span<const Index> s) : span<const Index>(s) {}
    template <size_t N>
    explicit Shape(const Index (&s)[N]) : span<const Index>(s) {}
  };

  /// Specifies the zero-origin bounds for the domain.
  ///
  /// This is equivalent to specifying a domain constraint of
  /// ``IndexDomain(shape)``.
  absl::Status Set(Shape value);

  /// Specifies the data storage layout.
  ///
  /// If additional chunk layout constraints are specified when existing chunk
  /// layout constraints are set, they are merged.
  ChunkLayout chunk_layout() const;
  explicit operator ChunkLayout() const { return chunk_layout(); }
  template <typename Option>
  std::enable_if_t<ChunkLayout::IsOption<Option>, absl::Status> Set(
      Option value) {
    TENSORSTORE_RETURN_IF_ERROR(MutableLayoutInternal().Set(std::move(value)));
    return ValidateLayoutInternal();
  }

  /// Strongly-typed alias of `SharedArrayView<const void>` for representing a
  /// `fill_value` constraint.
  struct FillValue : public SharedArrayView<const void> {
    FillValue() = default;
    explicit FillValue(SharedArrayView<const void> value)
        : SharedArrayView<const void>(std::move(value)) {}

    /// Compares two fill values for equality.
    friend bool operator==(const FillValue& a, const FillValue& b);
    friend bool operator!=(const FillValue& a, const FillValue& b) {
      return !(a == b);
    }
  };

  /// Specifies the fill value.
  ///
  /// The fill value data type must be convertible to the actual data type, and
  /// the shape must be broadcast-compatible with the domain.
  ///
  /// If an existing fill value has already been set as a constraint, it is an
  /// error to specify a different fill value (where the comparison is done
  /// after normalization by `UnbroadcastArray`).
  FillValue fill_value() const;
  explicit operator FillValue() const { return fill_value(); }
  absl::Status Set(FillValue value);

  /// Specifies the data codec.
  CodecSpec codec() const;
  explicit operator CodecSpec() const { return codec(); }
  absl::Status Set(CodecSpec value);

  /// Strongly-typed alias of `span<const std::optional<Unit>>` for representing
  /// dimension unit constraints.
  struct DimensionUnits : public span<const std::optional<Unit>> {
   public:
    explicit DimensionUnits() = default;
    explicit DimensionUnits(span<const std::optional<Unit>> s)
        : span<const std::optional<Unit>>(s) {}
    template <size_t N>
    explicit DimensionUnits(const std::optional<Unit> (&s)[N])
        : span<const std::optional<Unit>>(s) {}
    friend std::ostream& operator<<(std::ostream& os, DimensionUnits u);
    friend bool operator==(DimensionUnits a, DimensionUnits b);
    friend bool operator!=(DimensionUnits a, DimensionUnits b) {
      return !(a == b);
    }
    bool valid() const { return !this->empty(); }

    explicit operator DimensionUnitsVector() const {
      return DimensionUnitsVector(this->begin(), this->end());
    }
  };

  /// Specifies the physical quantity corresponding to a single index increment
  /// along each dimension.
  ///
  /// A value of `std::nullopt` indicates that the unit is
  /// unknown/unconstrained.  A dimension-less quantity can be indicated by a
  /// unit of `Unit()`.
  ///
  /// When creating a new TensorStore, the specified units may be stored as part
  /// of the metadata.
  ///
  /// When opening an existing TensorStore, the specified units serve as a
  /// constraint, to ensure the units are as expected.  Additionally, for
  /// drivers like neuroglancer_precomputed that support multiple scales, the
  /// desired scale can be selected by specifying constraints on the units.
  DimensionUnits dimension_units() const;
  absl::Status Set(DimensionUnits value);
  explicit operator DimensionUnits() const { return dimension_units(); }

  /// Merges in constraints from an existing schema.
  ///
  /// \id Schema
  absl::Status Set(Schema value);

  /// Evaluates to `true` for option types compatible with `Set`.  Supported
  /// types are:
  ///
  /// - `Schema`
  /// - `RankConstraint`
  /// - `DataType`, and `StaticDataType`
  /// - `IndexDomain`
  /// - `Schema::Shape`
  /// - `Schema::FillValue`
  /// - `CodecSpec`
  /// - `Schema::DimensionUnits`
  ///
  /// Additionally, all `ChunkLayout` options are also supported:
  ///
  /// - `ChunkLayout`
  /// - `ChunkLayout::GridOrigin`
  /// - `ChunkLayout::InnerOrder`
  /// - `ChunkLayout::GridViewFor`
  /// - `ChunkLayout::ChunkElementsFor`
  /// - `ChunkLayout::ChunkShapeFor`
  /// - `ChunkLayout::ChunkAspectRatioFor`
  template <typename T>
  static inline constexpr bool IsOption = ChunkLayout::IsOption<T>;

  /// Transforms a `Schema` by an index transform.
  ///
  /// Upon invocation, the input domain of `transform` corresponds to `this`.
  /// Upon return, `this` corresponds to the output space of `transform`.
  absl::Status TransformInputSpaceSchema(IndexTransformView<> transform);

  /// Transforms a `Schema` object by a `DimExpression`.
  template <typename Expr>
  friend std::enable_if_t<!IsIndexTransform<internal::remove_cvref_t<Expr>>,
                          Result<Schema>>
  ApplyIndexTransform(Expr&& expr, Schema schema) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto identity_transform,
                                 schema.GetTransformForIndexingOperation());
    if (!identity_transform.valid()) {
      // No constraints that would be affected by an index transform.
      return schema;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transform,
        std::forward<Expr>(expr)(std::move(identity_transform)));
    return ApplyIndexTransform(std::move(transform), std::move(schema));
  }

  friend Result<Schema> ApplyIndexTransform(IndexTransform<> transform,
                                            Schema schema);

  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Schema, FromJsonOptions,
                                          ToJsonOptions)

  /// "Pipeline" operator.
  ///
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then `operator|` applies ``y`` to the value of ``x``,
  /// returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
  template <typename Func>
  friend PipelineResultType<Schema, Func> operator|(Schema schema,
                                                    Func&& func) {
    return std::forward<Func>(func)(std::move(schema));
  }

  friend bool operator==(const Schema& a, const Schema& b);
  friend bool operator!=(const Schema& a, const Schema& b) { return !(a == b); }

  friend std::ostream& operator<<(std::ostream& os, const Schema& schema);

 private:
  ChunkLayout& MutableLayoutInternal();

 public:
  // Treat as private:

  // Returns an identity index transform over the domain/rank.
  //
  // If `domain().valid() == true`, returns `IdentityTransform(domain())`.
  //
  // Otherwise, if `rank() != dynamic_rank`, returns
  // `IdentityTransform(rank())`.
  //
  // Otherwise, returns a default-constructed (invalid) transform.
  Result<IndexTransform<>> GetTransformForIndexingOperation() const;

  absl::Status ValidateLayoutInternal();

  // Constraints other than `rank_` and `dtype_` are stored in a heap-allocated
  // `Impl` object.  It is expected that `rank` and `dtype` will be specified
  // in most cases, while other constraints may be less commonly used.  This
  // avoids bloating the size of `Schema` and allows efficient move
  // and copy-on-write, while also avoiding heap allocation in the common case
  // of specifying just `rank` and `dtype` constraints.
  struct Impl;
  friend void intrusive_ptr_increment(Impl* p);
  friend void intrusive_ptr_decrement(Impl* p);

  // Ensures `impl_` is non-null with a reference count of 1, copying if
  // necessary.
  //
  // \returns `*impl_`
  Impl& EnsureUniqueImpl();
  internal::IntrusivePtr<Impl> impl_;
  DimensionIndex rank_ = dynamic_rank;
  DataType dtype_;
};

// Specialize `Schema::IsOption<T>` for all supported option types
// (corresponding to `Schema::Set` overloads).

template <>
constexpr inline bool Schema::IsOption<RankConstraint> = true;

template <>
constexpr inline bool Schema::IsOption<DataType> = true;

template <typename T>
constexpr bool Schema::IsOption<StaticDataType<T>> = true;

template <DimensionIndex Rank, ContainerKind CKind>
constexpr bool Schema::IsOption<IndexDomain<Rank, CKind>> = true;

template <>
constexpr inline bool Schema::IsOption<Schema::Shape> = true;

template <>
constexpr inline bool Schema::IsOption<Schema::FillValue> = true;

template <>
constexpr inline bool Schema::IsOption<CodecSpec> = true;

template <>
constexpr inline bool Schema::IsOption<Schema::DimensionUnits> = true;

template <>
constexpr inline bool Schema::IsOption<Schema> = true;

namespace internal {

/// Combines the read and write chunk constraints of `schema` to choose a single
/// chunk grid for both reading and writing.
absl::Status ChooseReadWriteChunkGrid(MutableBoxView<> chunk_template,
                                      const Schema& schema);

}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::Schema)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::Schema)

#endif  // TENSORSTORE_SCHEMA_H_
