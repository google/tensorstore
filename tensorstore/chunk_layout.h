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

#ifndef TENSORSTORE_CHUNK_LAYOUT_H_
#define TENSORSTORE_CHUNK_LAYOUT_H_

#include <iosfwd>

#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/integer_range.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/maybe_hard_constraint.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies a precise chunk layout or constraints on a chunk layout.
///
/// A chunk layout is a hierarchical, regular grid defined over an index space.
///
/// At the coarsest level, a chunk layout specifies a chunk grid to which writes
/// should be aligned.  Each write chunk may then be optionally further
/// subdivided by a grid of read chunks.  Each read chunk may then be optionally
/// further subdivided by a grid of codec chunks.  Refer to the
/// :ref:`chunk-layout` documentation for details.
///
/// A precise chunk layout is defined by:
///
/// - The `grid_origin`, specifying the origin of the top-level (write) grid.
///   There is no separate origin for the read grid; it is assumed to be aligned
///   to the start of each write chunk.  There is also no separate origin for
///   the codec grid; while it may not be aligned to the start of each read -
///   chunk, the offset, if any, is not specified by the chunk layout.
///
/// - The `write_chunk_shape`, and `read_chunk_shape`, and `codec_chunk_shape`.
///
/// - The optional `inner_order` permutation specifying the data storage order
///   (e.g. C order or Fortran order) within read chunks.
///
/// The precise chunk layout may also be left unspecified, and several types of
/// hard and soft constraints on a chunk layout may be specified instead.  When
/// creating a new TensorStore, the driver determines the precise chunk layout
/// automatically based on the specified constraints as well as any
/// driver-specific requirements:
///
/// - Each dimension of `grid_origin` may be left unspecified, or specified as a
///   soft constraint rather than a hard constraint.
///
/// - Each dimension of the `write_chunk_shape`, `read_chunk_shape`, or
///   `codec_chunk_shape` may be left unspecified, or specified as a soft
///   constraint.
///
/// - Instead of specifying a precise write/read/codec chunk shape, an aspect
///   ratio and target number of elements per chunk may be specified.
///
/// - The `inner_order` permutation may be specified as a soft constraint rather
///   than a hard constraint.
///
/// Hard and soft constraints compose as follows:
///
/// - When specifying a new soft constraint:
///
///   - If there is no existing hard or soft constraint, the specified value is
///     set as a soft constraint.
///
///   - If there is already a hard or soft constraint, the new value is ignored.
///
/// - When specifying a new hard constraint:
///
///   - If there is no existing value, or an existing soft constraint, the
///     specified value is set as a hard constraint.
///
///   - If there is an existing hard constraint, it remains unchanged, and an
///     error is returned if it does not match the new hard constraint.
///
/// The `inner_order` constraint is set as an atomic value, even though it is
/// represented as a permutation vector.
///
/// Constraints are set using the `Set` overloads.  For example::
///
///     tensorstore::ChunkLayout layout;
///     TENSORSTORE_RETURN_IF_ERROR(layout.Set(
///         tensorstore::ChunkLayout::InnerOrder({0, 2, 1})));
///     TENSORSTORE_RETURN_IF_ERROR(layout.Set(
///         tensorstore::ChunkLayout::InnerOrder(
///             {0, 2, 1}, /*hard_constraint=*/false)}));
///     TENSORSTORE_RETURN_IF_ERROR(layout.Set(
///         tensorstore::ChunkLayout::ReadChunkElements(5)));
///     TENSORSTORE_RETURN_IF_ERROR(layout.Set(
///         tensorstore::ChunkLayout::ReadChunkShape(
///             {100, 200, 300})));
///
/// \relates Schema
class ChunkLayout {
 public:
  /// Specifies the type of operation to which a chunk grid applies.
  ///
  /// For example, with the neuroglancer_precomputed sharded format, writes must
  /// be aligned to entire shards for efficiency, while reads of individual
  /// chunks within shards are still fairly efficient.
  enum Usage : unsigned char {
    /// Chunk is efficient for read and write operations.
    kWrite = 0,

    /// Chunk shape is efficient for reading.
    kRead = 1,

    /// Chunk shape used by the codec, may affect compression ratio (e.g. the
    /// compression_segmentation_block_size for neuroglancer_precomputed
    /// format).
    kCodec = 2,
  };

  /// Identifies parameters that apply to multiple `Usage` values.
  ///
  /// \relates Usage
  constexpr static Usage kUnspecifiedUsage = static_cast<Usage>(3);

  // Range-based for loop-compatible range containing
  // ``{kWrite, kRead, kCodec}``.
  constexpr static internal::IntegerRange<Usage> kUsages =
      internal::IntegerRange<Usage>::Inclusive(kWrite, kCodec);

  constexpr static size_t kNumUsages = 3;

  constexpr static double kDefaultAspectRatioValue = 0;
  constexpr static Index kDefaultShapeValue = 0;

  /// Prints a string representation to an `std::ostream`.
  ///
  /// \relates Usage
  friend std::ostream& operator<<(std::ostream& os, Usage usage);

  /// Parses a string representation.
  ///
  /// \relates Usage
  static Result<Usage> ParseUsage(std::string_view s);

  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  /// Base type for specifying the target number of elements for a
  /// write/read/codec chunk.
  ///
  /// This type is defined as a subclass of `MaybeHardConstraintIndex` rather
  /// than an alias for the benefit of stronger typing of the constructor
  /// parameter/accessor of `GridView`.
  ///
  /// The actual constraints are specified using the usage-specific
  /// `ChunkElementsFor` type.
  struct ChunkElementsBase : public MaybeHardConstraintIndex {
    using MaybeHardConstraintIndex::MaybeHardConstraintIndex;
    explicit ChunkElementsBase(MaybeHardConstraintIndex base)
        : MaybeHardConstraintIndex(base) {}
  };

  /// Target number of elements in a chunk for the specified usage `U`.
  ///
  /// This is used in conjunction with any explicit shape constraints and aspect
  /// ratio constraints to determine a chunk shape automatically.
  ///
  /// Example::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///
  ///     // Sets a soft constraint on the target number of elements to 5000000
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkElements{
  ///             5000000, /*hard_constraint=*/false}));
  ///     EXPECT_EQ(5000000, constraints.read_chunk_elements());
  ///     EXPECT_EQ(false, constraints.read_chunk_elements().hard_constraint);
  ///
  ///     // Soft constraint ignored since it conflicts with existing value.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkElements{
  ///             4000000, /*hard_constraint=*/false}));
  ///
  ///     // Target number of elements remains the same.
  ///     EXPECT_EQ(5000000, constraints.read_chunk_elements());
  ///     EXPECT_EQ(false, constraints.read_chunk_elements().hard_constraint);
  ///
  ///     // Sets a hard constraint on the target number of elements to 6000000,
  ///     // overrides previous soft constraint.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkElements{
  ///             6000000, /*hard_constraint=*/false}));
  ///     EXPECT_EQ(6000000, constraints.read_chunk_elements());
  ///     EXPECT_EQ(true, constraints.read_chunk_elements().hard_constraint);
  ///
  ///     // Setting an incompatible hard constraint value is an error.
  ///     EXPECT_FALSE(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkElements{
  ///             7000000, /*hard_constraint=*/false}).ok());
  ///     EXPECT_EQ(6000000, constraints.read_chunk_elements());
  ///
  /// Note:
  ///
  ///   For this property, a hard constraint still just constrains the *target*
  ///   number of elements per chunk.  Any explicit constraints on the shape
  ///   still take precedence, as do any driver-specific constraints.
  template <Usage U>
  struct ChunkElementsFor : public ChunkElementsBase {
    using ChunkElementsBase::ChunkElementsBase;

    /// The target number of elements.  The default value of `kImplicit`
    /// indicates no constraint.  Specifying any other negative value is an
    /// error.  This type implicitly converts to an `Index` value equal to the
    /// `value` member.
    using ChunkElementsBase::value;

    /// Indicates whether this is a hard or soft constraint.
    using ChunkElementsBase::hard_constraint;
  };

  /// Target number of elements for both write and read chunks.
  using ChunkElements = ChunkElementsFor<kUnspecifiedUsage>;

  /// Target number of elements for write chunks.
  using WriteChunkElements = ChunkElementsFor<Usage::kWrite>;

  /// Target number of elements for read chunks.
  using ReadChunkElements = ChunkElementsFor<Usage::kRead>;

  /// Target number of elements for codec chunks.
  using CodecChunkElements = ChunkElementsFor<Usage::kCodec>;

  /// Base type for specifying explicit constraints on the write/read/codec
  /// chunk shape.
  ///
  /// This type is defined as a subclass of `MaybeHardConstraintSpan<Index>`
  /// rather than an alias for the benefit of stronger typing of the constructor
  /// parameter/accessor of `GridView`.
  ///
  /// The actual constraints are specified using the usage-specific
  /// `ChunkShapeFor` type.
  struct ChunkShapeBase : public MaybeHardConstraintSpan<Index> {
    using MaybeHardConstraintSpan<Index>::MaybeHardConstraintSpan;
    explicit ChunkShapeBase(MaybeHardConstraintSpan<Index> base)
        : MaybeHardConstraintSpan<Index>(base) {}
  };

  /// Constrains the chunk shape for the specified usage `U`.
  ///
  /// Example::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///
  ///     // Sets a hard constraint on the chunk size for dimension 0
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkShape{
  ///             {64, 0, 0}, /*hard_constraint=*/true}));
  ///     EXPECT_THAT(constraints.read_chunk_shape(),
  ///                 ::testing::ElementsAre(64, 0, 0));
  ///     EXPECT_EQ(tensorstore::DimensionSet({1, 0, 0}),
  ///               constraints.read_chunk_elements().hard_constraint);
  ///
  ///     // Sets a soft constraint on the chunk size for dimensions 0 and 2.
  ///     // The constraint on dimension 0 is ignored since the prior hard
  ///     // constraint takes precedence.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkShape{
  ///             {100, 0, 100}, /*hard_constraint=*/false}));
  ///     EXPECT_THAT(constraints.read_chunk_shape(),
  ///                 ::testing::ElementsAre(64, 0, 100));
  ///     EXPECT_EQ(tensorstore::DimensionSet({1, 0, 0}),
  ///               constraints.read_chunk_elements().hard_constraint);
  ///
  ///     // Sets a hard constraint on the chunk size for dimensions 1 and 2.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkShape{
  ///             {0, 50, 80}, /*hard_constraint=*/true}));
  ///     EXPECT_THAT(constraints.read_chunk_shape(),
  ///                 ::testing::ElementsAre(64, 50, 80));
  ///     EXPECT_EQ(tensorstore::DimensionSet({1, 1, 1}),
  ///               constraints.read_chunk_elements().hard_constraint);
  ///
  /// This type inherits from `span<const Index>`.
  ///
  /// The chunk size constraint for each dimension is specified as a
  /// non-negative integer.  The special value of 0 for a given dimension
  /// indicates no constraint on that dimension.  The special value of `-1`
  /// indicates that the size should equal the full extent of the domain, and is
  /// always treated as a soft constraint.  A rank of 0 indicates no
  /// constraints.  It is an error to specify negative values other than `-1`.
  template <Usage U>
  struct ChunkShapeFor : public ChunkShapeBase {
    using ChunkShapeBase::ChunkShapeBase;

    /// Bit vector specifying the dimensions for which the constraint is a hard
    /// constraint.
    using ChunkShapeBase::hard_constraint;
  };

  /// Constrains the shape for both write and read chunks.
  ///
  /// Note that unlike `ChunkAspectRatio`, this does not constrain the codec
  /// chunk shape, since it is unlikely that a user will want to set the same
  /// shape for both codec and read/write chunks.
  using ChunkShape = ChunkShapeFor<kUnspecifiedUsage>;

  /// Constrains the shape for write chunks.
  using WriteChunkShape = ChunkShapeFor<Usage::kWrite>;

  /// Constrains the shape for read chunks.
  using ReadChunkShape = ChunkShapeFor<Usage::kRead>;

  /// Constrains the shape for codec chunks.
  using CodecChunkShape = ChunkShapeFor<Usage::kCodec>;

  /// Base type for specifying constraints on the aspect ratio of the
  /// write/read/codec chunk shape.
  ///
  /// This type is defined as a subclass of `MaybeHardConstraintSpan<double>`
  /// rather than an alias for the benefit of stronger typing of the constructor
  /// parameter/accessor of `GridView`.
  ///
  /// The actual constraints are specified using the usage-specific
  /// `ChunkAspectRatioFor` type.
  struct ChunkAspectRatioBase : public MaybeHardConstraintSpan<double> {
    using MaybeHardConstraintSpan<double>::MaybeHardConstraintSpan;
    explicit ChunkAspectRatioBase(MaybeHardConstraintSpan<double> base)
        : MaybeHardConstraintSpan<double>(base) {}
  };

  /// Constrains the aspect ratio of the chunk shape for the specified usage
  /// `U`.
  ///
  /// Example::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///     // Sets a hard constraint on the aspect ratio for dimensions 0 and 1
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::ReadChunkAspectRatio{
  ///             {1, 1, 0}, /*hard_constraint=*/true}));
  ///
  /// This type inherits from `span<const double>`.
  ///
  /// The aspect ratio for each dimension is specified as a non-negative
  /// `double`.  A value of `0` for a given dimension indicates no constraint
  /// (which means the default aspect ratio of `1` will be used).  A rank of 0
  /// indicates no constraints.  It is an error to specify negative values.
  ///
  /// The actual chunk shape is determined as follows:
  ///
  /// 1. If the chunk size for a given dimension is explicitly constrained,
  ///    either by the `Grid::shape` property or by a driver-specific
  ///    constraint, that size is used and the aspect ratio value is ignored.
  ///
  /// 2. The chunk size of all dimensions ``i`` not otherwise constrained is
  ///    equal to
  ///    ``clip(round(factor * aspect_ratio[i]), 1, domain.shape()[i])``, where
  ///    ``factor`` is the number that makes the total number of elements per
  ///    chunk closest to the target number given by the `Grid::elements`
  ///    constraint.
  template <Usage U>
  struct ChunkAspectRatioFor : public ChunkAspectRatioBase {
    using ChunkAspectRatioBase::ChunkAspectRatioBase;
  };

  class GridView;

  /// Owned mutable representation of `shape`, `aspect_ratio` and `elements`
  /// constraints for a write/read/codec grid.
  class Grid {
   public:
    Grid() = default;
    Grid(const Grid&);
    Grid(Grid&&) = default;
    ~Grid();

    /// Assigns from another grid.
    Grid& operator=(const Grid& other);
    Grid& operator=(Grid&& other) = default;

    /// Represents the shape constraint.
    using Shape = ChunkShapeBase;

    /// Represents the aspect ratio constraint.
    using AspectRatio = ChunkAspectRatioBase;

    /// Represents the target number of elements constraint.
    using Elements = ChunkElementsBase;

    /// Specifies the rank constraint.
    DimensionIndex rank() const { return rank_; }
    absl::Status Set(RankConstraint value);

    /// Specifies the shape constraint.
    Shape shape() const {
      return shape_ ? Shape(span<const Index>(shape_.get(), rank_),
                            shape_hard_constraint_)
                    : Shape();
    }
    explicit operator Shape() const { return shape(); }
    absl::Status Set(Shape value);

    /// Specifies the aspect ratio constraint.
    AspectRatio aspect_ratio() const {
      return aspect_ratio_
                 ? AspectRatio(span<const double>(aspect_ratio_.get(), rank_),
                               aspect_ratio_hard_constraint_)
                 : AspectRatio();
    }
    explicit operator AspectRatio() const { return aspect_ratio(); }
    absl::Status Set(AspectRatio value);

    /// Specifies the target number of elements.
    Elements elements() const {
      return Elements(elements_, elements_hard_constraint_);
    }
    explicit operator Elements() const { return elements(); }
    absl::Status Set(Elements value);

    /// Merges in constraints from another `Grid` or `GridView`.
    ///
    /// \id GridView
    absl::Status Set(const GridView& value);

    /// Compares two chunk grid constraints for equality.
    friend bool operator==(const Grid& a, const Grid& b);
    friend bool operator!=(const Grid& a, const Grid& b) { return !(a == b); }

    TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Grid, FromJsonOptions,
                                            ToJsonOptions)

   private:
    friend class ChunkLayout;
    int8_t rank_ = dynamic_rank;
    bool elements_hard_constraint_ = false;
    std::unique_ptr<Index[]> shape_;
    std::unique_ptr<double[]> aspect_ratio_;
    DimensionSet shape_hard_constraint_;
    DimensionSet aspect_ratio_hard_constraint_;
    Index elements_ = kImplicit;
  };

  /// Base type for constraints applicable to a read/write/codec chunk grid.
  ///
  /// This combines constraints on the chunk shape, chunk shape aspect ratio,
  /// and chunk shape target number of elements.
  class GridView {
   public:
    /// Constructs a view of an unconstrained rank-0 grid.
    ///
    /// \id default
    explicit GridView() = default;

    /// Constructs from an existing grid, optionally converting to hard
    /// constraints.
    ///
    /// \id grid
    explicit GridView(const GridView& other, bool hard_constraint)
        : GridView(other) {
      if (!hard_constraint) {
        elements_hard_constraint_ = false;
        shape_hard_constraint_ = false;
        aspect_ratio_hard_constraint_ = false;
      }
    }
    explicit GridView(const Grid& grid, bool hard_constraint = true)
        : GridView(GridView(grid.shape(), grid.aspect_ratio(), grid.elements()),
                   hard_constraint) {}

    /// Constructs from individual constraints.
    ///
    /// \id components
    explicit GridView(ChunkShapeBase shape, ChunkAspectRatioBase aspect_ratio,
                      ChunkElementsBase elements)
        : shape_rank_(shape.size()),
          aspect_ratio_rank_(aspect_ratio.size()),
          elements_hard_constraint_(elements.hard_constraint),
          shape_hard_constraint_(shape.hard_constraint),
          aspect_ratio_hard_constraint_(aspect_ratio.hard_constraint),
          elements_(elements),
          shape_(shape.data()),
          aspect_ratio_(aspect_ratio.data()) {}
    explicit GridView(ChunkShapeBase shape)
        : GridView(shape, ChunkAspectRatioBase(), ChunkElementsBase()) {}
    explicit GridView(ChunkAspectRatioBase aspect_ratio)
        : GridView(ChunkShapeBase(), aspect_ratio, ChunkElementsBase()) {}
    explicit GridView(ChunkElementsBase elements)
        : GridView(ChunkShapeBase(), ChunkAspectRatioBase(), elements) {}

    /// Returns the shape constraint.
    ChunkShapeBase shape() const {
      return ChunkShapeBase(span<const Index>(shape_, shape_rank_),
                            shape_hard_constraint_);
    }

    /// Returns the aspect ratio constraint.
    ChunkAspectRatioBase aspect_ratio() const {
      return ChunkAspectRatioBase(
          span<const double>(aspect_ratio_, aspect_ratio_rank_),
          aspect_ratio_hard_constraint_);
    }

    /// Returns the target number of elements.
    ChunkElementsBase elements() const {
      return ChunkElementsBase(elements_, elements_hard_constraint_);
    }

   private:
    friend class ChunkLayout;

    // Length of the `shape_` array.
    int8_t shape_rank_ = 0;

    // Length of the `aspect_ratio_` array.  Stored separately from
    // `shape_rank_` in order to allow validation of the lengths to be
    // deferred.
    int8_t aspect_ratio_rank_ = 0;

    // Indicates whether the `elements_` value is a hard constraint.
    bool elements_hard_constraint_ = false;

    // Indicates which dimensions of `shape_` are hard constraints.
    DimensionSet shape_hard_constraint_;

    // Indicates which dimensions of `aspect_ratio_` are hard constraints.
    DimensionSet aspect_ratio_hard_constraint_;

    // Indicates the target number of elements per chunk.
    Index elements_ = kImplicit;

    // Pointer to array of length `shape_rank_`.
    const Index* shape_ = nullptr;

    // Pointer to array of length `aspect_ratio_rank_`.
    const double* aspect_ratio_ = nullptr;
  };

  /// Strongly-typed view that provides access to parameters specific to a
  /// particular `Usage`.
  template <Usage U>
  class GridViewFor : public GridView {
   public:
    /// Representation of the shape constraint.
    using Shape = ChunkShapeFor<U>;

    /// Representation of the aspect ratio constraint.
    using AspectRatio = ChunkAspectRatioFor<U>;

    /// Representation of the target number of elements constraint.
    using Elements = ChunkElementsFor<U>;

    using GridView::GridView;

    /// Constructs from an existing grid.
    explicit GridViewFor(GridView other) : GridView(other) {}

    /// Returns the shape constraint.
    Shape shape() const { return Shape(GridView::shape()); }

    /// Returns the aspect ratio constraint.
    AspectRatio aspect_ratio() const {
      return AspectRatio(GridView::aspect_ratio());
    }

    /// Returns the target number of elements.
    Elements elements() const { return Elements(GridView::elements()); }
  };

  /// Aliases of `GridViewFor` that provide access to parameters specific to a
  /// particular `Usage`.
  using Chunk = GridViewFor<kUnspecifiedUsage>;
  using WriteChunk = GridViewFor<Usage::kWrite>;
  using ReadChunk = GridViewFor<Usage::kRead>;
  using CodecChunk = GridViewFor<Usage::kCodec>;

  /// Specifies the data storage order within innermost chunks as a permutation
  /// of ``[0, ..., rank-1]``.
  ///
  /// Example::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///
  ///     // Sets a soft constraint on the storage order.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::InnerOrder{
  ///             {1, 2, 0}, /*hard_constraint=*/false}));
  ///     EXPECT_THAT(constraints.inner_order(),
  ///                 ::testing::ElementsAre(1, 2, 0));
  ///     EXPECT_EQ(false, constraints.inner_order().hard_constraint);
  ///
  ///     // Overrides the soft constraint with a hard constraint.
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::InnerOrder{
  ///             {1, 0, 2}, /*hard_constraint=*/true}));
  ///     EXPECT_THAT(constraints.inner_order(),
  ///                 ::testing::ElementsAre(1, 0, 2));
  ///     EXPECT_EQ(true, constraints.inner_order().hard_constraint);
  struct InnerOrder : public span<const DimensionIndex> {
    /// Constructs an unspecified order.
    ///
    /// \id default
    explicit InnerOrder() = default;

    /// Constructs from the specified order.
    ///
    /// \id order
    explicit InnerOrder(span<const DimensionIndex> s,
                        bool hard_constraint = true)
        : span<const DimensionIndex>(s), hard_constraint(hard_constraint) {}
    template <size_t N>
    explicit InnerOrder(const DimensionIndex (&s)[N],
                        bool hard_constraint = true)
        : span<const DimensionIndex>(s), hard_constraint(hard_constraint) {}

    /// Returns `true` if this specifies an order constraint.
    bool valid() const { return !this->empty(); }

    /// Compares two order constraints for equality.
    friend bool operator==(const InnerOrder& a, const InnerOrder& b) {
      return internal::RangesEqual(a, b) &&
             a.hard_constraint == b.hard_constraint;
    }
    friend bool operator!=(const InnerOrder& a, const InnerOrder& b) {
      return !(a == b);
    }

    /// Indicates whether the data storage order is a hard constraint.
    bool hard_constraint{false};
  };

  /// Specifies the base origin/offset of the chunk grid.
  ///
  /// Example::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///
  ///     // Sets a hard constraint on the origin for dimensions 0 and 1
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::GridOrigin{
  ///             {5, 6, kImplicit}));
  ///     EXPECT_THAT(constraints.grid_origin(),
  ///                 ::testing::ElementsAre(5, 6, kImplicit));
  ///     EXPECT_EQ(tensorstore::DimensionSet({1, 1, 0}),
  ///               constraints.grid_origin().hard_constraint);
  ///
  ///     // Sets a soft constraint on the origin for dimensions 0 and 2
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::GridOrigin{
  ///             {9, kImplicit, 10}, /*hard_constraint=*/false}));
  ///     EXPECT_THAT(constraints.grid_origin(),
  ///                 ::testing::ElementsAre(5, 6, 10));
  ///     EXPECT_EQ(tensorstore::DimensionSet({1, 1, 0}),
  ///               constraints.grid_origin().hard_constraint);
  ///
  /// Specifying the special value of `kImplicit` for a given dimension
  /// indicates no constraint.
  struct GridOrigin : public MaybeHardConstraintSpan<Index> {
    using MaybeHardConstraintSpan<Index>::MaybeHardConstraintSpan;

    using MaybeHardConstraintSpan<Index>::hard_constraint;
  };

  /// Specifies the desired aspect ratio for all chunk shapes (codec, read, and
  /// write).
  ///
  /// Equivalent to specifying `CodecChunkAspectRatio`, `ReadChunkAspectRatio`,
  /// and `WriteChunkAspectRatio`.
  ///
  /// Even with ``hard_constraint=true``, the aspect ratio is always treated as
  /// a preference rather than requirement, and is overridden by other
  /// constraints on chunk shapes.
  using ChunkAspectRatio = ChunkAspectRatioFor<kUnspecifiedUsage>;
  using WriteChunkAspectRatio = ChunkAspectRatioFor<Usage::kWrite>;
  using ReadChunkAspectRatio = ChunkAspectRatioFor<Usage::kRead>;
  using CodecChunkAspectRatio = ChunkAspectRatioFor<Usage::kCodec>;

  ChunkLayout() = default;
  explicit ChunkLayout(ChunkLayout layout, bool hard_constraint);

  /// Returns the rank constraint, or `dynamic_rank` if unspecified.
  DimensionIndex rank() const;

  /// Sets `box` to the precise write/read chunk template.
  ///
  /// For the purpose of this method, only hard constraints on `grid_origin()`
  /// and `(*this)[usage].shape()` are considered; soft constraints are ignored.
  ///
  /// For any dimension ``i`` where
  /// ``grid_origin().hard_constraint[i] == false`` or
  /// ``(*this)[usage].shape().hard_constraint[i] == false``, ``box[i]`` is set
  /// to `IndexInterval::Infinite()`.
  ///
  /// \param usage Must be either `kWrite` or `kRead`.
  /// \param box[out] Set to the chunk template.
  /// \error `absl::StatusCode::kInvalidArgument` if `box.rank() != rank()`.
  /// \error `absl::StatusCode::kInvalidArgument` if hard constraints on
  ///     `grid_origin` and the chunk shape would lead to an invalid box.
  absl::Status GetChunkTemplate(Usage usage, MutableBoxView<> box) const;

  absl::Status GetWriteChunkTemplate(MutableBoxView<> box) const {
    return GetChunkTemplate(kWrite, box);
  }

  absl::Status GetReadChunkTemplate(MutableBoxView<> box) const {
    return GetChunkTemplate(kRead, box);
  }

  /// Returns the inner order constraint.
  ///
  /// If the rank is unspecified, returns a rank-0 vector.
  InnerOrder inner_order() const;
  explicit operator InnerOrder() const { return inner_order(); }

  /// Sets the inner order constraint.
  ///
  /// \id InnerOrder
  absl::Status Set(InnerOrder value);

  /// Returns the grid origin constraint.
  ///
  /// If the rank is unspecified, returns a rank-0 vector.
  GridOrigin grid_origin() const;
  explicit operator GridOrigin() const { return grid_origin(); }

  /// Sets/updates the grid origin constraint.
  ///
  /// \id GridOrigin
  absl::Status Set(GridOrigin value);

  /// Returns the write chunk constraints.
  WriteChunk write_chunk() const;
  explicit operator WriteChunk() const { return write_chunk(); }

  /// Returns the read chunk constraints.
  ReadChunk read_chunk() const;
  explicit operator ReadChunk() const { return read_chunk(); }

  /// Returns the read chunk constraints.
  CodecChunk codec_chunk() const;
  explicit operator CodecChunk() const { return codec_chunk(); }

  /// Returns the chunk constraints for the given `usage`.
  ///
  /// \param usage Must be one of `Usage::kWrite`, `Usage::kRead`, or
  ///     `Usage::kRead`.  Must not be `kUnspecifiedUsage`.
  GridView operator[](Usage usage) const;

  /// Sets the grid constraints for the specified usage.
  ///
  /// - If `value` is of type `WriteChunk`, `ReadChunk`, or `CodecChunk`, the
  ///    constraints are set for the corresponding usage indicated by the type.
  ///
  /// - If `value` is of type `Chunk` (i.e. `U == kUnspecifiedUsage`), then the
  ///   constraints are set for the usage indicated by the run-time value
  ///   `value.usage()`.  If `value.usage() == kUnspecifiedUsage`, then the
  ///   `GridView::shape` and `GridView::elements` apply to write and read
  ///   chunks, and the `GridView::aspect_ratio` applies to write, read, and
  ///   codec chunks.
  ///
  /// For example, to specify constraints on write chunks::
  ///
  ///     tensorstore::ChunkLayout constraints;
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::WriteChunk(
  ///           tensorstore::ChunkLayout::ChunkShape({100, 200}))));
  ///
  /// Equivalently, specifying the usage at run time::
  ///
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::Chunk(
  ///           tensorstore::ChunkLayout::ChunkShapeBase({100, 200}),
  ///           tensorstore::ChunkLayout::kWrite)));
  ///
  /// To specify common constraints for write, read, and codec chunks::
  ///
  ///     TENSORSTORE_RETURN_IF_ERROR(constraints.Set(
  ///         tensorstore::ChunkLayout::Chunk(
  ///           tensorstore::ChunkLayout::ChunkAspectRatio({1, 2}),
  ///           tensorstore::ChunkLayout::ChunkElements(5000000))));
  ///
  /// Note that the aspect ratio of ``{1, 2}`` applies to write, read, and codec
  /// chunks, but a target number of elements of 5000000 is set only for write
  /// and read chunks.
  ///
  /// \id GridViewFor
  template <Usage U>
  absl::Status Set(const GridViewFor<U>& value);

  /// Returns the write chunk shape constraints.
  WriteChunkShape write_chunk_shape() const;
  explicit operator WriteChunkShape() const { return write_chunk_shape(); }

  /// Returns the read chunk shape constraints.
  ReadChunkShape read_chunk_shape() const;
  explicit operator ReadChunkShape() const { return read_chunk_shape(); }

  /// Returns the codec chunk shape constraints.
  CodecChunkShape codec_chunk_shape() const;
  explicit operator CodecChunkShape() const { return codec_chunk_shape(); }

  /// Returns the write chunk aspect ratio constraints.
  WriteChunkAspectRatio write_chunk_aspect_ratio() const;
  explicit operator WriteChunkAspectRatio() const {
    return write_chunk_aspect_ratio();
  }

  /// Returns the read chunk aspect ratio constraints.
  ReadChunkAspectRatio read_chunk_aspect_ratio() const;
  explicit operator ReadChunkAspectRatio() const {
    return read_chunk_aspect_ratio();
  }

  /// Returns the codec chunk aspect ratio constraints.
  CodecChunkAspectRatio codec_chunk_aspect_ratio() const;
  explicit operator CodecChunkAspectRatio() const {
    return codec_chunk_aspect_ratio();
  }

  /// Returns the write chunk target number of elements constraint.
  WriteChunkElements write_chunk_elements() const;
  explicit operator WriteChunkElements() const {
    return write_chunk_elements();
  }

  /// Returns the read chunk target number of elements constraint.
  ReadChunkElements read_chunk_elements() const;
  explicit operator ReadChunkElements() const { return read_chunk_elements(); }

  /// Returns the codec chunk target number of elements constraint.
  CodecChunkElements codec_chunk_elements() const;
  explicit operator CodecChunkElements() const {
    return codec_chunk_elements();
  }

  /// Sets/updates the chunk shape constraints for the given usage `U`.
  ///
  /// \id ChunkShapeFor
  template <Usage U>
  absl::Status Set(ChunkShapeFor<U> value) {
    return Set(GridViewFor<U>(value));
  }

  /// Sets/updates the chunk aspect ratio constraints for the given usage `U`.
  ///
  /// \id ChunkAspectRatioFor
  template <Usage U>
  absl::Status Set(ChunkAspectRatioFor<U> value) {
    return Set(GridViewFor<U>(value));
  }

  /// Sets/updates the chunk target number of elements constraint for the given
  /// usage `U`.
  ///
  /// \id ChunkElementsFor
  template <Usage U>
  absl::Status Set(ChunkElementsFor<U> value) {
    return Set(GridViewFor<U>(value));
  }

  /// Merges in additional chunk layout constraints.  Soft constraints of
  /// `*this` take precedence over soft constraints of `value`.
  ///
  /// \id ChunkLayout
  absl::Status Set(ChunkLayout value);

  /// Sets the rank.
  ///
  /// \id RankConstraint
  absl::Status Set(RankConstraint value);

  /// Validates and converts this layout into a precise chunk layout.
  ///
  /// - All dimensions of `grid_origin` must be specified as hard constraints.
  ///
  /// - Any write/read/codec chunk `GridView::shape` soft constraints are
  /// - cleared.
  ///
  /// - Any unspecified dimensions of the read chunk shape are set from the
  ///   write chunk shape.
  ///
  /// - Any write/read/codec chunk `GridView::aspect_ratio` or
  ///   `GridView::elements` constraints are cleared.
  absl::Status Finalize();

  /// Transforms a chunk layout for the output space of a transform to a
  /// corresponding chunk layout for the input space of the transform.
  ///
  /// \id transform
  friend Result<ChunkLayout> ApplyIndexTransform(
      IndexTransformView<> transform, ChunkLayout output_constraints);

  /// Transforms a chunk layout for the input space of a transform to a
  /// corresponding chunk layout for the output space of the transform.
  friend Result<ChunkLayout> ApplyInverseIndexTransform(
      IndexTransformView<> transform, ChunkLayout input_constraints);

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ChunkLayout, FromJsonOptions,
                                          ToJsonOptions)

  /// Transforms a `ChunkLayout` object by a `DimExpression`.
  ///
  /// \id expr
  template <typename Expr>
  friend std::enable_if_t<!IsIndexTransform<internal::remove_cvref_t<Expr>>,
                          Result<ChunkLayout>>
  ApplyIndexTransform(Expr&& expr, ChunkLayout constraints) {
    DimensionIndex rank = constraints.rank();
    if (rank == dynamic_rank) {
      // No constraints that would be affected by an index transform.
      return constraints;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transform,
        std::forward<Expr>(expr)(tensorstore::IdentityTransform(rank)));
    return ApplyIndexTransform(std::move(transform), std::move(constraints));
  }

  friend bool operator==(const ChunkLayout& a, const ChunkLayout& b);
  friend bool operator!=(const ChunkLayout& a, const ChunkLayout& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const ChunkLayout& x);

  /// "Pipeline" operator.
  ///
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then `operator|` applies ``y`` to the value of ``x``,
  /// returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
  template <typename Func>
  friend PipelineResultType<ChunkLayout, Func> operator|(ChunkLayout layout,
                                                         Func&& func) {
    return std::forward<Func>(func)(std::move(layout));
  }

  /// Evaluates to `true` for option types compatible with `Set`.  Supported
  /// types are:
  ///
  /// - `ChunkLayout`
  /// - `ChunkLayout::GridOrigin`
  /// - `ChunkLayout::InnerOrder`
  /// - `ChunkLayout::GridViewFor`
  /// - `ChunkLayout::ChunkElementsFor`
  /// - `ChunkLayout::ChunkShapeFor`
  /// - `ChunkLayout::ChunkAspectRatioFor`
  template <typename T>
  static inline constexpr bool IsOption = false;

  // Treat as private:

  struct Storage;
  friend void intrusive_ptr_increment(Storage* p);
  friend void intrusive_ptr_decrement(Storage* p);
  using StoragePtr = internal::IntrusivePtr<Storage>;
  StoragePtr storage_;
};

/// Specifies chunk layout constraints for a single usage specified at run time.
///
/// This may be passed to `ChunkLayout::Set` to set the grid for a single
/// runtime-specified usage.
///
/// \id kUnspecifiedUsage
template <>
class ChunkLayout::GridViewFor<ChunkLayout::kUnspecifiedUsage>
    : public GridView {
 public:
  explicit GridViewFor() = default;
  explicit GridViewFor(const Grid& grid, Usage usage = kUnspecifiedUsage)
      : GridView(grid), usage_(usage) {}
  explicit GridViewFor(const GridView& grid, Usage usage = kUnspecifiedUsage)
      : GridView(grid), usage_(usage) {}
  explicit GridViewFor(const GridView& other, bool hard_constraint,
                       Usage usage = kUnspecifiedUsage)
      : GridView(other, hard_constraint), usage_(usage) {}
  explicit GridViewFor(const Grid& other, bool hard_constraint,
                       Usage usage = kUnspecifiedUsage)
      : GridView(other, hard_constraint), usage_(usage) {}
  explicit GridViewFor(ChunkShapeBase shape, ChunkAspectRatioBase aspect_ratio,
                       ChunkElementsBase elements,
                       Usage usage = kUnspecifiedUsage)
      : GridView(shape, aspect_ratio, elements), usage_(usage) {}
  explicit GridViewFor(ChunkShapeBase shape, Usage usage = kUnspecifiedUsage)
      : GridView(shape), usage_(usage) {}
  explicit GridViewFor(ChunkAspectRatioBase aspect_ratio,
                       Usage usage = kUnspecifiedUsage)
      : GridView(aspect_ratio), usage_(usage) {}
  explicit GridViewFor(ChunkElementsBase elements,
                       Usage usage = kUnspecifiedUsage)
      : GridView(elements), usage_(usage) {}

  Usage usage() const { return usage_; }

 private:
  Usage usage_ = kUnspecifiedUsage;
};

template <>
constexpr bool ChunkLayout::IsOption<ChunkLayout> = true;

template <>
constexpr bool ChunkLayout::IsOption<RankConstraint> = true;

template <>
constexpr bool ChunkLayout::IsOption<ChunkLayout::InnerOrder> = true;

template <>
constexpr bool ChunkLayout::IsOption<ChunkLayout::GridOrigin> = true;

template <ChunkLayout::Usage U>
constexpr bool ChunkLayout::IsOption<ChunkLayout::GridViewFor<U>> = true;

template <ChunkLayout::Usage U>
constexpr bool ChunkLayout::IsOption<ChunkLayout::ChunkShapeFor<U>> = true;

template <ChunkLayout::Usage U>
constexpr bool ChunkLayout::IsOption<ChunkLayout::ChunkAspectRatioFor<U>> =
    true;

template <ChunkLayout::Usage U>
constexpr bool ChunkLayout::IsOption<ChunkLayout::ChunkElementsFor<U>> = true;

/// Sets `permutation` to a permutation that matches the dimension order of
/// `layout`.
///
/// Specifically, `permutation` is ordered by descending byte stride magnitude,
/// and then ascending dimension index.
///
/// \relates ChunkLayout
void SetPermutationFromStridedLayout(StridedLayoutView<> layout,
                                     span<DimensionIndex> permutation);

/// Sets `permutation` to ascending or descending order.
///
/// If `order == c_order`, sets `permutation` to
/// ``{0, 1, ..., permutation.size()-1}``.
///
/// Otherwise, sets `permutation` to ``{permutation.size()-1, ..., 1, 0}``.
///
/// \relates ChunkLayout
void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation);

/// Returns `true` if `permutation` is a valid permutation of
/// ``{0, 1, ..., permutation.size()-1}``.
///
/// \relates ChunkLayout
bool IsValidPermutation(span<const DimensionIndex> permutation);

/// Sets `inverse_perm` to the inverse permutation of `perm`.
///
/// \param perm[in] Pointer to array of length `rank`.
/// \param inverse_perm[out] Pointer to array of length `rank`.
/// \dchecks `IsValidPermutation({perm, rank})`.
/// \relates ChunkLayout
void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm);

/// Transforms a dimension order for the output space of `transform` to a
/// corresponding dimension order for the input space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `OutputIndexMethod::single_input_dimension` output
/// index maps, then `input_perm` is simply mapped from `output_perm` according
/// to this correspondence, and `TransformInputDimensionOrder` computes the
/// inverse.
///
/// More generally, a dimension ``input_dim`` in `input_perm` is ordered
/// ascending by the first index ``j`` for which the output dimension
/// ``output_perm[j]`` maps to ``input_dim`` via a
/// `OutputIndexMethod::single_input_dimension` output index map, and then by
/// dimension index.  Input dimensions with no corresponding output dimension
/// are ordered last.
///
/// \param transform The index transform.
/// \param output_perm Permutation of
///     ``{0, 1, ..., transform.output_rank()-1}``.
/// \param input_perm[out] Pointer to array of length `transform.input_rank()`.
/// \relates ChunkLayout
void TransformOutputDimensionOrder(IndexTransformView<> transform,
                                   span<const DimensionIndex> output_perm,
                                   span<DimensionIndex> input_perm);

/// Transforms a dimension order for the input space of `transform` to a
/// corresponding dimension order for the output space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `OutputIndexMethod::single_input_dimension` output
/// index maps, then `output_perm` is simply mapped from `input_perm` according
/// to this correspondence, and `TransformOutputDimensionOrder` computes the
/// inverse.
///
/// More generally, each output dimension ``output_dim`` mapped with a
/// `OutputIndexMethod::single_input_dimension` map is ordered ascending by
/// ``inv(input_perm)[output_dim]``, and then by dimension index.  Output
/// dimensions without a `OutputIndexMethod::single_input_dimension` map are
/// ordered last, and then by dimension index.
///
/// \param transform The index transform.
/// \param input_perm Permutation of ``{0, 1, ..., transform.input_rank()-1}``.
/// \param output_perm[out] Pointer to array of length
///     `transform.output_rank()`.
/// \relates ChunkLayout
void TransformInputDimensionOrder(IndexTransformView<> transform,
                                  span<const DimensionIndex> input_perm,
                                  span<DimensionIndex> output_perm);

namespace internal {

/// Chooses a regular grid according to the specified constraints.
///
/// If `shape_constraints.elements()` is unspecified, a default value (currently
/// ``1024*1024``, but may change in the future) is used.
///
/// \param origin_constraint If not empty, specifies the origin value for each
///     dimension.  A value of `kImplicit` indicates no constraint.
/// \param shape_constraints Optional constraints on the chunk shape.
/// \param domain Domain of the index space to be chunked.
/// \param chunk_template[out] Set to the chosen chunk origin and shape.
/// \dchecks `domain.rank() == chunk_template.rank()`
/// \error `absl::StatusCode::kInvalidArgument` if `domain.rank()` does not
///     match the rank of `origin_constraints` or `shape_constraints`.
absl::Status ChooseChunkGrid(span<const Index> origin_constraints,
                             ChunkLayout::GridView shape_constraints,
                             BoxView<> domain, MutableBoxView<> chunk_template);

/// Chooses a chunk shape according to the specified constraints.
///
/// If `shape_constraints.elements()` is unspecified, a default value (currently
/// 1024*1024, but may change in the future) is used.
///
/// \param shape_constraints Optional constraints on the chunk shape.
/// \param domain Domain of the index space to be chunked.
/// \param chunk_shape[out] Set to the chosen chunk shape.
/// \dchecks `domain.rank() == chunk_shape.size()`
/// \error `absl::StatusCode::kInvalidArgument` if `domain.rank()` does not
///     match the rank of `shape_constraints`.
absl::Status ChooseChunkShape(ChunkLayout::GridView shape_constraints,
                              BoxView<> domain, span<Index> chunk_shape);

/// Chooses a regular grid based on the combined read and write chunk
/// constraints.
///
/// \param constraints Optional constraints on the grid.
/// \param domain Domain of the index space to be chunked.
/// \param chunk_template[out] Set to the chosen chunk origin and shape.
/// \dchecks `domain.rank() == chunk_template.rank()`
/// \error `absl::StatusCode::kInvalidArgument` if `domain.rank()` does not
///     match the rank of `constraints`.
/// \error `absl::StatusCode::kInvalidArgument` if there is a conflict between
///     the read and write chunk constraints.
absl::Status ChooseReadWriteChunkGrid(const ChunkLayout& constraints,
                                      BoxView<> domain,
                                      MutableBoxView<> chunk_template);
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::ChunkLayout)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::ChunkLayout)

#endif  // TENSORSTORE_CHUNK_LAYOUT_H_
