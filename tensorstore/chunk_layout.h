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
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies a chunk layout.
///
/// A chunk layout is a hierarchical, regular grid defined over an index space.
///
/// At the coarsest level, a chunk layout specifies a chunk grid to which writes
/// should be aligned.  Each write chunk may then be optionally further
/// subdivided by a grid of read chunks.  Each read chunk may then be optionally
/// further subdivided by a grid of codec chunks.  Refer to the `chunk-layout`
/// documentation for details.
///
/// A chunk layout consists of:
///
/// - The `grid_origin`, specifying the origin of the top-level (write) grid.
///   There is no separate origin for the read grid; it is assumed to be aligned
///   to the start of each write chunk.  There is also no separate origin for
///   the codec grid; while it may not be aligned to the start of each read -
///   chunk, the offset, if any, is not specified by the chunk layout.
///
/// - The `write_chunk`, and `read_chunk`, and `codec_chunk` shape.
///
/// - The optional `inner_order` permutation specifying the data storage order
///   (e.g. C order or Fortran order) within read chunks.
class ChunkLayout {
  struct Storage;
  friend void intrusive_ptr_increment(Storage* p);
  friend void intrusive_ptr_decrement(Storage* p);
  using StoragePtr = internal::IntrusivePtr<Storage>;

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

  /// Range-based for loop-compatible range containing
  /// `{kWrite, kRead, kCodec}`.
  constexpr static internal::IntegerRange<Usage> kUsages =
      internal::IntegerRange<Usage>::Inclusive(kWrite, kCodec);

  constexpr static size_t kNumUsages = 3;

  friend std::ostream& operator<<(std::ostream& os, Usage usage);
  static Result<Usage> ParseUsage(std::string_view s);

  /// Owned mutable view of a chunk grid layout.
  ///
  /// Currently a grid is parameterized only by the chunk shape.
  class Grid {
   public:
    Grid() = default;

    /// Constructs a grid with an all-zero shape vector of the specified rank.
    ///
    /// \dchecks `IsValidRank(rank)`
    explicit Grid(DimensionIndex rank)
        : Grid(rank, GetConstantVector<Index, 0>()) {}

    explicit Grid(span<const Index> shape) : Grid(shape.size(), shape.data()) {}

    template <size_t N>
    explicit Grid(const Index (&shape)[N]) : Grid(N, shape) {
      static_assert(IsValidRank(N));
    }

    explicit Grid(DimensionIndex rank, const Index* shape);

    DimensionIndex rank() const { return Access::GetExtent(storage_); }

    span<const Index> shape() const { return Access::get<kShape>(&storage_); }
    span<Index> shape() { return Access::get<kShape>(&storage_); }

    /// Sets the shape to the specified value.
    ///
    /// \dchecks `IsValidRank(value.size())`
    Grid& shape(span<const Index> value);

    template <size_t N>
    Grid& shape(const Index (&value)[N]) {
      return shape(span<const Index>(value));
    }

    friend std::ostream& operator<<(std::ostream& os, const Grid& x);
    friend bool operator==(const Grid& a, const Grid& b);
    friend bool operator!=(const Grid& a, const Grid& b) { return !(a == b); }

   private:
    friend class ChunkLayout;
    using Storage =
        internal::MultiVectorStorageImpl<dynamic_rank, /*InlineSize=*/0, Index>;
    constexpr static size_t kShape = 0;
    using Access = internal::MultiVectorAccess<Storage>;
    Storage storage_;
  };

  /// Builder for constructing a `ChunkLayout`.
  ///
  /// Example usage:
  ///
  ///     tensorstore::ChunkLayout::Builder builder(3);
  ///     builder.origin({1, 2, 3});
  ///     builder.write_chunk().shape({1024, 1024, 1024});
  ///     builder.read_chunk().shape({64, 64, 64});
  ///     tensorstore::SetPermutation(c_order, builder.inner_order());
  ///     TENSORSTORE_ASSIGN_OR_RETURN(auto layout, builder.Finalize());
  class Builder {
   public:
    Builder() = default;
    explicit Builder(DimensionIndex rank);

    Builder(Builder&& other) = default;
    Builder(const Builder& other) = delete;
    Builder& operator=(Builder&& other) = default;
    Builder& operator=(const Builder& other);
    Builder& operator=(const ChunkLayout& other);

    ~Builder() = default;

    /// Returns true if this is non-null.
    bool valid() const { return static_cast<bool>(storage_); }

    /// Returns the rank.
    DimensionIndex rank() const;

    /// Returns the inner order permutation of length `rank`.
    ///
    /// May be set to a valid permutation to indicate a known inner chunk data
    /// order.  If not set to a valid permutation, indicates that the inner
    /// chunk data order is unknown.
    span<DimensionIndex> inner_order();

    /// Origin of the read/write grids.
    span<Index> grid_origin();

    /// Sets the read/write grid origin.
    ///
    /// \dchecks `value.size() == rank()`
    Builder& grid_origin(span<const Index> value);

    /// Same as above, but can be called with a braced list,
    /// e.g. `builder.origin({1, 2, 3})`.
    template <size_t N>
    Builder& origin(const Index (&origin)[N]) {
      return origin(span<const Index>(origin));
    }

    /// Returns the write chunk grid layout.
    ///
    /// \dchecks `valid()`
    Grid& write_chunk() { return (*this)[Usage::kWrite]; }

    /// Returns the read chunk grid layout.
    ///
    /// \dchecks `valid()`
    Grid& read_chunk() { return (*this)[Usage::kRead]; }

    /// Returns the codec chunk grid layout.
    ///
    /// \dchecks `valid()`
    Grid& codec_chunk() { return (*this)[Usage::kCodec]; }

    /// Returns the chunk grid layout for the given usage.
    ///
    /// \dchecks `valid()`
    Grid& operator[](Usage usage);

    Builder& write_chunk(const Grid& g);
    Builder& read_chunk(const Grid& g);
    Builder& codec_chunk(const Grid& g);
    Builder& inner_order(span<const DimensionIndex> permutation);
    template <size_t N>
    Builder& inner_order(const DimensionIndex (&permutation)[N]) {
      return inner_order(span<const DimensionIndex>(permutation));
    }

    Result<ChunkLayout> Finalize();

   private:
    StoragePtr storage_;
  };

  /// Constructs a null chunk layout.
  ChunkLayout() = default;

  /// Returns true if this is non-null.
  bool valid() const { return static_cast<bool>(storage_); }

  /// Rank of the index space to which this chunk layout applies.
  ///
  /// If `valid() == false`, returns `dynamic_rank`.
  DimensionIndex rank() const;

  /// Returns the read/write grid origin.
  span<const Index> grid_origin() const;

  /// Returns the permutation specifying the inner chunk layout order.
  ///
  /// If the inner layout is known, returns a permutation over
  /// `{0, ..., rank()-1}`.  The first dimension is the outermost dimension,
  /// while the last dimension is the innermost.  For example, lexicographic
  /// order (C order/row-major order) is specified as `{0, 1, ..., rank-1}`,
  /// while colexicographic order (Fortran order/column-major order) is
  /// specified as `{rank-1, ..., 1, 0}`.
  ///
  /// If the inner chunk layout is unknown, returns a zero-length span.
  span<const DimensionIndex> inner_order() const;

  /// Returns the write chunk grid layout.
  ///
  /// \dchecks `valid()`
  const Grid& write_chunk() const { return (*this)[Usage::kWrite]; }

  /// Returns the read chunk grid layout.
  ///
  /// \dchecks `valid()`
  const Grid& read_chunk() const { return (*this)[Usage::kRead]; }

  /// Returns the codec chunk grid layout.
  ///
  /// \dchecks `valid()`
  const Grid& codec_chunk() const { return (*this)[Usage::kCodec]; }

  BoxView<> write_chunk_template() const {
    return BoxView<>(grid_origin(), write_chunk().shape());
  }

  BoxView<> read_chunk_template() const {
    return BoxView<>(grid_origin(), read_chunk().shape());
  }

  /// Returns the grid layout for the given usage.
  ///
  /// \dchecks `valid()`
  const Grid& operator[](Usage usage) const;

  /// Transforms a chunk layout for the output space of a transform to a
  /// corresponding chunk layout for the input space of the transform.
  friend Result<ChunkLayout> ApplyIndexTransform(IndexTransformView<> transform,
                                                 ChunkLayout layout);

  /// Transforms a chunk layout for the input space of a transform to a
  /// corresponding chunk layout for the output space of the transform.
  friend Result<ChunkLayout> ApplyInverseIndexTransform(
      IndexTransformView<> transform, ChunkLayout layout);

  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ChunkLayout, FromJsonOptions,
                                          ToJsonOptions)

  friend bool operator==(const ChunkLayout& a, const ChunkLayout& b);
  friend bool operator!=(const ChunkLayout& a, const ChunkLayout& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const ChunkLayout& layout);

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  friend PipelineResultType<ChunkLayout, Func> operator|(ChunkLayout layout,
                                                         Func&& func) {
    return std::forward<Func>(func)(std::move(layout));
  }

 private:
  StoragePtr storage_;
};

/// Sets `permutation` to a permutation that matches the dimension order of
/// `layout`.
///
/// Specifically, `permutation` is ordered by descending byte stride magnitude,
/// and then ascending dimension index.
void SetPermutationFromStridedLayout(StridedLayoutView<> layout,
                                     span<DimensionIndex> permutation);

/// Sets `permutation` to ascending or descending order.
///
/// If `order == c_order`, sets `permutation` to
/// `{0, 1, ..., permutation.size()-1}`.
///
/// Otherwise, sets `permutation` to `{permutation.size()-1, ..., 1, 0}`.
void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation);

/// Returns `true` if `permutation` is a valid permutation of
/// `{0, 1, ..., permutation.size()-1}`.
bool IsValidPermutation(span<const DimensionIndex> permutation);

/// Sets `inverse_perm` to the inverse permutation of `perm`.
///
/// \param perm[in] Pointer to array of length `rank`.
/// \param inverse_perm[out] Pointer to array of length `rank`.
/// \dchecks `IsValidPermutation({perm, rank})`.
void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm);

/// Transforms a dimension order for the output space of `transform` to a
/// corresponding dimension order for the input space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `single_input_dimension` output index maps, then
/// `input_perm` is simply mapped from `output_perm` according to this
/// correspondence, and `TransformInputDimensionOrder` computes the inverse.
///
/// More generally, a dimension `input_dim` in `input_perm` is ordered ascending
/// by the first index `j` for which the output dimension `output_perm[j]` maps
/// to `input_dim` via a `single_input_dimension` output index map, and then by
/// dimension index.  Input dimensions with no corresponding output dimension
/// are ordered last.
///
/// \param transform The index transform.
/// \param output_perm Permutation of `{0, 1, ..., transform.output_rank()-1}`.
/// \param input_perm[out] Pointer to array of length `transform.input_rank()`.
void TransformOutputDimensionOrder(IndexTransformView<> transform,
                                   span<const DimensionIndex> output_perm,
                                   span<DimensionIndex> input_perm);

/// Transforms a dimension order for the input space of `transform` to a
/// corresponding dimension order for the output space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `single_input_dimension` output index maps, then
/// `output_perm` is simply mapped from `input_perm` according to this
/// correspondence, and `TransformOutputDimensionOrder` computes the inverse.
///
/// More generally, each output dimension `output_dim` mapped with a
/// `single_input_dimension` map is ordered ascending by
/// `inv(input_perm)[output_dim]`, and then by dimension index.  Output
/// dimensions without a `single_input_dimension` map are ordered last, and then
/// by dimension index.
///
/// \param transform The index transform.
/// \param input_perm Permutation of `{0, 1, ..., transform.input_rank()-1}`.
/// \param output_perm[out] Pointer to array of length
///     `transform.output_rank()`..
void TransformInputDimensionOrder(IndexTransformView<> transform,
                                  span<const DimensionIndex> input_perm,
                                  span<DimensionIndex> output_perm);

/// Transforms an origin value in a 1-d input space to the corresponding origin
/// in a 1-d output space.
///
/// The affine mapping from input to output coordinates is given by:
///
///     output = offset + stride * input
///
/// \param value Origin in the input space.
/// \param offset Offset in affine mapping.
/// \param stride Multiplier in affine mapping.
/// \returns Origin in output space.
/// \error `absl::StatusCode::kOutOfRange` if integer overflow occurs.
Result<Index> TransformInputOriginValue(Index value, Index offset,
                                        Index stride);

/// Transforms an origin value in a 1-d output space to the corresponding origin
/// in a 1-d input space.
///
/// This is the inverse of `TransformInputOriginValue`.
///
/// The affine mapping from input to output coordinates is given by:
///
///     output = offset + stride * input
///
/// \param value Origin in the output space.
/// \param offset Offset in affine mapping.
/// \param stride Multiplier in affine mapping.
/// \returns Origin in input space.
/// \error `absl::StatusCode::kOutOfRange` if integer overflow occurs.
Result<Index> TransformOutputOriginValue(Index value, Index offset,
                                         Index stride);

/// Transforms a size in a 1-d input space to the corresponding size in a 1-d
/// output space.
///
/// The affine mapping from input to output coordinates is given by:
///
///     output = offset + stride * input
///
/// Note that `offset` is not a parameter to this function since it does not
/// affect the result.
///
/// \param value Size in the input space.
/// \param stride Multiplier in affine mapping.
/// \returns Size in output space.
/// \error `absl::StatusCode::kOutOfRange` if integer overflow occurs.
Result<Index> TransformInputSizeValue(Index value, Index stride);

/// Transforms a size in a 1-d output space to the corresponding size in a 1-d
/// input space.
///
/// The affine mapping from input to output coordinates is given by:
///
///     output = offset + stride * input
///
/// Note that `offset` is not a parameter to this function since it does not
/// affect the result.
///
/// \param value Size in the output space.
/// \param stride Multiplier in affine mapping.
/// \returns Size in input space.
Index TransformOutputSizeValue(Index value, Index stride);

/// Returns the set of input dimensions that map to a unique output dimension
/// via a `single_input_dimension` output index map, and do not affect any other
/// output dimensions.
DimensionSet GetOneToOneInputDimensions(IndexTransformView<> transform);

}  // namespace tensorstore

#endif  // TENSORSTORE_CHUNK_LAYOUT_H_
