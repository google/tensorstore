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

#include "tensorstore/chunk_layout.h"

#include <algorithm>
#include <atomic>
#include <ostream>
#include <string_view>
#include <utility>

#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/dimension_indexed_json_binder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

namespace {
using Usage = ChunkLayout::Usage;
}  // namespace

ChunkLayout::Grid::Grid(DimensionIndex rank, const Index* shape) {
  assert(IsValidRank(rank));
  Access::Assign(&storage_, rank, shape);
}

ChunkLayout::Grid& ChunkLayout::Grid::shape(span<const Index> value) {
  assert(IsValidRank(value.size()));
  Access::Assign(&storage_, value.size(), value.data());
  return *this;
}

bool operator==(const ChunkLayout::Grid& a, const ChunkLayout::Grid& b) {
  return internal::RangesEqual(a.shape(), b.shape());
}

std::ostream& operator<<(std::ostream& os, const ChunkLayout::Grid& x) {
  return os << "{shape=" << x.shape() << "}";
}

/// Stored representation of `ChunkLayout`.
///
/// This is actually the header of the following variable-length structure:
///
///     Storage header;
///     Index origin[rank];
///     Index shapes[kNumUsages][rank];
///     DimensionIndex inner_order[rank];
struct ChunkLayout::Storage {
  explicit Storage(const DimensionIndex rank)
      : ref_count_(1),
        chunks_{Grid(rank), Grid(rank), Grid(rank)},
        rank_(rank) {}
  std::atomic<size_t> ref_count_;
  Grid chunks_[kNumUsages];
  const int8_t rank_;

  constexpr static size_t NumOriginElements(DimensionIndex rank) {
    return rank;
  }
  constexpr static size_t NumInnerOrderElements(DimensionIndex rank) {
    return rank;
  }
  constexpr static size_t TotalBytesAfterHeader(DimensionIndex rank) {
    return sizeof(Index) * NumOriginElements(rank) +
           sizeof(DimensionIndex) * NumInnerOrderElements(rank);
  }
  Index* origin() { return reinterpret_cast<Index*>(this + 1); }
  DimensionIndex* inner_order() {
    return reinterpret_cast<DimensionIndex*>(origin() +
                                             NumOriginElements(rank_));
  }
  static StoragePtr Allocate(DimensionIndex rank) {
    assert(rank >= 0 && rank < kMaxRank);
    const size_t total_bytes =
        // Header
        sizeof(Storage) +
        // Variable-level data
        TotalBytesAfterHeader(rank);
    StoragePtr ptr(static_cast<Storage*>(std::malloc(total_bytes)),
                   internal::adopt_object_ref);
    new (ptr.get()) Storage(rank);
    return ptr;
  }
  Grid& chunks(Usage usage) {
    size_t usage_index = static_cast<size_t>(usage);
    return chunks_[usage_index];
  }

  static void Assign(StoragePtr& target, const Storage* source) {
    if (!source) {
      target.reset();
      return;
    }
    const DimensionIndex rank = source->rank_;
    if (!target || target->rank_ != rank) {
      target = Storage::Allocate(rank);
      std::copy_n(source->chunks_, kNumUsages, target->chunks_);
    }
    std::memcpy(reinterpret_cast<void*>(target.get() + 1),
                reinterpret_cast<const void*>(source + 1),
                TotalBytesAfterHeader(rank));
  }
};

void intrusive_ptr_increment(ChunkLayout::Storage* p) {
  p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
}

void intrusive_ptr_decrement(ChunkLayout::Storage* p) {
  if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::destroy_at(p);
    std::free(p);
  }
}

ChunkLayout::Builder::Builder(DimensionIndex rank)
    : storage_(Storage::Allocate(rank)) {
  std::fill_n(storage_->origin(), rank, 0);

  // Set `inner_order` to an invalid permutation, to indicate that it is
  // unknown.
  std::fill_n(storage_->inner_order(), rank, DimensionIndex(-1));
}

ChunkLayout::Builder& ChunkLayout::Builder::operator=(const Builder& other) {
  Storage::Assign(storage_, other.storage_.get());
  return *this;
}

ChunkLayout::Builder& ChunkLayout::Builder::operator=(
    const ChunkLayout& other) {
  Storage::Assign(storage_, other.storage_.get());
  return *this;
}

DimensionIndex ChunkLayout::Builder::rank() const {
  return storage_ ? storage_->rank_ : dynamic_rank;
}

span<Index> ChunkLayout::Builder::grid_origin() {
  if (storage_) {
    return span(storage_->origin(), storage_->rank_);
  }
  return {};
}

span<DimensionIndex> ChunkLayout::Builder::inner_order() {
  if (storage_) {
    return span(storage_->inner_order(), storage_->rank_);
  }
  return {};
}

ChunkLayout::Grid& ChunkLayout::Builder::operator[](Usage usage) {
  assert(storage_);
  return storage_->chunks(usage);
}

ChunkLayout::Builder& ChunkLayout::Builder::write_chunk(const Grid& g) {
  write_chunk() = g;
  return *this;
}

ChunkLayout::Builder& ChunkLayout::Builder::read_chunk(const Grid& g) {
  read_chunk() = g;
  return *this;
}

ChunkLayout::Builder& ChunkLayout::Builder::codec_chunk(const Grid& g) {
  codec_chunk() = g;
  return *this;
}

ChunkLayout::Builder& ChunkLayout::Builder::grid_origin(
    span<const Index> value) {
  assert(storage_ && storage_->rank_ == value.size());
  std::copy_n(value.begin(), value.size(), storage_->origin());
  return *this;
}

ChunkLayout::Builder& ChunkLayout::Builder::inner_order(
    span<const DimensionIndex> permutation) {
  auto inner_order = this->inner_order();
  if (permutation.empty()) {
    std::fill(inner_order.begin(), inner_order.end(), DimensionIndex(-1));
  } else {
    assert(inner_order.size() == permutation.size());
    std::copy(permutation.begin(), permutation.end(), inner_order.begin());
  }
  return *this;
}

Result<ChunkLayout> ChunkLayout::Builder::Finalize() {
  assert(storage_);
  const DimensionIndex rank = storage_->rank_;
  span<Index> origin = this->grid_origin();
  // Validate origin
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    if (!IsFiniteIndex(origin[dim])) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Invalid origin: ", origin));
    }
  }

  for (Usage usage : ChunkLayout::kUsages) {
    auto status = [&]() -> absl::Status {
      auto shape = (*this)[usage].shape();
      if (shape.size() != rank) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Invalid rank, expected ", rank, " but received: ", shape.size()));
      }
      for (DimensionIndex dim = 0; dim < rank; ++dim) {
        const Index origin_value = origin[dim];
        Index& size_value = shape[dim];
        if (!IndexInterval::ValidSized(origin_value, size_value) ||
            !IsFiniteIndex(origin_value + size_value)) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Invalid origin/shape: origin=", origin, ", shape=", shape));
        }
        if (size_value == 0 && usage == Usage::kRead) {
          size_value = (*this).write_chunk().shape()[dim];
        }
      }
      return absl::OkStatus();
    }();
    if (!status.ok()) {
      return tensorstore::MaybeAnnotateStatus(
          status, tensorstore::StrCat("Invalid ", static_cast<Usage>(usage),
                                      " chunk grid"));
    }
  }
  if (rank > 0) {
    auto inner_order = this->inner_order();
    if (inner_order[0] == -1) {
      // No order specified.
      std::fill_n(inner_order.begin(), rank, -1);
    } else {
      if (!IsValidPermutation(inner_order)) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Invalid inner_order: ", inner_order));
      }
    }
  }

  auto write_chunk_shape = (*this).write_chunk().shape();
  auto read_chunk_shape = (*this).read_chunk().shape();
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    const Index read_size = read_chunk_shape[dim];
    const Index write_size = write_chunk_shape[dim];
    if (read_size == 0) continue;
    if ((write_size % read_size) != 0) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "write chunk shape ", write_chunk_shape,
          " is not a multiple of read chunk shape ", read_chunk_shape));
    }
  }
  ChunkLayout chunk_layout;
  chunk_layout.storage_ = std::move(storage_);
  return chunk_layout;
}

DimensionIndex ChunkLayout::rank() const {
  return storage_ ? storage_->rank_ : dynamic_rank;
}

span<const Index> ChunkLayout::grid_origin() const {
  if (storage_) {
    return span(storage_->origin(), storage_->rank_);
  }
  return {};
}

span<const DimensionIndex> ChunkLayout::inner_order() const {
  if (storage_) {
    const DimensionIndex rank = storage_->rank_;
    if (rank != 0) {
      const DimensionIndex* inner_order = storage_->inner_order();
      if (inner_order[0] != -1) {
        return span(inner_order, rank);
      }
    }
  }
  return {};
}

const ChunkLayout::Grid& ChunkLayout::operator[](Usage usage) const {
  assert(valid());
  return storage_->chunks(usage);
}

std::ostream& operator<<(std::ostream& os, const ChunkLayout& layout) {
  return os << ::nlohmann::json(layout).dump();
}

bool operator==(const ChunkLayout& a, const ChunkLayout& b) {
  const DimensionIndex rank = a.rank();
  if (b.rank() != rank) return false;
  if (rank <= 0) return true;
  for (Usage usage : ChunkLayout::kUsages) {
    if (a[usage] != b[usage]) return false;
  }
  if (!internal::RangesEqual(a.inner_order(), b.inner_order())) {
    return false;
  }
  return true;
}

Result<Index> TransformInputOriginValue(Index value, Index offset,
                                        Index stride) {
  if (stride < 0) value = value - 1;
  Index new_value;
  if (internal::MulOverflow(stride, value, &new_value) ||
      internal::AddOverflow(new_value, offset, &new_value) ||
      !IsFiniteIndex(new_value)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow transforming input origin ", value, " by offset ",
        offset, " and stride ", stride));
  }
  return new_value;
}

Result<Index> TransformOutputOriginValue(Index value, Index offset,
                                         Index stride) {
  Index new_value;
  if (internal::SubOverflow(value, offset, &new_value) ||
      !IsFiniteIndex(new_value)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow transforming output origin ", value, " by offset ",
        offset, " and stride ", stride));
  }
  new_value = CeilOfRatio(((stride > 0) ? new_value : new_value - 1), stride);
  return new_value;
}

Result<Index> TransformInputSizeValue(Index value, Index stride) {
  Index new_value;
  if (stride == std::numeric_limits<Index>::min() ||
      internal::MulOverflow(std::abs(stride), value, &new_value)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow computing abs(", stride, ") * ", value));
  }
  return new_value;
}

Index TransformOutputSizeValue(Index value, Index stride) {
  assert(stride != 0);
  const Index gcd = tensorstore::GreatestCommonDivisor(stride, value);
  return value / gcd;
}

namespace {
/// Transforms an output vector to an input vector.
///
/// Positions of `input` corresponding to input dimensions that don't map
/// one-to-one to an output dimension are left unassigned.  They should be
/// initialized to a default value before calling this function.
template <typename T, typename TransformOutput>
absl::Status TransformOutputVector(IndexTransformView<> transform,
                                   DimensionSet ignored_input_dims,
                                   span<const T> output, span<T> input,
                                   TransformOutput transform_output) {
  assert(output.size() == transform.output_rank());
  assert(input.size() == transform.input_rank());
  for (DimensionIndex output_dim = 0; output_dim < output.size();
       ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        // Constant maps have no effect on the input space.
        break;
      case OutputIndexMethod::array:
        // Array maps are ignored since chunking is not supported.
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        if (ignored_input_dims[input_dim]) {
          // Non one-to-one mappings are ignored.
          break;
        }
        TENSORSTORE_ASSIGN_OR_RETURN(
            input[input_dim],
            transform_output(output[output_dim], map.offset(), map.stride()),
            tensorstore::MaybeAnnotateStatus(
                _, tensorstore::StrCat("Error transforming output dimension ",
                                       output_dim, " -> input dimension ",
                                       input_dim)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

/// Transforms an input vector to an output vector.
///
/// Positions of `output` corresponding to output dimensions that don't map to a
/// single input dimension are left unassigned.  They should be initialized to a
/// default value before calling this function.
template <typename T, typename TransformInput>
absl::Status TransformInputVector(IndexTransformView<> transform,
                                  span<const T> input, span<T> output,
                                  TransformInput transform_input) {
  assert(output.size() == transform.output_rank());
  assert(input.size() == transform.input_rank());
  for (DimensionIndex output_dim = 0; output_dim < output.size();
       ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        // Constant maps have no effect on the input space.
        break;
      case OutputIndexMethod::array:
        // Array maps are ignored since chunking is not supported.
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        TENSORSTORE_ASSIGN_OR_RETURN(
            output[output_dim],
            transform_input(input[input_dim], map.offset(), map.stride()),
            tensorstore::MaybeAnnotateStatus(
                _, tensorstore::StrCat("Error transforming input dimension ",
                                       input_dim, " -> output dimension ",
                                       output_dim)));
        break;
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

Result<ChunkLayout> ApplyIndexTransform(IndexTransformView<> transform,
                                        ChunkLayout layout) {
  using Usage = ChunkLayout::Usage;
  const DimensionIndex input_rank = transform.input_rank(),
                       output_rank = transform.output_rank();
  if (layout.rank() != output_rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot apply index transform of rank ", input_rank, " -> ",
        output_rank, " to chunk layout of rank ", layout.rank()));
  }

  DimensionSet ignored_input_dims = ~GetOneToOneInputDimensions(transform);
  ChunkLayout::Builder builder(input_rank);

  // Transform origin
  TENSORSTORE_RETURN_IF_ERROR(TransformOutputVector<Index>(
      transform, ignored_input_dims, layout.grid_origin(),
      builder.grid_origin(), [](Index value, Index offset, Index stride) {
        return TransformOutputOriginValue(value, offset, stride);
      }));

  for (Usage usage : ChunkLayout::kUsages) {
    // Transform shape
    TENSORSTORE_RETURN_IF_ERROR(TransformOutputVector<Index>(
        transform, ignored_input_dims, layout[usage].shape(),
        builder[usage].shape(),
        [](Index value, Index offset, Index stride) -> Result<Index> {
          return TransformOutputSizeValue(value, stride);
        }));
  }
  if (auto orig_inner_order = layout.inner_order(); !orig_inner_order.empty()) {
    TransformOutputDimensionOrder(transform, orig_inner_order,
                                  builder.inner_order());
  }
  return std::move(builder).Finalize();
}

Result<ChunkLayout> ApplyInverseIndexTransform(IndexTransformView<> transform,
                                               ChunkLayout layout) {
  const DimensionIndex input_rank = transform.input_rank(),
                       output_rank = transform.output_rank();
  if (layout.rank() != input_rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot apply inverse index transform of rank ", input_rank, " -> ",
        output_rank, " to chunk layout of rank ", layout.rank()));
  }
  ChunkLayout::Builder builder(output_rank);

  // Transform origin
  TENSORSTORE_RETURN_IF_ERROR(TransformInputVector<Index>(
      transform, layout.grid_origin(), builder.grid_origin(),
      [](Index value, Index offset, Index stride) {
        return TransformInputOriginValue(value, offset, stride);
      }));

  for (Usage usage : ChunkLayout::kUsages) {
    // Transform shape
    TENSORSTORE_RETURN_IF_ERROR(TransformInputVector<Index>(
        transform, layout[usage].shape(), builder[usage].shape(),
        [](Index value, Index offset, Index stride) {
          return TransformInputSizeValue(value, stride);
        }));
  }
  if (auto orig_inner_order = layout.inner_order(); !orig_inner_order.empty()) {
    TransformInputDimensionOrder(transform, orig_inner_order,
                                 builder.inner_order());
  }
  return std::move(builder).Finalize();
}

namespace {

namespace jb = tensorstore::internal_json_binding;

constexpr auto UsageJsonBinder() {
  return jb::Enum<ChunkLayout::Usage, std::string_view>({
      {ChunkLayout::Usage::kWrite, "write"},
      {ChunkLayout::Usage::kRead, "read"},
      {ChunkLayout::Usage::kCodec, "codec"},
  });
}

constexpr auto DimensionIndexedSetSize = [](auto& layout, size_t size) {
  if (!layout.valid()) {
    layout = ChunkLayout::Builder(size);
  }
  return absl::OkStatus();
};

auto GridBinder(DimensionIndex& rank, ChunkLayout::Usage usage) {
  return jb::Object(jb::Member(
      "shape",
      jb::DimensionIndexedVector(
          &rank, [](auto& layout) { return layout.rank(); },
          DimensionIndexedSetSize,
          [usage](auto& layout, size_t i) -> decltype(auto) {
            return layout[usage].shape()[i];
          },
          jb::MapValue(jb::Integer<Index>(0), std::pair(Index(0), nullptr)))));
}

auto DefaultableGridBinder(DimensionIndex& rank, ChunkLayout::Usage usage) {
  return jb::DefaultPredicate(
      /*get_default=*/[](auto* layout) {},
      /*is_default=*/
      [usage](auto* layout) {
        auto shape = (*layout)[usage].shape();
        if (std::all_of(shape.begin(), shape.end(),
                        [](Index x) { return x == 0; })) {
          return true;
        }
        return (usage == ChunkLayout::kRead) &&
               internal::RangesEqual(shape,
                                     (*layout)[ChunkLayout::kWrite].shape());
      },
      GridBinder(rank, usage));
}

auto LayoutBinder(DimensionIndex& rank) {
  return jb::Object(
      jb::Member("grid_origin",
                 jb::DefaultPredicate(
                     /*get_default=*/[](auto* layout) {},
                     /*is_default=*/
                     [](auto* layout) {
                       auto origin = layout->grid_origin();
                       return std::all_of(origin.begin(), origin.end(),
                                          [](Index x) { return x == 0; });
                     },
                     jb::DimensionIndexedVector(
                         &rank, [](auto& layout) { return layout.rank(); },
                         DimensionIndexedSetSize,
                         [](auto& layout, size_t i) -> decltype(auto) {
                           return layout.grid_origin()[i];
                         }))),
      jb::Member("write_chunk",
                 DefaultableGridBinder(rank, ChunkLayout::kWrite)),
      jb::Member("read_chunk", DefaultableGridBinder(rank, ChunkLayout::kRead)),
      jb::Member("codec_chunk",
                 DefaultableGridBinder(rank, ChunkLayout::kCodec)),
      jb::Member("inner_order",
                 [&](auto is_loading, const auto& options, auto* obj, auto* j) {
                   if constexpr (is_loading) {
                     if (j->is_discarded() || j->is_null()) {
                       return absl::OkStatus();
                     }
                   } else {
                     if (obj->inner_order().empty()) return absl::OkStatus();
                   }
                   return jb::DimensionIndexedVector(
                       &rank, [](auto& layout) { return layout.rank(); },
                       DimensionIndexedSetSize,
                       [](auto& layout, size_t i) -> decltype(auto) {
                         return layout.inner_order()[i];
                       },
                       jb::Integer<DimensionIndex>(0, kMaxRank - 1))(
                       is_loading, options, obj, j);
                 }));
}

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ChunkLayout,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      if constexpr (is_loading) {
        if (j->is_discarded()) {
          *obj = ChunkLayout();
          return absl::OkStatus();
        }
        DimensionIndex rank = options.rank();
        ChunkLayout::Builder builder;
        TENSORSTORE_RETURN_IF_ERROR(
            LayoutBinder(rank)(is_loading, options, &builder, j));
        TENSORSTORE_ASSIGN_OR_RETURN(*obj, builder.Finalize());
      } else {
        if (!obj->valid()) return absl::OkStatus();
        DimensionIndex rank = obj->rank();
        TENSORSTORE_RETURN_IF_ERROR(
            LayoutBinder(rank)(is_loading, options, obj, j));
      }
      return absl::OkStatus();
    })

std::ostream& operator<<(std::ostream& os, ChunkLayout::Usage usage) {
  std::string_view s;
  UsageJsonBinder()(/*is_loading=*/std::false_type{},
                    /*options=*/jb::NoOptions{}, &usage, &s)
      .IgnoreError();
  return os << s;
}

Result<ChunkLayout::Usage> ChunkLayout::ParseUsage(std::string_view s) {
  Usage usage;
  TENSORSTORE_RETURN_IF_ERROR(UsageJsonBinder()(/*is_loading=*/std::true_type{},
                                                /*options=*/jb::NoOptions{},
                                                &usage, &s));
  return usage;
}

void SetPermutationFromStridedLayout(StridedLayoutView<> layout,
                                     span<DimensionIndex> permutation) {
  assert(layout.rank() == permutation.size());
  std::iota(permutation.begin(), permutation.end(), DimensionIndex(0));
  // Return the negative absolute value of the effective byte stride of
  // dimension `i`.  We use negative rather than positive absolute value to
  // avoid possible overflow.
  const auto get_effective_byte_stride_nabs = [&](DimensionIndex i) -> Index {
    const Index byte_stride = layout.byte_strides()[i];
    if (byte_stride > 0) return -byte_stride;
    return byte_stride;
  };
  // Sort in order of decreasing effective byte stride.
  std::stable_sort(permutation.begin(), permutation.end(),
                   [&](DimensionIndex a, DimensionIndex b) {
                     return get_effective_byte_stride_nabs(a) <
                            get_effective_byte_stride_nabs(b);
                   });
}

void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation) {
  if (order == c_order) {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = i;
    }
  } else {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = permutation.size() - 1 - i;
    }
  }
}

bool IsValidPermutation(span<const DimensionIndex> permutation) {
  DimensionSet seen_dims;
  const DimensionIndex rank = permutation.size();
  if (rank > kMaxRank) return false;
  for (DimensionIndex i = 0; i < rank; ++i) {
    DimensionIndex dim = permutation[i];
    if (dim < 0 || dim >= rank || seen_dims[dim]) {
      return false;
    }
    seen_dims[dim] = true;
  }
  return true;
}

void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm) {
  assert(IsValidPermutation(span(perm, rank)));
  for (DimensionIndex i = 0; i < rank; ++i) {
    inverse_perm[perm[i]] = i;
  }
}

void TransformOutputDimensionOrder(IndexTransformView<> transform,
                                   span<const DimensionIndex> output_perm,
                                   span<DimensionIndex> input_perm) {
  assert(transform.valid());
  assert(IsValidPermutation(output_perm));
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  assert(input_rank == input_perm.size());
  assert(output_rank == output_perm.size());
  // `min_output_dim[input_dim]` is the minimum index `i` such that
  // `output_perm[i]` is mapped by `transform` with a single_input_dimension map
  // to `input_dim`.
  DimensionIndex min_output_dim[kMaxRank];
  std::fill_n(min_output_dim, input_rank, kMaxRank);
  for (DimensionIndex orig_perm_i = 0; orig_perm_i < output_rank;
       ++orig_perm_i) {
    const DimensionIndex output_dim = output_perm[orig_perm_i];
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    const DimensionIndex input_dim = map.input_dimension();
    min_output_dim[input_dim] =
        std::min(min_output_dim[input_dim], orig_perm_i);
  }
  std::iota(input_perm.begin(), input_perm.end(), DimensionIndex(0));
  std::sort(input_perm.begin(), input_perm.end(),
            [&](DimensionIndex a, DimensionIndex b) {
              DimensionIndex a_ordinal = min_output_dim[a];
              DimensionIndex b_ordinal = min_output_dim[b];
              if (a_ordinal != b_ordinal) return a_ordinal < b_ordinal;
              return a < b;
            });
  assert(IsValidPermutation(input_perm));
}

void TransformInputDimensionOrder(IndexTransformView<> transform,
                                  span<const DimensionIndex> input_perm,
                                  span<DimensionIndex> output_perm) {
  assert(transform.valid());
  assert(IsValidPermutation(input_perm));
  [[maybe_unused]] const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  assert(input_rank == input_perm.size());
  assert(output_rank == output_perm.size());
  DimensionIndex inverse_input_perm[kMaxRank];
  InvertPermutation(input_rank, input_perm.data(), inverse_input_perm);
  std::iota(output_perm.begin(), output_perm.end(), DimensionIndex(0));
  const auto get_output_dim_ordinal = [&](DimensionIndex output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      return kMaxRank;
    }
    return inverse_input_perm[map.input_dimension()];
  };
  std::sort(output_perm.begin(), output_perm.end(),
            [&](DimensionIndex a, DimensionIndex b) {
              DimensionIndex a_ordinal = get_output_dim_ordinal(a);
              DimensionIndex b_ordinal = get_output_dim_ordinal(b);
              if (a_ordinal != b_ordinal) return a_ordinal < b_ordinal;
              return a < b;
            });
  assert(IsValidPermutation(output_perm));
}

DimensionSet GetOneToOneInputDimensions(IndexTransformView<> transform) {
  DimensionSet invalid_input_dims;
  DimensionSet seen_input_dims;
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex output_rank = transform.output_rank();
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        // Constant maps don't involve any input dimensions.
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        if (map.stride() == std::numeric_limits<Index>::min()) {
          // Stride is not invertible.
          invalid_input_dims[input_dim] = true;
          break;
        }
        if (seen_input_dims[input_dim]) {
          invalid_input_dims[input_dim] = true;
          break;
        }
        seen_input_dims[input_dim] = true;
        break;
      }
      case OutputIndexMethod::array: {
        const auto index_array = map.index_array();
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          if (index_array.byte_strides()[input_dim] != 0) {
            invalid_input_dims[input_dim] = true;
          }
        }
        break;
      }
    }
  }
  return seen_input_dims & ~invalid_input_dims;
}

}  // namespace tensorstore
