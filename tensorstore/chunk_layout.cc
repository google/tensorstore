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
#include <vector>

#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

namespace {
using Usage = ChunkLayout::Usage;
using Storage = ChunkLayout::Storage;
using StoragePtr = ChunkLayout::StoragePtr;
constexpr auto kNumUsages = ChunkLayout::kNumUsages;

namespace jb = tensorstore::internal_json_binding;

/// Bitvector offsets for tracking which non-vector fields have been specified
/// as hard constraints.
enum class HardConstraintBit {
  inner_order,
  write_chunk_elements,
  read_chunk_elements,
  codec_chunk_elements,
};

struct OriginValueTraits {
  using Element = Index;
  constexpr static Index kDefaultValue = kImplicit;
  constexpr static bool IsSoftConstraintValue(Index value) { return false; }
  constexpr static bool IsValid(Index x) {
    return x == kImplicit || IsFiniteIndex(x);
  }

  static Result<Index> TransformInputValue(Index value, Index offset,
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

  static Result<Index> TransformOutputValue(Index value, Index offset,
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
};

struct ShapeValueTraits {
  using Element = Index;
  constexpr static Index kDefaultValue = ChunkLayout::kDefaultShapeValue;
  constexpr static bool IsSoftConstraintValue(Index value) {
    return value == -1;
  }
  constexpr static bool IsValid(Index x) { return x == -1 || x >= 0; }

  static Result<Index> TransformInputValue(Index value, Index offset,
                                           Index stride) {
    Index new_value;
    if (stride == std::numeric_limits<Index>::min() ||
        internal::MulOverflow(std::abs(stride), value, &new_value)) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Integer overflow computing abs(", stride, ") * ", value));
    }
    return new_value;
  }

  static Result<Index> TransformOutputValue(Index value, Index offset,
                                            Index stride) {
    assert(stride != 0);
    const Index gcd = tensorstore::GreatestCommonDivisor(stride, value);
    return value / gcd;
  }
};

struct AspectRatioValueTraits {
  using Element = double;
  constexpr static double kDefaultValue = ChunkLayout::kDefaultAspectRatioValue;
  constexpr static bool IsSoftConstraintValue(double value) { return false; }
  constexpr static bool IsValid(double x) { return x >= 0; }

  static Result<double> TransformInputValue(double value, Index offset,
                                            Index stride) {
    return value * std::abs(static_cast<double>(stride));
  }

  static Result<double> TransformOutputValue(double value, Index offset,
                                             Index stride) {
    return value / std::abs(static_cast<double>(stride));
  }
};

// Data members of `ChunkLayout::Storage` that need to be copied on
// write.
//
// This separate base class simplifies the definition of the copy constructor.
struct ChunkLayoutData {
  // Rank of the layout, or `dynamic_rank` (-1) if unspecified.
  //
  // Determines the length of the variable-length `Storage` structure.  If
  // `rank_ == dynamic_rank`, and then the rank becomes known, the `Storage`
  // structure will be reallocated.
  int8_t rank_;
  SmallBitSet<8> hard_constraint_ = false;
  DimensionSet grid_origin_hard_constraint_;
  DimensionSet chunk_shape_hard_constraint_[kNumUsages];
  DimensionSet chunk_aspect_ratio_hard_constraint_[kNumUsages];
  Index chunk_elements_[kNumUsages] = {kImplicit, kImplicit, kImplicit};
};

bool IsHardConstraint(ChunkLayoutData& impl, HardConstraintBit bit) {
  return impl.hard_constraint_[static_cast<int>(bit)];
}

void SetHardConstraintBit(ChunkLayoutData& impl, HardConstraintBit bit) {
  impl.hard_constraint_[static_cast<int>(bit)] = true;
}

}  // namespace

/// Stored representation of `ChunkLayout`.
///
/// This is actually the header of the following variable-length structure:
///
///     Storage header;
///     Index grid_origin[rank];
///     Index chunk_shapes[kNumUsages][rank];
///     double chunk_aspect_ratios[kNumUsages][rank];
///     DimensionIndex inner_order[rank];
struct ChunkLayout::Storage : public ChunkLayoutData {
  explicit Storage(DimensionIndex rank)
      : ChunkLayoutData{static_cast<int8_t>(rank)} {
    Initialize();
  }
  Storage(const Storage& other) : ChunkLayoutData(other) {
    std::memcpy(static_cast<void*>(this + 1),
                static_cast<const void*>(&other + 1),
                TotalBytesAfterHeader(
                    std::max(DimensionIndex(0), DimensionIndex(rank_))));
  }

  Storage(const Storage& other, DimensionIndex new_rank)
      : ChunkLayoutData(other) {
    rank_ = new_rank;
    Initialize();
  }

  void Initialize() {
    if (DimensionIndex rank = rank_; rank > 0) {
      std::fill_n(this->grid_origin(), NumOriginElements(rank),
                  OriginValueTraits::kDefaultValue);
      std::fill_n(this->chunk_shapes(), NumShapeElements(rank),
                  ShapeValueTraits::kDefaultValue);
      std::fill_n(this->chunk_aspect_ratios(), NumAspectRatioElements(rank),
                  AspectRatioValueTraits::kDefaultValue);
      std::fill_n(this->inner_order(), NumInnerOrderElements(rank),
                  static_cast<DimensionIndex>(-1));
    }
  }

  constexpr static size_t NumOriginElements(DimensionIndex rank) {
    return rank;
  }
  constexpr static size_t NumShapeElements(DimensionIndex rank) {
    return kNumUsages * rank;
  }
  constexpr static size_t NumAspectRatioElements(DimensionIndex rank) {
    return kNumUsages * rank;
  }
  constexpr static size_t NumInnerOrderElements(DimensionIndex rank) {
    return rank;
  }
  constexpr static size_t TotalBytesAfterHeader(DimensionIndex rank) {
    return sizeof(Index) * NumOriginElements(rank) +
           sizeof(Index) * NumShapeElements(rank) +
           sizeof(double) * NumAspectRatioElements(rank) +
           sizeof(DimensionIndex) * NumInnerOrderElements(rank);
  }
  Index* grid_origin() { return reinterpret_cast<Index*>(this + 1); }
  Index* chunk_shapes() { return grid_origin() + rank_; }
  span<Index> chunk_shape(size_t usage_index) {
    return {chunk_shapes() + rank_ * usage_index, rank_};
  }
  double* chunk_aspect_ratios() {
    return reinterpret_cast<double*>(chunk_shapes() + NumShapeElements(rank_));
  }
  span<double> chunk_aspect_ratio(size_t usage_index) {
    return {chunk_aspect_ratios() + rank_ * usage_index, rank_};
  }
  DimensionIndex* inner_order() {
    return reinterpret_cast<DimensionIndex*>(chunk_aspect_ratios() +
                                             NumAspectRatioElements(rank_));
  }
  static StoragePtr Allocate(DimensionIndex rank) {
    rank = std::max(rank, DimensionIndex(0));
    assert(rank < kMaxRank);
    const size_t total_bytes =
        // Header
        sizeof(Storage) +
        // Variable-level data
        TotalBytesAfterHeader(rank);
    StoragePtr ptr(static_cast<Storage*>(std::malloc(total_bytes)),
                   internal::adopt_object_ref);
    return ptr;
  }

  /// Ensures that there are no other references to `*ptr`, copying if
  /// necessary.
  ///
  /// \param ptr[in,out] Pointer to storage, may be null.
  /// \param rank Required rank.  Assumed to be valid.
  /// \param storage_to_be_destroyed[out] Set to `ptr` if `ptr` is reassigned.
  ///     This ensures the storage remains valid in case we are performing some
  ///     type of self assignment operation.
  static Storage& EnsureUnique(StoragePtr& ptr, DimensionIndex rank,
                               StoragePtr& storage_to_be_destroyed) {
    if (!ptr) {
      ptr = Allocate(rank);
      new (ptr.get()) Storage(rank);
    } else if (ptr->ref_count_.load(std::memory_order_acquire) != 1) {
      auto new_ptr = Allocate(ptr->rank_);
      new (new_ptr.get()) Storage(*ptr);
      storage_to_be_destroyed = std::move(ptr);
      ptr = std::move(new_ptr);
    }
    return *ptr;
  }

  std::atomic<size_t> ref_count_{1};
};

void intrusive_ptr_increment(Storage* p) {
  p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
}

void intrusive_ptr_decrement(Storage* p) {
  if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::free(p);
  }
}

namespace {

void ClearHardConstraintBits(Storage& impl) {
  impl.hard_constraint_ = 0;
  impl.grid_origin_hard_constraint_ = false;
  for (int i = 0; i < kNumUsages; ++i) {
    impl.chunk_shape_hard_constraint_[i] = false;
    impl.chunk_aspect_ratio_hard_constraint_[i] = false;
  }
}

absl::Status RankMismatchError(DimensionIndex new_rank,
                               DimensionIndex existing_rank) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Rank ", new_rank, " does not match existing rank ", existing_rank));
}

/// Ensures that `ptr` is a unique reference to constraints storage with the
/// specified rank.  This provides copy-on-write behavior.
///
/// Upon successful return, `*ptr` may be safely modified.
///
/// \param ptr[in,out] Pointer to constraint storage, may be null.
/// \param rank Required rank.  Checked for validity.
/// \param storage_to_be_destroyed[out] Set to `ptr` if `ptr` is reassigned.
///     This ensures the storage remains valid in case we are performing some
///     type of self assignment operation.
absl::Status EnsureRank(StoragePtr& ptr, DimensionIndex rank,
                        StoragePtr& storage_to_be_destroyed) {
  TENSORSTORE_RETURN_IF_ERROR(tensorstore::ValidateRank(rank));
  if (!ptr || ptr->rank_ == rank) {
    Storage::EnsureUnique(ptr, rank, storage_to_be_destroyed);
    return absl::OkStatus();
  }
  if (ptr->rank_ == dynamic_rank) {
    // Storage was previously allocated only to hold `elements` values, because
    // the rank was unspecified.  We must reallocate in order to hold vectors.
    auto new_ptr = Storage::Allocate(rank);
    new (new_ptr.get()) Storage(*ptr, rank);
    storage_to_be_destroyed = std::move(ptr);
    ptr = std::move(new_ptr);
    return absl::OkStatus();
  }
  return RankMismatchError(rank, ptr->rank_);
}
template <typename T, typename U>
absl::Status MismatchError(const T& existing_value, const U& new_value) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "New hard constraint (", new_value,
      ") does not match existing hard constraint (", existing_value, ")"));
}

/// Merges additional per-dimension constraints from `in_vector` into
/// `out_vector`.
///
/// The rank and values in `in_vector` are assumed to already have been
/// validated.
///
/// \tparam Traits Class with `Element` type and
///     `static constexpr Element kDefaultValue` member specifying the default
///     value that indicates no constraint.
/// \param in_vector Additional hard/soft constraints.
/// \param out_vector[in,out] Pointer to array of length `in_vector.size()` into
///     which constraint values for each dimension will be merged.
/// \param out_hard_constraint[in,out] Indicates which values of `out_vector`
///     are hard constraints.  Will be updated.
template <typename Traits>
absl::Status MergeVectorInto(
    MaybeHardConstraintSpan<typename Traits::Element> in_vector,
    typename Traits::Element* out_vector, DimensionSet& out_hard_constraint) {
  using Element = typename Traits::Element;
  DimensionIndex rank = in_vector.size();
  // Check that the hard constraints are compatible before making any changes.
  // That way if there is an error, no changes are made.
  if (DimensionSet dims_to_check =
          in_vector.hard_constraint & out_hard_constraint;
      dims_to_check) {
    // Check for a mismatch.
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (!dims_to_check[i]) continue;
      Element x = in_vector[i];
      if (x != Traits::kDefaultValue && out_vector[i] != x) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "New hard constraint (", x, ") for dimension ", i,
            " does not match existing hard constraint (", out_vector[i], ")"));
      }
    }
  }
  for (DimensionIndex i = 0; i < rank; ++i) {
    Element x = in_vector[i];
    if (x == Traits::kDefaultValue) continue;
    const bool in_hard_constraint_value = in_vector.hard_constraint[i];
    if (in_hard_constraint_value || out_vector[i] == Traits::kDefaultValue) {
      out_vector[i] = x;
      out_hard_constraint[i] =
          out_hard_constraint[i] || in_hard_constraint_value;
    }
  }
  return absl::OkStatus();
}

/// Validates and merges additional per-dimension constraints from `in_vector`
/// into `out_vector`.
///
/// The rank of `in_vector` is assumed to already have been validated.
///
/// \tparam Traits Class with the following members:
///     `static constexpr Element kDefaultValue` member specifying the default
///     value that indicates no constraint, a
///     `static bool IsSoftConstraintValue(Element)` member indicating whether a
///     value should always be treated as a soft constraint, and a
///     `static bool IsValid(Element)` member indicating whether a value is
///     valid.
/// \param in_vector Additional hard/soft constraints.
/// \param out_vector[in,out] Pointer to array of length `in_vector.size()` into
///     which constraint values for each dimension will be merged.
/// \param out_hard_constraint[in,out] Indicates which values of `out_vector`
///     are hard constraints.  Will be updated.
template <typename Traits>
absl::Status ValidateAndMergeVectorInto(
    MaybeHardConstraintSpan<typename Traits::Element> in_vector,
    typename Traits::Element* out_vector, DimensionSet& out_hard_constraint) {
  using Element = typename Traits::Element;
  DimensionIndex rank = in_vector.size();
  if (rank == 0) return absl::OkStatus();
  for (DimensionIndex i = 0; i < in_vector.size(); ++i) {
    const Element value = in_vector[i];
    if (!Traits::IsValid(value)) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Invalid value for dimension ", i, ": ", in_vector));
    }
    if (Traits::IsSoftConstraintValue(value)) {
      in_vector.hard_constraint[i] = false;
    }
  }
  return MergeVectorInto<Traits>(in_vector, out_vector, out_hard_constraint);
}

ChunkLayout::ChunkShapeBase GetChunkShape(const ChunkLayout& self,
                                          Usage usage) {
  auto* storage = self.storage_.get();
  if (!storage || storage->rank_ <= 0) {
    return ChunkLayout::ChunkShapeBase();
  }
  const size_t usage_index = static_cast<size_t>(usage);
  return ChunkLayout::ChunkShapeBase(
      storage->chunk_shape(usage_index),
      storage->chunk_shape_hard_constraint_[usage_index]);
}

ChunkLayout::ChunkAspectRatioBase GetChunkAspectRatio(const ChunkLayout& self,
                                                      Usage usage) {
  auto* storage = self.storage_.get();
  if (!storage || storage->rank_ <= 0) {
    return ChunkLayout::ChunkAspectRatioBase();
  }
  const size_t usage_index = static_cast<size_t>(usage);
  return ChunkLayout::ChunkAspectRatioBase(
      storage->chunk_aspect_ratio(usage_index),
      storage->chunk_aspect_ratio_hard_constraint_[usage_index]);
}

constexpr inline HardConstraintBit GetChunkElementsHardConstraintBit(
    Usage usage) {
  return static_cast<HardConstraintBit>(
      static_cast<int>(HardConstraintBit::write_chunk_elements) +
      static_cast<int>(usage));
}

ChunkLayout::ChunkElementsBase GetChunkElements(const ChunkLayout& self,
                                                Usage usage) {
  auto* storage = self.storage_.get();
  if (!storage) return ChunkLayout::ChunkElementsBase();
  const size_t usage_index = static_cast<size_t>(usage);
  return ChunkLayout::ChunkElementsBase(
      storage->chunk_elements_[usage_index],
      IsHardConstraint(*storage, GetChunkElementsHardConstraintBit(usage)));
}

ChunkLayout::GridView GetGridConstraints(const ChunkLayout& self, Usage usage) {
  return ChunkLayout::GridView(GetChunkShape(self, usage),
                               GetChunkAspectRatio(self, usage),
                               GetChunkElements(self, usage));
}

/// Merges the specified inner order constraint into `self`.
///
/// \param self Constraints object into which to merge the constraint.
/// \param value Inner order constraint to add.
/// \param storage_to_be_destroyed[out] Set to the old value of `self.storage_`
///     if `self.storage_` is reassigned due to copy-on-write.  This ensures the
///     storage remains valid in case we are performing some type of self
///     assignment operation.
absl::Status SetInnerOrderInternal(ChunkLayout& self,
                                   ChunkLayout::InnerOrder value,
                                   StoragePtr& storage_to_be_destroyed) {
  if (!IsValidPermutation(value)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid permutation: ", value));
  }
  const DimensionIndex rank = value.size();
  TENSORSTORE_RETURN_IF_ERROR(
      EnsureRank(self.storage_, rank, storage_to_be_destroyed));
  auto& impl = *self.storage_;
  DimensionIndex* inner_order = impl.inner_order();
  if (inner_order[0] != -1) {
    if (!value.hard_constraint) return absl::OkStatus();
    if (IsHardConstraint(impl, HardConstraintBit::inner_order)) {
      if (!std::equal(value.data(), value.data() + rank, inner_order)) {
        return MismatchError(span<const DimensionIndex>(inner_order, rank),
                             span<const DimensionIndex>(value));
      }
      return absl::OkStatus();
    }
  }
  std::copy_n(value.begin(), rank, inner_order);
  if (value.hard_constraint) {
    SetHardConstraintBit(impl, HardConstraintBit::inner_order);
  }
  return absl::OkStatus();
}

absl::Status SetGridOriginInternal(ChunkLayout& self,
                                   MaybeHardConstraintSpan<Index> value,
                                   StoragePtr& storage_to_be_destroyed) {
  const DimensionIndex rank = value.size();
  TENSORSTORE_RETURN_IF_ERROR(
      EnsureRank(self.storage_, rank, storage_to_be_destroyed));
  return ValidateAndMergeVectorInto<OriginValueTraits>(
      value, self.storage_->grid_origin(),
      self.storage_->grid_origin_hard_constraint_);
}

absl::Status SetChunkShapeInternal(ChunkLayout& self,
                                   MaybeHardConstraintSpan<Index> value,
                                   Usage usage,
                                   StoragePtr& storage_to_be_destroyed) {
  const size_t usage_index = static_cast<size_t>(usage);
  const DimensionIndex rank = value.size();
  TENSORSTORE_RETURN_IF_ERROR(
      EnsureRank(self.storage_, rank, storage_to_be_destroyed));
  return ValidateAndMergeVectorInto<ShapeValueTraits>(
      value, self.storage_->chunk_shape(usage_index).data(),
      self.storage_->chunk_shape_hard_constraint_[usage_index]);
}

absl::Status SetChunkShape(ChunkLayout& self,
                           MaybeHardConstraintSpan<Index> value, Usage usage,
                           StoragePtr& storage_to_be_destroyed) {
  TENSORSTORE_RETURN_IF_ERROR(
      SetChunkShapeInternal(self, value, usage, storage_to_be_destroyed),
      tensorstore::MaybeAnnotateStatus(
          _, tensorstore::StrCat("Error setting ", usage, "_chunk shape")));
  return absl::OkStatus();
}

absl::Status SetChunkAspectRatioInternal(ChunkLayout& self,
                                         MaybeHardConstraintSpan<double> value,
                                         Usage usage,
                                         StoragePtr& storage_to_be_destroyed) {
  const size_t usage_index = static_cast<size_t>(usage);
  const DimensionIndex rank = value.size();
  TENSORSTORE_RETURN_IF_ERROR(
      EnsureRank(self.storage_, rank, storage_to_be_destroyed));
  return ValidateAndMergeVectorInto<AspectRatioValueTraits>(
      value, self.storage_->chunk_aspect_ratio(usage_index).data(),
      self.storage_->chunk_aspect_ratio_hard_constraint_[usage_index]);
}

absl::Status SetChunkAspectRatio(ChunkLayout& self,
                                 MaybeHardConstraintSpan<double> value,
                                 Usage usage,
                                 StoragePtr& storage_to_be_destroyed) {
  TENSORSTORE_RETURN_IF_ERROR(
      SetChunkAspectRatioInternal(self, value, usage, storage_to_be_destroyed),
      tensorstore::MaybeAnnotateStatus(
          _,
          tensorstore::StrCat("Error setting ", usage, "_chunk aspect_ratio")));
  return absl::OkStatus();
}

template <typename HardConstraintRef>
absl::Status SetChunkElementsInternal(Index& elements,
                                      HardConstraintRef is_hard_constraint,
                                      ChunkLayout::ChunkElementsBase value) {
  if (value.valid()) {
    if (value < 0) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Invalid value: ", value.value));
    }
    if (elements != kImplicit) {
      if (!value.hard_constraint) return absl::OkStatus();
      if (is_hard_constraint && elements != value.value) {
        return MismatchError(elements, value.value);
      }
    }
    elements = value.value;
    if (value.hard_constraint) {
      is_hard_constraint = true;
    }
  }
  return absl::OkStatus();
}

absl::Status SetChunkElementsInternal(ChunkLayout& self,
                                      ChunkLayout::ChunkElementsBase value,
                                      Usage usage,
                                      StoragePtr& storage_to_be_destroyed) {
  if (!value.valid()) return absl::OkStatus();
  auto& impl = Storage::EnsureUnique(self.storage_, dynamic_rank,
                                     storage_to_be_destroyed);
  return SetChunkElementsInternal(
      impl.chunk_elements_[static_cast<size_t>(usage)],
      impl.hard_constraint_[static_cast<size_t>(
          GetChunkElementsHardConstraintBit(usage))],
      value);
}

absl::Status SetChunkElements(ChunkLayout& self,
                              ChunkLayout::ChunkElementsBase value, Usage usage,
                              StoragePtr& storage_to_be_destroyed) {
  TENSORSTORE_RETURN_IF_ERROR(
      SetChunkElementsInternal(self, value, usage, storage_to_be_destroyed),
      tensorstore::MaybeAnnotateStatus(
          _, tensorstore::StrCat("Error setting ", usage, "_chunk elements")));
  return absl::OkStatus();
}

absl::Status SetGridConstraints(ChunkLayout& self,
                                const ChunkLayout::GridView& value, Usage usage,
                                StoragePtr& storage_to_be_destroyed) {
  if (value.shape().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        SetChunkShape(self, value.shape(), usage, storage_to_be_destroyed));
  }
  if (value.aspect_ratio().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(SetChunkAspectRatio(
        self, value.aspect_ratio(), usage, storage_to_be_destroyed));
  }
  if (value.elements().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(SetChunkElements(self, value.elements(), usage,
                                                 storage_to_be_destroyed));
  }
  return absl::OkStatus();
}

absl::Status SetChunkLayout(ChunkLayout& self, ChunkLayout other,
                            bool hard_constraint) {
  if (!other.storage_) return absl::OkStatus();
  if (!self.storage_) {
    self.storage_ = std::move(other.storage_);
    if (!hard_constraint) {
      StoragePtr storage_to_be_destroyed;
      ClearHardConstraintBits(Storage::EnsureUnique(
          self.storage_, self.storage_->rank_, storage_to_be_destroyed));
    }
    return absl::OkStatus();
  }
  {
    auto inner_order = other.inner_order();
    if (!hard_constraint) inner_order.hard_constraint = false;
    TENSORSTORE_RETURN_IF_ERROR(self.Set(inner_order));
  }
  {
    auto grid_origin = other.grid_origin();
    if (!hard_constraint) grid_origin.hard_constraint = false;
    TENSORSTORE_RETURN_IF_ERROR(self.Set(grid_origin));
  }
  // In this case `storage_to_be_destroyed` is not actually serving a purpose
  // because `other` already ensures its members remain alive.
  StoragePtr storage_to_be_destroyed;
  for (Usage usage : ChunkLayout::kUsages) {
    TENSORSTORE_RETURN_IF_ERROR(SetGridConstraints(
        self,
        ChunkLayout::GridView(GetGridConstraints(other, usage),
                              hard_constraint),
        usage, storage_to_be_destroyed));
  }
  return absl::OkStatus();
}

}  // namespace

DimensionIndex ChunkLayout::rank() const {
  if (storage_) return storage_->rank_;
  return dynamic_rank;
}

absl::Status ChunkLayout::Set(RankConstraint value) {
  if (value.rank == dynamic_rank) return absl::OkStatus();
  StoragePtr storage_to_be_destroyed;
  return EnsureRank(storage_, value.rank, storage_to_be_destroyed);
}

ChunkLayout::InnerOrder ChunkLayout::inner_order() const {
  if (storage_) {
    const DimensionIndex rank = storage_->rank_;
    if (rank > 0) {
      const DimensionIndex* inner_order = storage_->inner_order();
      if (inner_order[0] != -1) {
        return InnerOrder(
            span(inner_order, rank),
            IsHardConstraint(*storage_, HardConstraintBit::inner_order));
      }
    }
  }
  return InnerOrder();
}

absl::Status ChunkLayout::Set(InnerOrder value) {
  if (!value.valid()) return absl::OkStatus();
  StoragePtr storage_to_be_destroyed;
  TENSORSTORE_RETURN_IF_ERROR(
      SetInnerOrderInternal(*this, value, storage_to_be_destroyed),
      tensorstore::MaybeAnnotateStatus(_, "Error setting inner_order"));
  return absl::OkStatus();
}

ChunkLayout::GridOrigin ChunkLayout::grid_origin() const {
  if (storage_) {
    const DimensionIndex rank = storage_->rank_;
    if (rank > 0) {
      return GridOrigin(
          span<const Index>(storage_->grid_origin(), storage_->rank_),
          storage_->grid_origin_hard_constraint_);
    }
  }
  return GridOrigin();
}

absl::Status ChunkLayout::Set(GridOrigin value) {
  if (!value.valid()) return absl::OkStatus();
  StoragePtr storage_to_be_destroyed;
  TENSORSTORE_RETURN_IF_ERROR(
      SetGridOriginInternal(*this, value, storage_to_be_destroyed),
      tensorstore::MaybeAnnotateStatus(_, "Error setting grid_origin"));
  return absl::OkStatus();
}

#define TENSORSTORE_INTERNAL_DO_DEFINE_FOR_USAGE(NAME, USAGE)                 \
  template <>                                                                 \
  absl::Status ChunkLayout::Set<USAGE>(const GridViewFor<USAGE>& value) {     \
    StoragePtr storage_to_be_destroyed;                                       \
    return SetGridConstraints(*this, value, USAGE, storage_to_be_destroyed);  \
  }                                                                           \
  ChunkLayout::GridViewFor<USAGE> ChunkLayout::NAME##_chunk() const {         \
    return ChunkLayout::GridViewFor<USAGE>(GetGridConstraints(*this, USAGE)); \
  }                                                                           \
  ChunkLayout::ChunkShapeFor<USAGE> ChunkLayout::NAME##_chunk_shape() const { \
    return ChunkLayout::ChunkShapeFor<USAGE>(GetChunkShape(*this, USAGE));    \
  }                                                                           \
  ChunkLayout::ChunkAspectRatioFor<USAGE>                                     \
      ChunkLayout::NAME##_chunk_aspect_ratio() const {                        \
    return ChunkLayout::ChunkAspectRatioFor<USAGE>(                           \
        GetChunkAspectRatio(*this, USAGE));                                   \
  }                                                                           \
  ChunkLayout::ChunkElementsFor<USAGE> ChunkLayout::NAME##_chunk_elements()   \
      const {                                                                 \
    return ChunkLayout::ChunkElementsFor<USAGE>(                              \
        GetChunkElements(*this, USAGE));                                      \
  }                                                                           \
  /**/

TENSORSTORE_INTERNAL_DO_DEFINE_FOR_USAGE(write, Usage::kWrite)
TENSORSTORE_INTERNAL_DO_DEFINE_FOR_USAGE(read, Usage::kRead)
TENSORSTORE_INTERNAL_DO_DEFINE_FOR_USAGE(codec, Usage::kCodec)

#undef TENSORSTORE_INTERNAL_DO_DEFINE_FOR_USAGE

template <>
absl::Status ChunkLayout::Set<ChunkLayout::kUnspecifiedUsage>(
    const GridViewFor<ChunkLayout::kUnspecifiedUsage>& value) {
  // Necessary in case `value` refers to `this->storage_`.
  StoragePtr storage_to_be_destroyed;
  if (value.usage() == kUnspecifiedUsage) {
    TENSORSTORE_RETURN_IF_ERROR(SetGridConstraints(*this, value, Usage::kWrite,
                                                   storage_to_be_destroyed));
    TENSORSTORE_RETURN_IF_ERROR(SetGridConstraints(*this, value, Usage::kRead,
                                                   storage_to_be_destroyed));
    TENSORSTORE_RETURN_IF_ERROR(
        SetGridConstraints(*this, CodecChunk(value.aspect_ratio()),
                           Usage::kCodec, storage_to_be_destroyed));
    return absl::OkStatus();
  }
  return SetGridConstraints(*this, value, value.usage(),
                            storage_to_be_destroyed);
}

ChunkLayout::GridView ChunkLayout::operator[](Usage usage) const {
  assert(usage != kUnspecifiedUsage);
  return GetGridConstraints(*this, usage);
}

ChunkLayout::ChunkLayout(ChunkLayout layout, bool hard_constraint) {
  storage_ = std::move(layout.storage_);
  if (!hard_constraint && storage_) {
    StoragePtr storage_to_be_destroyed;
    ClearHardConstraintBits(Storage::EnsureUnique(storage_, storage_->rank_,
                                                  storage_to_be_destroyed));
  }
}

absl::Status ChunkLayout::Set(ChunkLayout value) {
  return SetChunkLayout(*this, value, /*hard_constraint=*/true);
}

namespace {

template <typename BaseBinder>
constexpr auto HardSoftMemberPairJsonBinder(const char* name,
                                            const char* soft_constraint_name,
                                            BaseBinder base_binder) {
  return jb::Sequence(
      jb::Member(name, base_binder(/*hard_constraint=*/true)),
      jb::Member(soft_constraint_name, base_binder(/*hard_constraint=*/false)));
}

template <typename ElementBinder>
constexpr auto DimensionIndexedFixedArrayJsonBinder(
    DimensionIndex& rank, ElementBinder element_binder) {
  return jb::DimensionIndexedVector(
      &rank,
      /*get_size=*/
      [](auto& x) -> size_t { ABSL_UNREACHABLE(); },  // COV_NF_LINE
      /*set_size=*/[](auto& x, size_t n) { return absl::OkStatus(); },
      /*get_element=*/
      [](auto& x, size_t i) -> decltype(auto) { return (&x)[i]; },
      element_binder);
}

template <typename Traits>
bool VectorIsDefault(span<const typename Traits::Element> vec) {
  return std::all_of(vec.begin(), vec.end(), [](typename Traits::Element x) {
    return x == Traits::kDefaultValue;
  });
}

bool GridConstraintsUnset(const ChunkLayout& self, Usage usage) {
  return VectorIsDefault<ShapeValueTraits>(GetChunkShape(self, usage)) &&
         VectorIsDefault<AspectRatioValueTraits>(
             GetChunkAspectRatio(self, usage)) &&
         !GetChunkElements(self, usage).valid();
}

bool AllRankDependentConstraintsUnset(Storage& storage) {
  const DimensionIndex rank = storage.rank_;
  if (rank <= 0) return true;
  if (storage.inner_order()[0] != -1) return false;
  if (auto* origin = storage.grid_origin();
      std::any_of(origin, origin + rank, [](auto x) {
        return x != OriginValueTraits::kDefaultValue;
      })) {
    return false;
  }
  if (auto* shapes = storage.chunk_shapes();
      std::any_of(shapes, shapes + Storage::NumShapeElements(rank), [](auto x) {
        return x != ShapeValueTraits::kDefaultValue;
      })) {
    return false;
  }
  if (auto* aspect_ratios = storage.chunk_aspect_ratios(); std::any_of(
          aspect_ratios, aspect_ratios + Storage::NumAspectRatioElements(rank),
          [](auto x) { return x != AspectRatioValueTraits::kDefaultValue; })) {
    return false;
  }
  return true;
}

bool AllConstraintsUnset(const ChunkLayout& self) {
  if (!self.storage_) return true;
  auto& storage = *self.storage_;
  if (storage.rank_ != dynamic_rank) return false;
  if (std::any_of(storage.chunk_elements_, storage.chunk_elements_ + kNumUsages,
                  [](Index x) { return x != kImplicit; })) {
    return false;
  }
  const DimensionIndex rank = storage.rank_;
  if (rank <= 0) return true;
  return AllRankDependentConstraintsUnset(storage);
}

template <typename Wrapper, typename Traits, typename Getter, typename Setter>
constexpr auto VectorJsonBinder(Getter getter, Setter setter) {
  using ElementType = typename Wrapper::value_type;
  return [=](bool hard_constraint) {
    return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
      constexpr auto element_binder = jb::MapValue(
          jb::DefaultBinder<>, std::pair(Traits::kDefaultValue, nullptr));
      if constexpr (is_loading) {
        if (j->is_discarded()) return absl::OkStatus();
        ElementType value[kMaxRank];
        DimensionIndex rank = dynamic_rank;
        TENSORSTORE_RETURN_IF_ERROR(DimensionIndexedFixedArrayJsonBinder(
            rank, element_binder)(is_loading, options, &value[0], j));
        return setter(*obj, Wrapper(span<const ElementType>(value, rank),
                                    hard_constraint));
      } else {
        auto vec = getter(*obj);
        if (!vec.valid()) {
          return absl::OkStatus();
        }
        ElementType new_vec[kMaxRank];
        bool has_value = false;
        for (DimensionIndex i = 0; i < vec.size(); ++i) {
          if (vec.hard_constraint[i] == hard_constraint &&
              vec[i] != Traits::kDefaultValue) {
            new_vec[i] = vec[i];
            has_value = true;
          } else {
            new_vec[i] = Traits::kDefaultValue;
          }
        }
        if (!has_value) return absl::OkStatus();
        span<const ElementType> new_span(new_vec, vec.size());
        return jb::Array(element_binder)(is_loading, options, &new_span, j);
      }
    };
  };
}

constexpr auto InnerOrderJsonBinder(bool hard_constraint) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    if constexpr (is_loading) {
      if (j->is_discarded() || j->is_null()) {
        return absl::OkStatus();
      }
      DimensionIndex value[kMaxRank];
      DimensionIndex rank = dynamic_rank;
      TENSORSTORE_RETURN_IF_ERROR(DimensionIndexedFixedArrayJsonBinder(
          rank, jb::Integer<DimensionIndex>(0, kMaxRank - 1))(
          is_loading, options, &value[0], j));
      StoragePtr storage_to_be_destroyed;
      return SetInnerOrderInternal(
          *obj,
          ChunkLayout::InnerOrder(span<const DimensionIndex>(value, rank),
                                  hard_constraint),
          storage_to_be_destroyed);
    } else {
      auto vec = obj->inner_order();
      if (vec.valid() && vec.hard_constraint == hard_constraint) {
        *j = static_cast<::nlohmann::json>(vec);
      }
      return absl::OkStatus();
    }
  };
}

constexpr auto StandaloneGridJsonBinder() {
  return jb::Object(
      HardSoftMemberPairJsonBinder(
          "shape", "shape_soft_constraint",
          VectorJsonBinder<ChunkLayout::ChunkShapeBase, ShapeValueTraits>(
              [](auto& self) { return self.shape(); },
              [](auto& self, ChunkLayout::ChunkShapeBase value) {
                return self.Set(value);
              })),
      HardSoftMemberPairJsonBinder(
          "aspect_ratio", "aspect_ratio_soft_constraint",
          VectorJsonBinder<ChunkLayout::ChunkAspectRatioBase,
                           AspectRatioValueTraits>(
              [](auto& self) { return self.aspect_ratio(); },
              [](auto& self, ChunkLayout::ChunkAspectRatioBase value) {
                return self.Set(value);
              })),
      HardSoftMemberPairJsonBinder(
          "elements", "elements_soft_constraint", [](bool hard_constraint) {
            return jb::GetterSetter(
                [=](auto& self) -> Index {
                  auto value = self.elements();
                  if (value.hard_constraint != hard_constraint)
                    return kImplicit;
                  return value.value;
                },
                [=](auto& self, Index value) {
                  return self.Set(
                      ChunkLayout::ChunkElementsBase(value, hard_constraint));
                },
                jb::DefaultPredicate<jb::kNeverIncludeDefaults>(
                    /*get_default=*/[](auto* obj) { *obj = kImplicit; },
                    /*is_default=*/[](auto*
                                          obj) { return *obj == kImplicit; }));
          }));
}

constexpr auto GridConstraintsJsonBinder(Usage usage) {
  return jb::Object(
      HardSoftMemberPairJsonBinder(
          "shape", "shape_soft_constraint",
          VectorJsonBinder<ChunkLayout::ChunkShapeBase, ShapeValueTraits>(
              [=](auto& self) { return GetChunkShape(self, usage); },
              [=](auto& self, ChunkLayout::ChunkShapeBase value) {
                StoragePtr storage_to_be_destroyed;
                if (usage != ChunkLayout::kUnspecifiedUsage) {
                  return SetChunkShapeInternal(self, value, usage,
                                               storage_to_be_destroyed);
                }
                TENSORSTORE_RETURN_IF_ERROR(SetChunkShapeInternal(
                    self, value, Usage::kWrite, storage_to_be_destroyed));
                TENSORSTORE_RETURN_IF_ERROR(SetChunkShapeInternal(
                    self, value, Usage::kRead, storage_to_be_destroyed));
                return absl::OkStatus();
              })),
      HardSoftMemberPairJsonBinder(
          "aspect_ratio", "aspect_ratio_soft_constraint",
          VectorJsonBinder<ChunkLayout::ChunkAspectRatioBase,
                           AspectRatioValueTraits>(
              [=](auto& self) { return GetChunkAspectRatio(self, usage); },
              [=](auto& self, ChunkLayout::ChunkAspectRatioBase value) {
                StoragePtr storage_to_be_destroyed;
                if (usage != ChunkLayout::kUnspecifiedUsage) {
                  return SetChunkAspectRatioInternal(self, value, usage,
                                                     storage_to_be_destroyed);
                }
                TENSORSTORE_RETURN_IF_ERROR(SetChunkAspectRatioInternal(
                    self, value, Usage::kWrite, storage_to_be_destroyed));
                TENSORSTORE_RETURN_IF_ERROR(SetChunkAspectRatioInternal(
                    self, value, Usage::kRead, storage_to_be_destroyed));
                TENSORSTORE_RETURN_IF_ERROR(SetChunkAspectRatioInternal(
                    self, value, Usage::kCodec, storage_to_be_destroyed));
                return absl::OkStatus();
              })),
      HardSoftMemberPairJsonBinder(
          "elements", "elements_soft_constraint", [=](bool hard_constraint) {
            return jb::GetterSetter(
                [=](auto& self) -> Index {
                  auto value = GetChunkElements(self, usage);
                  if (value.hard_constraint != hard_constraint)
                    return kImplicit;
                  return value.value;
                },
                [=](auto& self, Index value) {
                  ChunkLayout::ChunkElementsBase elements(value,
                                                          hard_constraint);
                  StoragePtr storage_to_be_destroyed;
                  if (usage != ChunkLayout::kUnspecifiedUsage) {
                    return SetChunkElementsInternal(self, elements, usage,
                                                    storage_to_be_destroyed);
                  }
                  TENSORSTORE_RETURN_IF_ERROR(SetChunkElementsInternal(
                      self, elements, Usage::kWrite, storage_to_be_destroyed));
                  TENSORSTORE_RETURN_IF_ERROR(SetChunkElementsInternal(
                      self, elements, Usage::kRead, storage_to_be_destroyed));
                  return absl::OkStatus();
                },
                jb::DefaultPredicate<jb::kNeverIncludeDefaults>(
                    /*get_default=*/[](auto* obj) { *obj = kImplicit; },
                    /*is_default=*/[](auto*
                                          obj) { return *obj == kImplicit; }));
          }));
}

constexpr auto DefaultableGridConstraintsJsonBinder(Usage usage) {
  return jb::DefaultPredicate<jb::kNeverIncludeDefaults>(
      /*get_default=*/[](auto* obj) {},
      /*is_default=*/
      [=](auto* obj) {
        if (!obj->storage_) return true;
        return GridConstraintsUnset(*obj, usage);
      },
      GridConstraintsJsonBinder(usage));
}

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ChunkLayout,
    jb::Object(jb::Member("rank",
                          jb::Compose<DimensionIndex>(
                              [](auto is_loading, const auto& options,
                                 auto* obj, auto* rank) {
                                if constexpr (is_loading) {
                                  return obj->Set(RankConstraint{*rank});
                                } else {
                                  const DimensionIndex rank_value = obj->rank();
                                  *rank = (rank_value == dynamic_rank ||
                                           !AllRankDependentConstraintsUnset(
                                               *obj->storage_))
                                              ? dynamic_rank
                                              : rank_value;
                                  return absl::OkStatus();
                                }
                              },
                              jb::ConstrainedRankJsonBinder)),
               HardSoftMemberPairJsonBinder("inner_order",
                                            "inner_order_soft_constraint",
                                            InnerOrderJsonBinder),
               HardSoftMemberPairJsonBinder(
                   "grid_origin", "grid_origin_soft_constraint",
                   VectorJsonBinder<ChunkLayout::GridOrigin, OriginValueTraits>(
                       [](auto& self) { return self.grid_origin(); },
                       [](auto& self, ChunkLayout::GridOrigin value) {
                         StoragePtr storage_to_be_destroyed;
                         return SetGridOriginInternal(self, value,
                                                      storage_to_be_destroyed);
                       })),
               jb::LoadSave(jb::Member("chunk",
                                       DefaultableGridConstraintsJsonBinder(
                                           ChunkLayout::kUnspecifiedUsage))),
               jb::Member("write_chunk",
                          DefaultableGridConstraintsJsonBinder(Usage::kWrite)),
               jb::Member("read_chunk",
                          DefaultableGridConstraintsJsonBinder(Usage::kRead)),
               jb::Member("codec_chunk",
                          DefaultableGridConstraintsJsonBinder(Usage::kCodec))))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ChunkLayout::Grid,
                                       StandaloneGridJsonBinder())

namespace {
/// Transforms a vector of soft/hard constraints for dimensions of the input
/// space to a corresponding vector of soft/hard constraints for the output
/// space.
///
/// \tparam Traits Class with `static constexpr Element kDefaultValue` member
///     specifying the default value to use if there is no constraint, and
///     `static Result<Element>(Element value, Index offset, Index stride)` that
///     transforms an input constraint to an output constraint.  This function
///     is only called for constraint values not equal to `kDefaultValue`.
/// \param transform The transform.
/// \param in_vec Vector of constraint values of length
///     `transform.input_rank()`.
/// \param in_hard_constraint Indicates which elements of `in_vec` are hard
///     constraints.
/// \param out_vec Vector of length `transform.output_rank()` to be set to the
///     constraint values for the output dimensions.
/// \param out_hard_constraint Will be set to indicate which elements of
///     `out_vec` are hard constraints.
template <typename Traits>
static absl::Status TransformInputVector(
    IndexTransformView<> transform, span<const typename Traits::Element> in_vec,
    DimensionSet in_hard_constraint, span<typename Traits::Element> out_vec,
    DimensionSet& out_hard_constraint) {
  using Element = typename Traits::Element;
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  assert(input_rank == in_vec.size());
  assert(output_rank == out_vec.size());
  // Copy `in_vec` in case it is aliased by `out_vec`.
  Element in_vec_copy[kMaxRank];
  std::copy_n(in_vec.begin(), input_rank, in_vec_copy);
  // Initialize output constraints to the default value, used in case there is
  // no corresponding input constraint.
  std::fill_n(out_vec.begin(), output_rank, Traits::kDefaultValue);
  out_hard_constraint = false;
  DimensionSet remaining_in_hard_constraint = in_hard_constraint;
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension ||
        map.stride() == 0) {
      // Output dimensions not mapped by a `single_input_dimension` map don't
      // correspond to a single input dimension and are therefore just left
      // unconstrained.
      continue;
    }
    const DimensionIndex input_dim = map.input_dimension();
    Element value = in_vec_copy[input_dim];
    remaining_in_hard_constraint[input_dim] = false;
    // If there is no constraint on the corresponding input dimension, just
    // leave the output dimension unconstrained.
    if (value == Traits::kDefaultValue) continue;
    TENSORSTORE_ASSIGN_OR_RETURN(
        value, Traits::TransformInputValue(value, map.offset(), map.stride()),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error transforming input dimension ",
                                   input_dim, " -> output dimension ",
                                   output_dim)));
    out_vec[output_dim] = value;
    // The output constraint is a hard constraint if, and only if, the input
    // constraint is a hard constraint.
    if (in_hard_constraint[input_dim] && value != Traits::kDefaultValue) {
      out_hard_constraint[output_dim] = true;
    }
  }
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    if (in_vec[input_dim] == Traits::kDefaultValue) continue;
    if (!remaining_in_hard_constraint[input_dim]) continue;
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "No output dimension corresponds to input dimension ", input_dim));
  }
  return absl::OkStatus();
}

/// Transforms a vector of soft/hard constraints for dimensions of the output
/// space to a corresponding vector of soft/hard constraints for the input
/// space.
///
/// \tparam Traits Class with `static constexpr Element kDefaultValue` member
///     specifying the default value to use if there is no constraint, and
///     `static Result<Element>(Element value, Index offset, Index stride)`
///     method that transforms an output constraint to an input constraint.
///     This function is only called for constraint values not equal to
///     `kDefaultValue`
/// \param transform The transform.
/// \param one_to_one_input_dims Must be equal to
///     `GetOneToOneInputDimensions(transform).one_to_one`.  This is a parameter
///     to avoid having to redundantly compute it multiple times.
/// \param out_vec Vector of constraint values of length
///     `transform.output_rank()`.
/// \param out_hard_constraint Indicates which elements of `out_vec` are hard
///     constraints.
/// \param in_vec Vector of length `transform.output_rank()` to be set to the
///     constraint values for the input dimensions.
/// \param in_hard_constraint Will be set to indicate which elements of `in_vec`
///     are hard constraints.
template <typename Traits>
static absl::Status TransformOutputVector(
    IndexTransformView<> transform, DimensionSet one_to_one_input_dims,
    span<const typename Traits::Element> out_vec,
    DimensionSet out_hard_constraint, span<typename Traits::Element> in_vec,
    DimensionSet& in_hard_constraint) {
  using Element = typename Traits::Element;
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex output_rank = transform.output_rank();
  assert(output_rank == out_vec.size());
  assert(input_rank == in_vec.size());
  // Copy `out_vec` in case it is aliased by `in_vec`.
  Element out_vec_copy[kMaxRank];
  std::copy_n(out_vec.begin(), output_rank, out_vec_copy);
  std::fill_n(in_vec.begin(), input_rank, Traits::kDefaultValue);
  in_hard_constraint = false;
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension ||
        map.stride() == 0) {
      // Output dimensions not mapped by a `single_input_dimension` map don't
      // correspond to a single input dimension and therefore we don't attempt
      // to propagate their constraints.
      continue;
    }
    const DimensionIndex input_dim = map.input_dimension();
    // If more than one output dimension maps to this input dimension (i.e. the
    // input dimension represents a diagonal), don't attempt to propagate the
    // output constraint.
    if (!one_to_one_input_dims[input_dim]) continue;
    Element value = out_vec_copy[output_dim];
    // If there is no constraint on the output dimension, just leave the input
    // dimension unconstrained.
    if (value == Traits::kDefaultValue) continue;
    TENSORSTORE_ASSIGN_OR_RETURN(
        value, Traits::TransformOutputValue(value, map.offset(), map.stride()),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error transforming output dimension ",
                                   output_dim, " -> input dimension ",
                                   input_dim)));
    in_vec[input_dim] = value;
    // The input constraint is a hard constraint if, and only if, the output
    // constraint is a hard constraint.
    if (value != Traits::kDefaultValue) {
      in_hard_constraint[input_dim] = out_hard_constraint[output_dim];
    }
  }
  return absl::OkStatus();
}

absl::Status TransformOutputGridConstraints(Storage& output_storage,
                                            Storage& input_storage,
                                            DimensionSet one_to_one_input_dims,
                                            IndexTransformView<> transform,
                                            size_t usage_index) {
  input_storage.chunk_elements_[usage_index] =
      output_storage.chunk_elements_[usage_index];
  TENSORSTORE_RETURN_IF_ERROR(
      TransformOutputVector<ShapeValueTraits>(
          transform, one_to_one_input_dims,
          output_storage.chunk_shape(usage_index),
          output_storage.chunk_shape_hard_constraint_[usage_index],
          input_storage.chunk_shape(usage_index),
          input_storage.chunk_shape_hard_constraint_[usage_index]),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming shape"));
  TENSORSTORE_RETURN_IF_ERROR(
      TransformOutputVector<AspectRatioValueTraits>(
          transform, one_to_one_input_dims,
          output_storage.chunk_aspect_ratio(usage_index),
          output_storage.chunk_aspect_ratio_hard_constraint_[usage_index],
          input_storage.chunk_aspect_ratio(usage_index),
          input_storage.chunk_aspect_ratio_hard_constraint_[usage_index]),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming aspect_ratio"));
  return absl::OkStatus();
}

absl::Status TransformInputGridConstraints(Storage& input_storage,
                                           Storage& output_storage,
                                           IndexTransformView<> transform,
                                           size_t usage_index) {
  output_storage.chunk_elements_[usage_index] =
      input_storage.chunk_elements_[usage_index];
  TENSORSTORE_RETURN_IF_ERROR(
      TransformInputVector<ShapeValueTraits>(
          transform, input_storage.chunk_shape(usage_index),
          input_storage.chunk_shape_hard_constraint_[usage_index],
          output_storage.chunk_shape(usage_index),
          output_storage.chunk_shape_hard_constraint_[usage_index]),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming shape"));
  TENSORSTORE_RETURN_IF_ERROR(
      TransformInputVector<AspectRatioValueTraits>(
          transform, input_storage.chunk_aspect_ratio(usage_index),
          input_storage.chunk_aspect_ratio_hard_constraint_[usage_index],
          output_storage.chunk_aspect_ratio(usage_index),
          output_storage.chunk_aspect_ratio_hard_constraint_[usage_index]),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming aspect_ratio"));
  return absl::OkStatus();
}

}  // namespace

Result<ChunkLayout> ApplyIndexTransform(IndexTransformView<> transform,
                                        ChunkLayout output_constraints) {
  if (!transform.valid() || !output_constraints.storage_) {
    return output_constraints;
  }
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex output_constraints_rank = output_constraints.rank();
  if (!RankConstraint::EqualOrUnspecified(output_constraints_rank,
                                          transform.output_rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot transform constraints of rank ", output_constraints_rank,
        " by index transform of rank ", input_rank, " -> ", output_rank));
  }
  if (output_constraints_rank <= 0) return output_constraints;
  ChunkLayout input_constraints;
  Storage* output_storage;
  if (output_rank == input_rank) {
    StoragePtr storage_to_be_destroyed;
    input_constraints = std::move(output_constraints);
    output_storage = &Storage::EnsureUnique(
        input_constraints.storage_, input_rank, storage_to_be_destroyed);
  } else {
    input_constraints.storage_ = Storage::Allocate(input_rank);
    new (input_constraints.storage_.get()) Storage(input_rank);
    output_storage = output_constraints.storage_.get();
  }
  input_constraints.storage_->hard_constraint_ =
      output_storage->hard_constraint_;
  if (auto* inner_order = output_storage->inner_order(); inner_order[0] != -1) {
    TransformOutputDimensionOrder(
        transform, {inner_order, output_rank},
        {input_constraints.storage_->inner_order(), input_rank});
  }
  DimensionSet one_to_one_input_dims =
      internal::GetOneToOneInputDimensions(transform).one_to_one;
  TENSORSTORE_RETURN_IF_ERROR(
      TransformOutputVector<OriginValueTraits>(
          transform, one_to_one_input_dims,
          span<const Index>(output_storage->grid_origin(), output_rank),
          output_storage->grid_origin_hard_constraint_,
          span<Index>(input_constraints.storage_->grid_origin(), input_rank),
          input_constraints.storage_->grid_origin_hard_constraint_),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming grid_origin"));
  for (size_t usage_index = 0; usage_index < kNumUsages; ++usage_index) {
    TENSORSTORE_RETURN_IF_ERROR(
        TransformOutputGridConstraints(
            *output_storage, *input_constraints.storage_, one_to_one_input_dims,
            transform, usage_index),
        tensorstore::MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error transforming ",
                                   static_cast<Usage>(usage_index), "_chunk")));
  }
  return input_constraints;
}

Result<ChunkLayout> ApplyInverseIndexTransform(IndexTransformView<> transform,
                                               ChunkLayout input_constraints) {
  if (!transform.valid() || !input_constraints.storage_) {
    return input_constraints;
  }
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex input_constraints_rank = input_constraints.rank();
  if (!RankConstraint::EqualOrUnspecified(input_constraints_rank,
                                          transform.input_rank())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot transform constraints of rank ", input_constraints_rank,
        " by index transform of rank ", input_rank, " -> ", output_rank));
  }
  if (input_constraints_rank <= 0) return input_constraints;
  ChunkLayout output_constraints;
  Storage* input_storage;
  if (output_rank == input_rank) {
    StoragePtr storage_to_be_destroyed;
    output_constraints = std::move(input_constraints);
    input_storage = &Storage::EnsureUnique(
        output_constraints.storage_, output_rank, storage_to_be_destroyed);
  } else {
    output_constraints.storage_ = Storage::Allocate(output_rank);
    new (output_constraints.storage_.get()) Storage(output_rank);
    input_storage = input_constraints.storage_.get();
  }
  output_constraints.storage_->hard_constraint_ =
      input_storage->hard_constraint_;
  if (auto* inner_order = input_storage->inner_order(); inner_order[0] != -1) {
    TransformInputDimensionOrder(
        transform, {inner_order, input_rank},
        {output_constraints.storage_->inner_order(), output_rank});
  }
  TENSORSTORE_RETURN_IF_ERROR(
      TransformInputVector<OriginValueTraits>(
          transform,
          span<const Index>(input_storage->grid_origin(), input_rank),
          input_storage->grid_origin_hard_constraint_,
          span<Index>(output_constraints.storage_->grid_origin(), output_rank),
          output_constraints.storage_->grid_origin_hard_constraint_),
      tensorstore::MaybeAnnotateStatus(_, "Error transforming grid_origin"));
  for (size_t usage_index = 0; usage_index < kNumUsages; ++usage_index) {
    TENSORSTORE_RETURN_IF_ERROR(
        TransformInputGridConstraints(*input_storage,
                                      *output_constraints.storage_, transform,
                                      usage_index),
        tensorstore::MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error transforming ",
                                   static_cast<Usage>(usage_index), "_chunk")));
  }
  return output_constraints;
}

bool operator==(const ChunkLayout& a, const ChunkLayout& b) {
  if (!a.storage_) {
    if (!b.storage_) return true;
    return AllConstraintsUnset(b);
  }
  if (!b.storage_) {
    return AllConstraintsUnset(a);
  }
  auto& a_storage = *a.storage_;
  auto& b_storage = *b.storage_;
  if (a_storage.hard_constraint_ != b_storage.hard_constraint_ ||
      a_storage.grid_origin_hard_constraint_ !=
          b_storage.grid_origin_hard_constraint_ ||
      !internal::RangesEqual(span(a_storage.chunk_shape_hard_constraint_),
                             span(b_storage.chunk_shape_hard_constraint_)) ||
      !internal::RangesEqual(
          span(a_storage.chunk_aspect_ratio_hard_constraint_),
          span(b_storage.chunk_aspect_ratio_hard_constraint_)) ||
      !std::equal(a_storage.chunk_elements_,
                  a_storage.chunk_elements_ + kNumUsages,
                  b_storage.chunk_elements_)) {
    return false;
  }
  const DimensionIndex rank = a_storage.rank_;
  if (rank <= 0 || rank != b_storage.rank_) {
    return AllRankDependentConstraintsUnset(a_storage) &&
           AllRankDependentConstraintsUnset(b_storage);
  }
  if (auto* a_inner_order = a_storage.inner_order(); !std::equal(
          a_inner_order, a_inner_order + rank, b_storage.inner_order())) {
    return false;
  }
  if (auto* a_origin = a_storage.grid_origin();
      !std::equal(a_origin, a_origin + rank, b_storage.grid_origin())) {
    return false;
  }
  if (auto* a_shapes = a_storage.chunk_shapes();
      !std::equal(a_shapes, a_shapes + Storage::NumShapeElements(rank),
                  b_storage.chunk_shapes())) {
    return false;
  }
  if (auto* a_aspect_ratios = a_storage.chunk_aspect_ratios();
      !std::equal(a_aspect_ratios,
                  a_aspect_ratios + Storage::NumAspectRatioElements(rank),
                  b_storage.chunk_aspect_ratios())) {
    return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const ChunkLayout& x) {
  return os << ::nlohmann::json(x).dump();
}

namespace internal {
constexpr Index kDefaultChunkElements = 1024 * 1024;

namespace {
void ChooseChunkSizeFromAspectRatio(span<const double> aspect_ratio,
                                    span<Index> chunk_shape,
                                    Index target_chunk_elements,
                                    BoxView<> domain) {
  const DimensionIndex rank = chunk_shape.size();
  assert(aspect_ratio.size() == rank);
  assert(domain.rank() == rank);
  // Determine upper bounds on the chunk size for each dimension.
  double max_chunk_shape[kMaxRank];
  for (DimensionIndex i = 0; i < rank; ++i) {
    double max_size = target_chunk_elements;
    if (IndexInterval bounds = domain[i]; IsFinite(bounds)) {
      max_size =
          std::min(max_size, std::max(1.0, static_cast<double>(bounds.size())));
    }
    max_size = std::min(max_size, 0x1.0p62);
    max_chunk_shape[i] = max_size;
  }

  // Computes the chunk size for a given dimension and scale factor.  We will
  // perform a binary search over `factor` values in order to determine the
  // optimal chunk sizes.
  const auto get_chunk_size = [&](DimensionIndex i, double factor) -> Index {
    if (const Index size = chunk_shape[i]; size != 0) return size;
    // Note: conversion from `double` to `Index` is guaranteed not to overflow
    // because `max_chunk_shape[i]` is guaranteed to be <=
    // `std::numeric_limits<Index>::max()`.
    return std::max(
        Index(1), static_cast<Index>(
                      std::min(aspect_ratio[i] * factor, max_chunk_shape[i])));
  };

  // Computes the total number of elements per chunk for a given factor.  This
  // guides the binary search over `factor` values.
  const auto get_total_elements = [&](double factor) -> Index {
    Index total = 1;
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (internal::MulOverflow(get_chunk_size(i, factor), total, &total)) {
        total = std::numeric_limits<Index>::max();
        break;
      }
    }
    return total;
  };

  // Determine minimum and maximum factors.  We will binary search over this
  // range to find the optimal factor.
  double min_factor_increment = std::numeric_limits<double>::infinity();
  double max_factor = 0;
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (chunk_shape[i] != 0) continue;
    const double factor = aspect_ratio[i];
    min_factor_increment = std::min(min_factor_increment, 1.0 / factor);
    max_factor = std::max(max_factor, max_chunk_shape[i] / factor);
  }
  // Add some leeway room to account for rounding.
  min_factor_increment /= 2;
  max_factor *= 2;

  double min_factor = min_factor_increment;
  Index min_factor_elements = get_total_elements(min_factor);
  Index max_factor_elements = get_total_elements(max_factor);

  // Binary search to find best factor.
  while (min_factor + min_factor_increment < max_factor) {
    double mid_factor = min_factor + (max_factor - min_factor) / 2.0;
    Index mid_factor_elements = get_total_elements(mid_factor);
    if (mid_factor_elements >= target_chunk_elements) {
      max_factor = mid_factor;
      max_factor_elements = mid_factor_elements;
    }
    if (mid_factor_elements <= target_chunk_elements) {
      min_factor = mid_factor;
      min_factor_elements = mid_factor_elements;
    }
  }

  // At this point, `min_factor` and `max_factor` are sufficiently close that
  // the corresponding chunk sizes differ by at most `1` in each dimension.
  // Choose factor that brings us closest to `target_chunk_elements`.
  const double factor =
      (std::abs(min_factor_elements - target_chunk_elements) <=
       std::abs(max_factor_elements - target_chunk_elements))
          ? min_factor
          : max_factor;

  for (DimensionIndex i = 0; i < rank; ++i) {
    chunk_shape[i] = get_chunk_size(i, factor);
  }
}
}  // namespace

absl::Status ChooseChunkShape(ChunkLayout::GridView shape_constraints,
                              BoxView<> domain, span<Index> chunk_shape) {
  const DimensionIndex rank = chunk_shape.size();
  assert(domain.rank() == rank);
  DimensionSet shape_hard_constraint = false;
  if (shape_constraints.shape().valid()) {
    if (shape_constraints.shape().size() != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank of constraints (", shape_constraints.shape().size(),
          ") does not match rank of domain (", rank, ")"));
    }
    std::copy_n(shape_constraints.shape().begin(), rank, chunk_shape.begin());
    shape_hard_constraint = shape_constraints.shape().hard_constraint;
  } else {
    std::fill_n(chunk_shape.begin(), rank, 0);
  }
  // Set the dimensions of chunk size that are explicitly constrained.
  for (DimensionIndex i = 0; i < rank; ++i) {
    Index& chunk_size = chunk_shape[i];
    if (chunk_size == 0) continue;
    if (chunk_size == -1) {
      IndexInterval bounds = domain[i];
      if (!IsFinite(bounds)) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Cannot match chunk size for dimension ", i,
                                " to unbounded domain ", bounds));
      }
      chunk_size = std::max(Index(1), bounds.size());
    }
  }

  if (std::any_of(chunk_shape.begin(), chunk_shape.end(),
                  [](Index x) { return x == 0; })) {
    // Choose chunk size of remaining dimensions according to
    // `shape_constraints.aspect_ratio()` and `shape_constraints.elements()`.

    // Determine the aspect ratio value to use for each dimension.
    double aspect_ratio[kMaxRank];
    if (shape_constraints.aspect_ratio().valid()) {
      if (shape_constraints.aspect_ratio().size() != rank) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Rank of constraints (", shape_constraints.aspect_ratio().size(),
            ") does not match rank of domain (", rank, ")"));
      }
      std::copy_n(shape_constraints.aspect_ratio().begin(), rank, aspect_ratio);
      for (DimensionIndex i = 0; i < rank; ++i) {
        if (aspect_ratio[i] == 0) {
          aspect_ratio[i] = 1;
        }
      }
    } else {
      std::fill_n(aspect_ratio, rank, 1);
    }

    Index target_chunk_elements = kDefaultChunkElements;
    if (shape_constraints.elements().valid()) {
      target_chunk_elements = shape_constraints.elements();
    }
    ChooseChunkSizeFromAspectRatio(span<const double>(aspect_ratio, rank),
                                   chunk_shape, target_chunk_elements, domain);
  }
  return absl::OkStatus();
}

absl::Status ChooseChunkGrid(span<const Index> origin_constraints,
                             ChunkLayout::GridView shape_constraints,
                             BoxView<> domain,
                             MutableBoxView<> chunk_template) {
  // First determine the chunk shape, which we can determine independent of the
  // grid origin.
  const DimensionIndex rank = chunk_template.rank();
  TENSORSTORE_RETURN_IF_ERROR(
      ChooseChunkShape(shape_constraints, domain, chunk_template.shape()));

  // Now that we have determined the chunk size, determine the grid origin.

  // For any dimension with an explicitly-constrained origin, just set the
  // origin to that.
  if (!origin_constraints.empty()) {
    if (origin_constraints.size() != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank of constraints (", origin_constraints.size(),
          ") does not match rank of domain (", rank, ")"));
    }
    std::copy_n(origin_constraints.begin(), rank,
                chunk_template.origin().begin());
  } else {
    std::fill_n(chunk_template.origin().begin(), rank, kImplicit);
  }

  // Choose a default origin for dimensions without an explicit origin
  // constraint.  Ensure that `0 <= grid_origin[i] < chunk_shape[i]`.
  for (DimensionIndex i = 0; i < rank; ++i) {
    Index& origin_value = chunk_template.origin()[i];
    if (origin_value == kImplicit) {
      const Index domain_origin_value = domain.origin()[i];
      if (domain_origin_value == -kInfIndex) {
        origin_value = 0;
      } else {
        origin_value =
            NonnegativeMod(domain_origin_value, chunk_template.shape()[i]);
      }
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        chunk_template[i],
        IndexInterval::Sized(origin_value, chunk_template.shape()[i]),
        tensorstore::MaybeAnnotateStatus(
            _, tensorstore::StrCat("Invalid chunk constraints for dimension ",
                                   i)));
  }
  return absl::OkStatus();
}

absl::Status ChooseReadWriteChunkGrid(const ChunkLayout& constraints,
                                      BoxView<> domain,
                                      MutableBoxView<> chunk_template) {
  ChunkLayout combined_constraints = constraints;
  TENSORSTORE_RETURN_IF_ERROR(
      combined_constraints.Set(
          ChunkLayout::ReadChunk(constraints.write_chunk())),
      tensorstore::MaybeAnnotateStatus(_,
                                       "write_chunk constraints not compatible "
                                       "with existing read_chunk constraints"));
  return ChooseChunkGrid(combined_constraints.grid_origin(),
                         combined_constraints.read_chunk(), domain,
                         chunk_template);
}

}  // namespace internal

absl::Status ChunkLayout::Finalize() {
  const DimensionIndex rank = this->rank();
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError("rank must be specified");
  }
  {
    StoragePtr storage_to_be_destroyed;
    Storage::EnsureUnique(storage_, rank, storage_to_be_destroyed);
  }
  auto& impl = *storage_;
  auto origin = impl.grid_origin();
  // Validate grid_origin
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    if (!impl.grid_origin_hard_constraint_[dim]) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "No grid_origin hard constraint for dimension ", dim));
    }
    if (!IsFiniteIndex(origin[dim])) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Invalid grid_origin: ", origin));
    }
  }

  for (Usage usage : ChunkLayout::kUsages) {
    const size_t usage_index = static_cast<size_t>(usage);
    auto status = [&]() -> absl::Status {
      auto shape = impl.chunk_shape(usage_index);
      auto& shape_hard_constraint =
          impl.chunk_shape_hard_constraint_[usage_index];
      for (DimensionIndex dim = 0; dim < rank; ++dim) {
        const Index origin_value = origin[dim];
        Index& size_value = shape[dim];
        if (!shape_hard_constraint[dim]) {
          size_value = 0;
        }
        if (!IndexInterval::ValidSized(origin_value, size_value) ||
            !IsFiniteIndex(origin_value + size_value)) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Invalid origin/shape: origin=", origin, ", shape=", shape));
        }
        if (size_value == 0 && usage == Usage::kRead) {
          auto write_shape =
              impl.chunk_shape(static_cast<size_t>(Usage::kWrite));
          size_value = write_shape[dim];
          shape_hard_constraint[dim] =
              impl.chunk_shape_hard_constraint_[static_cast<size_t>(
                  Usage::kWrite)][dim];
        }
      }

      // Clear aspect ratio and target number of elements since they are not
      // relevant to the finalized layout.
      impl.chunk_aspect_ratio_hard_constraint_[usage_index] = false;
      impl.hard_constraint_[static_cast<size_t>(
          GetChunkElementsHardConstraintBit(usage))] = false;
      impl.chunk_elements_[usage_index] = kImplicit;
      std::fill_n(impl.chunk_aspect_ratio(usage_index).begin(), rank, 0);
      return absl::OkStatus();
    }();
    if (!status.ok()) {
      return tensorstore::MaybeAnnotateStatus(
          status, tensorstore::StrCat("Invalid ", usage, " chunk grid"));
    }
  }

  auto write_chunk_shape = impl.chunk_shape(static_cast<size_t>(Usage::kWrite));
  auto read_chunk_shape = impl.chunk_shape(static_cast<size_t>(Usage::kRead));
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

  return absl::OkStatus();
}

namespace {

constexpr auto UsageJsonBinder() {
  return jb::Enum<ChunkLayout::Usage, std::string_view>({
      {ChunkLayout::Usage::kWrite, "write"},
      {ChunkLayout::Usage::kRead, "read"},
      {ChunkLayout::Usage::kCodec, "codec"},
  });
}

}  // namespace

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

ChunkLayout::Grid::Grid(const Grid& other)
    : rank_(other.rank_),
      elements_hard_constraint_(other.elements_hard_constraint_),
      shape_hard_constraint_(other.shape_hard_constraint_),
      aspect_ratio_hard_constraint_(other.aspect_ratio_hard_constraint_),
      elements_(other.elements_) {
  const DimensionIndex rank = other.rank_;
  if (rank > 0) {
    shape_.reset(new Index[rank]);
    std::copy_n(other.shape_.get(), rank, shape_.get());
    aspect_ratio_.reset(new double[rank]);
    std::copy_n(other.aspect_ratio_.get(), rank, aspect_ratio_.get());
  }
}

ChunkLayout::Grid& ChunkLayout::Grid::operator=(const Grid& other) {
  const DimensionIndex new_rank = other.rank_;
  if (new_rank <= 0) {
    shape_.reset();
    aspect_ratio_.reset();
  } else {
    if (new_rank != rank_) {
      shape_.reset(new Index[new_rank]);
      aspect_ratio_.reset(new double[new_rank]);
    }
    std::copy_n(other.shape_.get(), new_rank, shape_.get());
    std::copy_n(other.aspect_ratio_.get(), new_rank, aspect_ratio_.get());
  }
  rank_ = new_rank;
  elements_hard_constraint_ = other.elements_hard_constraint_;
  shape_hard_constraint_ = other.shape_hard_constraint_;
  aspect_ratio_hard_constraint_ = other.aspect_ratio_hard_constraint_;
  elements_ = other.elements_;
  return *this;
}

ChunkLayout::Grid::~Grid() = default;

absl::Status ChunkLayout::Grid::Set(RankConstraint value) {
  const DimensionIndex rank = value.rank;
  if (rank == dynamic_rank || rank == rank_) {
    return absl::OkStatus();
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(rank));
  if (!RankConstraint::EqualOrUnspecified(rank_, rank)) {
    return RankMismatchError(rank, rank_);
  }
  rank_ = rank;
  if (rank > 0) {
    shape_.reset(new Index[rank]);
    std::fill_n(shape_.get(), rank, ShapeValueTraits::kDefaultValue);
    aspect_ratio_.reset(new double[rank]);
    std::fill_n(aspect_ratio_.get(), rank,
                AspectRatioValueTraits::kDefaultValue);
  }
  return absl::OkStatus();
}

namespace {
template <typename Traits>
absl::Status SetVectorProperty(
    ChunkLayout::Grid& self, std::unique_ptr<typename Traits::Element[]>& vec,
    DimensionSet& hard_constraint,
    MaybeHardConstraintSpan<typename Traits::Element> value) {
  if (!value.valid()) return absl::OkStatus();
  const DimensionIndex rank = value.size();
  TENSORSTORE_RETURN_IF_ERROR(self.Set(RankConstraint(rank)));
  return ValidateAndMergeVectorInto<Traits>(value, vec.get(), hard_constraint);
}
}  // namespace

absl::Status ChunkLayout::Grid::Set(Shape value) {
  return SetVectorProperty<ShapeValueTraits>(*this, shape_,
                                             shape_hard_constraint_, value);
}

absl::Status ChunkLayout::Grid::Set(AspectRatio value) {
  return SetVectorProperty<AspectRatioValueTraits>(
      *this, aspect_ratio_, aspect_ratio_hard_constraint_, value);
}

absl::Status ChunkLayout::Grid::Set(Elements value) {
  return SetChunkElementsInternal<bool&>(elements_, elements_hard_constraint_,
                                         value);
}

absl::Status ChunkLayout::Grid::Set(const GridView& value) {
  TENSORSTORE_RETURN_IF_ERROR(Set(value.shape()));
  TENSORSTORE_RETURN_IF_ERROR(Set(value.aspect_ratio()));
  TENSORSTORE_RETURN_IF_ERROR(Set(value.elements()));
  return absl::OkStatus();
}

bool operator==(const ChunkLayout::Grid& a, const ChunkLayout::Grid& b) {
  const DimensionIndex rank = a.rank_;
  if (rank != b.rank_ ||
      a.elements_hard_constraint_ != b.elements_hard_constraint_ ||
      a.shape_hard_constraint_ != b.shape_hard_constraint_ ||
      a.aspect_ratio_hard_constraint_ != b.aspect_ratio_hard_constraint_ ||
      a.elements_ != b.elements_) {
    return false;
  }
  return rank <= 0 ||
         (std::equal(a.shape_.get(), a.shape_.get() + rank, b.shape_.get()) &&
          std::equal(a.aspect_ratio_.get(), a.aspect_ratio_.get() + rank,
                     b.aspect_ratio_.get()));
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

absl::Status ChunkLayout::GetChunkTemplate(Usage usage,
                                           MutableBoxView<> box) const {
  assert(usage == kRead || usage == kWrite);
  const DimensionIndex rank = this->rank();
  if (rank == dynamic_rank) {
    box.Fill();
    return absl::OkStatus();
  }
  if (rank != box.rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Rank of chunk layout (", rank, ") does not match expected rank (",
        box.rank(), ")"));
  }
  auto grid_origin = this->grid_origin();
  auto shape = (*this)[usage].shape();
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (grid_origin[i] == kImplicit || !grid_origin.hard_constraint[i] ||
        shape[i] == 0 || !shape.hard_constraint[i]) {
      box[i] = IndexInterval::Infinite();
      continue;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        box[i], IndexInterval::Sized(grid_origin[i], shape[i]),
        tensorstore::MaybeAnnotateStatus(
            _, tensorstore::StrCat(
                   "Incompatible grid origin/chunk shape for dimension ", i)));
  }
  return absl::OkStatus();
}

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::ChunkLayout,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::ChunkLayout>())
