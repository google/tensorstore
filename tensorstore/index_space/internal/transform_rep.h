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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_H_

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/index_space/transform_array_constraints.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
class IndexTransform;

template <DimensionIndex Rank, ContainerKind CKind>
class IndexDomain;

namespace internal_index_space {

struct IndexArrayData {
  SharedElementPointer<const Index> element_pointer;

  // Bounds on the index values that will be permitted.  These are not
  // calculated min/max values for the index array itself.  Rather, they reflect
  // bounds on the output space that have been propagated back to this index
  // array.  These bounds are only checked lazily.
  IndexInterval index_range;
  DimensionIndex rank_capacity;
  Index byte_strides[];

  StridedLayoutView<dynamic_rank, offset_origin> layout(
      BoxView<> input_domain) const {
    assert(rank_capacity >= input_domain.rank());
    return StridedLayoutView<dynamic_rank, offset_origin>(
        input_domain.rank(), input_domain.origin().data(),
        input_domain.shape().data(), byte_strides);
  }

  ArrayView<const Index, dynamic_rank, offset_origin> array_view(
      BoxView<> input_domain) const {
    return {element_pointer, layout(input_domain)};
  }

  SharedArrayView<const Index, dynamic_rank, offset_origin> shared_array_view(
      BoxView<> input_domain) const {
    return {element_pointer, layout(input_domain)};
  }
};
// Alignment of IndexArrayData must be >= 2 to ensure that the low bit of
// pointers to IndexArrayData is always 0.
static_assert(alignof(IndexArrayData) >= 2,
              "Platform has unsupported alignment.");

/// Defines a mapping from a multi-dimensional "input" index space to a single
/// "output" coordinate.
///
/// There are three types of mappings, corresponding to values of
/// `OutputIndexMethod`:
///
///   1. `constant`: the result is always `offset()`.
///
///   2. `single_input_dimension`: the result is `offset() + stride() *
///      input_indices[input_dimension()]`.
///
///   3. `array`: the result is `offset() + stride() * array(input_indices)`.
///
/// A full IndexTransform is defined using one OutputIndexMap per
/// output dimension.
///
/// This type is not exposed as part of the public API, because it operates at
/// too low of level to be used safely/conveniently.  Instead,
/// third_party/tensorstore/index_space/output_index_map.h defines a public API
/// that wraps this type.
class OutputIndexMap {
 public:
  OutputIndexMap() = default;

  OutputIndexMap(OutputIndexMap&& other)
      : value_(0), offset_(other.offset_), stride_(other.stride_) {
    std::swap(other.value_, value_);
  }

  OutputIndexMap& operator=(OutputIndexMap&& other) {
    std::swap(value_, other.value_);
    offset_ = other.offset_;
    stride_ = other.stride_;
    return *this;
  }

  OutputIndexMethod method() const {
    return value_ == 0  ? OutputIndexMethod::constant
           : value_ & 1 ? OutputIndexMethod::single_input_dimension
                        : OutputIndexMethod::array;
  }

  DimensionIndex input_dimension() const {
    assert(method() == OutputIndexMethod::single_input_dimension);
    return static_cast<DimensionIndex>(value_ >> 1);
  }

  const IndexArrayData& index_array_data() const {
    assert(method() == OutputIndexMethod::array);
    return *reinterpret_cast<IndexArrayData*>(value_);
  }

  IndexArrayData& index_array_data() {
    assert(method() == OutputIndexMethod::array);
    return *reinterpret_cast<IndexArrayData*>(value_);
  }

  Result<Index> operator()(span<const Index> input_indices) const;

  void SetConstant();
  void SetSingleInputDimension(DimensionIndex input_dim);

  /// Sets this map to array indexing.
  IndexArrayData& SetArrayIndexing(DimensionIndex rank);

  IndexArrayData& SetArrayIndexing(DimensionIndex rank,
                                   const IndexArrayData& other);

  void Assign(DimensionIndex rank, const OutputIndexMap& other);

  ~OutputIndexMap() { SetConstant(); }

  constexpr Index offset() const { return offset_; }
  constexpr Index stride() const { return stride_; }
  Index& offset() { return offset_; }
  Index& stride() { return stride_; }

 private:
  /// Represents one of three possible states:
  ///
  /// `value_ == 0`:
  ///   constant mapping
  ///
  /// `(value_ & 1) == 1`:
  ///   single_input_dimension mapping, with `input_dimension = value_ >> 1`.
  ///
  /// `value_ != 0 && (value_ & 1) == 0`:
  ///   value_ is a pointer to dynamically-allocated IndexArrayData instance.
  ///   The lowest bit of the pointer is guaranteed to be 0 since IndexArrayData
  ///   must have an alignment >= 2.
  std::uintptr_t value_ = 0;
  Index offset_, stride_;
};

class InputDimensionsView;
class InputDimensionRef;

/// The header data for the private representation of an index transform.
///
/// This struct is stored in the middle of a dynamically allocated block with
/// the following layout:
///
///     OutputIndexMap output_index_maps[output_rank_capacity];
///     TransformRep header;
///     Index input_origin[input_rank_capacity];
///     Index input_shape[input_rank_capacity];
///     std::string input_labels[input_rank_capacity];
///
/// The representation is reference counted, and IndexTransform provide a safe,
/// immutable public interface for accessing it.  However, internal functions
/// that holds a unique reference (i.e. `reference_count == 1`) may modify the
/// representation in place.
///
/// IndexTransformBuilder provides a safe builder interface for constructing a
/// new transform representation, which is then returned as an IndexTransform.
///
/// There are numerous invariants, documented below, that the representation
/// must satisfy while owned by an IndexTransform.  These invariants are
/// temporarily violated when building and operating on the representation, but
/// are always restored before returning an IndexTransform externally.
struct TransformRep {
  /// The input rank.
  /// \invariant `0 <= input_rank && input_rank <= input_rank_capacity`.
  std::int16_t input_rank;

  /// The output rank.
  /// \invariant `0 <= output_rank && output_rank <= output_rank_capacity`.
  std::int16_t output_rank;

  /// The length of the `input_origin`, `input_shape`, and `input_labels` arrays
  /// that immediately follow this TransformRep header in memory.
  ///
  /// \invariant `0 <= input_rank_capacity <= kMaxRank`.
  std::int16_t input_rank_capacity;

  /// The length of the `output_index_maps` array that precedes this
  /// TransformRep header in memory.
  ///
  /// \invariant `0 <= output_rank_capacity <= kMaxRank`.
  std::int16_t output_rank_capacity;

  /// Set of dimensions with implicit lower/upper bounds.
  DimensionSet implicit_lower_bounds;
  DimensionSet implicit_upper_bounds;

  /// Dimensions for which the lower or upper bound is implicit.
  DimensionSet implicit_dimensions() const {
    return implicit_lower_bounds | implicit_upper_bounds;
  }

  static_assert(kMaxRank * 2 <= 64);

  /// Reference count.
  ///
  /// \invariant `0 <= ref_count`.
  std::atomic<std::uint64_t> reference_count;

  /// Returns `true` if there is only one reference to this representation.
  bool is_unique() const {
    return reference_count.load(std::memory_order_acquire) == 1;
  }

  /// Returns a mutable view of the input dimension data.
  ///
  /// \param rank The input rank of the view.
  /// \dchecks `0 <= rank && rank < input_rank_capacity`.
  InputDimensionsView all_input_dimensions(DimensionIndex rank);

  /// Returns a mutable view of the data for an input dimension.
  ///
  /// \param i The input dimension index for which to return a view.
  /// \dchecks `0 <= i && i < input_rank_capacity`.
  InputDimensionRef input_dimension(DimensionIndex i);

  /// Returns the `input_origin` array of length `input_rank_capacity`.
  ///
  /// \invariant For `0 <= i && i < input_rank`,
  ///     `IndexInterval::ValidSized(input_origin()[i], input_shape()[i])`.
  span<Index> input_origin() {
    return span(reinterpret_cast<Index*>(this + 1), input_rank_capacity);
  }

  /// Returns the `input_shape` array of length `input_rank_capacity`.
  ///
  /// \invariant See the invariant specified for `input_origin`.
  span<Index> input_shape() {
    return span(reinterpret_cast<Index*>(this + 1) + input_rank_capacity,
                input_rank_capacity);
  }

  /// Returns a mutable view of the input domain for the first `rank` input
  /// dimensions.
  ///
  /// \dchecks `0 <= rank && rank <= input_rank_capacity`.
  MutableBoxView<> input_domain(DimensionIndex rank) {
    assert(0 <= rank && rank <= input_rank_capacity);
    return MutableBoxView<>(rank, input_origin().data(), input_shape().data());
  }

  /// Returns the `output_index_maps` array of length `output_rank_capacity`.
  ///
  /// \invariant For `0 <= i && i < output_rank`:
  ///     If `output_index_maps()[i].method() == single_input_dimension`:
  ///        `output_index_maps()[i].single_input_dimension()` must be in the
  ///        range `[0, input_rank)`.
  ///     If `output_index_maps()[i].method() == array`:
  ///        Let `index_array_data = output_index_maps()[i].index_array_data()`.
  ///        `index_array_data.rank_capacity` must be `>= input_rank`.
  ///        `index_array_data.element_pointer` and
  ///        `index_array_data.byte_strides[0:input_rank]` must specify a valid
  ///        array of shape `input_shape()[0:input_rank]`.
  span<OutputIndexMap> output_index_maps() {
    return span(reinterpret_cast<OutputIndexMap*>(this) - output_rank_capacity,
                output_rank_capacity);
  }

  /// Returns the `input_labels` array of length `input_rank_capacity`.
  span<std::string> input_labels() {
    return span(reinterpret_cast<std::string*>(input_shape().end()),
                input_rank_capacity);
  }

  /// Free a TransformRep allocated by Allocate.
  /// \param ptr A pointer returned by `Allocate` and not previously freed.
  /// \dchecks `ptr->reference_count != 0`
  static void Free(TransformRep* ptr);

  template <typename PointerType = TransformRep*>
  struct IntrusivePtrTraits {
    template <typename>
    using pointer = PointerType;

    static void increment(TransformRep* rep) {
      rep->reference_count.fetch_add(1, std::memory_order_acq_rel);
    }
    static void decrement(TransformRep* rep) {
      if (rep->reference_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        Free(rep);
      }
    }
  };

  template <ContainerKind CKind = container>
  using Ptr = std::conditional_t<
      CKind == view, TransformRep*,
      internal::IntrusivePtr<TransformRep, IntrusivePtrTraits<>>>;

  /// Allocate a new TransformRep with the specified capacities and type.
  ///
  /// Sets the `input_rank_capacity` and `output_rank_capacity` fields, and
  /// zero-initializes `implicit_lower_bounds` and `implicit_lower_bounds`, but
  /// does not initialize `input_rank` and `output_rank`.
  /// The `output_index_maps` array and `input_labels` array are default
  /// initialized.
  ///
  /// In case of insufficient memory, this function has the same behavior as
  /// `::operator new` (either throws `std::bad_alloc` or terminates the
  /// program).
  ///
  /// \dchecks `input_rank_capacity >= 0`
  /// \dchecks `output_rank_capacity >= 0`
  /// \returns A non-null transform representation pointer.
  static Ptr<> Allocate(DimensionIndex input_rank_capacity,
                        DimensionIndex output_rank_capacity);
};

#ifdef NDEBUG
// Don't check invariants when not in debug mode.
inline void DebugCheckInvariants(TransformRep* rep) {}
#else
/// Checks invariants of the representation in debug mode.
///
/// This must be called any time the internal representation of an index
/// transform is manipulated directly.
void DebugCheckInvariants(TransformRep* rep);
#endif

inline void NormalizeImplicitBounds(TransformRep& rep) {
  const auto mask = DimensionSet::UpTo(rep.input_rank);
  rep.implicit_lower_bounds &= mask;
  rep.implicit_upper_bounds &= mask;
}

// Check that OutputIndexMap and std::string don't have a greater alignment
// value than Index, as that would require more complicated logic for accessing
// the variable length fields than is currently implemented.  In practice these
// constraints should always be satisified.  If this code needs to work on a
// platform that doesn't satisfy these contraints, the more complicated logic
// could be implemented.
static_assert(alignof(OutputIndexMap) <= sizeof(Index),
              "Platform has unsupported alignment.");
static_assert(alignof(std::string) <= sizeof(Index),
              "Platform has unsupported alignment.");

/// Proxy mutable reference type representing all of the fields associated with
/// a single input dimension of an index transform.
///
/// This is useful for implementing operations that modify index transforms.
class InputDimensionRef {
 public:
  explicit InputDimensionRef(TransformRep* rep, DimensionIndex input_dim)
      : rep_(rep), input_dim_(input_dim) {}

  IndexIntervalRef domain() const {
    return rep_->input_domain(rep_->input_rank_capacity)[input_dim_];
  }

  OptionallyImplicitIndexInterval optionally_implicit_domain() const {
    return {domain(), implicit_lower_bound(), implicit_upper_bound()};
  }

  template <ContainerKind LabelCKind = view>
  IndexDomainDimension<LabelCKind> index_domain_dimension() const {
    return {optionally_implicit_domain(), label()};
  }

  DimensionSet::reference implicit_lower_bound() const {
    return rep_->implicit_lower_bounds[input_dim_];
  }

  DimensionSet::reference implicit_upper_bound() const {
    return rep_->implicit_upper_bounds[input_dim_];
  }

  std::string& label() const { return rep_->input_labels()[input_dim_]; }

  /// Deep assigns from another `InputDimensionRef`.
  const InputDimensionRef& operator=(const InputDimensionRef& other) const {
    domain() = other.domain();
    implicit_lower_bound() = other.implicit_lower_bound();
    implicit_upper_bound() = other.implicit_upper_bound();
    label() = other.label();
    return *this;
  }

  void SetEmptyLabel() const { label().clear(); }

  /// Deep assigns from an `IndexDomainDimension`.
  template <ContainerKind LabelCKind>
  const InputDimensionRef& operator=(
      const IndexDomainDimension<LabelCKind>& other) const {
    domain() = other.interval();
    implicit_lower_bound() = other.implicit_lower();
    implicit_upper_bound() = other.implicit_upper();
    // TODO(jbms): simplify this once std::string_view has been replaced by
    // std::string_view.
    label().assign(other.label().begin(), other.label().end());
    return *this;
  }

  template <ContainerKind LabelCKind = view>
  operator IndexDomainDimension<LabelCKind>() const {
    return index_domain_dimension<LabelCKind>();
  }

 private:
  TransformRep* const rep_;
  const DimensionIndex input_dim_;
};

/// Mutable view of all of the input dimension fields of an index transform.
///
/// This is useful in generic code that operates on array-like types that
/// provide `operator[]`.
class InputDimensionsView {
 public:
  explicit InputDimensionsView(TransformRep* rep, DimensionIndex input_rank)
      : rep_(rep), size_(input_rank) {}

  DimensionIndex size() const { return size_; }

  InputDimensionRef operator[](DimensionIndex i) const {
    assert(i >= 0 && i <= size_);
    return InputDimensionRef(rep_, i);
  }

 private:
  TransformRep* rep_;
  DimensionIndex size_;
};

inline ABSL_ATTRIBUTE_ALWAYS_INLINE InputDimensionsView
TransformRep::all_input_dimensions(DimensionIndex rank) {
  assert(rank >= 0 && rank <= input_rank_capacity);
  return InputDimensionsView(this, rank);
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE InputDimensionRef
TransformRep::input_dimension(DimensionIndex i) {
  assert(i >= 0 && i <= input_rank_capacity);
  return InputDimensionRef(this, i);
}

/// Copy assigns the representation in `dest` from `source`.
///
/// \param source Non-null pointer to the source representation.
/// \param dest Non-null pointer to the dest representation.
/// \dchecks `source->input_rank <= dest->input_rank_capacity`.
/// \dchecks `source->output_rank <= dest->output_rank_capacity`.
void CopyTransformRep(TransformRep* source, TransformRep* dest);

/// Copy assigns the domain in `dest` from `source`.
///
/// \param source Non-null pointer to the source representation.
/// \param dest Non-null pointer to the dest representation.
/// \dchecks `source->input_rank <= dest->input_rank_capacity`.
void CopyTransformRepDomain(TransformRep* source, TransformRep* dest);

/// Move assigns the representation in `dest` from `dest`.
///
/// This behaves the same as CopyTransformRep, except that it may modify the
/// `output_index_maps` and `input_labels` of `source`.
///
/// \param dest Non-null pointer to the dest representation.
/// \dchecks `source->input_rank <= dest->input_rank_capacity`.
/// \dchecks `source->output_rank <= dest->output_rank_capacity`.
void MoveTransformRep(TransformRep* source, TransformRep* dest);

/// Returns an equivalent transform that may safely be modified.
///
/// If `ptr->reference_count == 1`, returns `ptr`.  Otherwise, returns a
/// newly-allocated copy.
///
/// \param ptr Existing transform.
/// \param domain_only If true, the returned transform will have an output rank
///     of 0.
/// \dchecks `ptr != nullptr`
TransformRep::Ptr<> MutableRep(TransformRep::Ptr<> ptr,
                               bool domain_only = false);

/// Resets all output index maps to constant maps, freeing any associated index
/// arrays.
void ResetOutputIndexMaps(TransformRep* ptr);

/// Returns a transform with the specified capacities.
///
/// If `ptr->input_rank_capacity >= input_rank_capacity`,
/// `ptr->output_rank_capacity >= output_rank_capacity`, and
/// `ptr->reference_count == 1`, simply returns a new reference to `ptr`.
///
/// Otherwise, returns
/// `TransformRep::Allocate(input_rank_capacity, output_rank_capacity)`.
///
/// \param ptr Non-null pointer to an existing transform.
/// \param input_rank_capacity Required input rank capacity.
/// \param output_rank_capacity Required output rank capacity.
/// \param domain_only If true, `output_rank_capacity` is ignored and taken to
///     be 0 instead.  Additionally, if the existing `ptr` is returned, all of
///     its output index maps are reset in order to free any associated index
///     arrays.
/// \dchecks `ptr != nullptr`
/// \returns A non-null transform pointer.
TransformRep::Ptr<> NewOrMutableRep(TransformRep* ptr,
                                    DimensionIndex input_rank_capacity,
                                    DimensionIndex output_rank_capacity,
                                    bool domain_only = false);

/// Returns `true` if at least one dimension has an explicit zero-size domain,
/// i.e. the domain is empty and it cannot be resized to become non-empty.
///
/// Transforms for which this is true must not contain any index arrays, as the
/// index array output map representation does not support index arrays that
/// contain zero elements.
bool IsDomainExplicitlyEmpty(TransformRep* ptr);

/// Replaces all index array maps with constant maps to 0.
///
/// This is used in the case that `IsDomainExplicitlyEmpty(ptr) == true` to
/// ensure the invariant is satisfied.
void ReplaceAllIndexArrayMapsWithConstantMaps(TransformRep* ptr);

/// Checks if the domains of two transforms are equal.
///
/// A null pointer is considered equal only to a null pointer.
///
/// \param a Pointer to a transform, may be `nullptr`.
/// \param b Pointer to a transform, may be `nullptr`.
bool AreDomainsEqual(TransformRep* a, TransformRep* b);

/// Checks if two transforms are equal.
///
/// A null pointer is considered equal only to a null pointer.
///
/// \param a Pointer to a transform, may be `nullptr`.
/// \param b Pointer to a transform, may be `nullptr`.
bool AreEqual(TransformRep* a, TransformRep* b);

/// Writes a string representation of `transform` to `os`.
///
/// \param os The output stream.
/// \param transform Pointer to the representation, may be `nullptr`.
void PrintToOstream(std::ostream& os, TransformRep* transform);

/// Writes a string representation of the domain of `transform` to `os`.
///
/// \param os The output stream.
/// \param transform Pointer to the representation, may be `nullptr`.
void PrintDomainToOstream(std::ostream& os, TransformRep* transform);

/// Returns the set of input dimensions on which any index array output index
/// map depends.
DimensionSet GetIndexArrayInputDimensions(TransformRep* transform);

/// Returns a copy of `transform` with `implicit_lower_bounds` and
/// `implicit_upper_bounds` set to the specified values.
TransformRep::Ptr<> WithImplicitDimensions(TransformRep::Ptr<> transform,
                                           DimensionSet implicit_lower_bounds,
                                           DimensionSet implicit_upper_bounds,
                                           bool domain_only);

/// Computes the `output_indices` corresponding to the given `input_indices`.
///
/// \dchecks `data != nullptr`.
/// \dchecks `input_indices.size() == data->input_rank`.
/// \dchecks `output_indices.size() == data->output_rank()`.
/// \returns `OkStatus()` on success.
/// \error `absl::StatusCode::kOutOfRange` if `input_indices` is not contained
///     within the domain (implicit bounds are ignored).
/// \error `absl::StatusCode::kOutOfRange` if an array output index map results
///     in an index outside its `index_range` constraint.
absl::Status TransformIndices(TransformRep* data,
                              span<const Index> input_indices,
                              span<Index> output_indices);

/// Returns a transform with `input_rank==dims.size()` and `output_rank==0` in
/// which dimension `i` has the domain of dimension `dims[i]` of `*rep`.
///
/// This is used to implement `IndexDomain::operator[](span<const Index>)` and
/// the similar Python API.
///
/// \param rep[in] Non-null pointer to existing transform.
/// \param dims Vector of old dimension indices corresponding to the new
///     dimensions.
/// \dchecks `d >= 0 && d < rep->input_rank` for `d` in `dims`.
/// \dchecks All dimensions `d`  in `dims` are unique.
/// \returns The new transform.
TransformRep::Ptr<> GetSubDomain(TransformRep* rep,
                                 span<const DimensionIndex> dims);

/// Returns `true` if all labels in `labels` are empty.
bool IsUnlabeled(span<const std::string> labels);

/// Access helper used internally for getting and setting the `rep_` pointer
/// held by `IndexTransform`, and the `transform_` member of `IndexDomain`.
class TransformAccess {
 public:
  template <typename T>
  static TransformRep* rep(const T& x) {
    return internal::to_address(x.rep_);
  }

  template <ContainerKind TargetCKind, typename T>
  static auto rep_ptr(T&& x) {
    if constexpr (TargetCKind == view) {
      return rep(x);
    } else {
      return TransformRep::Ptr<>(std::forward<T>(x).rep_);
    }
  }

  template <typename T>
  static decltype(auto) rep_ptr(T&& x) {
    return (std::forward<T>(x).rep_);
  }

  template <DimensionIndex Rank, ContainerKind CKind>
  static IndexTransform<Rank, dynamic_rank, view> transform(
      const IndexDomain<Rank, CKind>& x) {
    return Make<IndexTransform<Rank, dynamic_rank, view>>(rep(x));
  }

  template <DimensionIndex Rank>
  static IndexTransform<Rank, dynamic_rank, container> transform(
      IndexDomain<Rank, container>&& x) {
    return Make<IndexTransform<Rank, dynamic_rank, container>>(
        std::move(x.rep_));
  }

  template <typename T>
  static T Make(TransformRep::Ptr<T::container_kind> ptr) {
    T t;
    t.rep_ = std::move(ptr);
    return t;
  }
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_REP_H_
