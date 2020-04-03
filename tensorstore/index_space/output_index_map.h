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

#ifndef TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_MAP_H_
#define TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_MAP_H_

#include "absl/base/macros.h"
#include "tensorstore/array.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"

namespace tensorstore {

/// Represents a const unowned view of an output index map for an index
/// transform, which specifies the how the index for a given output dimension is
/// computed from the input indices.
///
/// This view is valid only as long as the underlying index transform.
///
/// \see OutputIndexMapRange
template <DimensionIndex InputRank = dynamic_rank>
class OutputIndexMapRef {
 public:
  /// Represents a const view of an index array and associated bounds.
  class IndexArrayView {
   public:
    /// Returns a SharedArrayView representing the index array.
    ///
    /// The shape of the returned array is equal to the `input_shape()` of the
    /// index transform.
    ///
    /// If an unowned reference is sufficient, `array_ref()` can be used instead
    /// to avoid the cost of atomic reference count operations.
    SharedArrayView<const Index, InputRank, offset_origin> shared_array_ref()
        const {
      return {element_pointer(), layout()};
    }

    /// Returns an ArrayView representing the index array.
    ///
    /// The shape of the returned array is equal to the `input_shape()` of the
    /// index transform.
    ArrayView<const Index, InputRank, offset_origin> array_ref() const {
      return {element_pointer(), layout()};
    }

    /// Returns the element pointer to the start of the index array.
    const SharedElementPointer<const Index>& element_pointer() const {
      return index_array_data_->element_pointer;
    }

    /// Returns the bounding interval on the index array values.
    IndexInterval index_range() const { return index_array_data_->index_range; }

    /// Returns the rank of the index array, which is equal to the input rank of
    /// the index transform.
    StaticOrDynamicRank<InputRank> rank() const {
      return StaticRankCast<InputRank, unchecked>(
          static_cast<DimensionIndex>(rep_->input_rank));
    }

    /// Returns the strided layout of the index array.
    StridedLayoutView<InputRank, offset_origin> layout() const {
      return StridedLayoutView<InputRank, offset_origin>(
          rank(), rep_->input_origin().data(), rep_->input_shape().data(),
          index_array_data_->byte_strides);
    }

    /// Returns `layout().byte_strides()`.
    span<const Index, InputRank> byte_strides() const {
      return {index_array_data_->byte_strides, rank()};
    }

   private:
    template <DimensionIndex>
    friend class OutputIndexMapRef;

    explicit IndexArrayView(
        internal_index_space::IndexArrayData* index_array_data,
        internal_index_space::TransformRep* rep)
        : index_array_data_(index_array_data), rep_(rep) {}

    internal_index_space::IndexArrayData* index_array_data_;
    internal_index_space::TransformRep* rep_;
  };

  /// Constructs an invalid reference.
  ///
  /// No methods are valid on an invalid reference, except `operator=`.
  OutputIndexMapRef() = default;

  /// Returns the input rank of the index transform.
  StaticOrDynamicRank<InputRank> input_rank() const {
    return StaticRankCast<InputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }

  /// Returns the mapping method.
  OutputIndexMethod method() const { return map_->method(); }

  /// Returns the offset.
  Index offset() const { return map_->offset(); }

  /// Returns the stride.  This is ignored if `method() ==
  /// OutputIndexMethod::constant`.
  Index stride() const { return map_->stride(); }

  /// Returns the single input dimension used to compute the output index.
  /// \dchecks method() == OutputIndexMethod::single_input_dimension
  DimensionIndex input_dimension() const { return map_->input_dimension(); }

  /// Returns the index array and bounds.
  /// \dchecks `method() == OutputIndexMethod::array`
  IndexArrayView index_array() const {
    return IndexArrayView(&map_->index_array_data(), rep_);
  }

 private:
  template <DimensionIndex, DimensionIndex>
  friend class OutputIndexMapRange;
  template <DimensionIndex>
  friend class OutputIndexMapIterator;
  explicit OutputIndexMapRef(internal_index_space::OutputIndexMap* map,
                             internal_index_space::TransformRep* rep)
      : map_(map), rep_(rep) {}

  internal_index_space::OutputIndexMap* map_ = nullptr;
  internal_index_space::TransformRep* rep_ = nullptr;
};

/// Iterator type for OutputIndexMapRange.
///
/// Satisfies the standard library RandomAccessIterator concept, except that the
/// reference type is the proxy type OutputIndexMapRef, not a real reference
/// type.
template <DimensionIndex InputRank = dynamic_rank>
class OutputIndexMapIterator {
 public:
  using value_type = OutputIndexMapRef<InputRank>;
  using reference = OutputIndexMapRef<InputRank>;
  using difference_type = DimensionIndex;
  using pointer = value_type*;
  using iterator_category = std::random_access_iterator_tag;

  OutputIndexMapIterator() = default;

  OutputIndexMapRef<InputRank> operator*() const { return ref_; }
  const OutputIndexMapRef<InputRank>* operator->() const { return &ref_; }
  OutputIndexMapRef<InputRank> operator[](DimensionIndex n) const {
    auto new_ref = ref_;
    new_ref.map_ += n;
    return new_ref;
  }

  OutputIndexMapIterator& operator+=(DimensionIndex n) {
    ref_.map_ += n;
    return *this;
  }
  OutputIndexMapIterator& operator-=(DimensionIndex n) { return *this += (-n); }

  OutputIndexMapIterator& operator++() {
    ++ref_.map_;
    return *this;
  }
  OutputIndexMapIterator& operator--() {
    --ref_.map_;
    return *this;
  }
  OutputIndexMapIterator operator++(int) {
    auto temp = *this;
    ++ref_.map_;
    return temp;
  }
  OutputIndexMapIterator operator--(int) {
    auto temp = *this;
    --ref_.map_;
    return temp;
  }
  friend DimensionIndex operator-(OutputIndexMapIterator a,
                                  OutputIndexMapIterator b) {
    return a.map() - b.map();
  }
  friend OutputIndexMapIterator operator+(OutputIndexMapIterator it,
                                          DimensionIndex n) {
    it += n;
    return it;
  }
  friend OutputIndexMapIterator operator+(DimensionIndex n,
                                          OutputIndexMapIterator it) {
    it += n;
    return it;
  }
  friend OutputIndexMapIterator operator-(OutputIndexMapIterator it,
                                          DimensionIndex n) {
    it -= n;
    return it;
  }

  friend bool operator==(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() == b.map();
  }
  friend bool operator!=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() != b.map();
  }
  friend bool operator<(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() < b.map();
  }
  friend bool operator<=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() <= b.map();
  }
  friend bool operator>(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() > b.map();
  }
  friend bool operator>=(OutputIndexMapIterator a, OutputIndexMapIterator b) {
    return a.map() >= b.map();
  }

 private:
  internal_index_space::OutputIndexMap* map() const { return ref_.map_; }
  template <DimensionIndex, DimensionIndex>
  friend class OutputIndexMapRange;
  OutputIndexMapRef<InputRank> ref_;
  explicit OutputIndexMapIterator(internal_index_space::OutputIndexMap* map,
                                  internal_index_space::TransformRep* rep)
      : ref_(map, rep) {}
};

/// Represents a const unowned range view of the output index maps for an index
/// transform.
///
/// This range is valid only as long as the underlying index transform.
template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank>
class OutputIndexMapRange {
 public:
  using value_type = OutputIndexMapRef<InputRank>;
  using reference = value_type;
  using iterator = OutputIndexMapIterator<InputRank>;
  using difference_type = DimensionIndex;

  /// The static extent of this range, equal to the static output rank of the
  /// transform.
  constexpr static DimensionIndex extent = OutputRank;

  // Constructs an invalid output index map range.
  OutputIndexMapRange() = default;

  /// Returns the output rank of the transform.
  StaticOrDynamicRank<OutputRank> size() const {
    return StaticRankCast<OutputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->output_rank));
  }

  bool empty() const { return size() == 0; }

  iterator begin() const {
    return iterator(rep_->output_index_maps().data(), rep_);
  }

  iterator end() const {
    return iterator(rep_->output_index_maps().data() + size(), rep_);
  }

  /// Returns the output index map for dimension `output_dim`.
  /// \dchecks `0 <= output_dim && output_dim < size()`.
  OutputIndexMapRef<InputRank> operator[](DimensionIndex output_dim) const {
    ABSL_ASSERT(output_dim >= 0 && output_dim < size());
    return OutputIndexMapRef<InputRank>(
        rep_->output_index_maps().data() + output_dim, rep_);
  }

  /// Returns the input rank of the index transform.
  StaticOrDynamicRank<InputRank> input_rank() const {
    return StaticRankCast<InputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }

 private:
  template <DimensionIndex, DimensionIndex, ContainerKind CKind>
  friend class IndexTransform;
  explicit OutputIndexMapRange(internal_index_space::TransformRep* rep)
      : rep_(rep) {}
  internal_index_space::TransformRep* rep_ = nullptr;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_MAP_H_
