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

/// Multi-dimensional iteration facilities used internally only.

#ifndef TENSORSTORE_INTERNAL_ITERATE_H_
#define TENSORSTORE_INTERNAL_ITERATE_H_

#include <array>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/util/iterate.h"

namespace tensorstore {
namespace internal_iterate {

template <std::size_t N>
struct DimensionSizeAndStrides {
  Index size;
  std::array<Index, N> strides;
};

template <std::size_t N>
using StridedIterationLayout =
    absl::InlinedVector<DimensionSizeAndStrides<N>, internal::kNumInlinedDims>;

template <std::size_t N, DimensionIndex InnerRank>
struct InnerShapeAndStrides {
  std::array<Index, InnerRank> shape;
  std::array<std::array<Index, InnerRank>, N> strides;
};

}  // namespace internal_iterate

namespace internal {

/// Class used for applying an `Arity`-ary type-erased element-wise function
/// over `Arity` strided arrays.
///
/// For the specified `shape`, `strides`, and `constraints`, the constructor
/// precomputes an efficient iteration method.
///
/// Additionally, it determines whether the contiguous or strided variant of the
/// type-erased function should be used.
///
/// The `operator()` method can then be invoked one or more times with `Arity`
/// base pointers to iterate using the precomputed method.
///
/// Although both the construction and the actual iteration could be combined
/// into a single function, by splitting up the two steps, duplicate work is
/// avoided in the case that the same `shape`, `strides`, `constraints` and
/// `callbacks` are used multiple times, such as when iterating using index
/// space transforms.
template <std::size_t Arity>
class StridedLayoutFunctionApplyer {
  static_assert(Arity > 0 && Arity <= kMaxSupportedIterationArity,
                "Invalid arity.");

 public:
  /// Precomputes an iteration layout based on the specified constraints.
  /// \param shape The multi-dimensional shape over which to iterate.
  /// \param strides An array of strides specifying the layout for each array.
  /// \param constraints Specifies constraints on the iteration order.
  /// \param function The type-erased representation of the element-wise
  ///     function.
  explicit StridedLayoutFunctionApplyer(
      span<const Index> shape, std::array<const Index*, Arity> strides,
      IterationConstraints constraints,
      ElementwiseClosure<Arity, absl::Status*> function,
      std::array<std::ptrdiff_t, Arity> element_sizes);

  /// Precomputes an iteration layout using the specified dimension order.
  /// \param shape The multi-dimensional shape over which to iterate.
  /// \param dimension_order An array of indices into `shape` and `strides[i]`.
  /// \param strides An array of strides specifying the layout for each array.
  /// \param function The type-erased representation of the element-wise
  ///     function.
  explicit StridedLayoutFunctionApplyer(
      const Index* shape, span<const DimensionIndex> dimension_order,
      std::array<const Index*, Arity> strides,
      ElementwiseClosure<Arity, absl::Status*> function,
      std::array<std::ptrdiff_t, Arity> element_sizes);

  /// Invokes the element-wise function for each tuple of elements, using the
  /// specified base pointers.
  ArrayIterateResult operator()(
      std::array<ByteStridedPointer<void>, Arity> pointers,
      absl::Status* status) const;

  DimensionIndex outer_rank() const {
    return static_cast<DimensionIndex>(iteration_layout_.size());
  }

  Index inner_size() const { return inner_layout_.shape[0]; }

 private:
  struct WrappedFunction;
  internal_iterate::StridedIterationLayout<Arity> iteration_layout_;
  internal_iterate::InnerShapeAndStrides<Arity, 1> inner_layout_;
  void* context_;
  SpecializedElementwiseFunctionPointer<Arity, absl::Status*> callback_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ITERATE_H_
