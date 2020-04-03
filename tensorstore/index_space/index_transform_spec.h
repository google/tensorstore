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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_SPEC_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_SPEC_H_

#include <iosfwd>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

namespace internal_index_space {
absl::Status UnspecifiedTransformError();
}  // namespace internal_index_space

/// Specifies an IndexTransform or identity transform (of possibly unknown
/// rank).
///
/// This type is used with `Spec` to specify either an optional rank constraint
/// on the `TensorStore` to be opened, or an index transform (which implies a
/// rank constraint) to apply to the `TensorStore`.
///
/// While specifying `IdentityTransform(rank)` would serve a similar purpose of
/// specifying a rank constraint of `rank`, it would also clear the dimension
/// labels.
class IndexTransformSpec {
 public:
  using Transform = IndexTransform<>;

  /// Constructs an identity transform of unknown rank.
  IndexTransformSpec() : input_rank_(dynamic_rank) {}

  /// Constructs an identity transform of the specified `rank`.
  explicit IndexTransformSpec(DimensionIndex rank) : input_rank_(rank) {}

  /// Constructs from a possibly null transform.
  ///
  /// If `transform.valid() == true`, `*this` represents `transform`.
  /// Otherwise, `*this` represents an identity transform of rank `Rank`.
  template <DimensionIndex InputRank, DimensionIndex OutputRank,
            ContainerKind CKind>
  explicit IndexTransformSpec(
      IndexTransform<InputRank, OutputRank, CKind> transform)
      : transform_(std::move(transform)),
        input_rank_(transform_.valid() ? transform_.input_rank()
                                       : dynamic_rank) {}

  IndexTransformSpec& operator=(DimensionIndex rank) {
    return *this = IndexTransformSpec(rank);
  }

  template <DimensionIndex InputRank, DimensionIndex OutputRank,
            ContainerKind CKind>
  IndexTransformSpec& operator=(
      IndexTransform<InputRank, OutputRank, CKind> transform) {
    return *this = IndexTransformSpec(std::move(transform));
  }

  /// Returns the transform.
  ///
  /// May return a null transform to indicate an identity transform of rank
  /// `rank()`.  Otherwise, it is guaranteed that the `input_rank` of the
  /// returned transform is equal to `rank()`.
  const Transform& transform() const& { return transform_; }
  Transform transform() && { return std::move(transform_); }

  /// Returns the rank constraint, or `dynamic_rank` if unconstrained.
  DimensionIndex input_rank() const { return input_rank_; }

  DimensionIndex output_rank() const {
    return transform_.valid() ? transform_.output_rank() : input_rank_;
  }

  /// Applies a function that operates on an `IndexTransform` to an
  /// `IndexTransformSpec`.
  ///
  /// This definition allows DimExpression objects to be applied to
  /// `IndexTransformSpec` objects.
  ///
  /// \returns The transformed `IndexTransformSpec` on success.
  /// \error `absl::StatusCode::kInvalidArgument` if `transform()` is not valid.
  template <typename Expr>
  friend Result<IndexTransformSpec> ApplyIndexTransform(
      Expr&& expr, IndexTransformSpec spec) {
    if (!spec.transform().valid()) {
      return internal_index_space::UnspecifiedTransformError();
    }
    TENSORSTORE_ASSIGN_OR_RETURN(IndexTransform<> t,
                                 expr(std::move(spec).transform()));
    return IndexTransformSpec{t};
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const IndexTransformSpec& s);

  friend bool operator==(const IndexTransformSpec& a,
                         const IndexTransformSpec& b) {
    return a.transform() == b.transform() && a.input_rank() == b.input_rank();
  }

  friend bool operator!=(const IndexTransformSpec& a,
                         const IndexTransformSpec& b) {
    return !(a == b);
  }

 private:
  Transform transform_;
  DimensionIndex input_rank_;
};

/// Composes two IndexTransformSpec objects.
Result<IndexTransformSpec> ComposeIndexTransformSpecs(
    IndexTransformSpec b_to_c, IndexTransformSpec a_to_b);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_SPEC_H_
