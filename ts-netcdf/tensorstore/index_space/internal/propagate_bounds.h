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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_PROPAGATE_BOUNDS_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_PROPAGATE_BOUNDS_H_

#include "absl/status/status.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/dimension_set.h"

namespace tensorstore {
namespace internal_index_space {

/// Implementation of the `PropagateBounds` function in index_transform.h.
///
/// Refer to the documentation there.
absl::Status PropagateBounds(BoxView<> b, DimensionSet b_implicit_lower_bounds,
                             DimensionSet b_implicit_upper_bounds,
                             TransformRep* a_to_b, MutableBoxView<> a);

/// Implementation of the `PropagateExplicitBounds` function in
/// index_transform.h.
///
/// Same as above, except that `b_implicit_lower_bounds` and
/// `b_implicit_upper_bounds` are assumed to be all `false`.
absl::Status PropagateExplicitBounds(BoxView<> b, TransformRep* a_to_b,
                                     MutableBoxView<> a);

/// Implementation of the `PropagateBounds` function in index_transform.h.
///
/// Refer to the documentation there.
///
/// The output `a_implicit_{lower,upper}_bounds` bit vectors may alias
/// `a_to_b->implicit_{lower,upper}_bounds(a.rank())`.
absl::Status PropagateBounds(BoxView<> b, DimensionSet b_implicit_lower_bounds,
                             DimensionSet b_implicit_upper_bounds,
                             TransformRep* a_to_b, MutableBoxView<> a,
                             DimensionSet& a_implicit_lower_bounds,
                             DimensionSet& a_implicit_upper_bounds);

/// Implementation of `PropagateBoundsToTransform` function index_transform.h
///
/// Refer to the documentation there.
Result<TransformRep::Ptr<>> PropagateBoundsToTransform(
    BoxView<> b_domain, DimensionSet b_implicit_lower_bounds,
    DimensionSet b_implicit_upper_bounds, TransformRep::Ptr<> a_to_b);

/// Same as above, except that `b_implicit_lower_bounds` and
/// `b_implicit_upper_bounds` assumed to be all `false`, with the effect that
/// `implicit_lower_bounds` and `implicit_upper_bounds` of the returned
/// transform are all `false`.
Result<TransformRep::Ptr<>> PropagateExplicitBoundsToTransform(
    BoxView<> b_domain, TransformRep::Ptr<> a_to_b);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_PROPAGATE_BOUNDS_H_
