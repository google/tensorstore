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

#ifndef TENSORSTORE_INDEX_SPACE_H_
#define TENSORSTORE_INDEX_SPACE_H_

/// \file
/// Index Space
/// ===
///
/// We define an "index space" to be the Cartesian product `[inclusive_min(0),
/// inclusive_max(0)] * ... * [inclusive_min(rank-1), inclusive_max(rank-1)]` of
/// `rank` intervals of integer Index values representing the domains for each
/// of the `rank` dimensions.  The `inclusive_min` and `inclusive_max` bounds
/// for each dimension may be set to the special values of `-kInfIndex` and
/// `+kInfIndex`, respectively, which correspond to negative and positive
/// infinity and indicate a domain that is unbounded below and/or above.  Note
/// that the domain for a dimension may include negative numbers.
///
/// Dimensions of an index space may be identified by their DimensionIndex in
/// the range `[0, rank-1]`.  Dimensions also have string labels by which they
/// may be identified.
///
/// The in-memory array types SharedArray, SharedArrayView, and ArrayView
/// implicitly have an index space `[0, size(0)-1] * ... * [0, size(rank-1)-1]`.
///
/// Index Transform
/// ===
///
/// An "index transform" (represented by the IndexTransform and
/// IndexTransformView class templates) specifies an "input" index space, along
/// with a mapping from index vectors in the input domain to index vectors in an
/// "output" index space.  The rank of the input and output index spaces need
/// not be the same.  While the bounds of the input index space are represented
/// explicitly, the bounds of the output index space are implicit.
///
/// The mapping from index vectors in the input space to index vectors in the
/// output space is specified by a separate OutputIndexMap for each output
/// dimension.  An OutputIndexMap maps the `input_indices` index vector to a
/// single `output_index` into a given output dimension.  Three types of
/// mappings are supported:
///
///   1. `constant`, where the output index is a constant value that does not
///      depend on the input index vector, with a single `offset` parameter:
///
///          output_index = offset
///
///      The output range is the singleton set {offset}.
///
///   2. `single_input_dimension`, where the output index is an affine function
///      of a single component of the input index vector, with `offset` and
///      `stride` parameters:
///
///          output_index = offset + stride * input_indices[input_dimension]
///
///      The output range is defined by the same relation:
///
///          output_range = offset + stride * input_domain
///
///   3. `array`, where the output index is an affine function of the result of
///      indexing a strided index array by the input index vector, with
///      `offset`, `stride`, `index_array`, and `index_range` parameters:
///
///          output_index = offset
///                       + stride * CheckBounds(index_array(input_indices),
///                                              index_range)
///
///      Note that while the index_array is nominally indexed by the full
///      `input_indices` vector, the `byte_stride` value of the index array
///      corresponding to some of the input dimensions may be `0`, such that the
///      memory footprint of the index array need not be proportional to the
///      total number of index vectors in the input space.
///
///      The `index_range` IndexInterval specifies bounds on the elements in
///      `index_array` that are checked lazily when the index transform is used.
///
///      The output range is defined by:
///
///          output_range = offset + stride * index_range
///
/// This representation is capable of efficiently representing any composition
/// of the following types of index transforms:
///
///   1. Slicing a dimension by a single index, removing that dimension.
///
///   2. Slicing a dimension by an interval, with optional striding.
///
///   3. Translating (shifting) the domain of an input dimension.
///
///   4. Jointly indexing a subset of the dimensions using index arrays.
///
///   5. Extracting the diagonal of some subset of the dimensions.
///
///   6. Adding new dummy/singleton dimensions.
///
///   7. Reordering (transposing) the dimensions.
///
///   8. Changing the label of a dimension.
///
/// In particular, an index transform can represent any NumPy basic and advanced
/// integer indexing expression.  There is no direct support for NumPy-style
/// boolean array indexing, but it can be done indirectly by first converting
/// the boolean array to an integer index array.
///
/// Using index transforms
/// ===
///
/// Index transforms are useful for creating transformed views of in-memory
/// arrays (see TransformedArray) and tensor stores.  Rather than creating an
/// index transform explicitly, it is often more convenient to allow one to be
/// created implicitly by applying an "indexing expression" to an array or
/// tensor store, as described in a following section.
///
/// While it is possible to map individual index vectors using an index
/// transform, that is not a particularly efficient way to loop over an entire
/// transformed index space.  Instead, IterateOverTransformedArrays can be used
/// to efficiently apply an element-wise `n`-ary function to `n` transformed
/// array views.
///
/// Creating and composing index transforms
/// ===
///
/// There are several ways to create an index transform:
///
///   1. Using an IndexTransformBuilder, by explicitly specifying the input
///      domain, optional input dimension labels, and the output index maps.
///      This is useful for deserializing/converting from external
///      representations of index transforms.
///
///   2. The IdentityTransform and IdentityTransformLike functions can be used
///      to create an identity index transform over a given index space.
///
///   3. The ComposeTransforms function can be used to compose an index
///      transform from an input space B to an output space C with an index
///      transform from an input space A to B, to obtain a new index transform
///      from A to C.
///
///   4. A "dimension expression" (represented using the DimExpression class
///      template) specifies a subset of the input dimensions and a sequence of
///      one or more operations to apply to those dimensions.  A rich set of
///      operations are supported, including various forms of slicing and
///      dimension reordering.  A dimension expression is composed with an
///      existing index transform to produce a new index transform.
///
/// An index transform or dimension expression can also be composed directly
/// with any IndexTransformable object (such as an array, transformed array, or
/// tensor store) to obtain a transformed view of that object.

#include "tensorstore/index_space/dim_expression.h"  // IWYU pragma: export
#include "tensorstore/index_space/dimension_identifier.h"  // IWYU pragma: export
#include "tensorstore/index_space/index_transform.h"  // IWYU pragma: export
#include "tensorstore/index_space/index_transform_builder.h"  // IWYU pragma: export
#include "tensorstore/index_space/transformed_array.h"  // IWYU pragma: export

#endif  // TENSORSTORE_INDEX_SPACE_H_
