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

#ifndef TENSORSTORE_INDEX_SPACE_ALIGNMENT_H_
#define TENSORSTORE_INDEX_SPACE_ALIGNMENT_H_

/// \file
/// Utilities for aligning (and NumPy-style "broadcasting" of) the dimensions
/// between index transforms.

#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

enum DomainAlignmentOptions {
  /// Source and target domains must be equal (labels are ignored).
  none = 0,
  /// Source dimensions may be permuted based on their labels in order to align
  /// the source domain to the target domain.
  permute = 1,
  /// Source dimensions may be translated in order to align the source domain to
  /// the target domain.
  translate = 2,
  /// Source dimensions of size 1 do not have to match a target dimension, and
  /// not all target dimensions must match a source dimension.
  broadcast = 4,
  /// Dimensions may be permuted, translated, or broadcast to align the source
  /// domain to the target domain.
  all = 7,
};
constexpr inline DomainAlignmentOptions operator|(DomainAlignmentOptions a,
                                                  DomainAlignmentOptions b) {
  return static_cast<DomainAlignmentOptions>(static_cast<int>(a) |
                                             static_cast<int>(b));
}
constexpr inline DomainAlignmentOptions operator&(DomainAlignmentOptions a,
                                                  DomainAlignmentOptions b) {
  return static_cast<DomainAlignmentOptions>(static_cast<int>(a) &
                                             static_cast<int>(b));
}
constexpr inline bool operator!(DomainAlignmentOptions a) {
  return !static_cast<int>(a);
}

/// Attempts to align the `source` domain to match the `target` domain.
///
/// This is used to align dimensions for TensorStore read/write/copy operations.
///
/// First, a subset of dimensions of `source` is matched to a subset of
/// dimensions of `target`, according to one of two cases:
///
/// M1. At least one of `source` or `target` is entirely unlabeled (all
///     dimension labels are empty).  In this case, the last
///     `match_rank = min(source.rank(), target.rank())` dimensions of `source`
///     match in order to the last `match_rank` dimensions of `target`,
///     i.e. dimension `source.rank() - match_rank + i` of `source` matches to
///     dimension `target.rank() - match_rank + i` of `target`, for
///     `0 <= i < rank`.  This case also applies if `options` excludes
///     `DomainAlignmentOptions::permute`.
///
/// M2. Both `source` and `target` have at least one labeled dimension.  In this
///     case, dimensions of `source` and `target` with matching labels are
///     matched.  Any remaining labeled dimensions remain unmatched.  The
///     unlabeled dimensions of `source` are matched to the unlabeled dimensions
///     of `target` using the same method as in case M1 (right to left).
///
/// The matching is then validated as follows:
///
/// V1. For each match between a dimension `i` of `source` and a dimension `j`
///     of `target`, if `source.shape()[i] != target.shape()[j]`, the match is
///     dropped.  Note that if `source.shape()[i] != 1`, this leads to an error
///     in the following step (V2).
///
/// V2. For every unmatched dimension `i` of `source`, `source.shape()[i]` must
///     equal `1`.
///
/// If matching succeeds, a new `alignment` transform with an (input) domain
/// equal to `target` and an output rank equal to `source.rank()` is computed as
/// follows:
///
/// A1. For each dimension `j` of `target` with a matching dimension `i` of
///     `source`, output dimension `i` of `alignment` has a
///     `single_input_dimension` map to input dimension `j` with a stride of `1`
///     and offset of `source.origin()[i] - target.origin()[j]`.
///
/// A2. For every unmatched dimension `i` of `source`, output dimension `i` of
///     `alignment` is a `constant` map with an offset of `source.origin()[i]`.
///     (It must be the case that `source.shape()[i] == 1`.)
///
/// The return value is `alignment`.
///
/// Examples (input dimensions refer to dimensions of `target`, while output
/// dimensions refer to dimensions of `source`):
///
///   All unlabeled dimensions:
///
///     source: [3, 7), [5, 6), [4, 10)
///     target: [2, 6), [0, 4), [6, 12)
///     alignment: rank 3 -> 3, with:
///       output dimension 0 -> input dimension 0, offset 1
///       output dimension 1 -> constant 5
///       output dimension 2 -> input dimension 2, offset -2
///
///   All labeled dimensions:
///
///     source: "x": [3, 7), "y": [5, 6), "z": [4, 10)
///     target: "z": [6, 12), "x": [4, 8), "y": [0, 4)
///     alignment: rank 3 -> 3, with:
///       output dimension 0 -> input dimension 1, offset -1
///       output dimension 1 -> constant 5
///       output dimension 2 -> input dimension 0, offset -2
///
///   Partially labeled dimensions:
///
///     source: "x": [3, 7), "y": [5, 6), "": [4, 10)
///     target: "": [0, 10) "": [6, 12), "x": [4, 8), "y": [0, 4)
///     alignment: rank 4 -> 3, with:
///       output dimension 0 -> input dimension 2, offset -1
///       output dimension 1 -> constant 5
///       output dimension 2 -> input dimension 1, offset -2
///
///   Mismatched labeled dimensions:
///
///     source: "x": [3, 7), "y": [5, 6), "z": [4, 10)
///     target: "z": [6, 12), "w": [4, 8), "y": [0, 4)
///     ERROR: Unmatched source dimension 0 {"x": [3, 7)}
///            does not have a size of 1
///
/// \param source The source domain.
/// \param target The target domain to which `source` should be aligned.
/// \param options Specifies the transformations that are permitted.  By
///     default, all transformations (permutation, translation, broadcasting)
///     are permitted.
Result<IndexTransform<>> AlignDomainTo(
    IndexDomainView<> source, IndexDomainView<> target,
    DomainAlignmentOptions options = DomainAlignmentOptions::all);

/// Same as `AlignDomainTo` above, but instead of computing the `alignment`
/// transform, simply sets `source_matches[i] = j`, if `source` dimension `i`
/// matches to target dimension `j`, and sets `source_matches[i] = -1` if
/// `source` dimension `i` is unmatched.
///
/// \param source Source domain.
/// \param target Target domain.
/// \param source_matches[out] Array of length `source.rank()`, set to the
///     matches on success.  In the case of an error, the contents is
///     unspecified.
/// \param options Specifies the transformations that are permitted.  By
///     default, all transformations (permutation, translation, broadcasting)
///     are permitted.
/// \dchecks `source.valid()`
/// \dchecks `target.valid()`
/// \dchecks `source_matches.size() == source.rank()`
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if matching fails.
Status AlignDimensionsTo(
    IndexDomainView<> source, IndexDomainView<> target,
    span<DimensionIndex> source_matches,
    DomainAlignmentOptions options = DomainAlignmentOptions::all);

/// Returns `ComposeTransforms(source_transform, alignment)`, where `alignment`
/// is the result of
/// `AlignDomainTo(source_transform.domain(), target, options)`.
Result<IndexTransform<>> AlignTransformTo(
    IndexTransform<> source_transform, IndexDomainView<> target,
    DomainAlignmentOptions options = DomainAlignmentOptions::all);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_ALIGNMENT_H_
