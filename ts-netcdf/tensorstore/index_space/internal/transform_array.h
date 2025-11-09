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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_ARRAY_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_ARRAY_H_

#include "tensorstore/array.h"
#include "tensorstore/index_space/internal/transform_rep.h"

namespace tensorstore {
namespace internal_index_space {

/// Computes a strided array representation of a sub-region of a transformed
/// array.
///
/// The domain of the result array must be specified explicitly.  The
/// `TransformArrayPreservingOrigin` and `TransformArrayDiscardingOrigin`
/// functions provide convenience interfaces that compute the domain of the
/// result array automatically.
///
/// \param array The base array.
/// \param transform Pointer to index transform.  May be `nullptr` to indicate
///     an identity transform.  Otherwise, must have
///     `transform->output_rank() == array.rank()`.
/// \param result_origin[in] Non-null pointer to array of length
///     `transform->input_rank()`.  Specifies the origin of the sub-region to
///     extract.
/// \param result_shape[in] Non-null pointer to array of length
///     `transform->input_rank()`.  Specifies the shape of the sub-region to
///     extract.
/// \param result_byte_strides[out] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the byte strides for the new
///     array.
/// \param constraints_opt Specifies constraints on the strided array that is
///     returned.  If `std::nullopt`, there are no constraints and the returned
///     array may point to the existing array.  Otherwise, a new array is
///     allocated with its layout constrained by `*constraints`.
/// \returns An pointer to the element at the index vector `result_origin`.
/// \remark The returned element pointer corresponds to the origin index vector,
///     not to the zero index vector.
Result<SharedElementPointer<const void>> TransformArraySubRegion(
    const SharedArrayView<const void, dynamic_rank, offset_origin>& array,
    TransformRep* transform, const Index* result_origin,
    const Index* result_shape, Index* result_byte_strides,
    TransformArrayConstraints constraints);

/// Computes a strided array representation of a transformed array.
///
/// This computes the domain of the result array automatically by propagating
/// the domain of `array` back through `transform` using
/// `PropagateExplicitBounds`.
///
/// \param array The base array.
/// \param transform Pointer to index transform.  May be `nullptr` to indicate
///     an identity transform.  Otherwise, must have
///     `transform->output_rank() == array.rank()`.
/// \param result_origin[in] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the origin of the new array.
/// \param result_shape[out] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the shape of the new array.
/// \param result_byte_strides[out] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the byte strides of the new
///     array.
/// \param constraints Specifies constraints on the strided array that is
///     returned.  If `std::nullopt`, there are no constraints and the returned
///     array may point to the existing array.  Otherwise, a new array is
///     allocated with its layout constrained by `*constraints`.
/// \returns An pointer to the element at the zero vector.  The layout of the
///     returned array is given by `result_origin`, `result_shape`, and
///     `result_byte_strides`.
/// \remark This is a convenience interface on top of `TransformArraySubRegion`.
Result<SharedElementPointer<const void>> TransformArrayPreservingOrigin(
    SharedArrayView<const void, dynamic_rank, offset_origin> array,
    TransformRep* transform, Index* result_origin, Index* result_shape,
    Index* result_byte_strides, TransformArrayConstraints constraints);

/// Computes a strided array representation of a transformed array, translated
/// to have a zero origin.
///
/// This computes the domain of the result array automatically by propagating
/// the domain of `array` back through `transform` using
/// `PropagateExplicitBounds`.
///
/// \param array The base array.
/// \param transform Pointer to index transform.  May be `nullptr` to indicate
///     an identity transform.  Otherwise, must have
///     `transform->output_rank() == array.rank()`.
/// \param result_shape[out] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the shape of the new array.
/// \param result_byte_strides[out] Non-null pointer to array of length
///     `transform->input_rank()`.  Filled with the byte strides of the new
///     array.
/// \param constraints Specifies constraints on the strided array that is
///     returned.  If `std::nullopt`, there are no constraints and the returned
///     array may point to the existing array.  Otherwise, a new array is
///     allocated with its layout constrained by `*constraints`.
/// \returns An pointer to the element at the origin of the result array.
/// \remark This is a convenience interface on top of `TransformArraySubRegion`.
Result<SharedElementPointer<const void>> TransformArrayDiscardingOrigin(
    SharedArrayView<const void, dynamic_rank, offset_origin> array,
    TransformRep* transform, Index* result_shape, Index* result_byte_strides,
    TransformArrayConstraints constraints);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORM_ARRAY_H_
