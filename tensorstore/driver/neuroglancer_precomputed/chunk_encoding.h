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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_CHUNK_ENCODING_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_CHUNK_ENCODING_H_

#include <cstddef>
#include <string>

#include "tensorstore/array.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

/// Decodes a chunk.
///
/// \param chunk_indices Grid position of chunk (determines whether chunk is
///     clipped to volume bounds).
/// \param metadata Metadata (determines chunk format and volume bounds).
/// \param scale_index Scale index, in range `[0, metadata.scales.size())`.
/// \param chunk_layout Contiguous "czyx"-order layout of the returned chunk.
/// \param buffer Encoded chunk data.
/// \returns The decoded chunk, with layout equal to `chunk_layout`.
/// \error `absl::StatusCode::kInvalidArgument` if the encoded chunk is invalid.
Result<SharedArrayView<const void>> DecodeChunk(
    span<const Index> chunk_indices, const MultiscaleMetadata& metadata,
    std::size_t scale_index, StridedLayoutView<4> chunk_layout,
    absl::Cord buffer);

/// Encodes a chunk.
///
/// \param chunk_indices Grid position of chunk (determine whether chunk is
///     clipped to volume bounds).
/// \param metadata Metadata (determines chunk format and volume bounds).
/// \param scale_index Scale index, in range `[0, metadata.scales.size())`.
/// \param array Chunk data, in "czyx" order.
/// \returns The encoded chunk.
Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const MultiscaleMetadata& metadata,
                               std::size_t scale_index,
                               ArrayView<const void> array);

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_CHUNK_ENCODING_H_
