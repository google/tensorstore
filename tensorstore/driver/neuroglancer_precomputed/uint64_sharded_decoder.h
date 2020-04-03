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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_DECODER_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_DECODER_H_

/// \file
/// Support for decoding the neuroglancer_uint64_sharded_v1 format.
///
/// See description of format here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#sharded-format

#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

/// Decodes a minishard index.
///
/// \param input Encoded minishard index.
/// \param encoding Specifies the encoding used for `input`.
/// \returns The minishard index, sorted by `chunk_id`.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Result<std::vector<MinishardIndexEntry>> DecodeMinishardIndex(
    absl::string_view input, ShardingSpec::DataEncoding encoding);

/// Looks up the byte range associated with a given chunk id within a minishard
/// index.
///
/// \pre `minishard_index` is sorted by `chunk_id`.
/// \returns The byte range, or `std::nullopt` if `chunk_id` is not found in
///     `minishard_index`.
std::optional<ByteRange> FindChunkInMinishard(
    span<const MinishardIndexEntry> minishard_index, ChunkId chunk_id);

/// Decodes a string with a given `ShardingSpec::DataEncoding`.
///
/// \returns The decoded string.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Result<std::string> DecodeData(absl::string_view input,
                               ShardingSpec::DataEncoding encoding);

/// Decodes a shard index entry for a given minishard.
///
/// \param `absl::StatusCode::kFailedPrecondition` if `input` is corrupt.
/// \returns The byte range of the minishard index.
Result<ByteRange> DecodeShardIndexEntry(absl::string_view input);

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_DECODER_H_
