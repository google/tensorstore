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

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_decoder.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "tensorstore/internal/compression/zlib.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

Result<std::vector<MinishardIndexEntry>> DecodeMinishardIndex(
    const absl::Cord& input, ShardingSpec::DataEncoding encoding) {
  absl::Cord decoded_input;
  if (encoding != ShardingSpec::DataEncoding::raw) {
    TENSORSTORE_ASSIGN_OR_RETURN(decoded_input, DecodeData(input, encoding));
  } else {
    decoded_input = input;
  }
  if ((decoded_input.size() % 24) != 0) {
    return absl::InvalidArgumentError(
        StrCat("Invalid minishard index length: ", decoded_input.size()));
  }
  std::vector<MinishardIndexEntry> result(decoded_input.size() / 24);
  static_assert(sizeof(MinishardIndexEntry) == 24);
  auto decoded_flat = decoded_input.Flatten();
  ChunkId chunk_id{0};
  std::uint64_t byte_offset = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    auto& entry = result[i];
    chunk_id.value += absl::little_endian::Load64(decoded_flat.data() + i * 8);
    entry.chunk_id = chunk_id;
    byte_offset += absl::little_endian::Load64(decoded_flat.data() + i * 8 +
                                               8 * result.size());
    entry.byte_range.inclusive_min = byte_offset;
    byte_offset += absl::little_endian::Load64(decoded_flat.data() + i * 8 +
                                               16 * result.size());
    entry.byte_range.exclusive_max = byte_offset;
    if (!entry.byte_range.SatisfiesInvariants()) {
      return absl::InvalidArgumentError(
          StrCat("Invalid byte range in minishard index for chunk ",
                 entry.chunk_id.value, ": ", entry.byte_range));
    }
  }
  absl::c_sort(result,
               [](const MinishardIndexEntry& a, const MinishardIndexEntry& b) {
                 return a.chunk_id.value < b.chunk_id.value;
               });
  return result;
}

std::optional<ByteRange> FindChunkInMinishard(
    span<const MinishardIndexEntry> minishard_index, ChunkId chunk_id) {
  auto it =
      absl::c_lower_bound(minishard_index, chunk_id,
                          [](const MinishardIndexEntry& e, ChunkId chunk_id) {
                            return e.chunk_id.value < chunk_id.value;
                          });
  if (it == minishard_index.end() || it->chunk_id.value != chunk_id.value) {
    return std::nullopt;
  }
  return it->byte_range;
}

Result<absl::Cord> DecodeData(const absl::Cord& input,
                              ShardingSpec::DataEncoding encoding) {
  if (encoding == ShardingSpec::DataEncoding::raw) {
    return input;
  }
  absl::Cord uncompressed;
  TENSORSTORE_RETURN_IF_ERROR(
      zlib::Decode(input, &uncompressed, /*use_gzip_header=*/true));
  return uncompressed;
}

Result<ByteRange> DecodeShardIndexEntry(absl::string_view input) {
  if (input.size() != 16) {
    return absl::FailedPreconditionError(
        StrCat("Expected 16 bytes, but received: ", input.size(), " bytes"));
  }
  ByteRange r;
  r.inclusive_min = absl::little_endian::Load64(input.data());
  r.exclusive_max = absl::little_endian::Load64(input.data() + 8);
  if (!r.SatisfiesInvariants()) {
    return absl::FailedPreconditionError(
        StrCat("Shard index specified invalid byte range: ", r));
  }
  return r;
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
