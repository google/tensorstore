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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_decoder.h"

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/endian/endian_reading.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/cord_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

Result<std::vector<MinishardIndexEntry>> DecodeMinishardIndex(
    const absl::Cord& input, ShardingSpec::DataEncoding encoding) {
  absl::Cord decoded_storage;
  const absl::Cord* decoded_input = &input;
  if (encoding != ShardingSpec::DataEncoding::raw) {
    TENSORSTORE_ASSIGN_OR_RETURN(decoded_storage, DecodeData(input, encoding));
    decoded_input = &decoded_storage;
  }
  if ((decoded_input->size() % 24) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid minishard index length: %d", decoded_input->size()));
  }
  std::vector<MinishardIndexEntry> result(decoded_input->size() / 24);
  static_assert(sizeof(MinishardIndexEntry) == 24);
  riegeli::CordReader<const absl::Cord*> reader(decoded_input);
  ChunkId chunk_id{0};
  for (size_t i = 0; i < result.size(); ++i) {
    uint64_t delta = 0;
    riegeli::ReadLittleEndian<uint64_t>(reader, delta);
    chunk_id.value += delta;
    result[i].chunk_id = chunk_id;
  }
  for (size_t i = 0; i < result.size(); ++i) {
    uint64_t offset_delta = 0;
    riegeli::ReadLittleEndian<uint64_t>(reader, offset_delta);
    result[i].byte_range.inclusive_min = static_cast<int64_t>(offset_delta);
  }
  uint64_t byte_offset = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    auto& entry = result[i];
    uint64_t size_delta = 0;
    riegeli::ReadLittleEndian<uint64_t>(reader, size_delta);
    byte_offset += static_cast<uint64_t>(entry.byte_range.inclusive_min);
    entry.byte_range.inclusive_min = byte_offset;
    byte_offset += size_delta;
    entry.byte_range.exclusive_max = byte_offset;
    if (!entry.byte_range.SatisfiesInvariants()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid byte range in minishard index for chunk %d: %v",
          entry.chunk_id.value, absl::FormatStreamed(entry.byte_range)));
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

Result<ByteRange> DecodeShardIndexEntry(std::string_view input) {
  if (input.size() != 16) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Expected 16 bytes, but received: %d bytes", input.size()));
  }
  ByteRange r;
  r.inclusive_min = little_endian::Load64(input.data());
  r.exclusive_max = little_endian::Load64(input.data() + 8);
  if (!r.SatisfiesInvariants()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Shard index specified invalid byte range: %v",
                        absl::FormatStreamed(r)));
  }
  return r;
}

Result<ByteRange> DecodeShardIndexEntry(const absl::Cord& input) {
  if (auto flat = input.TryFlat()) {
    return DecodeShardIndexEntry(*flat);
  }
  if (input.size() != 16) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Expected 16 bytes, but received: %d bytes", input.size()));
  }
  char buf[16];
  internal::CopyCordToSpan(input, buf);
  return DecodeShardIndexEntry(std::string_view(buf, 16));
}

Result<std::vector<MinishardIndexEntry>>
DecodeMinishardIndexAndAdjustByteRanges(const absl::Cord& encoded,
                                        const ShardingSpec& sharding_spec) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto minishard_index,
      DecodeMinishardIndex(encoded, sharding_spec.minishard_index_encoding));
  for (auto& entry : minishard_index) {
    auto result = GetAbsoluteShardByteRange(entry.byte_range, sharding_spec);
    if (!result.ok()) {
      return StatusBuilder(std::move(result).status())
          .Format("Error decoding minishard index entry for chunk %d",
                  entry.chunk_id.value);
    }
    entry.byte_range = std::move(result).value();
  }
  return minishard_index;
}

namespace {
absl::Status SplitMinishard(const ShardingSpec& sharding_spec,
                            const absl::Cord& shard_data, uint64_t minishard,
                            span<const MinishardIndexEntry> minishard_index,
                            std::vector<EncodedChunk>& chunks) {
  std::optional<ChunkId> prev_chunk_id;
  for (const auto& existing_entry : minishard_index) {
    if (prev_chunk_id &&
        existing_entry.chunk_id.value == prev_chunk_id->value) {
      return absl::FailedPreconditionError(
          absl::StrFormat("Chunk %d occurs more than once in the minishard "
                          "index for minishard %d",
                          existing_entry.chunk_id.value, minishard));
    }
    prev_chunk_id = existing_entry.chunk_id;
    const auto GetChunkByteRange = [&]() -> Result<ByteRange> {
      TENSORSTORE_RETURN_IF_ERROR(
          OptionalByteRangeRequest(existing_entry.byte_range)
              .Validate(shard_data.size()));
      return existing_entry.byte_range;
    };
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_byte_range, GetChunkByteRange(),
        _.Format("Invalid existing byte range for chunk %d",
                 existing_entry.chunk_id.value));
    chunks.push_back(
        EncodedChunk{{minishard, existing_entry.chunk_id},
                     internal::GetSubCord(shard_data, chunk_byte_range)});
  }
  return absl::OkStatus();
}
}  // namespace

Result<std::vector<EncodedChunk>> SplitShard(const ShardingSpec& sharding_spec,
                                             const absl::Cord& shard_data) {
  std::vector<EncodedChunk> chunks;
  if (shard_data.empty()) return chunks;
  const uint64_t num_minishards = sharding_spec.num_minishards();
  if (shard_data.size() < num_minishards * 16) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Existing shard has size %d, but expected at least: %d",
                        shard_data.size(), num_minishards * 16));
  }
  riegeli::CordReader<const absl::Cord*> shard_index_reader(&shard_data);
  for (uint64_t minishard = 0; minishard < num_minishards; ++minishard) {
    char shard_index_entry[16];
    shard_index_reader.Read(16, shard_index_entry);
    const auto GetMinishardIndexByteRange = [&]() -> Result<ByteRange> {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto minishard_index_byte_range,
          DecodeShardIndexEntry(std::string_view(shard_index_entry, 16)));
      TENSORSTORE_ASSIGN_OR_RETURN(
          minishard_index_byte_range,
          GetAbsoluteShardByteRange(minishard_index_byte_range, sharding_spec));
      TENSORSTORE_RETURN_IF_ERROR(
          OptionalByteRangeRequest(minishard_index_byte_range)
              .Validate(shard_data.size()));
      return minishard_index_byte_range;
    };
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto minishard_ibr, GetMinishardIndexByteRange(),
        _.Format("Error decoding existing shard index entry for minishard %d",
                 minishard));
    if (minishard_ibr.size() == 0) continue;
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto minishard_index,
        DecodeMinishardIndexAndAdjustByteRanges(
            internal::GetSubCord(shard_data, minishard_ibr), sharding_spec),
        _.Format("Error decoding existing minishard index for minishard %d",
                 minishard));
    TENSORSTORE_RETURN_IF_ERROR(SplitMinishard(
        sharding_spec, shard_data, minishard, minishard_index, chunks));
  }
  return chunks;
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
