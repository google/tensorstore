// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/wrapping_writer.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

namespace jb = ::tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_BINDER(ShardIndexLocationJsonBinder,
                               jb::Enum<ShardIndexLocation, const char*>({
                                   {ShardIndexLocation::kStart, "start"},
                                   {ShardIndexLocation::kEnd, "end"},
                               }));

absl::Status ShardIndexEntry::Validate(EntryId entry_id) const {
  if (!IsMissing()) {
    uint64_t exclusive_max;
    if (internal::AddOverflow(offset, length, &exclusive_max) ||
        exclusive_max > std::numeric_limits<int64_t>::max()) {
      return absl::DataLossError(absl::StrFormat(
          "Invalid shard index entry %d with offset=%d, length=%d", entry_id,
          offset, length));
    }
  }
  return absl::OkStatus();
}

absl::Status ShardIndexEntry::Validate(EntryId entry_id,
                                       int64_t total_size) const {
  if (auto status = Validate(entry_id); !status.ok()) return status;
  auto byte_range = AsByteRange();
  if (byte_range.exclusive_max > total_size) {
    return absl::DataLossError(tensorstore::StrCat(
        "Shard index entry ", entry_id, " with byte range ", byte_range,
        " is invalid for shard of size ", total_size));
  }
  return absl::OkStatus();
}

Result<ShardIndex> DecodeShardIndex(const absl::Cord& input,
                                    const ShardIndexParameters& parameters) {
  assert(parameters.index_shape.back() == 2);
  SharedArray<const void> entries;
  TENSORSTORE_ASSIGN_OR_RETURN(
      entries,
      parameters.index_codec_state->DecodeArray(parameters.index_shape, input));
  if (!IsContiguousLayout(entries, c_order)) {
    entries = MakeCopy(entries);
  }
  return ShardIndex{
      StaticDataTypeCast<const uint64_t, unchecked>(std::move(entries))};
}

Result<ShardIndex> DecodeShardIndexFromFullShard(
    const absl::Cord& shard_data,
    const ShardIndexParameters& shard_index_parameters) {
  int64_t shard_index_size =
      shard_index_parameters.index_codec_state->encoded_size();
  if (shard_index_size > shard_data.size()) {
    return absl::DataLossError(absl::StrFormat(
        "Existing shard has size of %d bytes, but expected at least %d bytes",
        shard_data.size(), shard_index_size));
  }
  absl::Cord encoded_shard_index;
  switch (shard_index_parameters.index_location) {
    case ShardIndexLocation::kStart:
      encoded_shard_index = shard_data.Subcord(0, shard_index_size);
      break;
    case ShardIndexLocation::kEnd:
      encoded_shard_index = shard_data.Subcord(
          shard_data.size() - shard_index_size, shard_index_size);
      break;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto shard_index,
      DecodeShardIndex(encoded_shard_index, shard_index_parameters),
      tensorstore::MaybeAnnotateStatus(_, "Error decoding shard index"));
  return shard_index;
}

absl::Status EncodeShardIndex(riegeli::Writer& writer,
                              const ShardIndex& shard_index,
                              const ShardIndexParameters& parameters) {
  // Wrap `writer` to prevent `EncodeArray` from closing it.
  riegeli::WrappingWriter wrapping_writer{&writer};
  return parameters.index_codec_state->EncodeArray(shard_index.entries,
                                                   wrapping_writer);
}

absl::Status ValidateGridShape(span<const Index> grid_shape) {
  if (grid_shape.size() > kMaxRank - 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("grid rank of %d exceeds maximum of %d",
                        grid_shape.size(), kMaxRank - 1));
  }
  if (ProductOfExtents(grid_shape) > kMaxNumEntries) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("grid shape of ", grid_shape, " has more than ",
                            kMaxNumEntries, " entries"));
  }
  return absl::OkStatus();
}

Result<ZarrCodecChain::Ptr> InitializeIndexCodecChain(
    const ZarrCodecChainSpec& codec_chain_spec, DimensionIndex grid_rank,
    ZarrCodecChainSpec* resolved_codec_chain_spec) {
  if (grid_rank > kMaxRank - 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Rank of %d exceeds maximum ran of %d supported for sharding_indexed",
        grid_rank, kMaxRank - 1));
  }
  static const uint64_t fill_value{std::numeric_limits<uint64_t>::max()};
  internal_zarr3::ArrayCodecResolveParameters array_params;
  array_params.dtype = dtype_v<uint64_t>;
  array_params.rank = grid_rank + 1;
  array_params.fill_value =
      SharedArray<const void>(internal::UnownedToShared(&fill_value));
  internal_zarr3::BytesCodecResolveParameters bytes_params;
  return codec_chain_spec.Resolve(std::move(array_params), bytes_params,
                                  resolved_codec_chain_spec);
}

absl::Status ShardIndexParameters::InitializeIndexShape(
    span<const Index> grid_shape) {
  TENSORSTORE_RETURN_IF_ERROR(ValidateGridShape(grid_shape));
  num_entries = ProductOfExtents(grid_shape);
  index_shape.resize(grid_shape.size() + 1);
  std::copy(grid_shape.begin(), grid_shape.end(), index_shape.begin());
  index_shape.back() = 2;
  return absl::OkStatus();
}

absl::Status ShardIndexParameters::Initialize(
    const ZarrCodecChainSpec& codec_chain_spec, span<const Index> grid_shape,
    ZarrCodecChainSpec* resolved_codec_chain_spec) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      index_codec_chain,
      InitializeIndexCodecChain(codec_chain_spec, grid_shape.size(),
                                resolved_codec_chain_spec));
  return Initialize(*index_codec_chain, grid_shape);
  return absl::OkStatus();
}

absl::Status ShardIndexParameters::Initialize(const ZarrCodecChain& codec_chain,
                                              span<const Index> grid_shape) {
  // Avoid redundant assignment if `Initialize` is being called with
  // `index_codec_chain` already assigned.
  if (index_codec_chain.get() != &codec_chain) {
    index_codec_chain.reset(&codec_chain);
  }
  TENSORSTORE_RETURN_IF_ERROR(InitializeIndexShape(grid_shape));
  TENSORSTORE_ASSIGN_OR_RETURN(index_codec_state,
                               index_codec_chain->Prepare(index_shape));
  if (index_codec_state->encoded_size() == -1) {
    return absl::InvalidArgumentError(
        "Invalid index_codecs specified: only fixed-size encodings are "
        "supported");
  }
  return absl::OkStatus();
}

Result<ShardEntries> DecodeShard(
    const absl::Cord& shard_data,
    const ShardIndexParameters& shard_index_parameters) {
  const int64_t num_entries = shard_index_parameters.num_entries;
  ShardEntries entries;
  entries.entries.resize(num_entries);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto shard_index,
      DecodeShardIndexFromFullShard(shard_data, shard_index_parameters));
  for (int64_t i = 0; i < num_entries; ++i) {
    const auto entry_index = shard_index[i];
    if (entry_index.IsMissing()) continue;
    TENSORSTORE_RETURN_IF_ERROR(entry_index.Validate(i, shard_data.size()));
    entries.entries[i] =
        internal::GetSubCord(shard_data, entry_index.AsByteRange());
  }
  return entries;
}

Result<std::optional<absl::Cord>> EncodeShard(
    const ShardEntries& entries,
    const ShardIndexParameters& shard_index_parameters) {
  int64_t shard_index_size =
      shard_index_parameters.index_codec_state->encoded_size();
  absl::Cord shard_data;
  riegeli::CordWriter writer{&shard_data};
  auto shard_index_array = AllocateArray<uint64_t>(
      shard_index_parameters.index_shape, c_order, default_init);
  bool has_entry = false;
  uint64_t offset =
      shard_index_parameters.index_location == ShardIndexLocation::kStart
          ? shard_index_size
          : 0;
  for (size_t i = 0; i < entries.entries.size(); ++i) {
    const auto& entry = entries.entries[i];
    uint64_t entry_offset;
    uint64_t length;
    if (entry) {
      has_entry = true;
      length = entry->size();
      entry_offset = offset;
      offset += length;
      ABSL_CHECK(writer.Write(*entry));
    } else {
      entry_offset = std::numeric_limits<uint64_t>::max();
      length = std::numeric_limits<uint64_t>::max();
    }
    shard_index_array.data()[i * 2] = entry_offset;
    shard_index_array.data()[i * 2 + 1] = length;
  }
  if (!has_entry) return std::nullopt;
  switch (shard_index_parameters.index_location) {
    case ShardIndexLocation::kStart: {
      ABSL_CHECK(writer.Close());
      absl::Cord encoded_shard_index;
      riegeli::CordWriter index_writer{&encoded_shard_index};
      TENSORSTORE_RETURN_IF_ERROR(EncodeShardIndex(
          index_writer, ShardIndex{std::move(shard_index_array)},
          shard_index_parameters));
      ABSL_CHECK(index_writer.Close());
      encoded_shard_index.Append(std::move(shard_data));
      shard_data = std::move(encoded_shard_index);
      break;
    }
    case ShardIndexLocation::kEnd: {
      TENSORSTORE_RETURN_IF_ERROR(
          EncodeShardIndex(writer, ShardIndex{std::move(shard_index_array)},
                           shard_index_parameters));
      ABSL_CHECK(writer.Close());
      break;
    }
  }
  return shard_data;
}

}  // namespace zarr3_sharding_indexed
}  // namespace tensorstore
