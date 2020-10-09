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

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_encoder.h"

#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/function_view.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

absl::Cord EncodeMinishardIndex(
    span<const MinishardIndexEntry> minishard_index) {
  internal::FlatCordBuilder builder(minishard_index.size() * 24);
  ChunkId prev_chunk_id{0};
  std::uint64_t prev_offset = 0;
  for (std::ptrdiff_t i = 0; i < minishard_index.size(); ++i) {
    const auto& e = minishard_index[i];
    absl::little_endian::Store64(builder.data() + i * 8,
                                 e.chunk_id.value - prev_chunk_id.value);
    absl::little_endian::Store64(
        builder.data() + minishard_index.size() * 8 + i * 8,
        e.byte_range.inclusive_min - prev_offset);
    absl::little_endian::Store64(
        builder.data() + minishard_index.size() * 16 + i * 8,
        e.byte_range.exclusive_max - e.byte_range.inclusive_min);
    prev_chunk_id = e.chunk_id;
    prev_offset = e.byte_range.exclusive_max;
  }
  return std::move(builder).Build();
}

absl::Cord EncodeShardIndex(span<const ShardIndexEntry> shard_index) {
  internal::FlatCordBuilder builder(shard_index.size() * 16);
  for (std::ptrdiff_t i = 0; i < shard_index.size(); ++i) {
    const auto& e = shard_index[i];
    absl::little_endian::Store64(builder.data() + i * 16, e.inclusive_min);
    absl::little_endian::Store64(builder.data() + i * 16 + 8, e.exclusive_max);
  }
  return std::move(builder).Build();
}

ShardEncoder::ShardEncoder(const ShardingSpec& sharding_spec,
                           WriteFunction write_function)
    : sharding_spec_(sharding_spec),
      write_function_(std::move(write_function)),
      shard_index_(static_cast<size_t>(1) << sharding_spec_.minishard_bits),
      cur_minishard_(0),
      data_file_offset_(0) {}

ShardEncoder::ShardEncoder(const ShardingSpec& sharding_spec, absl::Cord& out)
    : ShardEncoder(sharding_spec, [&out](const absl::Cord& buffer) {
        out.Append(buffer);
        return absl::OkStatus();
      }) {}

namespace {
Result<std::uint64_t> EncodeData(
    const absl::Cord& input, ShardingSpec::DataEncoding encoding,
    FunctionView<Status(const absl::Cord& buffer)> write_function) {
  auto encoded = EncodeData(input, encoding);
  if (auto status = write_function(encoded); status.ok()) {
    return encoded.size();
  } else {
    return status;
  }
}
}  // namespace

Status ShardEncoder::FinalizeMinishard() {
  if (minishard_index_.empty()) return absl::OkStatus();
  auto uncompressed_minishard_index = EncodeMinishardIndex(minishard_index_);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto num_bytes,
      EncodeData(uncompressed_minishard_index,
                 sharding_spec_.minishard_index_encoding, write_function_));
  shard_index_[cur_minishard_] = {data_file_offset_,
                                  data_file_offset_ + num_bytes};
  data_file_offset_ += num_bytes;
  minishard_index_.clear();
  return absl::OkStatus();
}

Result<absl::Cord> ShardEncoder::Finalize() {
  TENSORSTORE_RETURN_IF_ERROR(FinalizeMinishard());
  return EncodeShardIndex(shard_index_);
}

Result<ByteRange> ShardEncoder::WriteUnindexedEntry(std::uint64_t minishard,
                                                    const absl::Cord& data,
                                                    bool compress) {
  if (minishard != cur_minishard_) {
    if (minishard < cur_minishard_) {
      return absl::InvalidArgumentError(StrCat("Minishard ", minishard,
                                               " cannot be written after ",
                                               cur_minishard_));
    }
    TENSORSTORE_RETURN_IF_ERROR(FinalizeMinishard());
    cur_minishard_ = minishard;
  }
  std::string output;
  auto start_offset = data_file_offset_;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto num_bytes, EncodeData(data,
                                 compress ? sharding_spec_.data_encoding
                                          : ShardingSpec::DataEncoding::raw,
                                 write_function_));
  data_file_offset_ += num_bytes;
  return ByteRange{start_offset, data_file_offset_};
}

Status ShardEncoder::WriteIndexedEntry(std::uint64_t minishard,
                                       ChunkId chunk_id, const absl::Cord& data,
                                       bool compress) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto byte_range,
                               WriteUnindexedEntry(minishard, data, compress));
  minishard_index_.push_back({chunk_id, byte_range});
  return absl::OkStatus();
}

ShardEncoder::~ShardEncoder() = default;

std::optional<absl::Cord> EncodeShard(const ShardingSpec& spec,
                                      span<const EncodedChunk> chunks) {
  absl::Cord shard_data;
  ShardEncoder encoder(spec, shard_data);
  for (const auto& chunk : chunks) {
    TENSORSTORE_CHECK_OK(
        encoder.WriteIndexedEntry(chunk.minishard_and_chunk_id.minishard,
                                  chunk.minishard_and_chunk_id.chunk_id,
                                  chunk.encoded_data, /*compress=*/false));
  }
  auto shard_index = encoder.Finalize().value();
  if (shard_data.empty()) return std::nullopt;
  shard_index.Append(shard_data);
  return shard_index;
}

absl::Cord EncodeData(const absl::Cord& input,
                      ShardingSpec::DataEncoding encoding) {
  if (encoding == ShardingSpec::DataEncoding::raw) {
    return input;
  }
  absl::Cord compressed;
  zlib::Options options;
  options.level = 9;
  options.use_gzip_header = true;
  zlib::Encode(input, &compressed, options);
  return compressed;
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
