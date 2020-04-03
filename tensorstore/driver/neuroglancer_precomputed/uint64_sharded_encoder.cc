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
#include "tensorstore/util/endian.h"
#include "tensorstore/util/function_view.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

void EncodeMinishardIndex(span<const MinishardIndexEntry> minishard_index,
                          std::string* out) {
  std::size_t base = out->size();
  out->resize(base + minishard_index.size() * 24);
  char* buf = out->data() + base;
  ChunkId prev_chunk_id{0};
  std::uint64_t prev_offset = 0;
  for (std::ptrdiff_t i = 0; i < minishard_index.size(); ++i) {
    const auto& e = minishard_index[i];
    absl::little_endian::Store64(buf + i * 8,
                                 e.chunk_id.value - prev_chunk_id.value);
    absl::little_endian::Store64(buf + minishard_index.size() * 8 + i * 8,
                                 e.byte_range.inclusive_min - prev_offset);
    absl::little_endian::Store64(
        buf + minishard_index.size() * 16 + i * 8,
        e.byte_range.exclusive_max - e.byte_range.inclusive_min);
    prev_chunk_id = e.chunk_id;
    prev_offset = e.byte_range.exclusive_max;
  }
}

void EncodeShardIndex(span<const ShardIndexEntry> shard_index,
                      std::string* out) {
  std::size_t base = out->size();
  out->resize(base + shard_index.size() * 16);
  char* buf = out->data() + base;
  for (std::ptrdiff_t i = 0; i < shard_index.size(); ++i) {
    const auto& e = shard_index[i];
    absl::little_endian::Store64(buf + i * 16, e.inclusive_min);
    absl::little_endian::Store64(buf + i * 16 + 8, e.exclusive_max);
  }
}

ShardEncoder::ShardEncoder(const ShardingSpec& sharding_spec,
                           WriteFunction write_function)
    : sharding_spec_(sharding_spec),
      write_function_(std::move(write_function)),
      shard_index_(static_cast<size_t>(1) << sharding_spec_.minishard_bits),
      cur_minishard_(0),
      data_file_offset_(0) {}

ShardEncoder::ShardEncoder(const ShardingSpec& sharding_spec, std::string* out)
    : ShardEncoder(sharding_spec, [out](absl::string_view buffer) {
        out->append(buffer.data(), buffer.size());
        return absl::OkStatus();
      }) {}

namespace {
Result<std::uint64_t> EncodeData(
    absl::string_view input, ShardingSpec::DataEncoding encoding,
    FunctionView<Status(absl::string_view buffer)> write_function) {
  if (encoding == ShardingSpec::DataEncoding::raw) {
    if (auto status = write_function(input); status.ok()) {
      return input.size();
    } else {
      return status;
    }
  }
  std::string compressed;
  zlib::Options options;
  options.level = 9;
  options.use_gzip_header = true;
  zlib::Encode(input, &compressed, options);
  if (auto status = write_function(compressed); status.ok()) {
    return compressed.size();
  } else {
    return status;
  }
}
}  // namespace

Status ShardEncoder::FinalizeMinishard() {
  if (minishard_index_.empty()) return absl::OkStatus();
  std::string uncompressed_minishard_index;
  EncodeMinishardIndex(minishard_index_, &uncompressed_minishard_index);
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

Result<std::string> ShardEncoder::Finalize() {
  TENSORSTORE_RETURN_IF_ERROR(FinalizeMinishard());
  std::string out;
  EncodeShardIndex(shard_index_, &out);
  return out;
}

Result<ByteRange> ShardEncoder::WriteUnindexedEntry(std::uint64_t minishard,
                                                    absl::string_view data,
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
                                       ChunkId chunk_id, absl::string_view data,
                                       bool compress) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto byte_range,
                               WriteUnindexedEntry(minishard, data, compress));
  minishard_index_.push_back({chunk_id, byte_range});
  return absl::OkStatus();
}

ShardEncoder::~ShardEncoder() = default;

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
