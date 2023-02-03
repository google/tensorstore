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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"

#include <algorithm>

#include "absl/base/optimization.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/murmurhash3.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

namespace {
namespace jb = tensorstore::internal_json_binding;
constexpr auto HashFunctionBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) {
  using HashFunction = ShardingSpec::HashFunction;
  return jb::Enum<HashFunction, const char*>({
      {HashFunction::identity, "identity"},
      {HashFunction::murmurhash3_x86_128, "murmurhash3_x86_128"},
  })(is_loading, options, obj, j);
};

constexpr auto DefaultableDataEncodingJsonBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      using DataEncoding = ShardingSpec::DataEncoding;
      return jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
          [](auto* v) { *v = DataEncoding::raw; }, DataEncodingJsonBinder)(
          is_loading, options, obj, j);
    };
}  // namespace

TENSORSTORE_DEFINE_JSON_BINDER(
    DataEncodingJsonBinder, jb::Enum<ShardingSpec::DataEncoding, const char*>({
                                {ShardingSpec::DataEncoding::raw, "raw"},
                                {ShardingSpec::DataEncoding::gzip, "gzip"},
                            }))

std::ostream& operator<<(std::ostream& os, ShardingSpec::HashFunction x) {
  // `ToJson` is guaranteed not to fail for this type.
  return os << jb::ToJson(x, HashFunctionBinder).value();
}

void to_json(::nlohmann::json& out,  // NOLINT
             ShardingSpec::HashFunction x) {
  // `ToJson` is guaranteed not to fail for this type.
  out = jb::ToJson(x, HashFunctionBinder).value();
}

std::ostream& operator<<(std::ostream& os, ShardingSpec::DataEncoding x) {
  // `ToJson` is guaranteed not to fail for this type.
  return os << jb::ToJson(x, DataEncodingJsonBinder).value();
}

std::ostream& operator<<(std::ostream& os, const ShardingSpec& x) {
  // `ToJson` is guaranteed not to fail for this type.
  return os << jb::ToJson(x).value();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ShardingSpec, [](auto is_loading,
                                                        const auto& options,
                                                        auto* obj, auto* j) {
  return jb::Object(
      jb::Member("@type",
                 jb::Constant([] { return "neuroglancer_uint64_sharded_v1"; })),
      jb::Member("preshift_bits", jb::Projection(&ShardingSpec::preshift_bits,
                                                 jb::Integer<int>(0, 64))),
      jb::Member("minishard_bits", jb::Projection(&ShardingSpec::minishard_bits,
                                                  jb::Integer<int>(0, 60))),
      jb::Member("shard_bits",
                 jb::Dependent([](auto is_loading, const auto& options,
                                  auto* obj, auto* j) {
                   return jb::Projection(
                       &ShardingSpec::shard_bits,
                       jb::Integer<int>(0, 64 - obj->minishard_bits));
                 })),
      jb::Member("hash", jb::Projection(&ShardingSpec::hash_function,
                                        HashFunctionBinder)),
      jb::Member("data_encoding",
                 jb::Projection(&ShardingSpec::data_encoding,
                                DefaultableDataEncodingJsonBinder)),
      jb::Member("minishard_index_encoding",
                 jb::Projection(&ShardingSpec::minishard_index_encoding,
                                DefaultableDataEncodingJsonBinder)))(
      is_loading, options, obj, j);
})

bool operator==(const ShardingSpec& a, const ShardingSpec& b) {
  return a.hash_function == b.hash_function &&
         a.preshift_bits == b.preshift_bits &&
         a.minishard_bits == b.minishard_bits && a.shard_bits == b.shard_bits &&
         a.data_encoding == b.data_encoding &&
         a.minishard_index_encoding == b.minishard_index_encoding;
}

std::string GetShardKey(const ShardingSpec& sharding_spec,
                        std::string_view prefix, std::uint64_t shard_number) {
  return internal::JoinPath(
      prefix,
      absl::StrFormat("%0*x.shard", CeilOfRatio(sharding_spec.shard_bits, 4),
                      shard_number));
}

namespace {

constexpr std::uint64_t ShiftRightUpTo64(std::uint64_t x, int amount) {
  if (amount == 64) return 0;
  return x >> amount;
}

std::uint64_t GetLowBitMask(int num_bits) {
  if (num_bits == 64) return ~std::uint64_t(0);
  return (std::uint64_t(1) << num_bits) - 1;
}

}  // namespace

std::uint64_t HashChunkId(ShardingSpec::HashFunction h, std::uint64_t key) {
  switch (h) {
    case ShardingSpec::HashFunction::identity:
      return key;
    case ShardingSpec::HashFunction::murmurhash3_x86_128: {
      std::uint32_t out[4] = {0, 0, 0};
      MurmurHash3_x86_128Hash64Bits(key, out);
      return (static_cast<std::uint64_t>(out[1]) << 32) | out[0];
    }
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

ChunkCombinedShardInfo GetChunkShardInfo(const ShardingSpec& sharding_spec,
                                         ChunkId chunk_id) {
  ChunkCombinedShardInfo result;
  const std::uint64_t hash_input =
      ShiftRightUpTo64(chunk_id.value, sharding_spec.preshift_bits);
  const std::uint64_t hash_output =
      HashChunkId(sharding_spec.hash_function, hash_input);
  result.shard_and_minishard =
      hash_output &
      GetLowBitMask(sharding_spec.minishard_bits + sharding_spec.shard_bits);
  return result;
}

ChunkSplitShardInfo GetSplitShardInfo(const ShardingSpec& sharding_spec,
                                      ChunkCombinedShardInfo combined_info) {
  ChunkSplitShardInfo result;
  result.minishard = combined_info.shard_and_minishard &
                     GetLowBitMask(sharding_spec.minishard_bits);
  result.shard = ShiftRightUpTo64(combined_info.shard_and_minishard,
                                  sharding_spec.minishard_bits) &
                 GetLowBitMask(sharding_spec.shard_bits);
  return result;
}

std::uint64_t ShardIndexSize(const ShardingSpec& sharding_spec) {
  return static_cast<std::uint64_t>(16) << sharding_spec.minishard_bits;
}

Result<ByteRange> GetAbsoluteShardByteRange(ByteRange relative_range,
                                            const ShardingSpec& sharding_spec) {
  const std::uint64_t offset = ShardIndexSize(sharding_spec);
  ByteRange result;
  if (internal::AddOverflow(relative_range.inclusive_min, offset,
                            &result.inclusive_min) ||
      internal::AddOverflow(relative_range.exclusive_max, offset,
                            &result.exclusive_max)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Byte range ", relative_range,
        " relative to the end of the shard index (", offset, ") is not valid"));
  }
  return result;
}

const EncodedChunk* FindChunk(span<const EncodedChunk> chunks,
                              MinishardAndChunkId minishard_and_chunk_id) {
  const auto chunk_it = std::lower_bound(
      chunks.begin(), chunks.end(), minishard_and_chunk_id,
      [](const auto& chunk, const auto& minishard_and_chunk_id) {
        return chunk.minishard_and_chunk_id < minishard_and_chunk_id;
      });
  if (chunk_it == chunks.end() ||
      chunk_it->minishard_and_chunk_id != minishard_and_chunk_id) {
    return nullptr;
  }
  return &*chunk_it;
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
