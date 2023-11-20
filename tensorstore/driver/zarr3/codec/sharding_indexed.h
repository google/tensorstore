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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_SHARDING_INDEXED_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_SHARDING_INDEXED_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr3 {

using ::tensorstore::zarr3_sharding_indexed::ShardIndexLocation;

// Codec that partitions each chunk into sub-chunks and supports efficient
// reading of each sub-chunk.
//
// The sub-chunks are accessed using the `zarr3_sharding_indexed` kvstore
// driver.
//
// https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html
class ShardingIndexedCodecSpec : public ZarrShardingCodecSpec {
 public:
  struct Options {
    std::optional<std::vector<Index>> sub_chunk_shape;
    std::optional<ZarrCodecChainSpec> index_codecs;
    std::optional<ZarrCodecChainSpec> sub_chunk_codecs;
    std::optional<ShardIndexLocation> index_location;
  };
  ShardingIndexedCodecSpec() = default;
  explicit ShardingIndexedCodecSpec(Options&& options)
      : options(std::move(options)) {}

  absl::Status MergeFrom(const ZarrCodecSpec& other, bool strict) override;
  ZarrCodecSpec::Ptr Clone() const override;

  absl::Status MergeSubChunkCodecsFrom(const ZarrCodecChainSpec& other,
                                       bool strict) override;

  const ZarrCodecChainSpec* GetSubChunkCodecs() const override;

  absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& array_info,
      ArrayCodecChunkLayoutInfo& decoded) const override;

  Result<ZarrArrayToBytesCodec::Ptr> Resolve(
      ArrayCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrArrayToBytesCodecSpec::Ptr* resolved_spec) const override;

  Options options;
};

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_SHARDING_INDEXED_H_
