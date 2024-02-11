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

#include "tensorstore/driver/zarr3/codec/sharding_indexed.h"

#include <stdint.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/bytes.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/crc32c.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/zarr3_sharding_indexed.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

absl::Status SubChunkRankMismatch(span<const Index> sub_chunk_shape,
                                  DimensionIndex outer_rank) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "sharding_indexed sub-chunk shape of ", sub_chunk_shape,
      " is not compatible with array of rank ", outer_rank));
}

absl::Status SubChunkShapeMismatch(span<const Index> sub_chunk_shape,
                                   span<const Index> chunk_shape) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "sharding_indexed sub-chunk shape of ", sub_chunk_shape,
      " does not evenly divide chunk shape of  ", chunk_shape));
}

namespace {
class ShardingIndexedCodec : public ZarrShardingCodec {
 public:
  explicit ShardingIndexedCodec(
      internal::ChunkGridSpecification&& sub_chunk_grid)
      : sub_chunk_grid_(std::move(sub_chunk_grid)) {}

  class State : public ZarrShardingCodec::PreparedState,
                public internal::LexicographicalGridIndexKeyParser {
   public:
    absl::Status EncodeArray(SharedArrayView<const void> decoded,
                             riegeli::Writer& writer) const final {
      return absl::InternalError("");
    }
    Result<SharedArray<const void>> DecodeArray(
        span<const Index> decoded_shape, riegeli::Reader& reader) const final {
      return absl::InternalError("");
    }

    kvstore::DriverPtr GetSubChunkKvstore(
        kvstore::DriverPtr parent, std::string parent_key,
        const Executor& executor,
        internal::CachePool::WeakPtr cache_pool) const override {
      zarr3_sharding_indexed::ShardedKeyValueStoreParameters params;
      params.base_kvstore = std::move(parent);
      params.base_kvstore_path = std::move(parent_key);
      params.executor = executor;
      params.cache_pool = std::move(cache_pool);
      params.index_params = shard_index_params_;
      return zarr3_sharding_indexed::GetShardedKeyValueStore(std::move(params));
    }

    const LexicographicalGridIndexKeyParser& GetSubChunkStorageKeyParser()
        const final {
      return *this;
    }

    std::string FormatKey(span<const Index> grid_indices) const final {
      return zarr3_sharding_indexed::IndicesToKey(grid_indices);
    }

    bool ParseKey(std::string_view key, span<Index> grid_indices) const final {
      return zarr3_sharding_indexed::KeyToIndices(key, grid_indices);
    }

    Index MinGridIndexForLexicographicalOrder(
        DimensionIndex dim, IndexInterval grid_interval) const final {
      return 0;
    }

    // Reference to parent codec, to ensure that `this->sub_chunk_grid` remains
    // valid.
    internal::IntrusivePtr<const ZarrShardingCodec> parent_codec_;

    std::vector<Index> sub_chunk_grid_shape_;
    ZarrCodecChain::PreparedState::Ptr codec_state_;
    zarr3_sharding_indexed::ShardIndexParameters shard_index_params_;
  };

  Result<ZarrArrayToBytesCodec::PreparedState::Ptr> Prepare(
      span<const Index> decoded_shape) const final {
    span<const Index> sub_chunk_shape = sub_chunk_grid_.components[0].shape();
    if (decoded_shape.size() != sub_chunk_shape.size()) {
      return SubChunkRankMismatch(sub_chunk_shape, decoded_shape.size());
    }
    auto state = internal::MakeIntrusivePtr<State>();
    state->parent_codec_.reset(this);
    auto& sub_chunk_grid_shape = state->sub_chunk_grid_shape_;
    sub_chunk_grid_shape.resize(decoded_shape.size());
    for (DimensionIndex i = 0; i < sub_chunk_shape.size(); ++i) {
      if (decoded_shape[i] % sub_chunk_shape[i] != 0) {
        return SubChunkShapeMismatch(sub_chunk_shape, decoded_shape);
      }
      const int64_t grid_size = decoded_shape[i] / sub_chunk_shape[i];
      sub_chunk_grid_shape[i] = grid_size;
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        state->codec_state_, sub_chunk_codec_chain_->Prepare(sub_chunk_shape));
    state->sub_chunk_grid = &sub_chunk_grid_;
    state->sub_chunk_codec_chain = sub_chunk_codec_chain_.get();
    state->sub_chunk_codec_state = state->codec_state_.get();
    state->shard_index_params_.index_location = index_location_;
    TENSORSTORE_RETURN_IF_ERROR(state->shard_index_params_.Initialize(
        *index_codec_chain_, sub_chunk_grid_shape));
    return {std::in_place, std::move(state)};
  }

  internal::ChunkGridSpecification sub_chunk_grid_;
  ZarrCodecChain::Ptr sub_chunk_codec_chain_;
  ZarrCodecChain::Ptr index_codec_chain_;
  ShardIndexLocation index_location_;
};
}  // namespace

absl::Status ShardingIndexedCodecSpec::MergeFrom(const ZarrCodecSpec& other,
                                                 bool strict) {
  using Self = ShardingIndexedCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  TENSORSTORE_RETURN_IF_ERROR(MergeConstraint<&Options::sub_chunk_shape>(
      "chunk_shape", options, other_options));
  TENSORSTORE_RETURN_IF_ERROR(
      internal_zarr3::MergeZarrCodecSpecs(options.index_codecs,
                                          other_options.index_codecs, strict),
      tensorstore::MaybeAnnotateStatus(_, "Incompatible \"index_codecs\""));
  TENSORSTORE_RETURN_IF_ERROR(
      internal_zarr3::MergeZarrCodecSpecs(
          options.sub_chunk_codecs, other_options.sub_chunk_codecs, strict),
      tensorstore::MaybeAnnotateStatus(_, "Incompatible sub-chunk \"codecs\""));
  TENSORSTORE_RETURN_IF_ERROR(MergeConstraint<&Options::index_location>(
      "index_location", options, other_options));
  return absl::OkStatus();
}

absl::Status ShardingIndexedCodecSpec::MergeSubChunkCodecsFrom(
    const ZarrCodecChainSpec& other, bool strict) {
  if (!options.sub_chunk_codecs) {
    options.sub_chunk_codecs = other;
    return absl::OkStatus();
  }
  return options.sub_chunk_codecs->MergeFrom(other, strict);
}

ZarrCodecSpec::Ptr ShardingIndexedCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<ShardingIndexedCodecSpec>(*this);
}

const ZarrCodecChainSpec* ShardingIndexedCodecSpec::GetSubChunkCodecs() const {
  return options.sub_chunk_codecs ? &*options.sub_chunk_codecs : nullptr;
}

absl::Status ShardingIndexedCodecSpec::GetDecodedChunkLayout(
    const ArrayDataTypeAndShapeInfo& array_info,
    ArrayCodecChunkLayoutInfo& decoded) const {
  ArrayDataTypeAndShapeInfo sub_chunk_info;
  if (options.sub_chunk_shape &&
      !RankConstraint::Implies(options.sub_chunk_shape->size(),
                               array_info.rank)) {
    return SubChunkRankMismatch(*options.sub_chunk_shape, array_info.rank);
  }
  sub_chunk_info.dtype = array_info.dtype;
  sub_chunk_info.rank = array_info.rank;
  if (options.sub_chunk_shape) {
    std::copy(options.sub_chunk_shape->begin(), options.sub_chunk_shape->end(),
              sub_chunk_info.shape.emplace().begin());
  }
  if (options.sub_chunk_codecs) {
    TENSORSTORE_RETURN_IF_ERROR(options.sub_chunk_codecs->GetDecodedChunkLayout(
        sub_chunk_info, decoded));
  }
  return absl::OkStatus();
}

namespace {
ZarrCodecChainSpec DefaultIndexCodecChainSpec() {
  ZarrCodecChainSpec codecs;
  codecs.array_to_bytes = DefaultBytesCodec();
  codecs.bytes_to_bytes.push_back(
      internal::MakeIntrusivePtr<const Crc32cCodecSpec>());
  return codecs;
}
}  // namespace

Result<ZarrArrayToBytesCodec::Ptr> ShardingIndexedCodecSpec::Resolve(
    ArrayCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrArrayToBytesCodecSpec::Ptr* resolved_spec) const {
  ShardingIndexedCodecSpec::Options* resolved_options = nullptr;
  if (resolved_spec) {
    auto* resolved_spec_ptr = new ShardingIndexedCodecSpec;
    resolved_options = &resolved_spec_ptr->options;
    resolved_spec->reset(resolved_spec_ptr);
  }
  span<const Index> sub_chunk_shape;
  if (options.sub_chunk_shape) {
    sub_chunk_shape = *options.sub_chunk_shape;
  } else if (decoded.read_chunk_shape) {
    sub_chunk_shape =
        span<const Index>(decoded.read_chunk_shape->data(), decoded.rank);
  } else {
    return absl::InvalidArgumentError("\"chunk_shape\" must be specified");
  }
  if (sub_chunk_shape.size() != decoded.rank) {
    return SubChunkRankMismatch(sub_chunk_shape, decoded.rank);
  }
  internal::ChunkGridSpecification::ComponentList components;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto broadcast_fill_value,
      BroadcastArray(decoded.fill_value, sub_chunk_shape));
  components.emplace_back(std::move(broadcast_fill_value),
                          Box<>(sub_chunk_shape.size()));
  components.back().fill_value_comparison_kind =
      EqualityComparisonKind::identical;
  auto codec = internal::MakeIntrusivePtr<ShardingIndexedCodec>(
      internal::ChunkGridSpecification(std::move(components)));
  codec->index_location_ =
      options.index_location.value_or(ShardIndexLocation::kEnd);
  if (resolved_options) {
    resolved_options->sub_chunk_shape = codec->sub_chunk_grid_.chunk_shape;
    resolved_options->index_location = codec->index_location_;
  }
  auto set_up_codecs =
      [&](const ZarrCodecChainSpec& sub_chunk_codecs) -> absl::Status {
    ArrayCodecResolveParameters sub_chunk_decoded;
    sub_chunk_decoded.dtype = decoded.dtype;
    sub_chunk_decoded.rank = decoded.rank;
    sub_chunk_decoded.fill_value = std::move(decoded.fill_value);
    if (decoded.read_chunk_shape) {
      std::copy_n(decoded.read_chunk_shape->begin(), decoded.rank,
                  sub_chunk_decoded.read_chunk_shape.emplace().begin());
    }
    if (decoded.codec_chunk_shape) {
      std::copy_n(decoded.codec_chunk_shape->begin(), decoded.rank,
                  sub_chunk_decoded.codec_chunk_shape.emplace().begin());
    }
    if (decoded.inner_order) {
      std::copy_n(decoded.inner_order->begin(), decoded.rank,
                  sub_chunk_decoded.inner_order.emplace().begin());
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        codec->sub_chunk_codec_chain_,
        sub_chunk_codecs.Resolve(
            std::move(sub_chunk_decoded), encoded,
            resolved_options ? &resolved_options->sub_chunk_codecs.emplace()
                             : nullptr));
    return absl::OkStatus();
  };
  TENSORSTORE_RETURN_IF_ERROR(
      set_up_codecs(options.sub_chunk_codecs ? *options.sub_chunk_codecs
                                             : ZarrCodecChainSpec{}),
      tensorstore::MaybeAnnotateStatus(_, "Error resolving sub-chunk codecs"));

  auto set_up_index_codecs =
      [&](const ZarrCodecChainSpec& index_codecs) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        codec->index_codec_chain_,
        zarr3_sharding_indexed::InitializeIndexCodecChain(
            index_codecs, sub_chunk_shape.size(),
            resolved_options ? &resolved_options->index_codecs.emplace()
                             : nullptr));
    return absl::OkStatus();
  };
  TENSORSTORE_RETURN_IF_ERROR(
      set_up_index_codecs(options.index_codecs ? *options.index_codecs
                                               : DefaultIndexCodecChainSpec()),
      tensorstore::MaybeAnnotateStatus(_, "Error resolving index_codecs"));
  return {std::in_place, std::move(codec)};
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = ShardingIndexedCodecSpec;
  using Options = Self::Options;
  namespace jb = ::tensorstore::internal_json_binding;
  RegisterCodec<Self>(
      "sharding_indexed",
      jb::Projection<&Self::options>(jb::Sequence(
          jb::Member("chunk_shape", jb::Projection<&Options::sub_chunk_shape>(
                                        OptionalIfConstraintsBinder(
                                            jb::Array(jb::Integer<Index>(1))))),
          jb::Member("index_codecs", jb::Projection<&Options::index_codecs>(
                                         OptionalIfConstraintsBinder())),
          jb::Member("codecs", jb::Projection<&Options::sub_chunk_codecs>(
                                   OptionalIfConstraintsBinder())),
          jb::Member(
              "index_location",
              jb::Projection<&Options::index_location>(
                  [](auto is_loading, const auto& options, auto* obj, auto* j) {
                    // For compatibility with implementations that don't support
                    // `index_location`, don't include it in the stored
                    // representation when the value is equal to the default of
                    // "end".
                    if constexpr (!is_loading) {
                      if (!options.constraints &&
                          *obj == ShardIndexLocation::kEnd) {
                        return absl::OkStatus();
                      }
                    }
                    return jb::Validate([](const auto& options, auto* obj) {
                      if (!options.constraints) {
                        if (!obj->has_value()) *obj = ShardIndexLocation::kEnd;
                      }
                      return absl::OkStatus();
                    })(is_loading, options, obj, j);
                  }))
          //
          )));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
