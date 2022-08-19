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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_METADATA_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_METADATA_H_

/// \file
/// Metadata handling for the Neuroglancer precomputed format.
///
/// Refer to the specification here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

constexpr inline const char kAtSignTypeId[] = "@type";
constexpr inline const char kChunkSizeId[] = "chunk_size";
constexpr inline const char kChunkSizesId[] = "chunk_sizes";
constexpr inline const char kCompressedSegmentationBlockSizeId[] =
    "compressed_segmentation_block_size";
constexpr inline const char kDataTypeId[] = "data_type";
constexpr inline const char kDriverId[] = "neuroglancer_precomputed";
constexpr inline const char kEncodingId[] = "encoding";
constexpr inline const char kJpegQualityId[] = "jpeg_quality";
constexpr inline const char kKeyId[] = "key";
constexpr inline const char kMetadataKey[] = "info";
constexpr inline const char kMultiscaleMetadataId[] = "multiscale_metadata";
constexpr inline const char kNumChannelsId[] = "num_channels";
constexpr inline const char kPathId[] = "path";
constexpr inline const char kResolutionId[] = "resolution";
constexpr inline const char kScaleIndexId[] = "scale_index";
constexpr inline const char kScaleMetadataId[] = "scale_metadata";
constexpr inline const char kScalesId[] = "scales";
constexpr inline const char kShardingId[] = "sharding";
constexpr inline const char kSizeId[] = "size";
constexpr inline const char kTypeId[] = "type";
constexpr inline const char kVoxelOffsetId[] = "voxel_offset";

/// Don't change this, since it would affect the behavior when jpeg_quality
/// isn't specified explicitly.
constexpr inline int kDefaultJpegQuality = 75;

using ShardingSpec = ::tensorstore::neuroglancer_uint64_sharded::ShardingSpec;

/// Tag type that specifies an unsharded volume.
struct NoShardingSpec {
  friend bool operator==(NoShardingSpec, NoShardingSpec) { return true; }
  friend bool operator!=(NoShardingSpec, NoShardingSpec) { return false; }

  /// Converts to JSON.  Note that the value of `null` to indicate no sharding
  /// is only used within the TensorStore Spec, not in the actual stored
  /// metadata.
  friend void to_json(::nlohmann::json& out,  // NOLINT
                      const NoShardingSpec& s) {
    out = nullptr;
  }
  friend void to_json(::nlohmann::json& out,  // NOLINT
                      const std::variant<NoShardingSpec, ShardingSpec>& s);
};

/// Parsed representation of a single entry in the "scales" array of a
/// multiscale volume.
struct ScaleMetadata {
  enum class Encoding {
    raw,
    jpeg,
    compressed_segmentation,
  };

  friend std::string_view to_string(Encoding e);
  friend std::ostream& operator<<(std::ostream& os, Encoding e) {
    return os << std::string(to_string(e));
  }

  friend void to_json(::nlohmann::json& out,  // NOLINT
                      Encoding e) {
    out = std::string(to_string(e));
  }

  /// Equal to `"key"` member of JSON metadata.
  std::string key;
  /// Bounds in xyz order.
  Box<3> box;
  std::vector<std::array<Index, 3>> chunk_sizes;
  Encoding encoding;
  int jpeg_quality = kDefaultJpegQuality;
  std::array<Index, 3> compressed_segmentation_block_size{};
  std::variant<NoShardingSpec, ShardingSpec> sharding;
  std::array<double, 3> resolution;

  /// Additional members excluding those listed above.  These are preserved when
  /// re-writing the metadata.
  ::nlohmann::json::object_t extra_attributes;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ScaleMetadata,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults)
};

/// Parsed representation of the multiscale volume `info` metadata file.
struct MultiscaleMetadata {
  std::string type;
  DataType dtype;
  Index num_channels;
  std::vector<ScaleMetadata> scales;

  /// Extra JSON members (excluding the parsed members above).  These are
  /// preserved when re-writing the metadata.
  ::nlohmann::json::object_t extra_attributes;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(MultiscaleMetadata,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults)
};

/// Specifies constraints on the non-scale-specific metadata for
/// opening/creating a multiscale volume.
struct MultiscaleMetadataConstraints {
  std::optional<std::string> type;
  DataType dtype;
  std::optional<Index> num_channels;

  /// Additional JSON members required in the multiscale metadata.
  ::nlohmann::json::object_t extra_attributes;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(MultiscaleMetadataConstraints,
                                          JsonSerializationOptions,
                                          JsonSerializationOptions)
};

/// Specifies constraints on the per-scale metadata for opening/creating a
/// multiscale volume.
struct ScaleMetadataConstraints {
  std::optional<std::string> key;
  std::optional<Box<3>> box;
  std::optional<std::array<Index, 3>> chunk_size;
  std::optional<std::array<double, 3>> resolution;
  std::optional<ScaleMetadata::Encoding> encoding;
  std::optional<int> jpeg_quality;
  std::optional<std::array<Index, 3>> compressed_segmentation_block_size;
  std::optional<std::variant<NoShardingSpec, ShardingSpec>> sharding;
  /// Additional JSON members required in the scale metadata.
  ::nlohmann::json::object_t extra_attributes;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ScaleMetadataConstraints,
                                          JsonSerializationOptions,
                                          JsonSerializationOptions)
};

/// Specifies constraints for opening/creating a multiscale volume.
struct OpenConstraints {
  MultiscaleMetadataConstraints multiscale;
  ScaleMetadataConstraints scale;
  std::optional<std::size_t> scale_index;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OpenConstraints,
                                          JsonSerializationOptions,
                                          JsonSerializationOptions,
                                          ::nlohmann::json::object_t)
};

struct NeuroglancerPrecomputedCodecSpec
    : public tensorstore::internal::CodecDriverSpec {
 public:
  constexpr static char id[] = "neuroglancer_precomputed";
  std::optional<ScaleMetadata::Encoding> encoding;
  std::optional<int> jpeg_quality;
  std::optional<ShardingSpec::DataEncoding> shard_data_encoding;

  CodecSpec Clone() const final;
  absl::Status DoMergeFrom(const internal::CodecDriverSpec& other_base) final;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(NeuroglancerPrecomputedCodecSpec,
                                          FromJsonOptions, ToJsonOptions,
                                          ::nlohmann::json::object_t)
};

/// Returns the compatibility key.
///
/// The compatibility key encodes all parameters that affect the
/// encoding/decoding of chunks.
std::string GetMetadataCompatibilityKey(const MultiscaleMetadata& metadata,
                                        std::size_t scale_index,
                                        const std::array<Index, 3>& chunk_size);

/// Validates that scale `scale_index` of `existing_metadata` is compatible with
/// the same scale of `new_metadata`.
///
/// \pre `scale_index < existing_metadata.scales.size()`
/// \returns `absl::Status()` on success.
/// \error `absl::StatusCode::kFailedPrecondition` if
///     `scale_index >= new_metadata.scales.size()` or the scales are not
///     compatible.
absl::Status ValidateMetadataCompatibility(
    const MultiscaleMetadata& existing_metadata,
    const MultiscaleMetadata& new_metadata, std::size_t scale_index,
    const std::array<Index, 3>& chunk_size);

/// Attempts to create a new scale.
///
/// \returns The new metadata and the new scale index.
/// \error `absl::StatusCode::kAlreadyExists` if the scale already exists.
/// \error `absl::StatusCode::kFailedPrecondition` if `constraints` are not
///     satisfied.
/// \error `absl::StatusCode::kInvalidArgument` if `constraints` are not valid
///     for creating a new scale.
Result<std::pair<std::shared_ptr<MultiscaleMetadata>, std::size_t>> CreateScale(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints, const Schema& schema);

/// Validates that the specified scale is compatible with `schema`.
absl::Status ValidateMetadataSchema(const MultiscaleMetadata& metadata,
                                    size_t scale_index,
                                    span<const Index, 3> chunk_size_xyz,
                                    const Schema& schema);

CodecSpec GetCodecFromMetadata(const MultiscaleMetadata& metadata,
                               size_t scale_index);

/// Returns the combined domain from `existing_metadata`, `constraints` and
/// `schema`.
///
/// If the domain is unspecified, returns a null domain.
///
/// This function uses the same dimension order as the schema,
/// i.e. `{"x", "y", "z", "channel"}`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `constraints` is inconsistent
///     with `schema`.
Result<IndexDomain<>> GetEffectiveDomain(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints, const Schema& schema);

/// Returns the combined domain and chunk layout from `existing_metadata`,
/// `constraints` and `schema`.
///
/// This function uses the same dimension order as the schema,
/// i.e. `{"x", "y", "z", "channel"}`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `constraints` is inconsistent
///     with `schema`.
Result<std::pair<IndexDomain<>, ChunkLayout>> GetEffectiveDomainAndChunkLayout(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints, const Schema& schema);

/// Returns the combined codec spec from from `constraints` and `schema`.
///
/// \returns Non-null pointer.
/// \error `absl::StatusCode::kInvalidArgument` if `constraints` is inconsistent
///     with `schema`.
Result<internal::CodecDriverSpec::PtrT<NeuroglancerPrecomputedCodecSpec>>
GetEffectiveCodec(const OpenConstraints& constraints, const Schema& schema);

/// Returns the combined dimension units from `constraints` and `schema`.
///
/// This function uses the same dimension order as the schema,
/// i.e. `{"x", "y", "z", "channel"}`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `constraints` is inconsistent
///     with `schema`.
Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const OpenConstraints& constraints, const Schema& schema);

/// Attempts to open an existing scale.
///
/// \param The existing metadata.
/// \param constraints Constraints specifying the scale to open.
/// \param schema Schema constraints.
/// \returns The scale index that is compatible with `constraints`.
/// \error `absl::StatusCode::kNotFound` if no such scale is found.
/// \error `absl::StatusCode::kFailedPrecondition` if constraints are not
///     satisfied.
Result<std::size_t> OpenScale(const MultiscaleMetadata& metadata,
                              const OpenConstraints& constraints,
                              const Schema& schema);

/// Resolves `scale_key` relative to `key_prefix`.
///
/// Treats `scale_key` as a relative path, handling ".." if possible components.
///
/// \param key_prefix The key prefix.  The `info` file containing `scale_key` is
///     assumed to have a path of `key_prefix + "/info"`.
/// \param scale_key The scale `key` from the `info` file to resolve.
/// \returns The resolved path.
std::string ResolveScaleKey(std::string_view key_prefix,
                            std::string_view scale_key);

/// Validates that `dtype` is supported by the Neuroglancer precomputed
/// format.
///
/// \dchecks `dtype.valid()`
absl::Status ValidateDataType(DataType dtype);

/// Returns the number of bits used for each index in the compressed Z index
/// representation of chunk indices.
std::array<int, 3> GetCompressedZIndexBits(span<const Index, 3> shape,
                                           span<const Index, 3> chunk_size);

/// Returns the compressed z index of a chunk index vector.
///
/// \param bits The number of bits to use for each index.
/// \pre `indices[i] < 2**bits[i]` for `0 <= i < 3`
std::uint64_t EncodeCompressedZIndex(span<const Index, 3> indices,
                                     std::array<int, 3> bits);

struct ShardChunkHierarchy {
  /// Number of bits for each dimension in compressed Morton code.
  std::array<int, 3> z_index_bits;
  /// Shape of entire volume in base chunks, in xyz order.
  std::array<Index, 3> grid_shape_in_chunks;
  /// Shape of a minishard in base chunks, in xyz order.
  std::array<Index, 3> minishard_shape_in_chunks;
  /// Shape of a shard in base chunks, in xyz order.
  std::array<Index, 3> shard_shape_in_chunks;
  /// Number of initial bits that do not contribute to the shard number.
  int non_shard_bits;
  /// Number of bits after `non_shard_bits` that contribute to shard number.
  int shard_bits;
};

/// Computes the minishard and shard grids.
///
/// \param sharding_spec The sharding spec.
/// \param volume_shape The volume shape, xyz.
/// \param chunk_shape The chunk shape, xyz.
/// \returns `true` if the minishards and shards correspond to rectangular
///     regions, or `false` otherwise.
bool GetShardChunkHierarchy(const ShardingSpec& sharding_spec,
                            span<const Index, 3> volume_shape,
                            span<const Index, 3> chunk_shape,
                            ShardChunkHierarchy& hierarchy);

/// Returns a function that computes the number of chunks in a given shard of a
/// sharded volume.
///
/// This is used to optimize the writeback behavior of
/// `Uint64ShardedKeyValueStore`.
///
/// Returns a null function if shards do not correspond to rectangular regions
/// of the volume.
///
/// \param sharding_spec The sharding spec.
/// \param volume_shape The volume shape, xyz.
/// \param chunk_shape The chunk shape, xyz.
std::function<std::uint64_t(std::uint64_t shard)>
GetChunksPerVolumeShardFunction(const ShardingSpec& sharding_spec,
                                span<const Index, 3> volume_shape,
                                span<const Index, 3> chunk_shape);

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_neuroglancer_precomputed::OpenConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_neuroglancer_precomputed::OpenConstraints)

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_METADATA_H_
