// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/driver/tiff/metadata.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json_binding/json_binding.h"  // For AnyCodecSpec
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// ResolveMetadata function implementation is moved to tiff_driver.cc

namespace tensorstore {
namespace internal_tiff {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal_tiff_kvstore::CompressionType;
using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::PlanarConfigType;
using ::tensorstore::internal_tiff_kvstore::TiffParseResult;

// Anonymous namespace for helper functions used only by
// CreateMetadataFromParseResult
namespace {
// Maps TIFF SampleFormat and BitsPerSample to TensorStore DataType.
Result<DataType> GetDataTypeFromTiff(const ImageDirectory& dir) {
  if (dir.samples_per_pixel == 0 || dir.bits_per_sample.empty() ||
      dir.sample_format.empty()) {
    return absl::FailedPreconditionError(
        "Incomplete TIFF metadata for data type");
  }
  // Assume uniform bits/format per sample for simplicity in this scaffold.
  uint16_t bits = dir.bits_per_sample[0];
  uint16_t format = dir.sample_format[0];

  // Check consistency if multiple samples exist
  for (size_t i = 1; i < dir.samples_per_pixel; ++i) {
    if (i >= dir.bits_per_sample.size() || i >= dir.sample_format.size() ||
        dir.bits_per_sample[i] != bits || dir.sample_format[i] != format) {
      return absl::UnimplementedError(
          "Varying bits_per_sample or sample_format per channel not yet "
          "supported");
    }
  }

  switch (format) {
    case static_cast<uint16_t>(
        internal_tiff_kvstore::SampleFormatType::kUnsignedInteger):
      if (bits == 8) return dtype_v<uint8_t>;
      if (bits == 16) return dtype_v<uint16_t>;
      if (bits == 32) return dtype_v<uint32_t>;
      if (bits == 64) return dtype_v<uint64_t>;
      break;
    case static_cast<uint16_t>(
        internal_tiff_kvstore::SampleFormatType::kSignedInteger):
      if (bits == 8) return dtype_v<int8_t>;
      if (bits == 16) return dtype_v<int16_t>;
      if (bits == 32) return dtype_v<int32_t>;
      if (bits == 64) return dtype_v<int64_t>;
      break;
    case static_cast<uint16_t>(
        internal_tiff_kvstore::SampleFormatType::kIEEEFloat):
      if (bits == 32) return dtype_v<tensorstore::dtypes::float32_t>;
      if (bits == 64) return dtype_v<tensorstore::dtypes::float64_t>;
      break;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      StrCat("Unsupported TIFF data type: bits=", bits, ", format=", format));
}

// Gets the rank based on the ImageDirectory and PlanarConfiguration.
DimensionIndex GetRankFromTiff(const ImageDirectory& dir) {
  // Only support chunky for now
  if (dir.planar_config != static_cast<uint16_t>(PlanarConfigType::kChunky)) {
    return dynamic_rank;  // Indicate error or inability to determine
  }
  return (dir.samples_per_pixel > 1) ? 3 : 2;  // Y, X, [C]
}

// Gets the shape based on the ImageDirectory and PlanarConfiguration.
Result<std::vector<Index>> GetShapeFromTiff(const ImageDirectory& dir) {
  if (dir.planar_config != static_cast<uint16_t>(PlanarConfigType::kChunky)) {
    return absl::InternalError(
        "GetShapeFromTiff called with unsupported planar config");
  }
  std::vector<Index> shape;
  shape = {dir.height, dir.width};  // Y, X
  if (dir.samples_per_pixel > 1) {
    shape.push_back(static_cast<Index>(dir.samples_per_pixel));  // C
  }
  return shape;
}

// Gets chunk shape based on ImageDirectory and PlanarConfiguration.
Result<std::vector<Index>> GetChunkShapeFromTiff(const ImageDirectory& dir) {
  if (dir.planar_config != static_cast<uint16_t>(PlanarConfigType::kChunky)) {
    return absl::InternalError(
        "GetChunkShapeFromTiff called with unsupported planar config");
  }
  std::vector<Index> chunk_shape;
  Index tile_h = dir.tile_height > 0 ? dir.tile_height : dir.rows_per_strip;
  Index tile_w = dir.tile_width > 0 ? dir.tile_width : dir.width;

  chunk_shape = {tile_h, tile_w};  // Y, X
  if (dir.samples_per_pixel > 1) {
    chunk_shape.push_back(static_cast<Index>(dir.samples_per_pixel));  // C
  }
  return chunk_shape;
}

// Gets inner order based on ImageDirectory and PlanarConfiguration. (Fastest
// varying last)
Result<std::vector<DimensionIndex>> GetInnerOrderFromTiff(
    const ImageDirectory& dir) {
  if (dir.planar_config != static_cast<uint16_t>(PlanarConfigType::kChunky)) {
    return absl::InternalError(
        "GetInnerOrderFromTiff called with unsupported planar config");
  }
  DimensionIndex rank = GetRankFromTiff(dir);
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError(
        "Could not determine rank for inner order");
  }
  std::vector<DimensionIndex> inner_order(rank);
  if (rank == 3) {            // Y, X, C
    inner_order = {2, 1, 0};  // C faster than X faster than Y
  } else {                    // Y, X
    inner_order = {1, 0};     // X faster than Y
  }
  return inner_order;
}
}  // namespace

Result<std::shared_ptr<TiffMetadata>> CreateMetadataFromParseResult(
    const TiffParseResult& parse_result, uint32_t ifd_index) {
  auto metadata = std::make_shared<TiffMetadata>();
  metadata->ifd_index = ifd_index;
  metadata->num_ifds = 1;  // Default for single IFD interpretation

  // 1. Select and Validate IFD
  if (ifd_index >= parse_result.image_directories.size()) {
    return absl::NotFoundError(tensorstore::StrCat(
        "Requested IFD index ", ifd_index, " not found in TIFF file (found ",
        parse_result.image_directories.size(), " IFDs)"));
  }
  const ImageDirectory& img_dir = parse_result.image_directories[ifd_index];

  // 2. Validate Planar Configuration and Compression.
  uint16_t raw_planar_config = img_dir.planar_config;
  if (raw_planar_config != static_cast<uint16_t>(PlanarConfigType::kChunky)) {
    return absl::UnimplementedError(
        tensorstore::StrCat("PlanarConfiguration=", raw_planar_config,
                            " is not supported yet (only Chunky=1)"));
  }
  metadata->planar_config = PlanarConfigType::kChunky;

  uint16_t raw_compression = img_dir.compression;
  if (raw_compression != static_cast<uint16_t>(CompressionType::kNone)) {
    return absl::UnimplementedError(
        tensorstore::StrCat("Compression type ", raw_compression,
                            " is not supported yet (only None=1)"));
  }
  metadata->compression_type = CompressionType::kNone;

  // 3. Determine Core Properties from ImageDirectory
  metadata->rank = GetRankFromTiff(img_dir);
  if (metadata->rank == dynamic_rank) {
    return absl::InternalError("Failed to determine rank");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(metadata->shape, GetShapeFromTiff(img_dir));
  TENSORSTORE_ASSIGN_OR_RETURN(metadata->dtype, GetDataTypeFromTiff(img_dir));
  metadata->samples_per_pixel = img_dir.samples_per_pixel;

  // 4. Determine Basic Chunk Layout
  {
    ChunkLayout& layout = metadata->chunk_layout;
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(RankConstraint{metadata->rank}));
    TENSORSTORE_ASSIGN_OR_RETURN(std::vector<Index> chunk_shape,
                                 GetChunkShapeFromTiff(img_dir));
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::ChunkShape(chunk_shape)));
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(
        ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(metadata->rank))));
    TENSORSTORE_ASSIGN_OR_RETURN(std::vector<DimensionIndex> inner_order,
                                 GetInnerOrderFromTiff(img_dir));
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::InnerOrder(inner_order)));
    // Don't finalize yet, schema constraints will be merged later
  }

  // 5. Initialize Codec Spec (Default)
  // The actual compression type is stored directly in
  // metadata->compression_type. The CodecSpec will be populated/validated later
  // during ResolveMetadata when merging with schema constraints. For now,
  // initialize as default.
  metadata->codec_spec = CodecSpec();

  // 6. Initialize other fields to default
  metadata->dimension_units.resize(metadata->rank);  // Unknown units
  // Fill value will be determined later based on schema

  // 7. OME-XML / User Interpretation Hooks (Future)
  // TODO: Parse OME-XML here if present in ImageDescription tag.
  // TODO: Apply user interpretation flags here if they affect basic properties.

  return metadata;  // Return the partially filled metadata object
}

absl::Status ValidateMetadataSchema(const TiffMetadata& metadata,
                                    const Schema& schema) {
  // Rank
  if (!RankConstraint::EqualOrUnspecified(metadata.rank, schema.rank())) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Rank specified by schema (", schema.rank(),
                            ") does not match rank of resolved TIFF metadata (",
                            metadata.rank, ")"));
  }

  // Domain
  if (schema.domain().valid()) {
    IndexDomainBuilder builder(metadata.rank);
    builder.shape(metadata.shape);
    // TODO: Add labels if supported
    builder.implicit_upper_bounds(
        true);  // Assuming TIFF dims are typically resizable
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
    TENSORSTORE_RETURN_IF_ERROR(
        MergeIndexDomains(schema.domain(), domain_from_metadata),
        MaybeAnnotateStatus(
            _, "Mismatch between schema domain and resolved TIFF dimensions"));
  }

  // Data Type
  if (!IsPossiblySameDataType(metadata.dtype, schema.dtype())) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "dtype from resolved TIFF metadata (", metadata.dtype,
        ") does not match dtype in schema (", schema.dtype(), ")"));
  }

  // Chunk Layout
  // The compatibility check is implicitly handled when merging schema
  // constraints into the layout during the ResolveMetadata step (in
  // driver.cc).

  // Codec
  // Compatibility was checked during ResolveMetadata when merging schema
  // constraints.

  // Fill Value
  // Compatibility was checked during ResolveMetadata when setting the fill
  // value. Remove the incorrect ValidateFillValue call.

  // Dimension Units
  if (schema.dimension_units().valid()) {
    // Validate that the schema dimension units are compatible with the resolved
    // one.
    DimensionUnitsVector merged_units = metadata.dimension_units;
    TENSORSTORE_RETURN_IF_ERROR(
        MergeDimensionUnits(merged_units, schema.dimension_units()),
        internal::ConvertInvalidArgumentToFailedPrecondition(
            MaybeAnnotateStatus(_,
                                "dimension_units from schema are incompatible "
                                "with resolved TIFF metadata")));
    // Check if merging resulted in changes (indicates incompatibility if strict
    // matching needed) if (merged_units != metadata.dimension_units) { ...
    // return error ... }
  }

  return absl::OkStatus();
}

}  // namespace internal_tiff
}  // namespace tensorstore
