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

#ifndef TENSORSTORE_DRIVER_TIFF_METADATA_H_
#define TENSORSTORE_DRIVER_TIFF_METADATA_H_

#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"  // Needed for ValidateMetadataSchema declaration
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_tiff {

/// Represents the resolved and interpreted metadata for a TIFF TensorStore.
/// This structure holds the information needed by the driver after parsing
/// TIFF tags, potentially OME-XML, and applying user specifications.
struct TiffMetadata {
  // Which IFD this metadata corresponds to.
  uint32_t ifd_index;

  // Number of IFDs represented (1 for single IFD mode, >1 for stacked mode).
  uint32_t num_ifds = 1;

  // Core TensorStore Schema components
  /// Length of `shape`, `axes` and `chunk_shape` if any are specified.  If none
  /// are specified, equal to `dynamic_rank`.
  DimensionIndex rank;

  // Derived shape (e.g. [C,Y,X] or [Y,X,C] or [Y,X], ...)
  std::vector<Index> shape;

  DataType dtype;
  // Derived chunk layout including order.
  ChunkLayout chunk_layout;

  // Represents compression
  CodecSpec codec_spec;

  // From user spec or default
  SharedArray<const void> fill_value;

  // Derived from TIFF/OME/user spec
  DimensionUnitsVector dimension_units;

  // Information retained from TIFF for reference/logic
  internal_tiff_kvstore::CompressionType compression_type;
  internal_tiff_kvstore::PlanarConfigType planar_config;
  uint16_t samples_per_pixel;

  // TODO: Add fields for parsed OME-XML metadata if needed in the future.
  // std::shared_ptr<OmeXmlStruct> ome_metadata;

  // TODO: Add fields representing user overrides/interpretations if needed.
  // e.g., bool ifd_is_z_dimension;
  TiffMetadata() = default;
};

/// Specifies constraints on the TIFF metadata required when opening.
struct TiffMetadataConstraints {
  std::optional<DataType> dtype;
  std::optional<std::vector<Index>> shape;
  DimensionIndex rank = dynamic_rank;  // Track rank from constraints
  std::vector<std::string> axes;
  std::vector<Index> chunk_shape;

  // Specifies which IFD (Image File Directory) to open. Defaults to 0.
  uint32_t ifd_index = 0;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TiffMetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

/// Creates a basic `TiffMetadata` object by interpreting a single IFD
/// from the parsed TIFF structure. Performs initial checks for unsupported
/// features based solely on the TIFF tags.
///
/// \param parse_result The result of parsing the TIFF structure via
/// TiffDirectoryCache.
/// \param ifd_index The specific IFD to interpret.
/// \returns A shared pointer to the basic metadata object.
/// \error `absl::StatusCode::kNotFound` if `ifd_index` is invalid.
/// \error `absl::StatusCode::kUnimplemented` if unsupported features are
/// detected.
/// \error `absl::StatusCode::kInvalidArgument` if required tags are missing or
//      inconsistent within the IFD.
Result<std::shared_ptr<TiffMetadata>> CreateMetadataFromParseResult(
    const internal_tiff_kvstore::TiffParseResult& parse_result,
    uint32_t ifd_index);

/// Validates that the resolved `TiffMetadata` is compatible with Schema
/// constraints.
/// This is typically called after the final metadata object is resolved.
///
/// \param metadata The resolved TIFF metadata.
/// \param schema The schema constraints to validate against.
/// \error `absl::StatusCode::kFailedPrecondition` if constraints are violated.
absl::Status ValidateMetadataSchema(const TiffMetadata& metadata,
                                    const Schema& schema);

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff::TiffMetadataConstraints)

#endif  // TENSORSTORE_DRIVER_TIFF_METADATA_H_
