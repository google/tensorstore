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
#include "tensorstore/driver/tiff/compressor.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_tiff {

/// Options specified in the `TiffDriverSpec` that guide interpretation.
struct TiffSpecOptions {
  // Specifies which IFD (Image File Directory) to open. Defaults to 0.
  uint32_t ifd_index = 0;

  // --- Future extensions ---
  // enum class IfdHandling { kSingle, kStackZ } ifd_handling =
  // IfdHandling::kSingle; bool use_ome_metadata = true; // Default to using OME
  // if present?

  // --- JSON Binding ---
  // Make options configurable via JSON in the driver spec.
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TiffSpecOptions,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

/// Represents the resolved and interpreted metadata for a TIFF TensorStore.
/// This structure holds the information needed by the driver after parsing
/// TIFF tags, potentially OME-XML, and applying user specifications.
struct TiffMetadata {
  // Which IFD this metadata corresponds to.
  uint32_t ifd_index;

  // Number of IFDs represented (1 for single IFD mode, >1 for stacked mode).
  uint32_t num_ifds = 1;

  // Core TensorStore Schema components
  DimensionIndex rank = dynamic_rank;

  // Derived shape (e.g. [C,Y,X] or [Y,X,C] or [Y,X], ...)
  std::vector<Index> shape;

  DataType dtype;

  // Derived chunk layout including order.
  ChunkLayout chunk_layout;

  // Represents compression
  Compressor compressor;

  // From user spec or default
  SharedArray<const void> fill_value;

  // Derived from TIFF/OME/user spec
  DimensionUnitsVector dimension_units;

  // Information retained from TIFF for reference/logic
  internal_tiff_kvstore::Endian endian;
  internal_tiff_kvstore::CompressionType compression_type;
  internal_tiff_kvstore::PlanarConfigType planar_config;
  uint16_t samples_per_pixel;

  // Pre-calculated layout order enum (C or Fortran) based on finalized
  // chunk_layout.inner_order
  ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c;

  // Returns `true` if a byte‑swap is required on this platform.
  bool NeedByteSwap() const {
    constexpr bool kHostIsBig =
        (tensorstore::endian::native == tensorstore::endian::big);

    return (endian == internal_tiff_kvstore::Endian::kBig) ^ kHostIsBig;
  }

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

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TiffMetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

// Represents the codec specification specifically for the TIFF driver.
// It primarily stores the compression type used.
class TiffCodecSpec : public internal::CodecDriverSpec {
 public:
  // Unique identifier for the TIFF codec driver spec.
  constexpr static char id[] = "tiff";

  // Specifies the compression type, if constrained by the spec.
  // If std::nullopt, the compression type is unconstrained by this spec.
  std::optional<internal_tiff_kvstore::CompressionType> compression_type;

  // Virtual method overrides from CodecDriverSpec
  CodecSpec Clone() const override;
  absl::Status DoMergeFrom(
      const internal::CodecDriverSpec& other_base) override;

  // JSON Binding support
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TiffCodecSpec, FromJsonOptions,
                                          ToJsonOptions,
                                          ::nlohmann::json::object_t)

  friend bool operator==(const TiffCodecSpec& a, const TiffCodecSpec& b);
};

inline bool operator!=(const TiffCodecSpec& a, const TiffCodecSpec& b) {
  return !(a == b);
}

/// Resolves the final metadata by interpreting parsed TIFF data according
/// to spec options and merging with schema constraints.
///
/// \param source The parsed TIFF directory structure.
/// \param options User-specified interpretation options from the driver spec.
/// \param schema General TensorStore schema constraints.
/// \returns The final, resolved metadata for the driver.
Result<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
    const internal_tiff_kvstore::TiffParseResult& source,
    const TiffSpecOptions& options, const Schema& schema);

/// Validates the final resolved metadata against explicit user constraints
/// provided in the driver spec.
///
/// \param resolved_metadata The final metadata produced by `ResolveMetadata`.
/// \param user_constraints Constraints provided by the user in the spec.
/// \error `absl::StatusCode::kFailedPrecondition` if constraints are violated.
absl::Status ValidateResolvedMetadata(
    const TiffMetadata& resolved_metadata,
    const TiffMetadataConstraints& user_constraints);

/// Computes the effective domain based on spec options, constraints, and
/// schema. If the rank or shape cannot be determined from the inputs, returns
/// an unknown domain.
///
/// \param options TIFF-specific interpretation options (currently unused here).
/// \param constraints User constraints on the final metadata (e.g., shape).
/// \param schema General schema constraints (e.g., domain, rank).
/// \returns The best estimate of the domain based on the spec, or an error if
///     constraints conflict.
Result<IndexDomain<>> GetEffectiveDomain(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema);

/// Computes the effective chunk layout based on spec options, constraints, and
/// schema.
///
/// \param options TIFF-specific interpretation options (currently unused here).
/// \param constraints User constraints on the final metadata (e.g.,
/// chunk_shape).
/// \param schema General schema constraints (e.g., chunk layout).
/// \returns The best estimate of the chunk layout based on the spec, or an
/// error if constraints conflict. Returns a default layout if rank is unknown.
Result<ChunkLayout> GetEffectiveChunkLayout(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema);

/// Computes the effective codec spec based on spec options, constraints, and
/// schema.
///
/// Returns a default TIFF codec (uncompressed) if no constraints are provided.
///
/// \param options TIFF-specific interpretation options (currently unused here).
/// \param constraints User constraints on the final metadata (e.g.,
/// compression).
/// \param schema General schema constraints (e.g., codec spec).
/// \returns The best estimate of the codec spec based on the spec, or an error
///     if constraints conflict.
Result<internal::CodecDriverSpec::PtrT<TiffCodecSpec>> GetEffectiveCodec(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema);

/// Computes the effective dimension units based on spec options, constraints,
/// and schema.
///
/// \param options TIFF-specific interpretation options (currently unused here).
/// \param constraints User constraints on the final metadata (e.g., units).
/// \param schema General schema constraints (e.g., dimension_units).
/// \returns The best estimate of the dimension units based on the spec, or an
///     error if constraints conflict. Returns unknown units if rank is unknown
///     or units are unspecified.
Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema);

/// Computes the effective data type based on constraints and schema.
///
/// \param constraints User constraints on the final metadata (e.g., dtype).
/// \param schema General schema constraints (e.g., dtype).
/// \returns The effective data type. Returns `DataType()` (invalid) if neither
///     input specifies a data type. Returns an error if constraints conflict.
Result<DataType> GetEffectiveDataType(
    const TiffMetadataConstraints& constraints, const Schema& schema);

/// Decodes a raw (potentially compressed) chunk buffer based on TIFF metadata.
///
/// \param metadata The resolved metadata for the TIFF dataset.
/// \param buffer The raw Cord containing the bytes for a single tile/strip.
/// \returns The decoded chunk as a SharedArray, or an error.
Result<SharedArray<const void>> DecodeChunk(const TiffMetadata& metadata,
                                            absl::Cord buffer);

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff::TiffMetadataConstraints)

#endif  // TENSORSTORE_DRIVER_TIFF_METADATA_H_
