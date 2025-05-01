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
  /// Options specific to multi-IFD stacking mode.
  struct IfdStackingOptions {
    // Specifies the labels for the dimensions represented by the IFD sequence.
    // Required if `ifd_stacking` is specified.
    std::vector<std::string> dimensions;

    // Explicitly defines the size of each corresponding dimension in
    // `dimensions`. Must have the same length as `dimensions`. Required if
    // `dimensions.size() > 1` and OME-XML is not used/found. Optional if
    // `dimensions.size() == 1` (can use `ifd_count` instead).
    std::optional<std::vector<Index>> dimension_sizes;

    // Specifies the total number of IFDs involved in the stack OR the size of
    // the single dimension if `dimensions.size() == 1` and `dimension_sizes`
    // is absent. If specified along with `dimension_sizes`, their product must
    // match `ifd_count`.
    std::optional<uint32_t> ifd_count;

    // Specifies the order of stacked dimensions within the flat IFD sequence.
    // Must be a permutation of `dimensions`. Defaults to the order in
    // `dimensions` with the last dimension varying fastest.
    std::optional<std::vector<std::string>> ifd_sequence_order;

    // Member binding for serialization/reflection (used internally)
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.dimensions, x.dimension_sizes, x.ifd_count,
               x.ifd_sequence_order);
    };
  };

  // Use EITHER ifd_index OR ifd_stacking. Default is single IFD mode
  // (ifd_index=0). The JSON binder will enforce mutual exclusion.

  // Option A: Single IFD Mode (default behavior if ifd_stacking is absent)
  // Specifies which IFD to open.
  uint32_t ifd_index = 0;

  // Option B: Multi-IFD Stacking Mode
  // Interprets a sequence of IFDs as additional TensorStore dimensions.
  std::optional<IfdStackingOptions> ifd_stacking;

  // Optional Sample Dimension Label
  // Specifies the conceptual label for the dimension derived from
  // SamplesPerPixel when SamplesPerPixel > 1. If omitted, a default ('c') is
  // used internally.
  std::optional<std::string> sample_dimension_label;

  // Future: OME-XML Control
  // bool use_ome_xml = true;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(TiffSpecOptions,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.ifd_index, x.ifd_stacking, x.sample_dimension_label);
  };
};

/// Stores information about the mapping between final TensorStore dimensions.
struct TiffDimensionMapping {
  /// TensorStore dimension index corresponding to logical height (Y).
  std::optional<DimensionIndex> ts_y_dim;
  /// TensorStore dimension index corresponding to logical width (X).
  std::optional<DimensionIndex> ts_x_dim;
  /// TensorStore dimension index corresponding to the sample dimension (if spp
  /// > 1).
  std::optional<DimensionIndex> ts_sample_dim;

  /// Maps stacked dimension labels (from ifd_stacking.dimensions) to their
  /// corresponding TensorStore dimension indices.
  std::map<std::string, DimensionIndex> ts_stacked_dims;

  /// Maps TensorStore dimension indices back to conceptual labels (e.g., "z",
  /// "t", "y", "x", "c") Useful for debugging or potentially reconstructing
  /// spec.
  std::vector<std::string> labels_by_ts_dim;
};

/// Represents the resolved and interpreted metadata for a TIFF TensorStore.
/// This structure holds the information needed by the driver after parsing
/// TIFF tags, potentially OME-XML, and applying user specifications.
struct TiffMetadata {
  // Which IFD was used as the base (0 unless single IFD mode requested specific
  // one).
  uint32_t base_ifd_index;

  // Number of IFDs used (1 for single IFD mode, >1 for stacked mode).
  uint32_t num_ifds_read = 1;  // Reflects IFDs actually parsed/validated

  // Parsed stacking options, if multi-IFD mode was used.
  std::optional<TiffSpecOptions::IfdStackingOptions> stacking_info;

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

  std::vector<std::string> dimension_labels;

  // Dimension mapping.
  TiffDimensionMapping dimension_mapping;

  // Information retained from TIFF for reference/logic
  internal_tiff_kvstore::Endian endian;
  internal_tiff_kvstore::CompressionType compression_type;
  internal_tiff_kvstore::PlanarConfigType planar_config;
  uint16_t samples_per_pixel;

  // Chunk sizes from base IFD.
  uint32_t ifd0_chunk_width;
  uint32_t ifd0_chunk_height;

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

/// Computes the effective compressor object by merging the compression type
/// derived from TIFF tags with constraints from the schema's CodecSpec.
///
/// \param compression_type The compression type read from the TIFF file's tags.
/// \param schema_codec The CodecSpec provided via the Schema object, which may
///     contain constraints or overrides.
/// \returns The resolved Compressor object (JsonSpecifiedCompressor::Ptr),
/// which
///     will be nullptr if the final resolved type is kNone (raw) or if an
///     unsupported/unregistered compressor type is specified.
/// \error `absl::StatusCode::kInvalidArgument` if `schema_codec` conflicts with
///     `compression_type`.
/// \error `absl::StatusCode::kUnimplemented` if the resolved compressor type
///     is not supported by the current build.
Result<Compressor> GetEffectiveCompressor(
    internal_tiff_kvstore::CompressionType compression_type,
    const CodecSpec& schema_codec);

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

/// Validates that `dtype` is supported by the TIFF driver.
///
/// Checks if the data type corresponds to a standard TIFF SampleFormat
/// and BitsPerSample combination (uint8/16/32/64, int8/16/32/64, float32/64).
absl::Status ValidateDataType(DataType dtype);

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffSpecOptions::IfdStackingOptions)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff::TiffSpecOptions::IfdStackingOptions)

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffSpecOptions)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff::TiffSpecOptions)

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff::TiffMetadataConstraints)

#endif  // TENSORSTORE_DRIVER_TIFF_METADATA_H_
