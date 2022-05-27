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

#ifndef TENSORSTORE_DRIVER_N5_METADATA_H_
#define TENSORSTORE_DRIVER_N5_METADATA_H_

#include <string>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_n5 {

/// Decoded representation of N5 metadata.
///
/// Per the specification:
/// https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
///
///   A dataset is a group with the mandatory attributes:
///
///    * dimensions (e.g. [100, 200, 300]),
///
///    * blockSize (e.g. [64, 64, 64]),
///
///    * dataType (one of {uint8, uint16, uint32, uint64, int8, int16, int32,
///      int64, float32, float64})
///
///    * compression as a struct with the mandatory attribute type that
///      specifies the compression scheme, currently available are:
///
///      * raw (no parameters),
///      * bzip2 with parameters
///        * blockSize ([1-9], default 9)
///      * gzip with parameters
///        * level (integer, default -1)
///      * lz4 with parameters
///        * blockSize (integer, default 65536)
///      * xz with parameters
///        * preset (integer, default 6).
class N5Metadata {
 public:
  // The following members are common to `N5Metadata` and
  // `N5MetadataConstraints`, except that in `N5MetadataConstraints` some are
  // `std::optional`-wrapped.

  /// Length of `shape`, `axes` and `chunk_shape`.
  DimensionIndex rank = dynamic_rank;

  /// Specifies the current shape of the full volume.
  std::vector<Index> shape;

  /// Specifies the dimension labels.
  std::vector<std::string> axes;

  struct UnitsAndResolution {
    /// Specifies the base unit for each dimension.
    std::optional<std::vector<std::string>> units;

    /// Specifies the resolution (i.e. multiplier for the base unit) for each
    /// dimension.
    std::optional<std::vector<double>> resolution;
  };

  UnitsAndResolution units_and_resolution;

  /// Specifies the chunk size (corresponding to the `"blockSize"` attribute)
  /// and the in-memory layout of a full chunk (always C order).
  std::vector<Index> chunk_shape;

  Compressor compressor;
  DataType dtype;

  /// Contains all additional attributes, excluding attributes parsed into the
  /// data members above.
  ::nlohmann::json::object_t extra_attributes;

  // Derived members computed from `chunk_shape` and `dtype`:

  StridedLayout<> chunk_layout;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(N5Metadata,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
  std::string GetCompatibilityKey() const;
};

/// Representation of partial metadata/metadata constraints specified as the
/// "metadata" member in the DriverSpec.
class N5MetadataConstraints {
 public:
  // The following members are common to `N5Metadata` and
  // `N5MetadataConstraints`, except that in `N5MetadataConstraints` some are
  // `std::optional`-wrapped.

  /// Length of `shape`, `axes` and `chunk_shape` if any are specified.  If none
  /// are specified, equal to `dynamic_rank`.
  DimensionIndex rank = dynamic_rank;

  /// Specifies the current shape of the full volume.
  std::optional<std::vector<Index>> shape;

  /// Specifies the dimension labels.
  std::optional<std::vector<std::string>> axes;

  N5Metadata::UnitsAndResolution units_and_resolution;

  /// Specifies the chunk size (corresponding to the `"blockSize"` attribute)
  /// and the in-memory layout of a full chunk (always C order).
  std::optional<std::vector<Index>> chunk_shape;

  std::optional<Compressor> compressor;
  std::optional<DataType> dtype;

  /// Contains all additional attributes, excluding attributes parsed into the
  /// data members above.
  ::nlohmann::json::object_t extra_attributes;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(N5MetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

class N5CodecSpec : public internal::CodecDriverSpec {
 public:
  constexpr static char id[] = "n5";

  CodecSpec Clone() const final;
  absl::Status DoMergeFrom(const internal::CodecDriverSpec& other_base) final;

  std::optional<Compressor> compressor;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(N5CodecSpec, FromJsonOptions,
                                          ToJsonOptions,
                                          ::nlohmann::json::object_t)
};

/// Validates that `metadata` is consistent with `constraints`.
absl::Status ValidateMetadata(const N5Metadata& metadata,
                              const N5MetadataConstraints& constraints);

/// Converts `metadata_constraints` to a full metadata object.
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const N5Metadata>> GetNewMetadata(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Validates that `schema` is compatible with `metadata`.
absl::Status ValidateMetadataSchema(const N5Metadata& metadata,
                                    const Schema& schema);

/// Sets chunk layout constraints implied by `rank` and `chunk_shape`.
absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout);

/// Returns the combined domain from `metadata_constraints` and `schema`.
///
/// If the domain is unspecified, returns a null domain.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<IndexDomain<>> GetEffectiveDomain(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the combined chunk layout from `metadata_constraints` and `schema`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<ChunkLayout> GetEffectiveChunkLayout(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the combined codec spec from `metadata_constraints` and `schema`.
///
/// \returns Non-null pointer.
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<internal::CodecDriverSpec::PtrT<N5CodecSpec>> GetEffectiveCodec(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the codec from the specified metadata.
CodecSpec GetCodecFromMetadata(const N5Metadata& metadata);

/// Combines the units and resolution fields into a dimension units vector.
DimensionUnitsVector GetDimensionUnits(
    DimensionIndex metadata_rank,
    const N5Metadata::UnitsAndResolution& units_and_resolution);

/// Returns the combined dimension units from `units_and_resolution` and
/// `schema_units`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `units_and_resolution` is
///     inconsistent with `schema_units`.
Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    DimensionIndex metadata_rank,
    const N5Metadata::UnitsAndResolution& units_and_resolution,
    Schema::DimensionUnits schema_units);

/// Decodes a chunk.
///
/// The layout of the returned array is only valid as long as `metadata`.
Result<SharedArrayView<const void>> DecodeChunk(const N5Metadata& metadata,
                                                absl::Cord buffer);

/// Encodes a chunk.
Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const N5Metadata& metadata,
                               ArrayView<const void> array);

/// Validates that `dtype` is supported by N5.
///
/// \dchecks `dtype.valid()`
absl::Status ValidateDataType(DataType dtype);

}  // namespace internal_n5
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_n5::N5MetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_n5::N5MetadataConstraints)

#endif  // TENSORSTORE_DRIVER_N5_METADATA_H_
