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

#ifndef TENSORSTORE_DRIVER_ZARR3_METADATA_H_
#define TENSORSTORE_DRIVER_ZARR3_METADATA_H_

/// \file
/// Support for encoding/decoding the JSON metadata for zarr arrays
/// See: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#metadata

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

// Defines how chunks map to keys in the underlying kvstore.
//
// https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#chunk-key-encoding
struct ChunkKeyEncoding {
  enum Kind {
    kDefault,
    kV2,
  };
  Kind kind;
  char separator;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ChunkKeyEncoding,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  friend bool operator==(const ChunkKeyEncoding& a, const ChunkKeyEncoding& b) {
    return a.kind == b.kind && a.separator == b.separator;
  }
  friend bool operator!=(const ChunkKeyEncoding& a, const ChunkKeyEncoding& b) {
    return !(a == b);
  }
};

struct FillValueJsonBinder {
  DataType data_type;

  absl::Status operator()(std::true_type is_loading,
                          internal_json_binding::NoOptions,
                          SharedArray<const void>* obj,
                          ::nlohmann::json* j) const;

  absl::Status operator()(std::false_type is_loading,
                          internal_json_binding::NoOptions,
                          const SharedArray<const void>* obj,
                          ::nlohmann::json* j) const;
};

struct ZarrMetadata {
  // The following members are common to `ZarrMetadata` and
  // `ZarrMetadataConstraints`, except that in `ZarrMetadataConstraints` some
  // are `std::optional`-wrapped.

  DimensionIndex rank = dynamic_rank;

  int zarr_format;
  std::vector<Index> shape;
  DataType data_type;
  ::nlohmann::json::object_t user_attributes;
  std::optional<DimensionUnitsVector> dimension_units;
  std::vector<std::optional<std::string>> dimension_names;
  ChunkKeyEncoding chunk_key_encoding;
  std::vector<Index> chunk_shape;
  ZarrCodecChainSpec codec_specs;
  SharedArray<const void> fill_value;
  ::nlohmann::json::object_t unknown_extension_attributes;

  std::string GetCompatibilityKey() const;

  ZarrCodecChain::Ptr codecs;
  ZarrCodecChain::PreparedState::Ptr codec_state;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrMetadata,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

struct ZarrMetadataConstraints {
  ZarrMetadataConstraints() = default;
  explicit ZarrMetadataConstraints(const ZarrMetadata& metadata);

  DimensionIndex rank = dynamic_rank;

  std::optional<int> zarr_format;
  std::optional<std::vector<Index>> shape;
  std::optional<DataType> data_type;
  ::nlohmann::json::object_t user_attributes;
  std::optional<DimensionUnitsVector> dimension_units;
  std::optional<std::vector<std::optional<std::string>>> dimension_names;
  std::optional<ChunkKeyEncoding> chunk_key_encoding;
  std::optional<std::vector<Index>> chunk_shape;
  std::optional<ZarrCodecChainSpec> codec_specs;
  std::optional<SharedArray<const void>> fill_value;
  ::nlohmann::json::object_t unknown_extension_attributes;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrMetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

/// Validates metadata, initializes `metadata.codecs`.
absl::Status ValidateMetadata(ZarrMetadata& metadata);

absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrMetadataConstraints& constraints);

/// Returns the combined domain from `metadata_constraints` and `schema`.
///
/// If the domain is unspecified, returns a null domain.
///
/// \param dimension_names_ignored[out] If non-null, the pointee is set to
///     indicate if the dimension names specified by `metadata_constraints` were
///     used.
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<IndexDomain<>> GetEffectiveDomain(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema,
    bool* dimension_names_used = nullptr);

/// Sets chunk layout constraints implied by `dtype`, `rank`, `chunk_shape`, and
/// `codecs`.
absl::Status SetChunkLayoutFromMetadata(
    DataType dtype, DimensionIndex rank,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, ChunkLayout& chunk_layout);

/// Returns the combined chunk layout from `metadata_constraints` and `schema`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<ChunkLayout> GetEffectiveChunkLayout(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the stored dimension units, or default unspecified units.
Result<DimensionUnitsVector> GetDimensionUnits(
    DimensionIndex rank,
    const std::optional<DimensionUnitsVector>& dimension_units_constraints);

/// Returns the combined dimension units from `dimension_units_constraints` and
/// `schema_units`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `dimension_units_constraints`
///     is inconsistent with `schema_units`.
Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    DimensionIndex rank,
    const std::optional<DimensionUnitsVector>& dimension_units_constraints,
    Schema::DimensionUnits schema_units);

/// Returns the combined codec spec from `metadata_constraints` and `schema`.
///
/// \returns Non-null pointer.
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<internal::CodecDriverSpec::PtrT<TensorStoreCodecSpec>> GetEffectiveCodec(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the codec from the specified metadata.
CodecSpec GetCodecFromMetadata(const ZarrMetadata& metadata);

/// Validates that `schema` is compatible with `metadata`.
absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    const Schema& schema);

/// Converts `metadata_constraints` to a full metadata object.
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const ZarrMetadata>> GetNewMetadata(
    const ZarrMetadataConstraints& metadata_constraints, const Schema& schema);

absl::Status ValidateDataType(DataType dtype);

}  // namespace internal_zarr3
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrMetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_zarr3::ZarrMetadataConstraints)

#endif  // TENSORSTORE_DRIVER_ZARR3_METADATA_H_
