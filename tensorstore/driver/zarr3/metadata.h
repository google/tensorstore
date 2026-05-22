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

#include <stddef.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/dtype.h"
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
  ZarrDType zarr_dtype;
  bool allow_missing_dtype = false;
  FillValueJsonBinder() = default;
  explicit FillValueJsonBinder(ZarrDType zarr_dtype,
                               bool allow_missing_dtype = false);
  explicit FillValueJsonBinder(DataType dtype,
                               bool allow_missing_dtype = false);

  absl::Status operator()(std::true_type is_loading,
                          internal_json_binding::NoOptions,
                          std::vector<SharedArray<const void>>* obj,
                          ::nlohmann::json* j) const;

  absl::Status operator()(std::false_type is_loading,
                          internal_json_binding::NoOptions,
                          const std::vector<SharedArray<const void>>* obj,
                          ::nlohmann::json* j) const;

 private:
  absl::Status DecodeSingle(::nlohmann::json& j, DataType data_type,
                            SharedArray<const void>& out) const;
  absl::Status EncodeSingle(const SharedArray<const void>& arr,
                            DataType data_type,
                            ::nlohmann::json& j) const;
};

struct SpecRankAndFieldInfo;

struct ZarrMetadata {
  // The following members are common to `ZarrMetadata` and
  // `ZarrMetadataConstraints`, except that in `ZarrMetadataConstraints` some
  // are `std::optional`-wrapped.

  DimensionIndex rank = dynamic_rank;

  int zarr_format;
  std::vector<Index> shape;
  ZarrDType zarr_dtype;
  ::nlohmann::json::object_t user_attributes;
  std::optional<DimensionUnitsVector> dimension_units;
  std::vector<std::optional<std::string>> dimension_names;
  ChunkKeyEncoding chunk_key_encoding;
  std::vector<Index> chunk_shape;
  ZarrCodecChainSpec codec_specs;
  std::vector<SharedArray<const void>> fill_value;
  ::nlohmann::json::object_t unknown_extension_attributes;

  std::string GetCompatibilityKey() const;

  ZarrCodecChain::Ptr codecs;
  ZarrCodecChain::PreparedState::Ptr codec_state;
  std::array<DimensionIndex, kMaxRank> inner_order;

  // Inner trailing dimensions appended to `chunk_shape` to form the codec
  // resolution shape.  Empty for plain scalar single-field arrays.  For
  // single-field arrays whose dtype carries an `rN`-style `field_shape`, this
  // mirrors that per-field `field_shape`.  For multi-field structs and for
  // the byte-substituted view produced by `GetVoidMetadata`, this is
  // `{bytes_per_outer_element}` -- the single trailing byte dimension that
  // pins the struct's interleaved byte layout into the codec chain.
  //
  // Conceptually `chunk_shape` describes the array's chunked dimensions and
  // `field_shape` describes any inner (per-element) dimensions; the two are
  // concatenated to form the rank the codec chain is resolved against.  This
  // replaces the previous implicit "shape prefix" handling where the trailing
  // byte dimension was ad-hoc-appended at every consumer site.
  //
  // Not part of the persisted zarr.json -- derived during `ValidateMetadata`.
  std::vector<Index> field_shape;

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
  std::optional<ZarrDType> zarr_dtype;
  ::nlohmann::json::object_t user_attributes;
  std::optional<DimensionUnitsVector> dimension_units;
  std::optional<std::vector<std::optional<std::string>>> dimension_names;
  std::optional<ChunkKeyEncoding> chunk_key_encoding;
  std::optional<std::vector<Index>> chunk_shape;
  std::optional<ZarrCodecChainSpec> codec_specs;
  std::optional<std::vector<SharedArray<const void>>> fill_value;
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
    const SpecRankAndFieldInfo& info,
    std::optional<span<const Index>> chunk_shape,
    const ZarrCodecChainSpec* codecs, ChunkLayout& chunk_layout);
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
                                    size_t field_index, const Schema& schema);
absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    const Schema& schema);

/// Converts `metadata_constraints` to a full metadata object.
///
/// When `open_as_void` is true, the persisted metadata is constructed for the
/// natural data type representation (i.e. unchanged from `metadata_constraints`),
/// but the user-visible domain rank is extended by one trailing byte dimension.
/// `GetNewMetadata` strips that trailing dimension from the user-supplied schema
/// shape/chunk_layout, and skips the dtype/domain compatibility check against
/// the schema (the caller is expected to re-validate against the void-substituted
/// metadata).
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const ZarrMetadata>> GetNewMetadata(
    const ZarrMetadataConstraints& metadata_constraints,
    const Schema& schema, std::string_view selected_field,
    bool open_as_void);

absl::Status ValidateDataType(DataType dtype);

/// Returns the index of `selected_field` within `dtype.fields`.
///
/// If `selected_field` is empty, requires the dtype to have a single field and
/// returns 0.
Result<size_t> GetFieldIndex(const ZarrDType& zarr_dtype,
                             std::string_view selected_field);

/// Returns a synthetic single-field "void view" `ZarrDType` whose only field
/// has dtype `byte` and `field_shape = {bytes_per_outer_element}`.  This is
/// the representation handed to the chunk cache when `open_as_void` is in use,
/// and is treated by all downstream code as a normal `field_shape`-bearing
/// field (no separate `open_as_void` plumbing required).
ZarrDType MakeVoidDType(Index bytes_per_outer_element);

/// Packs `per_field_fill` (one fill value per field of `dtype`) into a single
/// 1-D byte `SharedArray` of length `dtype.bytes_per_outer_element`, following
/// the struct's `byte_offset` layout.
///
/// The byte order of each typed field in the packed buffer matches the byte
/// order that the bytes codec would write for a chunk encoded under the
/// supplied `codec_specs` -- for a single non-byte scalar field this means
/// the codec's configured endian; for multi-field structs and `rN` raw byte
/// fields it is native-endian (matching the chunk cache's CopyArray-based
/// per-field packing path).  This guarantees that the synthetic void fill
/// returned for a missing chunk is byte-identical to what `read` returns for
/// a present chunk.
SharedArray<const void> MakeVoidFillValue(
    const ZarrDType& zarr_dtype, const ZarrCodecChainSpec& codec_specs,
    span<const SharedArray<const void>> per_field_fill);

/// Returns a void-access view of `metadata`: a new `ZarrMetadata` whose
/// `data_type` is `MakeVoidDType(...)`, whose `fill_value` is a single byte
/// array packed from each original field's per-field fill, and whose codec
/// chain has been re-resolved against the substituted byte data type with an
/// extra innermost dimension of `bytes_per_outer_element`.  Persisted state
/// (`shape`, `chunk_shape`, `chunk_key_encoding`, etc.) is unchanged.
///
/// Also validates that the resulting codec chain is acceptable for raw byte
/// access (see `ValidateVoidCodecChain`).
Result<std::shared_ptr<const ZarrMetadata>> GetVoidMetadata(
    const ZarrMetadata& metadata);

/// Validates that `codec_specs` is compatible with `open_as_void` access.
///
/// The only structural precondition is that, after unwinding any
/// `sharding_indexed` layers, the innermost array-to-bytes codec is the
/// `bytes` codec.  The previous "preserves dtype" and "preserves trailing
/// byte dim" rules on array-to-array codecs are now structurally guaranteed
/// by the codec resolution architecture: the chain exposes inner
/// (`field_shape`) dims to a-to-a codecs only via `inner_shape` (which they
/// must propagate verbatim), and `Resolve` forces every a-to-a codec to
/// preserve the substituted byte dtype.
absl::Status ValidateVoidCodecChain(const ZarrCodecChainSpec& codec_specs);

struct SpecRankAndFieldInfo {
  /// Full rank of the TensorStore, if known. Equal to the chunked rank plus
  /// the field rank.
  DimensionIndex full_rank = dynamic_rank;

  /// Number of chunked dimensions (the array's original rank).
  DimensionIndex chunked_rank = dynamic_rank;

  /// Number of field dimensions contributed by `field->field_shape`.
  DimensionIndex field_rank = dynamic_rank;

  /// Data type field, or `nullptr` if unknown.
  const ZarrDType::Field* field = nullptr;
};

/// Validates and computes derived rank fields in `info`.
absl::Status ValidateSpecRankAndFieldInfo(SpecRankAndFieldInfo& info);

/// Gets spec rank and field info from metadata constraints.
///
/// When `open_as_void` is true, `info.field_rank` is 1 (the synthetic bytes
/// dimension contributed by `MakeVoidDType`).  Otherwise `info.field_rank` is
/// derived from the selected field's `field_shape`.
///
/// \param metadata Metadata constraints.
/// \param selected_field The field to access. Must be empty if `open_as_void`
///     is true.
/// \param schema Schema constraints.
/// \param open_as_void If true, opens the array as raw bytes.
/// \error `absl::StatusCode::kInvalidArgument` if both `selected_field` is
///     non-empty and `open_as_void` is true.
Result<SpecRankAndFieldInfo> GetSpecRankAndFieldInfo(
    const ZarrMetadataConstraints& metadata, std::string_view selected_field,
    const Schema& schema, bool open_as_void);

SpecRankAndFieldInfo GetSpecRankAndFieldInfo(const ZarrMetadata& metadata,
                                             size_t field_index);

/// Sets schema dtype and rank constraints based on metadata constraints.
///
/// \param metadata_constraints The metadata constraints.
/// \param selected_field The selected field name, or empty.  Must be empty
///     if `open_as_void` is true.
/// \param open_as_void If true, opens the array as raw bytes.
/// \param schema The schema to update (modified in place).
/// \error `absl::StatusCode::kInvalidArgument` if both `selected_field` is
///     non-empty and `open_as_void` is true.
absl::Status TrySetMetadataConstraintsOnSchema(
    const ZarrMetadataConstraints& metadata_constraints,
    std::string_view selected_field, bool open_as_void, Schema& schema);

}  // namespace internal_zarr3
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrMetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_zarr3::ZarrMetadataConstraints)

#endif  // TENSORSTORE_DRIVER_ZARR3_METADATA_H_
