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

#ifndef TENSORSTORE_DRIVER_ZARR_SPEC_H_
#define TENSORSTORE_DRIVER_ZARR_SPEC_H_

/// \file
/// Facilities related to parsing a zarr array DriverSpec.

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/codec_spec.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr {

class ZarrCodecSpec : public internal::CodecDriverSpec {
 public:
  constexpr static char id[] = "zarr";

  CodecSpec Clone() const final;
  absl::Status DoMergeFrom(const internal::CodecDriverSpec& other_base) final;

  std::optional<Compressor> compressor;
  std::optional<std::nullptr_t> filters;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrCodecSpec, FromJsonOptions,
                                          ToJsonOptions,
                                          ::nlohmann::json::object_t)
};

/// Validates that `metadata` is consistent with `constraints`.
///
/// \returns `OkStatus()` if `metadata` is consistent.
/// \error `absl::StatusCode::kFailedPrecondition` if `metadata` is
///     inconsistent.
absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrPartialMetadata& constraints);

/// An empty string indicates the singleton field if the dtype does not have
/// fields.
using SelectedField = std::string;

/// Creates zarr metadata from the given constraints.
///
/// \param partial_metadata Constraints in the form of partial zarr metadata.
/// \param selected_field The field to which `schema` applies.
/// \param schema Schema constraints for the `selected_field`.
Result<ZarrMetadataPtr> GetNewMetadata(
    const ZarrPartialMetadata& partial_metadata,
    const SelectedField& selected_field, const Schema& schema);

struct SpecRankAndFieldInfo {
  /// Full rank of the TensorStore, if known.  Equal to the chunked rank plus
  /// the field rank.
  DimensionIndex full_rank = dynamic_rank;

  /// Number of chunked dimensions.
  DimensionIndex chunked_rank = dynamic_rank;

  /// Number of field dimensions.
  DimensionIndex field_rank = dynamic_rank;

  /// Data type field, or `nullptr` if unknown.
  const ZarrDType::Field* field = nullptr;
};

absl::Status ValidateSpecRankAndFieldInfo(SpecRankAndFieldInfo& info);

Result<SpecRankAndFieldInfo> GetSpecRankAndFieldInfo(
    const ZarrPartialMetadata& metadata, const SelectedField& selected_field,
    const Schema& schema);

SpecRankAndFieldInfo GetSpecRankAndFieldInfo(const ZarrMetadata& metadata,
                                             size_t field_index);

/// Returns the combined domain from `metadata_shape` and `schema`.
///
/// \param info Rank and field information from metadata/schema.
/// \param metadata_shape The `shape` metadata field, if specified.
/// \param schema Schema constraints.
Result<IndexDomain<>> GetDomainFromMetadata(
    const SpecRankAndFieldInfo& info,
    std::optional<span<const Index>> metadata_shape, const Schema& schema);

absl::Status SetChunkLayoutFromMetadata(
    const SpecRankAndFieldInfo& info, std::optional<span<const Index>> chunks,
    std::optional<ContiguousLayoutOrder> order, ChunkLayout& chunk_layout);

CodecSpec GetCodecSpecFromMetadata(const ZarrMetadata& metadata);

/// Validates that `schema` is compatible with the specified field of
/// `metadata`.
absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    size_t field_index, const Schema& schema);

/// Parses a selected field JSON specification.
///
/// The selected field specification indicates the single field in a multi-field
/// zarr array to read/write.
///
/// \param value Selected field JSON specification.  Valid values are `null`
///     (which indicates that the zarr array must have only a single field) and
///     non-empty strings indicating the field label.
/// \returns The selected field label, or an empty string if `value` is null.
Result<SelectedField> ParseSelectedField(const ::nlohmann::json& value);

/// Returns the numeric index of the field.
///
/// \param dtype The parsed zarr "dtype" specification.
/// \param selected_field The label of the field, or an empty string to indicate
///     that the zarr array must have only a single field.
/// \returns The field index.
/// \error `absl::StatusCode::kFailedPrecondition` if `selected_field` is not
///     valid.
Result<std::size_t> GetFieldIndex(const ZarrDType& dtype,
                                  const SelectedField& selected_field);

/// Encodes a field index as a `SelectedField` JSON specification.
///
/// This is the inverse of `GetCompatibleField`.
///
/// \param field_index The field index.
/// \param dtype The zarr dtype specification.
/// \dchecks `field_index >= 0 && field_index < dtype.fields.size()`
/// \returns the field name associated with `field_index`, or the empty string
///     otherwise.
SelectedField EncodeSelectedField(std::size_t field_index,
                                  const ZarrDType& dtype);

/// Determines the order permutation for the given contiguous layout order
/// value.
///
/// \param chunked_rank Number of (outer) chunked dimensions, which use the
///     layout specified `order`.  The inner array dimensions are always in C
///     (lexicographic) order.
/// \param order The contiguous layout order value.
/// \param permutation[out] Set to the permutation, specifies the full rank.
void GetChunkInnerOrder(DimensionIndex chunked_rank,
                        ContiguousLayoutOrder order,
                        span<DimensionIndex> permutation);

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_SPEC_H_
