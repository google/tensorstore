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

#include <nlohmann/json.hpp>
#include "tensorstore/codec_spec.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr {

class ZarrCodecSpec : public CodecSpec {
 public:
  constexpr static char id[] = "zarr";
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
Status ValidateMetadata(const ZarrMetadata& metadata,
                        const ZarrPartialMetadata& constraints);

Result<ZarrMetadataPtr> GetNewMetadata(
    const ZarrPartialMetadata& partial_metadata, DataType data_type_constraint);

/// Specifies how chunk index vectors are encoded as keys.
///
/// Index vectors are encoded in order as their base-10 ASCII representation,
/// separated by either "." or "/".
///
/// The zarr metadata format does not specify the key encoding; therefore, it
/// must be specified separately when opening the driver.
enum class ChunkKeyEncoding {
  kDotSeparated = 0,
  kSlashSeparated = 1,
};

TENSORSTORE_DECLARE_JSON_BINDER(ChunkKeyEncodingJsonBinder, ChunkKeyEncoding,
                                internal_json_binding::NoOptions,
                                internal_json_binding::NoOptions)

/// An empty string indicates the singleton field if the dtype does not have
/// fields.
using SelectedField = std::string;

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

/// Returns the numeric index of the field, and validates its data type.
///
/// \param dtype The parsed zarr "dtype" specification.
/// \param data_type_constraint If `data_type_constraint.valid() == true`,
///     constrains the data type of the selected field.
/// \param selected_field The label of the field, or an empty string to indicate
///     that the zarr array must have only a single field.
/// \returns The field index.
/// \error `absl::StatusCode::kFailedPrecondition` if `selected_field` is not
///     valid for `dtype`.
/// \error `absl::StatusCode::kFailedPrecondition` if `data_type_constraint` is
///     not satisfied.
Result<std::size_t> GetCompatibleField(const ZarrDType& dtype,
                                       DataType data_type_constraint,
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

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_SPEC_H_
