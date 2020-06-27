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

#ifndef TENSORSTORE_DRIVER_ZARR_METADATA_H_
#define TENSORSTORE_DRIVER_ZARR_METADATA_H_

/// \file
/// Support for encoding/decoding the JSON metadata for zarr arrays
/// See: https://zarr.readthedocs.io/en/stable/spec/v2.html

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr {

/// Derived layout information computed from the `ZarrDType`, the
/// `ContiguousLayoutOrder`, and the chunk shape.
struct ZarrChunkLayout {
  /// Derived layout information for a single `ZarrDType::Field`.
  struct Field {
    /// Strided layout for `full_chunk_shape` where the last
    /// `field_shape.size()` dimensions are in contiguous C order (with a base
    /// stride equal to the underlying element size), while the first
    /// `chunk_shape.size()` dimensions are in either Fortran or C order
    /// depending on the `order` metadata property, with a base stride of
    /// `bytes_per_outer_element`.  This layout matches the actual encoded
    /// representation of a chunk (the fields are interleaved).
    StridedLayout<> encoded_chunk_layout;

    /// Same as `encoded_chunk_layout`, except that the base stride for the
    /// first `chunk_shape.size()` dimensions is `num_bytes`.  This layout
    /// matches the representation used in the chunk cache (fields are not
    /// interleaved).
    StridedLayout<> decoded_chunk_layout;

    /// The concatenation of the chunk shape with `field_shape`.
    span<const Index> full_chunk_shape() const {
      return decoded_chunk_layout.shape();
    }
  };

  /// Number of "outer" elements consisting of a single value of each
  /// field. Each such value may itself be an inner array.  This is simply the
  /// product of the `chunk_shape` dimensions.
  Index num_outer_elements;

  /// Total bytes per chunk.
  Index bytes_per_chunk;

  std::vector<Field> fields;
};

struct ZarrMetadata {
 public:
  std::uint64_t zarr_format;

  /// Overall shape of array.
  std::vector<Index> shape;

  /// Chunk shape.  Must have same length as `shape`.
  std::vector<Index> chunks;

  ZarrDType dtype;
  Compressor compressor;

  /// Encoded layout of chunk.
  ContiguousLayoutOrder order;
  // std::vector<ZarrFilter> filters;

  /// Fill values for each of the fields.  Must have same length as
  /// `dtype.fields`.
  std::vector<SharedArray<const void>> fill_values;

  /// Derived information computed from `dtype`, `order`, and `chunks`.
  ZarrChunkLayout chunk_layout;

  friend void to_json(::nlohmann::json& out,  // NOLINT
                      const ZarrMetadata& metadata);

  /// Appends to `*out` a string that corresponds to the equivalence
  /// relationship defined by `IsMetadataCompatible`.
  friend void EncodeCacheKeyAdl(std::string* out, const ZarrMetadata& metadata);
};

using ZarrMetadataPtr = std::shared_ptr<ZarrMetadata>;

/// Parses a zarr metadata JSON specification.
///
/// \param metadata[out] Non-null pointer set to the parsed metadata on success.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not valid.
Status ParseMetadata(const nlohmann::json& j, ZarrMetadata* metadata);

/// Computes the derived `ZarrChunkLayout` from the specified `dtype`, `order`,
/// and `chunk_shape` values.
///
/// \error `absl::StatusCode::kInvalidArgument` if the chunk or data type size
///     is too large.
Result<ZarrChunkLayout> ComputeChunkLayout(const ZarrDType& dtype,
                                           ContiguousLayoutOrder order,
                                           span<const Index> chunk_shape);

/// Encodes the field fill values as a zarr metadata "fill_value" JSON
/// specification.
///
/// An unspecified fill value is indicated by a null array.  The fill fill
/// values must be specified, or all fill values must be unspecified; it is not
/// valid to specify the fill values for some, but not all fields.
///
/// \dchecks `fill_values.size() == dtype.fields.size()`.
/// \pre `fill_values[i].data_type() == dtype.fields[i].data_type` for
///     `0 <= i < dtype.fields.size()`.
::nlohmann::json EncodeFillValue(
    const ZarrDType& dtype, span<const SharedArray<const void>> fill_values);

/// Parses a zarr metadata "fill_value" JSON specification.
///
/// \param j The fill value specification.
/// \param dtype The decoded zarr data type specification associated with the
///     fill value specification.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not valid.
Result<std::vector<SharedArray<const void>>> ParseFillValue(
    const nlohmann::json& j, const ZarrDType& dtype);

/// Parses a zarr metadata "zarr_format" JSON specification.
///
/// Currently the only valid value is `2`.
///
/// \param value The "zarr_format" JSON specification.
/// \error `absl::StatusCode::kInvalidArgument` if `value` is not valid.
Result<std::uint64_t> ParseZarrFormat(const nlohmann::json& value);

/// Parses a zarr metadata "filters" JSON specification.
///
/// Currently the only supported value is `null`.
///
/// \param value The "filters" JSON specification.
/// \error `absl::StatusCode::kInvalidArgument` if `value` is not valid.
Status ParseFilters(const nlohmann::json& value);

/// Encodes a layout order as a zarr metadata "order" JSON specification.
///
/// \returns `"C"` or `"F"`.
::nlohmann::json EncodeOrder(ContiguousLayoutOrder order);

/// Parses a zarr metadata "order" JSON specification.
///
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not valid.
Result<ContiguousLayoutOrder> ParseOrder(const nlohmann::json& j);

/// Parses a zarr metadata "chunks" JSON specification.
///
/// \param value The "chunks" JSON specification.
/// \param rank[in,out] Must be non-null.  On input, if
///     `*rank != absl::nullopt`, `**rank` specifies the expected number of
///     dimensions.  On successful return, `**rank` is set to `shape->size()`.
/// \param shape[out] Must be non-null.  Pointer to shape vector to be filled
///     with the parsed chunk shape.
/// \error `absl::StatusCode::kInvalidArgument` if `value` is not valid.
Status ParseChunkShape(const ::nlohmann::json& value,
                       absl::optional<DimensionIndex>* rank,
                       std::vector<Index>* shape);

/// Parses a zarr metadata "shape" JSON specification.
///
/// \param value The "chunks" JSON specification.
/// \param rank[in,out] Must be non-null.  On input, if
///     `*rank != absl::nullopt`, `**rank` specifies the expected number of
///     dimensions.  On successful return, `**rank` is set to `shape->size()`.
/// \param shape[out] Must be non-null.  Pointer to shape vector to be filled
///     with the parsed chunk shape.
/// \error `absl::StatusCode::kInvalidArgument` if `value` is not valid.
Status ParseShape(const ::nlohmann::json& value,
                  absl::optional<DimensionIndex>* rank,
                  std::vector<Index>* shape);

/// Encodes per-field arrays into an encoded zarr chunk.
///
/// \param metadata Metadata associated with the chunk.
/// \param components Vector of per-field arrays.
/// \dchecks `components.size() == metadata.dtype.fields.size()`
/// \returns The encoded chunk.
Result<absl::Cord> EncodeChunk(const ZarrMetadata& metadata,
                               span<const ArrayView<const void>> components);

/// Decodes an encoded zarr chunk into per-field arrays.
///
/// \param metadata Metadata associated with the chunk.
/// \param buffer The buffer to decode.
/// \returns A vector of length `metadata.dtype.fields.size()`.
/// \error `absl::StatusCode::kInvalidArgument` if `buffer` is not a valid
///     encoded zarr chunk according to `metadata`.
Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
    const ZarrMetadata& metadata, absl::Cord buffer);

/// Returns `true` if `a` and `b` are compatible, meaning stored data created
/// with `a` can be read using `b`.
bool IsMetadataCompatible(const ZarrMetadata& a, const ZarrMetadata& b);

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_METADATA_H_
