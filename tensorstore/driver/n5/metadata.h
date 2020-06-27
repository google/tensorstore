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

#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/strided_layout.h"
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
  DimensionIndex rank() const { return chunk_layout.rank(); }

  /// Specifies the current shape of the full volume.
  std::vector<Index> shape;

  /// Specifies the chunk size (corresponding to the `"blockSize"` attribute)
  /// and the in-memory layout of a full chunk (always C order).
  StridedLayout<> chunk_layout;

  DataType data_type;

  Compressor compressor;

  /// Contains all attributes, including the `"dimensions"`, `"blockSize"`,
  /// `"dataType"`, and `"compression"` attributes corresponding to the other
  /// data members of this class.
  ::nlohmann::json::object_t attributes;

  /// Parses an N5 metadata JSON specification, i.e. the contents of the
  /// `"attributes.json"` file for an N5 dataset.
  ///
  /// \param metadata[out] Non-null pointer set to the parsed metadata on
  ///     success.
  /// \error `absl::StatusCode::kInvalidArgument` if `j` is not valid.
  static Result<N5Metadata> Parse(::nlohmann::json j);

  friend void to_json(::nlohmann::json& out,  // NOLINT
                      const N5Metadata& metadata) {
    out = metadata.attributes;
  }

  std::string GetCompatibilityKey() const;
};

/// Representation of partial metadata/metadata constraints specified as the
/// "metadata" member in the DriverSpec.
class N5MetadataConstraints {
 public:
  std::optional<std::vector<Index>> shape;
  std::optional<std::vector<Index>> chunk_shape;
  std::optional<Compressor> compressor;
  /// Specifies the data type, or may be invalid to indicate an unspecified data
  /// type.
  DataType data_type;

  ::nlohmann::json::object_t attributes;

  /// Parses a partial N5 metadata specification from a TensorStore open
  /// specification.
  static Result<N5MetadataConstraints> Parse(::nlohmann::json j);

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(N5MetadataConstraints,
                                          internal::json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

/// Validates that `metadata` is consistent with `constraints`.
Status ValidateMetadata(const N5Metadata& metadata,
                        const N5MetadataConstraints& constraints);

/// Converts `metadata_constraints` to a full metadata object.
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const N5Metadata>> GetNewMetadata(
    const N5MetadataConstraints& metadata_constraints);

/// Decodes a chunk.
///
/// The layout of the returned array is only valid as long as `metadata`.
Result<SharedArrayView<const void>> DecodeChunk(const N5Metadata& metadata,
                                                absl::Cord buffer);

/// Encodes a chunk.
Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const N5Metadata& metadata,
                               ArrayView<const void> array);

/// Validates that `data_type` is supported by N5.
///
/// \dchecks `data_type.valid()`
Status ValidateDataType(DataType data_type);

}  // namespace internal_n5
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_N5_METADATA_H_
