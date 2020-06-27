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

#include "tensorstore/driver/n5/metadata.h"

#include <optional>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/strings/str_join.h"
#include "tensorstore/internal/container_to_shared.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_n5 {

namespace {

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,   DataTypeId::uint16_t, DataTypeId::uint32_t,
    DataTypeId::uint64_t,  DataTypeId::int8_t,   DataTypeId::int16_t,
    DataTypeId::int32_t,   DataTypeId::int64_t,  DataTypeId::float32_t,
    DataTypeId::float64_t,
};

Status ParseIndexVector(const ::nlohmann::json& value,
                        std::optional<DimensionIndex>* rank,
                        std::vector<Index>* vec, Index min_value,
                        Index max_value) {
  return internal::JsonParseArray(
      value,
      [&](std::ptrdiff_t size) {
        if (*rank) {
          TENSORSTORE_RETURN_IF_ERROR(
              internal::JsonValidateArrayLength(size, **rank));
        } else {
          *rank = size;
        }
        vec->resize(size);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
        return internal::JsonRequireInteger<Index>(
            v, &(*vec)[i], /*strict=*/true, min_value, max_value);
      });
}
}  // namespace

Status ParseChunkShape(const ::nlohmann::json& value,
                       std::optional<DimensionIndex>* rank,
                       std::vector<Index>* shape) {
  return ParseIndexVector(
      value, rank, shape, /*min_value=*/1,
      /*max_value=*/std::numeric_limits<std::uint32_t>::max());
}

Status ParseShape(const ::nlohmann::json& value,
                  std::optional<DimensionIndex>* rank,
                  std::vector<Index>* shape) {
  return ParseIndexVector(value, rank, shape, /*min_value=*/0, kInfIndex);
}

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

Status ParseDataType(const ::nlohmann::json& value, DataType* data_type) {
  std::string s;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireValueAs(value, &s));
  auto x = GetDataType(s);
  if (!x.valid() || !absl::c_linear_search(kSupportedDataTypes, x.id())) {
    return absl::InvalidArgumentError(StrCat(
        QuoteString(s),
        " is not one of the supported data types: ", GetSupportedDataTypes()));
  }
  *data_type = x;
  return absl::OkStatus();
}

Result<N5Metadata> N5Metadata::Parse(::nlohmann::json j) {
  std::optional<DimensionIndex> rank;
  N5Metadata metadata;
  if (auto* obj = j.get_ptr<::nlohmann::json::object_t*>()) {
    metadata.attributes = std::move(*obj);
  } else {
    return internal_json::ExpectedError(j, "object");
  }
  std::vector<Index> chunk_shape;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, "dimensions", [&](const ::nlohmann::json& value) {
        return ParseShape(value, &rank, &metadata.shape);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, "blockSize", [&](const ::nlohmann::json& value) {
        return ParseChunkShape(value, &rank, &chunk_shape);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, "dataType", [&](const ::nlohmann::json& value) {
        return ParseDataType(value, &metadata.data_type);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, "compression", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.compressor,
                                     Compressor::FromJson(value));
        return absl::OkStatus();
      }));
  // Per the specification:
  // https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
  //
  // Chunks cannot be larger than 2GB (2<sup>31</sup>Bytes).

  // While this limit may apply to compressed data, we will also apply it
  // to the uncompressed data.
  const Index max_num_elements =
      (static_cast<std::size_t>(1) << 31) / metadata.data_type.size();
  if (ProductOfExtents(span(chunk_shape)) > max_num_elements) {
    return absl::InvalidArgumentError(
        StrCat("\"blockSize\" of ", span(chunk_shape), " with data type of ",
               metadata.data_type, " exceeds maximum chunk size of 2GB"));
  }
  InitializeContiguousLayout(fortran_order, metadata.data_type.size(),
                             chunk_shape, &metadata.chunk_layout);
  return metadata;
}

std::string N5Metadata::GetCompatibilityKey() const {
  ::nlohmann::json::object_t obj;
  span<const Index> chunk_shape = chunk_layout.shape();
  obj.emplace("blockSize", ::nlohmann::json::array_t(chunk_shape.begin(),
                                                     chunk_shape.end()));
  obj.emplace("dataType", data_type.name());
  obj.emplace("compression", compressor);
  return ::nlohmann::json(obj).dump();
}

Result<N5MetadataConstraints> N5MetadataConstraints::Parse(::nlohmann::json j) {
  std::optional<DimensionIndex> rank;
  N5MetadataConstraints metadata;
  if (auto* obj = j.get_ptr<::nlohmann::json::object_t*>()) {
    metadata.attributes = std::move(*obj);
  } else {
    return internal_json::ExpectedError(j, "object");
  }
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      metadata.attributes, "dimensions", [&](const ::nlohmann::json& value) {
        return ParseShape(value, &rank, &metadata.shape.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      metadata.attributes, "blockSize", [&](const ::nlohmann::json& value) {
        return ParseChunkShape(value, &rank, &metadata.chunk_shape.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      metadata.attributes, "dataType", [&](const ::nlohmann::json& value) {
        return ParseDataType(value, &metadata.data_type);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      metadata.attributes, "compression", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.compressor,
                                     Compressor::FromJson(value));
        return absl::OkStatus();
      }));
  return metadata;
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(N5MetadataConstraints,
                                       [](auto is_loading, const auto& options,
                                          auto* obj, auto* j) {
                                         if constexpr (is_loading) {
                                           // TODO(jbms): Convert this to use
                                           // JSON binding framework for
                                           // parsing.
                                           TENSORSTORE_ASSIGN_OR_RETURN(
                                               *obj, Parse(*j));
                                         } else {
                                           *j = obj->attributes;
                                         }
                                         return absl::OkStatus();
                                       })

std::size_t GetChunkHeaderSize(const N5Metadata& metadata) {
  // Per the specification:
  // https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
  //
  // Chunks are stored in the following binary format:
  //
  //    * mode (uint16 big endian, default = 0x0000, varlength = 0x0001)
  //
  //    * number of dimensions (uint16 big endian)
  //
  //    * dimension 1[,...,n] (uint32 big endian)
  //
  //    * [ mode == varlength ? number of elements (uint32 big endian) ]
  //
  //    * compressed data (big endian)
  return                                        //
      2 +                                       // mode
      2 +                                       // number of dimensions
      sizeof(std::uint32_t) * metadata.rank();  // dimensions
}

Result<SharedArrayView<const void>> DecodeChunk(const N5Metadata& metadata,
                                                absl::Cord buffer) {
  // TODO(jbms): Currently, we do not check that `buffer.size()` is less than
  // the 2GiB limit, although we do implicitly check that the decoded array data
  // within the chunk is within the 2GiB limit due to the checks on the block
  // size.  Determine if this is an important limit.
  SharedArrayView<const void> array;
  array.layout() = metadata.chunk_layout;
  const std::size_t header_size = GetChunkHeaderSize(metadata);
  if (buffer.size() < header_size) {
    return absl::InvalidArgumentError(
        StrCat("Expected header of length ", header_size,
               ", but chunk has size ", buffer.size()));
  }
  auto header_cord = buffer.Subcord(0, header_size);
  auto header = header_cord.Flatten();
  std::uint16_t mode = absl::big_endian::Load16(header.data());
  switch (mode) {
    case 0:  // default
      break;
    case 1:  // varlength
      return absl::InvalidArgumentError("varlength chunks not supported");
    default:
      return absl::InvalidArgumentError(
          StrCat("Unexpected N5 chunk mode: ", mode));
  }
  std::uint16_t num_dims = absl::big_endian::Load16(header.data() + 2);
  if (num_dims != metadata.rank()) {
    return absl::InvalidArgumentError(StrCat("Received chunk with ", num_dims,
                                             " dimensions but expected ",
                                             metadata.rank()));
  }
  Array<const void, dynamic_rank(internal::kNumInlinedDims)> encoded_array;
  encoded_array.layout().set_rank(metadata.rank());
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    encoded_array.shape()[i] =
        absl::big_endian::Load32(header.data() + 4 + i * 4);
  }
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    if (encoded_array.shape()[i] > metadata.chunk_layout.shape()[i]) {
      return absl::InvalidArgumentError(StrCat(
          "Received chunk of size ", encoded_array.shape(),
          " which exceeds blockSize of ", metadata.chunk_layout.shape()));
    }
  }
  size_t decoded_offset = header_size;
  if (metadata.compressor) {
    // TODO(jbms): Change compressor interface to allow the output size to be
    // specified.
    absl::Cord decoded;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Decode(
        buffer.Subcord(header_size, buffer.size() - header_size), &decoded,
        metadata.data_type.size()));
    buffer = std::move(decoded);
    decoded_offset = 0;
  }
  const std::size_t expected_data_size =
      encoded_array.num_elements() * metadata.data_type.size();
  if (buffer.size() - decoded_offset != expected_data_size) {
    return absl::InvalidArgumentError(StrCat(
        "Uncompressed chunk data has length ", buffer.size() - decoded_offset,
        ", but expected length to be ", expected_data_size));
  }
  // TODO(jbms): Try to avoid unnecessary copies in encoding and decoding.  This
  // applies to other chunk drivers as well.
  if (absl::c_equal(encoded_array.shape(), metadata.chunk_layout.shape())) {
    // Chunk is full size.  Attempt to decode in place.  Transfer ownership of
    // the existing `buffer` string into `decoded_array`.
    auto decoded_array =
        internal::TryViewCordAsArray(buffer, decoded_offset, metadata.data_type,
                                     endian::big, metadata.chunk_layout);
    if (decoded_array.valid()) return decoded_array;
  }
  // Partial chunk, must copy.
  auto flat_buffer = buffer.Flatten();
  ComputeStrides(fortran_order, metadata.data_type.size(),
                 encoded_array.shape(), encoded_array.byte_strides());
  encoded_array.element_pointer() = ElementPointer<const void>(
      static_cast<const void*>(flat_buffer.data() + decoded_offset),
      metadata.data_type);
  SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(
          metadata.chunk_layout.num_elements(), value_init, metadata.data_type),
      metadata.chunk_layout);
  ArrayView<void> partial_decoded_array(
      full_decoded_array.element_pointer(),
      StridedLayoutView<>{encoded_array.shape(),
                          metadata.chunk_layout.byte_strides()});
  internal::DecodeArray(encoded_array, endian::big, partial_decoded_array);
  return full_decoded_array;
}

Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const N5Metadata& metadata,
                               ArrayView<const void> array) {
  assert(absl::c_equal(metadata.chunk_layout.shape(), array.shape()));
  assert(chunk_indices.size() == array.rank());
  Array<void, dynamic_rank(internal::kNumInlinedDims)> encoded_array;
  encoded_array.layout().set_rank(array.rank());
  for (DimensionIndex i = 0; i < array.rank(); ++i) {
    const Index full_size = array.shape()[i];
    encoded_array.shape()[i] =
        std::min(full_size, metadata.shape[i] - full_size * chunk_indices[i]);
  }
  const std::size_t header_size = GetChunkHeaderSize(metadata);
  internal::FlatCordBuilder encoded(encoded_array.num_elements() *
                                    metadata.data_type.size());
  encoded_array.element_pointer() = ElementPointer<void>(
      static_cast<void*>(encoded.data()), metadata.data_type);
  InitializeContiguousLayout(fortran_order, metadata.data_type.size(),
                             &encoded_array.layout());
  ArrayView<const void> partial_source_array(
      array.element_pointer(),
      StridedLayoutView<>(encoded_array.shape(), array.byte_strides()));
  internal::EncodeArray(partial_source_array, encoded_array, endian::big);
  auto encoded_cord = std::move(encoded).Build();
  if (metadata.compressor) {
    absl::Cord compressed;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Encode(
        std::move(encoded_cord), &compressed, metadata.data_type.size()));
    encoded_cord = std::move(compressed);
  }
  internal::FlatCordBuilder header(header_size);
  // Write header
  absl::big_endian::Store16(header.data(), 0);  // mode: 0x0 = default
  const DimensionIndex rank = metadata.rank();
  absl::big_endian::Store16(header.data() + 2, rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    absl::big_endian::Store32(header.data() + 4 + i * 4,
                              encoded_array.shape()[i]);
  }
  auto full_cord = std::move(header).Build();
  full_cord.Append(std::move(encoded_cord));
  return full_cord;
}

Status ValidateMetadata(const N5Metadata& metadata,
                        const N5MetadataConstraints& constraints) {
  const auto MetadataMismatchError = [](const char* name, const auto& expected,
                                        const auto& actual) -> Status {
    return absl::FailedPreconditionError(
        StrCat("Expected ", QuoteString(name), " of ",
               ::nlohmann::json(expected).dump(),
               " but received: ", ::nlohmann::json(actual).dump()));
  };

  if (constraints.shape && !absl::c_equal(metadata.shape, *constraints.shape)) {
    return MetadataMismatchError("dimensions", *constraints.shape,
                                 metadata.shape);
  }
  if (constraints.chunk_shape &&
      !absl::c_equal(metadata.chunk_layout.shape(), *constraints.chunk_shape)) {
    return MetadataMismatchError("blockSize", *constraints.chunk_shape,
                                 metadata.attributes.at("blockSize"));
  }
  if (constraints.data_type.valid() &&
      constraints.data_type != metadata.data_type) {
    return MetadataMismatchError("dataType", constraints.data_type.name(),
                                 metadata.data_type.name());
  }
  if (constraints.compressor && ::nlohmann::json(*constraints.compressor) !=
                                    ::nlohmann::json(metadata.compressor)) {
    return MetadataMismatchError("compression", *constraints.compressor,
                                 metadata.compressor);
  }
  return absl::OkStatus();
}

Result<std::shared_ptr<const N5Metadata>> GetNewMetadata(
    const N5MetadataConstraints& metadata_constraints) {
  if (!metadata_constraints.shape) {
    return absl::InvalidArgumentError("\"dimensions\" must be specified");
  }
  if (!metadata_constraints.chunk_shape) {
    return absl::InvalidArgumentError("\"blockSize\" must be specified");
  }
  if (!metadata_constraints.data_type.valid()) {
    return absl::InvalidArgumentError("\"dataType\" must be specified");
  }
  if (!metadata_constraints.compressor) {
    return absl::InvalidArgumentError("\"compression\" must be specified");
  }
  auto metadata = std::make_shared<N5Metadata>();
  metadata->attributes = metadata_constraints.attributes;
  metadata->shape = *metadata_constraints.shape;

  InitializeContiguousLayout(
      fortran_order, metadata_constraints.data_type.size(),
      *metadata_constraints.chunk_shape, &metadata->chunk_layout);
  metadata->data_type = metadata_constraints.data_type;
  metadata->compressor = *metadata_constraints.compressor;
  return metadata;
}

Status ValidateDataType(DataType data_type) {
  if (!absl::c_linear_search(kSupportedDataTypes, data_type.id())) {
    return absl::InvalidArgumentError(
        StrCat(data_type, " data type is not one of the supported data types: ",
               GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

}  // namespace internal_n5
}  // namespace tensorstore
