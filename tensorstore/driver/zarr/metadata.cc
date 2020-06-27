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

#include "tensorstore/driver/zarr/metadata.h"

#include <cassert>
#include <memory>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/escaping.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/internal/container_to_shared.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json.h"

namespace tensorstore {
namespace internal_zarr {

Result<ContiguousLayoutOrder> ParseOrder(const nlohmann::json& j) {
  if (j.is_string()) {
    std::string value = j.get<std::string>();
    if (value == "C") return ContiguousLayoutOrder::c;
    if (value == "F") return ContiguousLayoutOrder::fortran;
  }
  return absl::InvalidArgumentError(
      StrCat("Expected \"C\" or \"F\", but received: ", j.dump()));
}

::nlohmann::json EncodeOrder(ContiguousLayoutOrder order) {
  return order == c_order ? "C" : "F";
}

template <typename T>
inline T ConvertFromScalarArray(ElementPointer<const void> x) {
  TENSORSTORE_UNREACHABLE;
}

template <typename T, typename U0, typename... U>
inline T ConvertFromScalarArray(ElementPointer<const void> x) {
  if (x.data_type() == DataTypeOf<U0>()) {
    return static_cast<T>(*static_cast<const U0*>(x.data()));
  }
  return ConvertFromScalarArray<T, U...>(x);
}

Result<double> DecodeFloat(const nlohmann::json& j) {
  double value;
  if (j == "NaN") {
    value = std::numeric_limits<double>::quiet_NaN();
  } else if (j == "Infinity") {
    value = std::numeric_limits<double>::infinity();
  } else if (j == "-Infinity") {
    value = -std::numeric_limits<double>::infinity();
  } else if (j.is_number()) {
    value = j.get<double>();
  } else {
    return absl::InvalidArgumentError(
        StrCat("Invalid floating-point value: ", j.dump()));
  }
  return value;
}

Result<std::vector<SharedArray<const void>>> ParseFillValue(
    const nlohmann::json& j, const ZarrDType& dtype) {
  std::vector<SharedArray<const void>> fill_values;
  fill_values.resize(dtype.fields.size());
  if (j.is_null()) return fill_values;
  if (!dtype.has_fields) {
    assert(dtype.fields.size() == 1);
    auto& field = dtype.fields[0];
    const char type_indicator = field.encoded_dtype[1];
    switch (type_indicator) {
      case 'f': {
        TENSORSTORE_ASSIGN_OR_RETURN(double value, DecodeFloat(j));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.data_type)
                .value();
        return fill_values;
      }
      case 'i': {
        std::int64_t value;
        const std::size_t num_bits = 8 * field.data_type->size - 1;
        const std::uint64_t max_value = static_cast<std::int64_t>(
            (static_cast<std::uint64_t>(1) << num_bits) - 1);
        const std::int64_t min_value = static_cast<std::int64_t>(-1)
                                       << num_bits;
        TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireInteger(
            j, &value, /*strict=*/true, min_value, max_value));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.data_type)
                .value();
        return fill_values;
      }
      case 'u': {
        std::uint64_t value;
        const std::size_t num_bits = 8 * field.data_type->size;
        const std::uint64_t max_value =
            (static_cast<std::uint64_t>(2) << (num_bits - 1)) - 1;
        TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireInteger(
            j, &value, /*strict=*/true, 0, max_value));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(value), c_order, field.data_type)
                .value();
        return fill_values;
      }
      case 'b': {
        bool value;
        TENSORSTORE_RETURN_IF_ERROR(
            internal::JsonRequireValueAs(j, &value, /*strict=*/true));
        fill_values[0] = MakeScalarArray<bool>(value);
        return fill_values;
      }
      case 'c': {
        double values[2];
        if (!j.is_array()) {
          // Fallthrough to allow base64 encoding.
          break;
        }
        TENSORSTORE_RETURN_IF_ERROR(internal::JsonParseArray(
            j,
            [](std::ptrdiff_t size) {
              return internal::JsonValidateArrayLength(size, 2);
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              TENSORSTORE_ASSIGN_OR_RETURN(values[i], DecodeFloat(v));
              return absl::OkStatus();
            }));
        fill_values[0] =
            MakeCopy(MakeScalarArrayView(complex128_t(values[0], values[1])),
                     c_order, field.data_type)
                .value();
        return fill_values;
      }
    }
  }
  // Decode as Base64
  std::string b64_decoded;
  if (!j.is_string() ||
      !absl::Base64Unescape(j.get<std::string>(), &b64_decoded) ||
      static_cast<Index>(b64_decoded.size()) != dtype.bytes_per_outer_element) {
    return absl::InvalidArgumentError(
        StrCat("Expected ", dtype.bytes_per_outer_element,
               " base64-encoded bytes, but received: ", j.dump()));
  }
  for (size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    auto& field = dtype.fields[field_i];
    DataType r = field.data_type;
    auto fill_value = AllocateArray(field.field_shape, ContiguousLayoutOrder::c,
                                    default_init, r);
    internal::DecodeArray(
        ArrayView<const void>(
            {static_cast<const void*>(b64_decoded.data() + field.byte_offset),
             r},
            fill_value.layout()),
        field.endian, fill_value);
    fill_values[field_i] = std::move(fill_value);
  }
  return fill_values;
}

::nlohmann::json EncodeFloat(double value) {
  if (std::isnan(value)) return "NaN";
  if (value == std::numeric_limits<double>::infinity()) return "Infinity";
  if (value == -std::numeric_limits<double>::infinity()) return "-Infinity";
  return value;
}

::nlohmann::json EncodeFillValue(
    const ZarrDType& dtype, span<const SharedArray<const void>> fill_values) {
  assert(dtype.fields.size() == static_cast<size_t>(fill_values.size()));
  if (!dtype.has_fields) {
    assert(dtype.fields.size() == 1);
    const auto& field = dtype.fields[0];
    const auto& fill_value = fill_values[0];
    if (!fill_value.valid()) return nullptr;
    const char type_indicator = field.encoded_dtype[1];
    switch (type_indicator) {
      case 'f': {
        double value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return EncodeFloat(value);
      }
      case 'c': {
        complex128_t value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return ::nlohmann::json::array_t{EncodeFloat(value.real()),
                                         EncodeFloat(value.imag())};
      }
      case 'i':
      case 'u':
      case 'b': {
        ::nlohmann::json value;
        TENSORSTORE_CHECK_OK(
            CopyConvertedArray(fill_value, MakeScalarArrayView(value)));
        return value;
      }
    }
  }
  // Compute base-64 encoding of fill values.
  std::vector<unsigned char> buffer(dtype.bytes_per_outer_element);
  for (size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    const auto& field = dtype.fields[field_i];
    const auto& fill_value = fill_values[field_i];
    if (!fill_value.valid()) return nullptr;
    DataType r = field.data_type;
    Array<void> encoded_fill_value(
        {static_cast<void*>(buffer.data() + field.byte_offset), r},
        field.field_shape);
    internal::EncodeArray(fill_value, encoded_fill_value, field.endian);
  }
  std::string b64_encoded;
  absl::Base64Escape(
      absl::string_view(reinterpret_cast<const char*>(buffer.data()),
                        buffer.size()),
      &b64_encoded);
  return b64_encoded;
}

Result<ZarrChunkLayout> ComputeChunkLayout(const ZarrDType& dtype,
                                           ContiguousLayoutOrder order,
                                           span<const Index> chunk_shape) {
  ZarrChunkLayout layout;
  layout.fields.resize(dtype.fields.size());
  layout.num_outer_elements = ProductOfExtents(chunk_shape);
  if (layout.num_outer_elements == std::numeric_limits<Index>::max()) {
    return absl::InvalidArgumentError(
        StrCat("Product of chunk dimensions ", chunk_shape, " is too large"));
  }
  if (internal::MulOverflow(dtype.bytes_per_outer_element,
                            layout.num_outer_elements,
                            &layout.bytes_per_chunk)) {
    return absl::InvalidArgumentError(
        StrCat("Total number of bytes per chunk is too large"));
  }

  for (std::size_t field_i = 0; field_i < dtype.fields.size(); ++field_i) {
    auto& field = dtype.fields[field_i];
    auto& field_layout = layout.fields[field_i];
    const DimensionIndex inner_rank = field.field_shape.size();
    const DimensionIndex total_rank = chunk_shape.size() + inner_rank;
    const auto initialize_layout = [&](StridedLayout<>* strided_layout,
                                       Index outer_element_stride) {
      strided_layout->set_rank(total_rank);
      std::copy(field.field_shape.begin(), field.field_shape.end(),
                std::copy(chunk_shape.begin(), chunk_shape.end(),
                          strided_layout->shape().begin()));
      // Compute strides for inner field dimensions.
      ComputeStrides(ContiguousLayoutOrder::c, field.data_type->size,
                     strided_layout->shape().last(inner_rank),
                     strided_layout->byte_strides().last(inner_rank));
      // Compute strides for outer dimensions.
      ComputeStrides(order, outer_element_stride, chunk_shape,
                     strided_layout->byte_strides().first(chunk_shape.size()));
    };
    initialize_layout(&field_layout.decoded_chunk_layout, field.num_bytes);
    initialize_layout(&field_layout.encoded_chunk_layout,
                      dtype.bytes_per_outer_element);
  }
  return layout;
}

namespace {
Status ParseIndexVector(const ::nlohmann::json& value,
                        absl::optional<DimensionIndex>* rank,
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
                       absl::optional<DimensionIndex>* rank,
                       std::vector<Index>* shape) {
  return ParseIndexVector(value, rank, shape, /*min_value=*/1,
                          /*max_value=*/kInfIndex);
}

Status ParseShape(const ::nlohmann::json& value,
                  absl::optional<DimensionIndex>* rank,
                  std::vector<Index>* shape) {
  return ParseIndexVector(value, rank, shape, /*min_value=*/0,
                          /*max_value=*/kInfIndex);
}

Status ParseFilters(const nlohmann::json& value) {
  if (!value.is_null()) {
    return absl::InvalidArgumentError("Filters not supported");
  }
  return absl::OkStatus();
}

Result<std::uint64_t> ParseZarrFormat(const nlohmann::json& value) {
  int result;
  if (auto status = internal::JsonRequireInteger(value, &result,
                                                 /*strict=*/true, 2, 2);
      status.ok()) {
    return result;
  } else {
    return status;
  }
}

Status ParseMetadata(const nlohmann::json& j, ZarrMetadata* metadata) {
  absl::optional<DimensionIndex> rank;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
      j, {"zarr_format", "shape", "chunks", "dtype", "compressor", "fill_value",
          "order", "filters"}));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "zarr_format", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata->zarr_format,
                                     internal_zarr::ParseZarrFormat(value));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "shape", [&](const ::nlohmann::json& value) {
        return ParseShape(value, &rank, &metadata->shape);
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "chunks", [&](const ::nlohmann::json& value) {
        return ParseChunkShape(value, &rank, &metadata->chunks);
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "dtype", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata->dtype, ParseDType(value));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "compressor", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata->compressor,
                                     Compressor::FromJson(value));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "fill_value", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata->fill_values,
                                     ParseFillValue(value, metadata->dtype));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "order", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata->order, ParseOrder(value));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, "filters", [&](const ::nlohmann::json& value) {
        return internal_zarr::ParseFilters(value);
      }));

  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->chunk_layout,
      ComputeChunkLayout(metadata->dtype, metadata->order, metadata->chunks));

  return absl::OkStatus();
}

void to_json(::nlohmann::json& out,  // NOLINT,
             const ZarrMetadata& metadata) {
  out = ::nlohmann::json::object_t{
      {"zarr_format", metadata.zarr_format},
      {"shape", metadata.shape},
      {"chunks", metadata.chunks},
      {"dtype", metadata.dtype},
      {"compressor", ::nlohmann::json(metadata.compressor)},
      {"fill_value", EncodeFillValue(metadata.dtype, metadata.fill_values)},
      {"order", EncodeOrder(metadata.order)},
      {"filters", nullptr},
  };
}

// Two decoding strategies:  raw decoder and custom decoder.  Initially we will
// only support raw decoder.

Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
    const ZarrMetadata& metadata, absl::Cord buffer) {
  const size_t num_fields = metadata.dtype.fields.size();
  if (metadata.compressor) {
    absl::Cord decoded;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Decode(
        buffer, &decoded, metadata.dtype.bytes_per_outer_element));
    buffer = std::move(decoded);
  }
  if (static_cast<Index>(buffer.size()) !=
      metadata.chunk_layout.bytes_per_chunk) {
    return absl::InvalidArgumentError(StrCat(
        "Uncompressed chunk is ", buffer.size(), " bytes, but should be ",
        metadata.chunk_layout.bytes_per_chunk, " bytes"));
  }
  absl::InlinedVector<SharedArrayView<const void>, 1> field_arrays(num_fields);

  bool must_copy = false;
  // First, attempt to create arrays that reference the cord without copying.
  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = metadata.dtype.fields[field_i];
    const auto& field_layout = metadata.chunk_layout.fields[field_i];
    field_arrays[field_i] = internal::TryViewCordAsArray(
        buffer, field.byte_offset, field.data_type, field.endian,
        field_layout.encoded_chunk_layout);
    if (!field_arrays[field_i].valid()) {
      must_copy = true;
      break;
    }
  }
  if (must_copy) {
    auto flat_buffer = buffer.Flatten();
    for (size_t field_i = 0; field_i < num_fields; ++field_i) {
      const auto& field = metadata.dtype.fields[field_i];
      const auto& field_layout = metadata.chunk_layout.fields[field_i];
      ArrayView<const void> source_array{
          ElementPointer<const void>(
              static_cast<const void*>(flat_buffer.data() + field.byte_offset),
              field.data_type),
          field_layout.encoded_chunk_layout};
      field_arrays[field_i] = internal::CopyAndDecodeArray(
          source_array, field.endian, field_layout.decoded_chunk_layout);
    }
  }
  return field_arrays;
}

Result<absl::Cord> EncodeChunk(const ZarrMetadata& metadata,
                               span<const ArrayView<const void>> components) {
  const size_t num_fields = metadata.dtype.fields.size();
  internal::FlatCordBuilder output_builder(
      metadata.chunk_layout.bytes_per_chunk);
  ByteStridedPointer<void> data_ptr = output_builder.data();
  for (size_t field_i = 0; field_i < num_fields; ++field_i) {
    const auto& field = metadata.dtype.fields[field_i];
    const auto& field_layout = metadata.chunk_layout.fields[field_i];
    ArrayView<void> encoded_array{
        {data_ptr + field.byte_offset, field.data_type},
        field_layout.encoded_chunk_layout};
    internal::EncodeArray(components[field_i], encoded_array, field.endian);
  }
  auto output = std::move(output_builder).Build();
  if (metadata.compressor) {
    absl::Cord encoded;
    TENSORSTORE_RETURN_IF_ERROR(metadata.compressor->Encode(
        output, &encoded, metadata.dtype.bytes_per_outer_element));
    return encoded;
  }
  return output;
}

bool IsMetadataCompatible(const ZarrMetadata& a, const ZarrMetadata& b) {
  // Rank must be the same.
  if (a.shape.size() != b.shape.size()) return false;
  auto a_json = ::nlohmann::json(a);
  auto b_json = ::nlohmann::json(b);
  // Shape is allowed to differ.
  a_json.erase("shape");
  b_json.erase("shape");
  return a_json == b_json;
}

void EncodeCacheKeyAdl(std::string* out, const ZarrMetadata& metadata) {
  auto json = ::nlohmann::json(metadata);
  json["shape"] = metadata.shape.size();
  out->append(json.dump());
}

}  // namespace internal_zarr
}  // namespace tensorstore
