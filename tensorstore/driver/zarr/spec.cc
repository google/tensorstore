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

#include "tensorstore/driver/zarr/spec.h"

#include "tensorstore/internal/json.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_zarr {

Result<ZarrPartialMetadata> ParsePartialMetadata(const ::nlohmann::json& j) {
  ZarrPartialMetadata metadata;
  absl::optional<DimensionIndex> rank;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
      j, {"zarr_format", "chunks", "shape", "compressor", "order", "dtype",
          "filters", "fill_value"}));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "zarr_format", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.zarr_format,
                                     internal_zarr::ParseZarrFormat(value));
        return absl::OkStatus();
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "chunks", [&](const ::nlohmann::json& value) {
        return internal_zarr::ParseChunkShape(value, &rank,
                                              &metadata.chunks.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "shape", [&](const ::nlohmann::json& value) {
        return internal_zarr::ParseShape(value, &rank,
                                         &metadata.shape.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "compressor", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            metadata.compressor, internal_zarr::Compressor::FromJson(value));
        return absl::OkStatus();
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "order", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.order,
                                     internal_zarr::ParseOrder(value));
        return absl::OkStatus();
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "dtype", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.dtype,
                                     internal_zarr::ParseDType(value));
        return absl::OkStatus();
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "filters", [&](const ::nlohmann::json& value) {
        return internal_zarr::ParseFilters(value);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "fill_value", [&](const ::nlohmann::json& value) {
        metadata.fill_value = value;
        return absl::OkStatus();
      }));
  return metadata;
}

namespace {
template <typename T>
Status MetadataMismatchError(const char* name, const T& expected,
                             const T& actual) {
  return absl::FailedPreconditionError(StrCat(
      "Expected ", QuoteString(name), " of ", ::nlohmann::json(expected).dump(),
      " but received: ", ::nlohmann::json(actual).dump()));
}
}  // namespace

Status ValidateMetadata(const ZarrMetadata& metadata,
                        const ZarrPartialMetadata& constraints) {
  if (constraints.zarr_format &&
      *constraints.zarr_format != metadata.zarr_format) {
    return MetadataMismatchError("zarr_format", *constraints.zarr_format,
                                 metadata.zarr_format);
  }
  if (constraints.shape && *constraints.shape != metadata.shape) {
    return MetadataMismatchError("shape", *constraints.shape, metadata.shape);
  }
  if (constraints.chunks && *constraints.chunks != metadata.chunks) {
    return MetadataMismatchError("chunks", *constraints.chunks,
                                 metadata.chunks);
  }
  if (constraints.compressor && ::nlohmann::json(*constraints.compressor) !=
                                    ::nlohmann::json(metadata.compressor)) {
    return MetadataMismatchError("compressor", *constraints.compressor,
                                 metadata.compressor);
  }
  if (constraints.order && *constraints.order != metadata.order) {
    return MetadataMismatchError("order", EncodeOrder(*constraints.order),
                                 EncodeOrder(metadata.order));
  }
  if (constraints.dtype && ::nlohmann::json(*constraints.dtype) !=
                               ::nlohmann::json(metadata.dtype)) {
    return MetadataMismatchError("dtype", *constraints.dtype, metadata.dtype);
  }
  if (constraints.fill_value) {
    auto encoded_fill_value =
        EncodeFillValue(metadata.dtype, metadata.fill_values);
    if (*constraints.fill_value != encoded_fill_value) {
      return MetadataMismatchError("fill_value", *constraints.fill_value,
                                   encoded_fill_value);
    }
  }
  return absl::OkStatus();
}

Result<ZarrMetadataPtr> GetNewMetadata(
    const ZarrPartialMetadata& partial_metadata,
    DataType data_type_constraint) {
  ZarrMetadataPtr metadata = std::make_shared<ZarrMetadata>();
  metadata->zarr_format = partial_metadata.zarr_format.value_or(2);
  if (!partial_metadata.shape) {
    return absl::InvalidArgumentError("\"shape\" must be specified");
  }
  metadata->shape = *partial_metadata.shape;

  // TODO(jbms): Handle special `true` and `false` values for `chunks` as the
  // zarr library does, and provide a default chunk size.
  if (!partial_metadata.chunks) {
    return absl::InvalidArgumentError("\"chunks\" must be specified");
  }
  metadata->chunks = *partial_metadata.chunks;

  // TODO(jbms): infer dtype from `data_type_constraint` if specified.
  if (!partial_metadata.dtype) {
    return absl::InvalidArgumentError("\"dtype\" must be specified");
  }
  metadata->dtype = *partial_metadata.dtype;
  // TODO(jbms): use default compressor as in zarr library (blosc)
  if (!partial_metadata.compressor) {
    return absl::InvalidArgumentError("\"compressor\" must be specified");
  }
  metadata->compressor = *partial_metadata.compressor;
  metadata->order = partial_metadata.order.value_or(c_order);
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->fill_values,
      ParseFillValue(partial_metadata.fill_value.value_or(nullptr),
                     metadata->dtype),
      tensorstore::MaybeAnnotateStatus(
          _, "Error parsing object member \"fill_value\""));
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->chunk_layout,
      ComputeChunkLayout(metadata->dtype, metadata->order, metadata->chunks));
  return metadata;
}

namespace {
std::string GetFieldNames(const ZarrDType& dtype) {
  std::vector<std::string> field_names;
  for (const auto& field : dtype.fields) {
    field_names.push_back(field.name);
  }
  return ::nlohmann::json(field_names).dump();
}
}  // namespace

Result<std::size_t> GetFieldIndex(const ZarrDType& dtype,
                                  const SelectedField& selected_field) {
  if (selected_field.empty()) {
    if (dtype.fields.size() != 1) {
      return absl::FailedPreconditionError(StrCat(
          "Must specify a \"field\" that is one of: ", GetFieldNames(dtype)));
    }
    return 0;
  }
  if (!dtype.has_fields) {
    return absl::FailedPreconditionError(
        StrCat("Requested field ", QuoteString(selected_field),
               " but dtype does not have named fields"));
  }
  for (std::size_t field_index = 0; field_index < dtype.fields.size();
       ++field_index) {
    if (dtype.fields[field_index].name == selected_field) return field_index;
  }
  return absl::FailedPreconditionError(
      StrCat("Requested field ", QuoteString(selected_field),
             " is not one of: ", GetFieldNames(dtype)));
}

Result<std::size_t> GetCompatibleField(const ZarrDType& dtype,
                                       DataType data_type_constraint,
                                       const SelectedField& selected_field) {
  TENSORSTORE_ASSIGN_OR_RETURN(std::size_t field_index,
                               GetFieldIndex(dtype, selected_field));
  auto& field = dtype.fields[field_index];
  if (data_type_constraint.valid() && data_type_constraint != field.data_type) {
    return absl::FailedPreconditionError(StrCat(
        "Expected field to have data type of ", data_type_constraint.name(),
        " but the actual data type is: ", field.data_type.name()));
  }
  return field_index;
}

Result<ChunkKeyEncoding> ParseKeyEncoding(const ::nlohmann::json& value) {
  if (const auto* s = value.get_ptr<const std::string*>()) {
    if (*s == "/") return ChunkKeyEncoding::kSlashSeparated;
    if (*s == ".") return ChunkKeyEncoding::kDotSeparated;
  }
  return absl::InvalidArgumentError(
      StrCat("Expected \".\" or \"/\", but received: ", value.dump()));
}

void to_json(::nlohmann::json& j,  // NOLINT
             ChunkKeyEncoding key_encoding) {
  j = (key_encoding == ChunkKeyEncoding::kSlashSeparated) ? "/" : ".";
}

Result<SelectedField> ParseSelectedField(const ::nlohmann::json& value) {
  if (value.is_null()) return std::string{};
  if (const auto* s = value.get_ptr<const std::string*>()) {
    if (!s->empty()) return *s;
  }
  return absl::InvalidArgumentError(
      StrCat("Expected null or non-empty string, but received: ",
             ::nlohmann::json(value).dump()));
}

SelectedField EncodeSelectedField(std::size_t field_index,
                                  const ZarrDType& dtype) {
  assert(field_index >= 0 && field_index < dtype.fields.size());
  const auto& field = dtype.fields[field_index];
  return field.name;
}

}  // namespace internal_zarr
}  // namespace tensorstore
