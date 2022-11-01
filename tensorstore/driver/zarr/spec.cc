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

#include "absl/status/status.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr {

using ::tensorstore::internal::MetadataMismatchError;
namespace jb = tensorstore::internal_json_binding;

CodecSpec ZarrCodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<ZarrCodecSpec>(*this);
}

absl::Status ZarrCodecSpec::DoMergeFrom(
    const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(ZarrCodecSpec)) {
    return absl::InvalidArgumentError("");
  }
  auto& other = static_cast<const ZarrCodecSpec&>(other_base);
  // Set filters if set in either spec.
  if (other.filters) {
    filters = nullptr;
  }

  if (other.compressor) {
    if (!compressor) {
      compressor = other.compressor;
    } else if (!internal_json::JsonSame(::nlohmann::json(*compressor),
                                        ::nlohmann::json(*other.compressor))) {
      return absl::InvalidArgumentError("\"compressor\" does not match");
    }
  }
  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ZarrCodecSpec,
    jb::Sequence(
        jb::Member("compressor", jb::Projection(&ZarrCodecSpec::compressor)),
        jb::Member("filters", jb::Projection(&ZarrCodecSpec::filters))))

namespace {
const internal::CodecSpecRegistration<ZarrCodecSpec> encoding_registration;

}  // namespace

void GetChunkInnerOrder(DimensionIndex chunked_rank,
                        ContiguousLayoutOrder order,
                        span<DimensionIndex> permutation) {
  assert(chunked_rank <= permutation.size());
  const DimensionIndex inner_rank = permutation.size() - chunked_rank;
  std::iota(permutation.begin(), permutation.end(), DimensionIndex(0));
  if (order == fortran_order) {
    std::reverse(permutation.begin(), permutation.end() - inner_rank);
  }
}

absl::Status ValidateMetadata(const ZarrMetadata& metadata,
                              const ZarrPartialMetadata& constraints) {
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
    return MetadataMismatchError("order",
                                 tensorstore::StrCat(*constraints.order),
                                 tensorstore::StrCat(metadata.order));
  }
  if (constraints.dtype && ::nlohmann::json(*constraints.dtype) !=
                               ::nlohmann::json(metadata.dtype)) {
    return MetadataMismatchError("dtype", *constraints.dtype, metadata.dtype);
  }
  if (constraints.fill_value) {
    auto a = EncodeFillValue(metadata.dtype, *constraints.fill_value);
    auto b = EncodeFillValue(metadata.dtype, metadata.fill_value);
    if (a != b) {
      return MetadataMismatchError("fill_value", a, b);
    }
  }
  if (constraints.dimension_separator && metadata.dimension_separator &&
      *constraints.dimension_separator != *metadata.dimension_separator) {
    return MetadataMismatchError("dimension_separator",
                                 *constraints.dimension_separator,
                                 *metadata.dimension_separator);
  }
  return absl::OkStatus();
}

Result<ZarrMetadataPtr> GetNewMetadata(
    const ZarrPartialMetadata& partial_metadata,
    const SelectedField& selected_field, const Schema& schema) {
  ZarrMetadataPtr metadata = std::make_shared<ZarrMetadata>();
  metadata->zarr_format = partial_metadata.zarr_format.value_or(2);
  metadata->dimension_separator = partial_metadata.dimension_separator.value_or(
      DimensionSeparator::kDotSeparated);

  // Determine the new zarr metadata based on `partial_metadata` and `schema`.
  // Note that zarr dtypes can be NumPy "structured data types"
  // (https://numpy.org/doc/stable/user/basics.rec.html), which can specify
  // multiple named fields, each of which may itself be an array field.  In that
  // case, the full domain of the field has additional dimensions beyond the
  // chunked dimensions constrained by `partial_metadata.chunks` and
  // `partial_metadata.shape`.  Therefore, we must determine the dtype first
  // before validating the domain.

  size_t selected_field_index = 0;
  if (partial_metadata.dtype) {
    // If a zarr dtype is specified explicitly, determine the field index.  If a
    // multi-field zarr dtype is desired, it must be specified explicitly.
    TENSORSTORE_ASSIGN_OR_RETURN(
        selected_field_index,
        GetFieldIndex(*partial_metadata.dtype, selected_field));
    metadata->dtype = *partial_metadata.dtype;
  } else {
    if (!selected_field.empty()) {
      return absl::InvalidArgumentError(
          "\"dtype\" must be specified in \"metadata\" if \"field\" is "
          "specified");
    }
    if (!schema.dtype().valid()) {
      return absl::InvalidArgumentError("\"dtype\" must be specified");
    }
    // Choose a zarr dtype from `schema.dtype()`.
    metadata->dtype.fields.resize(1);
    metadata->dtype.has_fields = false;
    auto& field = metadata->dtype.fields[0];
    TENSORSTORE_ASSIGN_OR_RETURN(
        static_cast<ZarrDType::BaseDType&>(field),
        internal_zarr::ChooseBaseDType(schema.dtype()));
    TENSORSTORE_RETURN_IF_ERROR(ValidateDType(metadata->dtype));
  }
  auto& field = metadata->dtype.fields[selected_field_index];

  SpecRankAndFieldInfo info;
  info.full_rank = schema.rank();
  info.chunked_rank = partial_metadata.rank;
  info.field = &field;
  TENSORSTORE_RETURN_IF_ERROR(ValidateSpecRankAndFieldInfo(info));

  // Determine domain.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetDomainFromMetadata(info, partial_metadata.shape, schema),
      tensorstore::MaybeAnnotateStatus(_, "Invalid domain"));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }
  assert(info.chunked_rank != dynamic_rank);
  metadata->shape.assign(domain.shape().begin(),
                         domain.shape().begin() + info.chunked_rank);
  metadata->rank = info.chunked_rank;

  // Determine chunk shape.
  ChunkLayout chunk_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      info, partial_metadata.chunks, partial_metadata.order, chunk_layout));

  {
    Index ignored_offset[kMaxRank];
    Index chunk_shape[kMaxRank];
    TENSORSTORE_RETURN_IF_ERROR(internal::ChooseReadWriteChunkGrid(
        chunk_layout, domain.box(),
        MutableBoxView<>(info.full_rank, ignored_offset, chunk_shape)));
    metadata->chunks.assign(chunk_shape, chunk_shape + info.chunked_rank);
  }

  // Determine compressor.
  auto codec_spec = internal::CodecDriverSpec::Make<ZarrCodecSpec>();
  if (partial_metadata.compressor) {
    codec_spec->compressor = partial_metadata.compressor;
  }
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  if (codec_spec->compressor) {
    metadata->compressor = std::move(*codec_spec->compressor);
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(metadata->compressor,
                                 Compressor::FromJson({{"id", "blosc"}}));
  }

  metadata->filters = nullptr;

  // Determine storage order within chunk.
  {
    DimensionIndex order[kMaxRank];
    metadata->order = c_order;
    if (auto inner_order = chunk_layout.inner_order(); inner_order.valid()) {
      GetChunkInnerOrder(metadata->rank, c_order, span(order, info.full_rank));
      if (std::equal(inner_order.begin(), inner_order.end(), order)) {
        metadata->order = c_order;
      } else {
        GetChunkInnerOrder(metadata->rank, fortran_order,
                           span(order, info.full_rank));
        if (std::equal(inner_order.begin(), inner_order.end(), order)) {
          metadata->order = fortran_order;
        } else if (inner_order.hard_constraint) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Invalid \"inner_order\" constraint: ", inner_order));
        }
      }
    }
  }

  // Determine fill value.
  if (partial_metadata.fill_value) {
    metadata->fill_value = *partial_metadata.fill_value;
  } else if (auto fill_value = schema.fill_value(); fill_value.valid()) {
    const auto status = [&] {
      if (metadata->dtype.fields.size() > 1) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Cannot specify fill_value through schema "
                                "for structured zarr data type ",
                                ::nlohmann::json(metadata->dtype).dump()));
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto broadcast_fill_value,
          tensorstore::BroadcastArray(fill_value, field.field_shape));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto converted_fill_value,
          tensorstore::MakeCopy(std::move(broadcast_fill_value),
                                skip_repeated_elements, field.dtype));
      metadata->fill_value.push_back(std::move(converted_fill_value));
      return absl::OkStatus();
    }();
    TENSORSTORE_RETURN_IF_ERROR(
        status, tensorstore::MaybeAnnotateStatus(_, "Invalid fill_value"));
  } else {
    metadata->fill_value.resize(metadata->dtype.fields.size());
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateMetadataSchema(*metadata, selected_field_index, schema));
  return metadata;
}

absl::Status ValidateSpecRankAndFieldInfo(SpecRankAndFieldInfo& info) {
  if (info.field) {
    info.field_rank = info.field->field_shape.size();
  }

  if (info.full_rank == dynamic_rank) {
    info.full_rank = RankConstraint::Add(info.chunked_rank, info.field_rank);
    if (info.full_rank != dynamic_rank) {
      TENSORSTORE_RETURN_IF_ERROR(ValidateRank(info.full_rank));
    }
  }

  if (!RankConstraint::LessEqualOrUnspecified(info.chunked_rank,
                                              info.full_rank) ||
      !RankConstraint::LessEqualOrUnspecified(info.field_rank,
                                              info.full_rank) ||
      !RankConstraint::EqualOrUnspecified(
          info.full_rank,
          RankConstraint::Add(info.chunked_rank, info.field_rank))) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Rank specified by schema (", info.full_rank,
                            ") is not compatible with metadata"));
  }

  if (info.chunked_rank == dynamic_rank) {
    info.chunked_rank =
        RankConstraint::Subtract(info.full_rank, info.field_rank);
  }
  if (info.field_rank == dynamic_rank) {
    info.field_rank =
        RankConstraint::Subtract(info.full_rank, info.chunked_rank);
  }

  return absl::OkStatus();
}

Result<SpecRankAndFieldInfo> GetSpecRankAndFieldInfo(
    const ZarrPartialMetadata& metadata, const SelectedField& selected_field,
    const Schema& schema) {
  SpecRankAndFieldInfo info;

  info.full_rank = schema.rank();
  info.chunked_rank = metadata.rank;

  if (metadata.dtype) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        size_t field_index, GetFieldIndex(*metadata.dtype, selected_field));
    info.field = &metadata.dtype->fields[field_index];
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateSpecRankAndFieldInfo(info));

  return info;
}

SpecRankAndFieldInfo GetSpecRankAndFieldInfo(const ZarrMetadata& metadata,
                                             size_t field_index) {
  SpecRankAndFieldInfo info;
  info.chunked_rank = metadata.rank;
  info.field = &metadata.dtype.fields[field_index];
  info.field_rank = info.field->field_shape.size();
  info.full_rank = info.chunked_rank + info.field_rank;
  return info;
}

Result<IndexDomain<>> GetDomainFromMetadata(
    const SpecRankAndFieldInfo& info,
    std::optional<span<const Index>> metadata_shape, const Schema& schema) {
  const DimensionIndex full_rank = info.full_rank;
  auto schema_domain = schema.domain();
  if (full_rank == dynamic_rank ||
      (!schema_domain.valid() && ((info.chunked_rank != 0 && !metadata_shape) ||
                                  (info.field_rank != 0 && !info.field)))) {
    // If metadata only specifies constraints on a subset of the dimensions, and
    // the schema does not already specify a domain, don't return a
    // partially-specified domain.
    return schema_domain;
  }
  IndexDomainBuilder builder(full_rank);
  span<Index> shape = builder.shape();
  std::fill(shape.begin(), shape.end(), kInfIndex + 1);
  DimensionSet implicit_upper_bounds = true;
  if (metadata_shape) {
    assert(metadata_shape->size() == info.chunked_rank);
    std::copy_n(metadata_shape->begin(), info.chunked_rank, shape.begin());
  }
  if (info.field) {
    const DimensionIndex field_rank = info.field_rank;
    for (DimensionIndex i = 0; i < field_rank; ++i) {
      implicit_upper_bounds[full_rank - field_rank + i] = false;
    }
    std::copy_n(info.field->field_shape.begin(), field_rank,
                shape.end() - field_rank);
  }
  builder.implicit_upper_bounds(implicit_upper_bounds);
  TENSORSTORE_ASSIGN_OR_RETURN(auto domain, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(
      domain, MergeIndexDomains(std::move(domain), schema_domain));
  return WithImplicitDimensions(domain, false, implicit_upper_bounds);
}

absl::Status SetChunkLayoutFromMetadata(
    const SpecRankAndFieldInfo& info, std::optional<span<const Index>> chunks,
    std::optional<ContiguousLayoutOrder> order, ChunkLayout& chunk_layout) {
  const DimensionIndex full_rank = info.full_rank;
  if (full_rank == dynamic_rank) {
    return absl::OkStatus();
  }
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint(full_rank)));
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(full_rank))));

  // Set chunk_shape
  {
    Index chunk_shape[kMaxRank];
    std::fill_n(chunk_shape, full_rank, DimensionIndex(0));
    if (chunks) {
      assert(info.chunked_rank == chunks->size());
      std::copy_n(chunks->begin(), info.chunked_rank, chunk_shape);
    }
    if (info.field) {
      std::copy_n(info.field->field_shape.begin(), info.field_rank,
                  chunk_shape + info.chunked_rank);
    }
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
        ChunkLayout::ChunkShape(span(chunk_shape, full_rank))));
  }

  // Set inner_order
  if (order && info.chunked_rank != dynamic_rank) {
    DimensionIndex inner_order[kMaxRank];
    GetChunkInnerOrder(info.chunked_rank, *order, span(inner_order, full_rank));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
        ChunkLayout::InnerOrder(span(inner_order, full_rank))));
  }
  return absl::OkStatus();
}

CodecSpec GetCodecSpecFromMetadata(const ZarrMetadata& metadata) {
  auto codec = internal::CodecDriverSpec::Make<ZarrCodecSpec>();
  codec->compressor = metadata.compressor;
  codec->filters = nullptr;
  return codec;
}

absl::Status ValidateMetadataSchema(const ZarrMetadata& metadata,
                                    size_t field_index, const Schema& schema) {
  auto info = GetSpecRankAndFieldInfo(metadata, field_index);
  const auto& field = metadata.dtype.fields[field_index];

  if (!RankConstraint::EqualOrUnspecified(schema.rank(), info.full_rank)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Rank is ", info.full_rank,
                            ", but schema specifies rank of ", schema.rank()));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(dtype, field.dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("dtype from metadata (", field.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (schema.domain().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        GetDomainFromMetadata(info, metadata.shape, schema),
        tensorstore::MaybeAnnotateStatus(
            _, "domain from metadata does not match domain in schema"));
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecSpecFromMetadata(metadata);
    TENSORSTORE_RETURN_IF_ERROR(
        codec.MergeFrom(schema_codec),
        tensorstore::MaybeAnnotateStatus(
            _, "codec from metadata does not match codec in schema"));
  }

  if (auto schema_fill_value = schema.fill_value(); schema_fill_value.valid()) {
    const auto& fill_value = metadata.fill_value[field_index];
    if (!fill_value.valid()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Invalid fill_value: schema requires fill value of ",
          schema_fill_value, ", but metadata specifies no fill value"));
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto broadcast_fill_value,
        tensorstore::BroadcastArray(schema_fill_value, field.field_shape));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto converted_fill_value,
        tensorstore::MakeCopy(std::move(broadcast_fill_value),
                              skip_repeated_elements, field.dtype));
    if (!AreArraysSameValueEqual(converted_fill_value, fill_value)) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Invalid fill_value: schema requires fill value of ",
          converted_fill_value, ", but metadata specifies fill value of ",
          fill_value));
    }
  }

  if (auto chunk_layout = schema.chunk_layout();
      chunk_layout.rank() != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
        info, metadata.chunks, metadata.order, chunk_layout));
    if (chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError("codec_chunk_shape not supported");
    }
  }

  if (auto schema_units = schema.dimension_units(); schema_units.valid()) {
    if (std::any_of(schema_units.begin(), schema_units.end(),
                    [](const auto& u) { return u.has_value(); })) {
      return absl::InvalidArgumentError(
          "Dimension units not supported by zarr driver");
    }
  }

  return absl::OkStatus();
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
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Must specify a \"field\" that is one of: ", GetFieldNames(dtype)));
    }
    return 0;
  }
  if (!dtype.has_fields) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Requested field ", QuoteString(selected_field),
                            " but dtype does not have named fields"));
  }
  for (std::size_t field_index = 0; field_index < dtype.fields.size();
       ++field_index) {
    if (dtype.fields[field_index].name == selected_field) return field_index;
  }
  return absl::FailedPreconditionError(
      tensorstore::StrCat("Requested field ", QuoteString(selected_field),
                          " is not one of: ", GetFieldNames(dtype)));
}

Result<SelectedField> ParseSelectedField(const ::nlohmann::json& value) {
  if (value.is_null()) return std::string{};
  if (const auto* s = value.get_ptr<const std::string*>()) {
    if (!s->empty()) return *s;
  }
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Expected null or non-empty string, but received: ",
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
