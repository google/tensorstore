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
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_n5 {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal::MetadataMismatchError;

CodecSpec N5CodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<N5CodecSpec>(*this);
}

absl::Status N5CodecSpec::DoMergeFrom(
    const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(N5CodecSpec)) {
    return absl::InvalidArgumentError("");
  }
  auto& other = static_cast<const N5CodecSpec&>(other_base);
  if (other.compressor) {
    if (!compressor) {
      compressor = other.compressor;
    } else if (!internal_json::JsonSame(::nlohmann::json(*compressor),
                                        ::nlohmann::json(*other.compressor))) {
      return absl::InvalidArgumentError("\"compression\" does not match");
    }
  }
  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    N5CodecSpec,
    jb::Sequence(jb::Member("compression",
                            jb::Projection(&N5CodecSpec::compressor))))

namespace {

const internal::CodecSpecRegistration<N5CodecSpec> encoding_registration;

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,   DataTypeId::uint16_t, DataTypeId::uint32_t,
    DataTypeId::uint64_t,  DataTypeId::int8_t,   DataTypeId::int16_t,
    DataTypeId::int32_t,   DataTypeId::int64_t,  DataTypeId::float32_t,
    DataTypeId::float64_t,
};

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

absl::Status ValidateMetadata(N5Metadata& metadata) {
  // Per the specification:
  // https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
  //
  // Chunks cannot be larger than 2GB (2<sup>31</sup>Bytes).

  // While this limit may apply to compressed data, we will also apply it
  // to the uncompressed data.
  const Index max_num_elements =
      (static_cast<std::size_t>(1) << 31) / metadata.dtype.size();
  if (ProductOfExtents(span(metadata.chunk_shape)) > max_num_elements) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "\"blockSize\" of ", span(metadata.chunk_shape), " with data type of ",
        metadata.dtype, " exceeds maximum chunk size of 2GB"));
  }
  InitializeContiguousLayout(fortran_order, metadata.dtype.size(),
                             span<const Index>(metadata.chunk_shape),
                             &metadata.chunk_layout);
  return absl::OkStatus();
}

constexpr auto MetadataJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using T = internal::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }
    return jb::Object(
        jb::Member(
            "dimensions",
            jb::Projection(&T::shape, maybe_optional(jb::ShapeVector(rank)))),
        jb::Member("blockSize",
                   jb::Projection(&T::chunk_shape,
                                  maybe_optional(jb::ChunkShapeVector(rank)))),
        jb::Member(
            "dataType",
            jb::Projection(&T::dtype, maybe_optional(jb::Validate(
                                          [](const auto& options, auto* obj) {
                                            return ValidateDataType(*obj);
                                          },
                                          jb::DataTypeJsonBinder)))),
        jb::Member("compression", jb::Projection(&T::compressor)),
        jb::Member("axes", jb::Projection(
                               &T::axes,
                               maybe_optional(jb::DimensionLabelVector(rank)))),
        jb::Projection<&T::units_and_resolution>(jb::Sequence(
            jb::Member("units",
                       jb::Projection<&N5Metadata::UnitsAndResolution::units>(
                           jb::Optional(jb::DimensionIndexedVector(rank)))),
            jb::Member(
                "resolution",
                jb::Projection<&N5Metadata::UnitsAndResolution::resolution>(
                    jb::Optional(jb::DimensionIndexedVector(rank)))))),
        jb::Projection(&T::extra_attributes))(is_loading, options, obj, j);
  };
};

}  // namespace

std::string N5Metadata::GetCompatibilityKey() const {
  ::nlohmann::json::object_t obj;
  span<const Index> chunk_shape = chunk_layout.shape();
  obj.emplace("blockSize", ::nlohmann::json::array_t(chunk_shape.begin(),
                                                     chunk_shape.end()));
  obj.emplace("dataType", dtype.name());
  obj.emplace("compression", compressor);
  return ::nlohmann::json(obj).dump();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    N5Metadata, jb::Validate([](const auto& options,
                                auto* obj) { return ValidateMetadata(*obj); },
                             MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(N5MetadataConstraints,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))

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
  return                                      //
      2 +                                     // mode
      2 +                                     // number of dimensions
      sizeof(std::uint32_t) * metadata.rank;  // dimensions
}

Result<SharedArray<const void>> DecodeChunk(const N5Metadata& metadata,
                                            absl::Cord buffer) {
  // TODO(jbms): Currently, we do not check that `buffer.size()` is less than
  // the 2GiB limit, although we do implicitly check that the decoded array data
  // within the chunk is within the 2GiB limit due to the checks on the block
  // size.  Determine if this is an important limit.
  const std::size_t header_size = GetChunkHeaderSize(metadata);
  if (buffer.size() < header_size) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected header of length ", header_size,
                            ", but chunk has size ", buffer.size()));
  }
  std::unique_ptr<riegeli::Reader> reader =
      std::make_unique<riegeli::CordReader<>>(&buffer);
  uint16_t mode;
  uint16_t num_dims;
  if (!riegeli::ReadBigEndian16(*reader, mode) ||
      !riegeli::ReadBigEndian16(*reader, num_dims)) {
    return reader->status();
  }
  switch (mode) {
    case 0:  // default
      break;
    case 1:  // varlength
      return absl::InvalidArgumentError("varlength chunks not supported");
    default:
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Unexpected N5 chunk mode: ", mode));
  }
  if (num_dims != metadata.rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Received chunk with ", num_dims,
                            " dimensions but expected ", metadata.rank));
  }
  Index encoded_shape_buffer[kMaxRank];
  span<Index> encoded_shape(&encoded_shape_buffer[0], num_dims);
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    uint32_t size;
    if (!riegeli::ReadBigEndian32(*reader, size)) {
      return reader->status();
    }
    encoded_shape[i] = size;
  }
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    if (encoded_shape[i] > metadata.chunk_layout.shape()[i]) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Received chunk of size ", encoded_shape,
          " which exceeds blockSize of ", metadata.chunk_layout.shape()));
    }
  }
  if (metadata.compressor) {
    reader = metadata.compressor->GetReader(std::move(reader),
                                            metadata.dtype.size());
  }
  SharedArray<const void> decoded_array;
  if (absl::c_equal(encoded_shape, metadata.chunk_layout.shape())) {
    // Decode full array.
    TENSORSTORE_ASSIGN_OR_RETURN(
        decoded_array,
        internal::DecodeArrayEndian(*reader, metadata.dtype,
                                    metadata.chunk_layout.shape(), endian::big,
                                    fortran_order));
  } else {
    auto array = AllocateArray(metadata.chunk_layout.shape(), fortran_order,
                               value_init, metadata.dtype);
    ArrayView<void> partial_decoded_array(
        array.element_pointer(),
        StridedLayoutView<>{encoded_shape, array.byte_strides()});
    TENSORSTORE_RETURN_IF_ERROR(internal::DecodeArrayEndian(
        *reader, endian::big, fortran_order, partial_decoded_array));
    decoded_array = std::move(array);
  }
  if (!reader->VerifyEndAndClose()) {
    return reader->status();
  }
  return decoded_array;
}

Result<absl::Cord> EncodeChunk(const N5Metadata& metadata,
                               SharedArrayView<const void> array) {
  assert(absl::c_equal(metadata.chunk_layout.shape(), array.shape()));
  absl::Cord encoded;
  std::unique_ptr<riegeli::Writer> writer =
      std::make_unique<riegeli::CordWriter<>>(&encoded);

  // Write header
  // mode: 0x0 = default
  if (!riegeli::WriteBigEndian16(0, *writer) ||
      !riegeli::WriteBigEndian16(metadata.rank, *writer)) {
    return writer->status();
  }
  for (DimensionIndex i = 0; i < array.rank(); ++i) {
    if (!riegeli::WriteBigEndian32(array.shape()[i], *writer)) {
      return writer->status();
    }
  }
  if (metadata.compressor) {
    writer = metadata.compressor->GetWriter(std::move(writer),
                                            metadata.dtype.size());
  }
  // Always write chunks as full size, to avoid race conditions or data loss
  // in the event of a concurrent resize.
  if (!internal::EncodeArrayEndian(array, endian::big, fortran_order,
                                   *writer)) {
    return writer->status();
  }
  if (!writer->Close()) return writer->status();
  return encoded;
}

absl::Status ValidateMetadata(const N5Metadata& metadata,
                              const N5MetadataConstraints& constraints) {
  if (constraints.shape && !absl::c_equal(metadata.shape, *constraints.shape)) {
    return MetadataMismatchError("dimensions", *constraints.shape,
                                 metadata.shape);
  }
  if (constraints.axes && !absl::c_equal(metadata.axes, *constraints.axes)) {
    return MetadataMismatchError("axes", *constraints.axes, metadata.axes);
  }
  if (constraints.chunk_shape &&
      !absl::c_equal(metadata.chunk_layout.shape(), *constraints.chunk_shape)) {
    return MetadataMismatchError("blockSize", *constraints.chunk_shape,
                                 metadata.chunk_shape);
  }
  if (constraints.dtype && *constraints.dtype != metadata.dtype) {
    return MetadataMismatchError("dataType", constraints.dtype->name(),
                                 metadata.dtype.name());
  }
  if (constraints.compressor && ::nlohmann::json(*constraints.compressor) !=
                                    ::nlohmann::json(metadata.compressor)) {
    return MetadataMismatchError("compression", *constraints.compressor,
                                 metadata.compressor);
  }
  if (constraints.units_and_resolution.units &&
      metadata.units_and_resolution.units !=
          constraints.units_and_resolution.units) {
    return MetadataMismatchError(
        "units", *constraints.units_and_resolution.units,
        metadata.units_and_resolution.units
            ? ::nlohmann::json(*metadata.units_and_resolution.units)
            : ::nlohmann::json(::nlohmann::json::value_t::discarded));
  }
  if (constraints.units_and_resolution.resolution &&
      metadata.units_and_resolution.resolution !=
          constraints.units_and_resolution.resolution) {
    return MetadataMismatchError(
        "resolution", *constraints.units_and_resolution.resolution,
        metadata.units_and_resolution.resolution
            ? ::nlohmann::json(*metadata.units_and_resolution.resolution)
            : ::nlohmann::json(::nlohmann::json::value_t::discarded));
  }
  return internal::ValidateMetadataSubset(constraints.extra_attributes,
                                          metadata.extra_attributes);
}

Result<IndexDomain<>> GetEffectiveDomain(
    DimensionIndex rank, std::optional<span<const Index>> shape,
    std::optional<span<const std::string>> axes, const Schema& schema) {
  auto domain = schema.domain();
  if (!shape && !axes && !domain.valid()) {
    if (schema.rank() == 0) return {std::in_place, 0};
    // No information about the domain available.
    return {std::in_place};
  }

  // Rank is already validated by caller.
  assert(RankConstraint::EqualOrUnspecified(schema.rank(), rank));
  IndexDomainBuilder builder(std::max(schema.rank().rank, rank));
  if (shape) {
    builder.shape(*shape);
    builder.implicit_upper_bounds(true);
  } else {
    builder.origin(GetConstantVector<Index, 0>(builder.rank()));
  }
  if (axes) {
    builder.labels(*axes);
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(domain,
                               MergeIndexDomains(domain, domain_from_metadata),
                               tensorstore::MaybeAnnotateStatus(
                                   _, "Mismatch between metadata and schema"));
  return WithImplicitDimensions(domain, false, true);
  return domain;
}

Result<IndexDomain<>> GetEffectiveDomain(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema) {
  return GetEffectiveDomain(metadata_constraints.rank,
                            metadata_constraints.shape,
                            metadata_constraints.axes, schema);
}

absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout) {
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint{rank}));
  rank = chunk_layout.rank();
  if (rank == dynamic_rank) return absl::OkStatus();

  // n5 always uses Fortran (colexicographic) inner order
  {
    DimensionIndex inner_order[kMaxRank];
    for (DimensionIndex i = 0; i < rank; ++i) {
      inner_order[i] = rank - i - 1;
    }
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::InnerOrder(span(inner_order, rank))));
  }
  if (chunk_shape) {
    assert(chunk_shape->size() == rank);
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::ChunkShape(*chunk_shape)));
  }
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));
  return absl::OkStatus();
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    const Schema& schema) {
  auto chunk_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(
      SetChunkLayoutFromMetadata(rank, chunk_shape, chunk_layout));
  return chunk_layout;
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema) {
  assert(RankConstraint::EqualOrUnspecified(metadata_constraints.rank,
                                            schema.rank()));
  return GetEffectiveChunkLayout(
      std::max(metadata_constraints.rank, schema.rank().rank),
      metadata_constraints.chunk_shape, schema);
}

Result<internal::CodecDriverSpec::PtrT<N5CodecSpec>> GetEffectiveCodec(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema) {
  auto codec_spec = internal::CodecDriverSpec::Make<N5CodecSpec>();
  if (metadata_constraints.compressor) {
    codec_spec->compressor = *metadata_constraints.compressor;
  }
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  return codec_spec;
}

CodecSpec GetCodecFromMetadata(const N5Metadata& metadata) {
  auto codec_spec = internal::CodecDriverSpec::Make<N5CodecSpec>();
  codec_spec->compressor = metadata.compressor;
  return CodecSpec(std::move(codec_spec));
}

DimensionUnitsVector GetDimensionUnits(
    DimensionIndex metadata_rank,
    const N5Metadata::UnitsAndResolution& units_and_resolution) {
  if (metadata_rank == dynamic_rank) return {};
  DimensionUnitsVector dimension_units(metadata_rank);
  if (units_and_resolution.units) {
    assert(units_and_resolution.units->size() == metadata_rank);
    assert(!units_and_resolution.resolution ||
           units_and_resolution.resolution->size() == metadata_rank);
    for (DimensionIndex i = 0; i < metadata_rank; ++i) {
      dimension_units[i] = Unit(units_and_resolution.resolution
                                    ? (*units_and_resolution.resolution)[i]
                                    : 1.0,
                                (*units_and_resolution.units)[i]);
    }
  }
  return dimension_units;
}

namespace {
absl::Status ValidateDimensionUnitsResolution(
    span<const std::optional<Unit>> dimension_units,
    const N5Metadata::UnitsAndResolution& units_and_resolution) {
  if (!units_and_resolution.units && units_and_resolution.resolution) {
    // Since `units` wasn't specified, `GetDimensionUnits` did not set any
    // dimension units from the metadata.  But we can still validate the
    // resolution separately.
    for (DimensionIndex i = 0; i < dimension_units.size(); ++i) {
      const auto& unit = dimension_units[i];
      if (!unit) continue;
      if ((*units_and_resolution.resolution)[i] != unit->multiplier) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"resolution\" from metadata ",
            span(*units_and_resolution.resolution),
            " does not match dimension units from schema ",
            tensorstore::DimensionUnitsToString(dimension_units)));
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    DimensionIndex metadata_rank,
    const N5Metadata::UnitsAndResolution& units_and_resolution,
    Schema::DimensionUnits schema_units) {
  DimensionUnitsVector dimension_units =
      GetDimensionUnits(metadata_rank, units_and_resolution);
  if (schema_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::MergeDimensionUnits(dimension_units, schema_units));
    TENSORSTORE_RETURN_IF_ERROR(ValidateDimensionUnitsResolution(
        dimension_units, units_and_resolution));
  }
  return dimension_units;
}

Result<std::shared_ptr<const N5Metadata>> GetNewMetadata(
    const N5MetadataConstraints& metadata_constraints, const Schema& schema) {
  auto metadata = std::make_shared<N5Metadata>();

  // Set domain
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetEffectiveDomain(metadata_constraints, schema));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }
  const DimensionIndex rank = metadata->rank = domain.rank();
  metadata->shape.assign(domain.shape().begin(), domain.shape().end());
  metadata->axes.assign(domain.labels().begin(), domain.labels().end());

  // Set dtype
  auto dtype = schema.dtype();
  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(dtype));
  metadata->dtype = dtype;

  // Set chunk shape
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto chunk_layout, GetEffectiveChunkLayout(metadata_constraints, schema));
  metadata->chunk_shape.resize(rank);
  {
    Index chunk_origin[kMaxRank];
    TENSORSTORE_RETURN_IF_ERROR(tensorstore::internal::ChooseReadWriteChunkGrid(
        chunk_layout, domain.box(),
        MutableBoxView<>(rank, chunk_origin, metadata->chunk_shape.data())));
  }

  // Set compressor
  TENSORSTORE_ASSIGN_OR_RETURN(auto codec_spec,
                               GetEffectiveCodec(metadata_constraints, schema));
  if (codec_spec->compressor) {
    metadata->compressor = *codec_spec->compressor;
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        metadata->compressor,
        Compressor::FromJson({{"type", "blosc"},
                              {"cname", "lz4"},
                              {"clevel", 5},
                              {"shuffle", dtype.size() == 1 ? 2 : 1}}));
  }

  // Set `units_and_resolution`.
  {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto dimension_units,
        GetEffectiveDimensionUnits(metadata_constraints.rank,
                                   metadata_constraints.units_and_resolution,
                                   schema.dimension_units()));
    metadata->units_and_resolution = metadata_constraints.units_and_resolution;
    if (std::any_of(dimension_units.begin(), dimension_units.end(),
                    [](const auto& unit) { return unit.has_value(); })) {
      if (!metadata->units_and_resolution.units) {
        metadata->units_and_resolution.units.emplace(rank);
      }
      if (!metadata->units_and_resolution.resolution) {
        metadata->units_and_resolution.resolution.emplace(rank, 1);
      }
      for (DimensionIndex i = 0; i < rank; ++i) {
        const auto& unit = dimension_units[i];
        if (!unit) continue;
        (*metadata->units_and_resolution.resolution)[i] = unit->multiplier;
        (*metadata->units_and_resolution.units)[i] = unit->base_unit;
      }
    }
  }

  metadata->extra_attributes = metadata_constraints.extra_attributes;
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(*metadata, schema));
  return metadata;
}

absl::Status ValidateMetadataSchema(const N5Metadata& metadata,
                                    const Schema& schema) {
  if (!RankConstraint::EqualOrUnspecified(metadata.rank, schema.rank())) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Rank specified by schema (", schema.rank(),
        ") does not match rank specified by metadata (", metadata.rank, ")"));
  }

  if (schema.domain().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(GetEffectiveDomain(
        metadata.rank, metadata.shape, metadata.axes, schema));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(metadata.dtype, dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("dtype from metadata (", metadata.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecFromMetadata(metadata);
    TENSORSTORE_RETURN_IF_ERROR(
        codec.MergeFrom(schema_codec),
        internal::ConvertInvalidArgumentToFailedPrecondition(
            tensorstore::MaybeAnnotateStatus(
                _, "codec from metadata does not match codec in schema")));
  }

  if (schema.chunk_layout().rank() != dynamic_rank) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_layout,
        GetEffectiveChunkLayout(metadata.rank, metadata.chunk_shape, schema));
    if (chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError("codec_chunk_shape not supported");
    }
  }

  if (schema.fill_value().valid()) {
    return absl::InvalidArgumentError("fill_value not supported by N5 format");
  }

  if (auto schema_units = schema.dimension_units(); schema_units.valid()) {
    auto dimension_units =
        GetDimensionUnits(metadata.rank, metadata.units_and_resolution);
    DimensionUnitsVector schema_units_vector(schema_units);
    TENSORSTORE_RETURN_IF_ERROR(
        MergeDimensionUnits(schema_units_vector, dimension_units),
        internal::ConvertInvalidArgumentToFailedPrecondition(_));
    if (schema_units_vector != dimension_units) {
      return absl::FailedPreconditionError(
          tensorstore::StrCat("Dimension units in metadata ",
                              DimensionUnitsToString(dimension_units),
                              " do not match dimension units in schema ",
                              DimensionUnitsToString(schema_units)));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDataType(DataType dtype) {
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

}  // namespace internal_n5
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_n5::N5MetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_n5::N5MetadataConstraints>())
