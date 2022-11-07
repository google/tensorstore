
#include "tensorstore/driver/ometiff/metadata.h"

// ToDo - Clean up headers
#include <iostream>
#include <optional>
#include <tiffio.h>
#include <tiffio.hxx>
#include <sstream>
#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"

namespace tensorstore {
namespace internal_ometiff {
using internal::MetadataMismatchError;
namespace jb = tensorstore::internal_json_binding;
namespace {

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

absl::Status ValidateMetadata(OmeTiffMetadata& metadata) {
  InitializeContiguousLayout(c_order, metadata.dtype.size(),
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
        jb::Member("dimOrder",
                   jb::Projection(&T::dim_order)),
        jb::Member(
            "dataType",
            jb::Projection(&T::dtype, maybe_optional(jb::Validate(
                                          [](const auto& options, auto* obj) {
                                            return ValidateDataType(*obj);
                                          },
                                          jb::DataTypeJsonBinder)))),
            jb::Projection(&T::extra_attributes)
        )(is_loading, options, obj, j);
  };
};

}  // namespace

size_t OmeTiffMetadata::GetIfdIndex(size_t z, size_t c, size_t t) const{
	size_t nz=shape[2], nc=shape[1], nt=shape[0], ifd_offset;
  auto it = ifd_lookup_table.find(std::make_tuple(0,0,0));
  if (it == ifd_lookup_table.end()) {
    ifd_offset = 0;
  } else {
    ifd_offset = it->second;
  }
  
  it = ifd_lookup_table.find(std::make_tuple(z,c,t));
  if (it!=ifd_lookup_table.end()){return it->second;}
	else {
		switch (dim_order)
		{
			case 1:
				return nz*nt*c + nz*t + z + ifd_offset;
				break;
			case 2:
				return nz*nc*t + nz*c + z + ifd_offset;
				break;
			case 4:
				return nt*nc*z + nt*c + t + ifd_offset;
				break;
			case 8:
				return nt*nz*c + nt*z + t + ifd_offset;
				break;
			case 16:
				return nc*nt*z + nc*t + c + ifd_offset;
				break;
			case 32:
				return nc*nz*t + nc*z + c + ifd_offset;
				break;
			
			default:
				return nz*nt*c + nz*t + z + ifd_offset;
				break;
		}
	}
}

std::string OmeTiffMetadata::GetCompatibilityKey() const {
  // need to figure out what goes here
  ::nlohmann::json::object_t obj;
  span<const Index> chunk_shape = chunk_layout.shape();
  obj.emplace("blockSize", ::nlohmann::json::array_t(chunk_shape.begin(),
                                                     chunk_shape.end()));
  obj.emplace("dataType", dtype.name());
  return ::nlohmann::json(obj).dump();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    OmeTiffMetadata,
    jb::Validate([](const auto& options,
                    auto* obj) { return ValidateMetadata(*obj); },
                 MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(OmeTiffMetadataConstraints,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))

Result<std::shared_ptr<const OmeTiffMetadata>> GetNewMetadata(
    const OmeTiffMetadataConstraints& metadata_constraints,
    const Schema& schema) {
  auto metadata = std::make_shared<OmeTiffMetadata>();

  // Set domain
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetEffectiveDomain(metadata_constraints, schema));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }
  const DimensionIndex rank = metadata->rank = domain.rank();
  metadata->shape.assign(domain.shape().begin(), domain.shape().end());

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
  if(metadata_constraints.dim_order){
    metadata->dim_order = *metadata_constraints.dim_order;
  }
  
  metadata->extra_attributes = metadata_constraints.extra_attributes;
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(*metadata, schema));
  return metadata;
}

absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout) {
  // ToDo - Need to understand and reimplement
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint{rank}));
  rank = chunk_layout.rank();
  if (rank == dynamic_rank) return absl::OkStatus();

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

Result<SharedArrayView<const void>> DecodeChunk(const OmeTiffMetadata& metadata,
                                                absl::Cord buffer) {


    auto decoded_array =
        internal::TryViewCordAsArray(buffer, 0, metadata.dtype,
                                     endian::little, metadata.chunk_layout);
    if (decoded_array.valid()){
      return decoded_array;
    }
    else {
    SharedArrayView<void> full_decoded_array(
      internal::AllocateAndConstructSharedElements(
          metadata.chunk_layout.num_elements(), value_init, metadata.dtype),
      metadata.chunk_layout);
      return full_decoded_array;   
    }

}

Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const OmeTiffMetadata& metadata,
                               ArrayView<const void> array) {
  return absl::UnimplementedError(
            "Writing OMETiff is not supported");
}

absl::Status ValidateMetadata(const OmeTiffMetadata& metadata,
                              const OmeTiffMetadataConstraints& constraints) {
  if (constraints.shape && !absl::c_equal(metadata.shape, *constraints.shape)) {
    return MetadataMismatchError("dimensions", *constraints.shape,
                                 metadata.shape);
  }

  if (constraints.dim_order && *constraints.dim_order!=metadata.dim_order) {
    return MetadataMismatchError("dimOrder", *constraints.dim_order,
                                 metadata.dim_order);
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
  return internal::ValidateMetadataSubset(constraints.extra_attributes,
                                          metadata.extra_attributes);
  //return absl::OkStatus();
}

Result<IndexDomain<>> GetEffectiveDomain(DimensionIndex rank,
                                         std::optional<span<const Index>> shape,
                                         const Schema& schema) {
  auto domain = schema.domain();
  if (!shape && !domain.valid()) {
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

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(domain,
                               MergeIndexDomains(domain, domain_from_metadata),
                               tensorstore::MaybeAnnotateStatus(
                                   _, "Mismatch between metadata and schema"));
  return WithImplicitDimensions(domain, false, true);
  return domain;
}

Result<IndexDomain<>> GetEffectiveDomain(
    const OmeTiffMetadataConstraints& metadata_constraints,
    const Schema& schema) {
  return GetEffectiveDomain(metadata_constraints.rank,
                            metadata_constraints.shape, schema);
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
    const OmeTiffMetadataConstraints& metadata_constraints,
    const Schema& schema) {
  assert(RankConstraint::EqualOrUnspecified(metadata_constraints.rank,
                                            schema.rank()));
  return GetEffectiveChunkLayout(
      std::max(metadata_constraints.rank, schema.rank().rank),
      metadata_constraints.chunk_shape, schema);
}

absl::Status ValidateMetadataSchema(const OmeTiffMetadata& metadata,
                                    const Schema& schema) {
  if (!RankConstraint::EqualOrUnspecified(metadata.rank, schema.rank())) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Rank specified by schema (", schema.rank(),
        ") does not match rank specified by metadata (", metadata.rank, ")"));
  }

  if (schema.domain().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        GetEffectiveDomain(metadata.rank, metadata.shape, schema));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(metadata.dtype, dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("dtype from metadata (", metadata.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
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
    return absl::InvalidArgumentError(
        "fill_value not supported by OmeTiff format");
  }

  return absl::OkStatus();
}

absl::Status ValidateDataType(DataType dtype) {
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(
        StrCat(dtype, " data type is not one of the supported data types: ",
               GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}
}  // namespace internal_ometiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_ometiff::OmeTiffMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_ometiff::OmeTiffMetadataConstraints>())