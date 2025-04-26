// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/driver/tiff/metadata.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_tiff {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal_tiff_kvstore::CompressionType;
using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::PlanarConfigType;
using ::tensorstore::internal_tiff_kvstore::SampleFormatType;
using ::tensorstore::internal_tiff_kvstore::TiffParseResult;

ABSL_CONST_INIT internal_log::VerboseFlag tiff_metadata_logging(
    "tiff_metadata");

CodecSpec TiffCodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<TiffCodecSpec>(*this);
}

absl::Status TiffCodecSpec::DoMergeFrom(
    const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(TiffCodecSpec)) {
    return absl::InvalidArgumentError("Cannot merge non-TIFF codec spec");
  }
  const auto& other = static_cast<const TiffCodecSpec&>(other_base);

  if (other.compression_type.has_value()) {
    if (!compression_type.has_value()) {
      compression_type = other.compression_type;
    } else if (*compression_type != *other.compression_type) {
      // Allow merging if one specifies 'raw' (kNone) and the other doesn't
      // specify? Or require exact match or one empty? Let's require exact match
      // or one empty.
      if (*compression_type != CompressionType::kNone &&
          *other.compression_type != CompressionType::kNone) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "TIFF compression type mismatch: existing=",
            static_cast<int>(*compression_type),
            ", new=", static_cast<int>(*other.compression_type)));
      }
      // If one is kNone and the other isn't, take the non-kNone one.
      if (*compression_type == CompressionType::kNone) {
        compression_type = other.compression_type;
      }
    }
  }
  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TiffCodecSpec,
    jb::Object(jb::Member(
        "compression", jb::Projection<&TiffCodecSpec::compression_type>(
                           jb::Optional(jb::Enum<CompressionType, std::string>({
                               {CompressionType::kNone, "raw"},
                               {CompressionType::kLZW, "lzw"},
                               {CompressionType::kDeflate, "deflate"},
                               {CompressionType::kPackBits, "packbits"}
                               // TODO: Add other supported types
                           }))))))

bool operator==(const TiffCodecSpec& a, const TiffCodecSpec& b) {
  // Two specs are equal if their compression_type members are equal.
  return a.compression_type == b.compression_type;
}

namespace {
const internal::CodecSpecRegistration<TiffCodecSpec> registration;

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,   DataTypeId::uint16_t, DataTypeId::uint32_t,
    DataTypeId::uint64_t,  DataTypeId::int8_t,   DataTypeId::int16_t,
    DataTypeId::int32_t,   DataTypeId::int64_t,  DataTypeId::float32_t,
    DataTypeId::float64_t,
    // Note: Complex types are typically not standard TIFF.
    // Note: Boolean might be mapped to uint8 with specific interpretation,
    //       but let's require explicit numeric types for now.
};

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

// Maps TIFF SampleFormat and BitsPerSample to TensorStore DataType.
Result<DataType> GetDataTypeFromTiff(const ImageDirectory& dir) {
  if (dir.samples_per_pixel == 0 || dir.bits_per_sample.empty() ||
      dir.sample_format.empty()) {
    return absl::FailedPreconditionError(
        "Incomplete TIFF metadata for data type");
  }
  // Accept either identical (most files) or uniformly 8‑bit unsigned channels
  auto uniform_bits = dir.bits_per_sample[0];
  auto uniform_format = dir.sample_format[0];
  for (size_t i = 1; i < dir.samples_per_pixel; ++i) {
    if (dir.bits_per_sample[i] != uniform_bits ||
        dir.sample_format[i] != uniform_format) {
      // allow common RGB 8‑bit + Alpha 8‑bit case
      if (uniform_bits == 8 && dir.bits_per_sample[i] == 8 &&
          uniform_format ==
              static_cast<uint16_t>(SampleFormatType::kUnsignedInteger) &&
          dir.sample_format[i] == uniform_format) {
        continue;
      }
      return absl::UnimplementedError(
          "Mixed bits/sample or sample_format is not supported yet");
    }
  }

  switch (uniform_format) {
    case static_cast<uint16_t>(SampleFormatType::kUnsignedInteger):
      if (uniform_bits == 8) return dtype_v<uint8_t>;
      if (uniform_bits == 16) return dtype_v<uint16_t>;
      if (uniform_bits == 32) return dtype_v<uint32_t>;
      if (uniform_bits == 64) return dtype_v<uint64_t>;
      break;
    case static_cast<uint16_t>(SampleFormatType::kSignedInteger):
      if (uniform_bits == 8) return dtype_v<int8_t>;
      if (uniform_bits == 16) return dtype_v<int16_t>;
      if (uniform_bits == 32) return dtype_v<int32_t>;
      if (uniform_bits == 64) return dtype_v<int64_t>;
      break;
    case static_cast<uint16_t>(SampleFormatType::kIEEEFloat):
      if (uniform_bits == 32) return dtype_v<tensorstore::dtypes::float32_t>;
      if (uniform_bits == 64) return dtype_v<tensorstore::dtypes::float64_t>;
      break;
    case static_cast<uint16_t>(
        SampleFormatType::kUndefined):  // Might be complex, not standard TIFF
      break;                            // Fall through to error
    default:
      break;
  }
  return absl::InvalidArgumentError(
      StrCat("Unsupported TIFF data type: bits=", uniform_bits,
             ", format=", uniform_format));
}

// Gets the shape and sets rank based on the ImageDirectory and
// PlanarConfiguration.
Result<std::vector<Index>> GetShapeAndRankFromTiff(const ImageDirectory& dir,
                                                   DimensionIndex& rank) {
  const bool chunky =
      dir.planar_config == static_cast<uint16_t>(PlanarConfigType::kChunky);
  const bool multi_channel = dir.samples_per_pixel > 1;

  if (chunky) {
    rank = multi_channel ? 3 : 2;
    std::vector<Index> shape = {static_cast<Index>(dir.height),
                                static_cast<Index>(dir.width)};
    if (multi_channel)
      shape.push_back(static_cast<Index>(dir.samples_per_pixel));
    return shape;
  } else {                         // planar == 2
    rank = multi_channel ? 3 : 2;  // (rare but legal: planar 1‑sample strips)
    std::vector<Index> shape;
    if (multi_channel)
      shape.push_back(static_cast<Index>(dir.samples_per_pixel));
    shape.push_back(static_cast<Index>(dir.height));
    shape.push_back(static_cast<Index>(dir.width));
    return shape;
  }
}

// Gets chunk shape based on ImageDirectory and PlanarConfiguration.
// Determines the chunk‑shape implied by the TIFF tags.
//
// For planar‑configuration images the channel dimension is represented
// as a size‑1 chunk axis so that every chunk contains a single C‑plane.
Result<std::vector<Index>> GetChunkShapeFromTiff(
    const ImageDirectory& directory, DimensionIndex resolved_rank,
    bool planar_dimension_leading) {
  Index tile_height = 0;
  Index tile_width = 0;

  if (directory.tile_width > 0 && directory.tile_height > 0) {
    tile_height = static_cast<Index>(directory.tile_height);
    tile_width = static_cast<Index>(directory.tile_width);
  } else {
    // Classic strips
    if (directory.rows_per_strip == 0) {
      return absl::InvalidArgumentError(
          "RowsPerStrip tag is zero while TileWidth/TileLength missing");
    }
    tile_height = static_cast<Index>(directory.rows_per_strip);
    tile_width = static_cast<Index>(directory.width);

    // RowsPerStrip must evenly partition the image height.
    if (directory.height % tile_height != 0) {
      return absl::InvalidArgumentError(StrCat("RowsPerStrip (", tile_height,
                                               ") must divide ImageLength (",
                                               directory.height, ")"));
    }
  }

  if (tile_height <= 0 || tile_width <= 0) {
    return absl::InvalidArgumentError(
        StrCat("Invalid tile/strip dimensions: height=", tile_height,
               ", width=", tile_width));
  }
  if (tile_height > directory.height || tile_width > directory.width) {
    return absl::InvalidArgumentError(
        "Tile/strip size exceeds image dimensions");
  }

  std::vector<Index> chunk_shape;
  chunk_shape.reserve(resolved_rank);

  const bool multi_channel = directory.samples_per_pixel > 1;

  if (planar_dimension_leading && multi_channel) {
    chunk_shape.push_back(1);  // leading C‑slice per chunk
  }

  chunk_shape.push_back(tile_height);  // Y
  chunk_shape.push_back(tile_width);   // X

  if (!planar_dimension_leading && multi_channel) {
    chunk_shape.push_back(
        directory.samples_per_pixel);  // trailing C‑slice per chunk
  }

  // Final invariant check
  if (static_cast<DimensionIndex>(chunk_shape.size()) != resolved_rank) {
    return absl::InternalError(
        StrCat("Derived chunk_shape rank (", chunk_shape.size(),
               ") does not match resolved rank (", resolved_rank, ")"));
  }
  return chunk_shape;
}

// Gets inner order based on ImageDirectory and PlanarConfiguration. (Fastest
// varying last)
Result<std::vector<DimensionIndex>> GetInnerOrderFromTiff(DimensionIndex rank) {
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError(
        "Could not determine rank for inner order");
  }
  std::vector<DimensionIndex> inner_order(rank);
  // TIFF stores chunky data as Y,X,C with C varying fastest.
  // TensorStore uses C-order (last index fastest) by default.
  // So, the natural inner order is [C, X, Y] -> [2, 1, 0] for rank 3
  // or [X, Y] -> [1, 0] for rank 2.
  for (DimensionIndex i = 0; i < rank; ++i) {
    inner_order[i] = rank - 1 - i;
  }
  return inner_order;
}

// Returns ContiguousLayoutOrder::c  or  ContiguousLayoutOrder::fortran
// for a given permutation.  Any mixed/blocked order is rejected.
Result<ContiguousLayoutOrder> GetLayoutOrderFromInnerOrder(
    span<const DimensionIndex> inner_order) {
  if (PermutationMatchesOrder(inner_order, ContiguousLayoutOrder::c)) {
    return ContiguousLayoutOrder::c;
  }
  if (PermutationMatchesOrder(inner_order, ContiguousLayoutOrder::fortran)) {
    return ContiguousLayoutOrder::fortran;
  }
  return absl::UnimplementedError(
      StrCat("Inner order ", inner_order,
             " is not a pure C or Fortran permutation; "
             "mixed-strides currently unimplemented"));
}

// Helper to convert CompressionType enum to string ID for registry lookup
Result<std::string_view> CompressionTypeToStringId(CompressionType type) {
  // Use a map for easy extension
  static const absl::flat_hash_map<CompressionType, std::string_view> kMap = {
      {CompressionType::kNone, "raw"},
      {CompressionType::kLZW, "lzw"},
      {CompressionType::kDeflate, "deflate"},
      {CompressionType::kPackBits, "packbits"},
  };
  auto it = kMap.find(type);
  if (it == kMap.end()) {
    return absl::UnimplementedError(
        tensorstore::StrCat("TIFF compression type ", static_cast<int>(type),
                            " not mapped to string ID"));
  }
  return it->second;
}

}  // namespace

// Implement JSON binder for TiffMetadataConstraints here
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TiffMetadataConstraints,
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      using T = absl::remove_cvref_t<decltype(*obj)>;
      DimensionIndex* rank = nullptr;
      if constexpr (is_loading.value) {  // Check if loading JSON
        rank = &obj->rank;
      }
      return jb::Object(
          jb::Member("dtype", jb::Projection<&T::dtype>(
                                  jb::Optional(jb::DataTypeJsonBinder))),
          // Pass the potentially non-const rank to ShapeVector
          jb::Member("shape", jb::Projection<&T::shape>(
                                  jb::Optional(jb::ShapeVector(rank))))
          // No need to explicitly bind 'rank', as ShapeVector manages it.
          )(is_loading, options, obj, j);
    })

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    tensorstore::internal_tiff::TiffSpecOptions,
    jb::Object(jb::Member(
        "ifd",  // Use "ifd" as the JSON key for ifd_index
        jb::Projection<&tensorstore::internal_tiff::TiffSpecOptions::ifd_index>(
            jb::DefaultValue([](auto* v) { *v = 0; })))
               // Add future options here, e.g.:
               // jb::Member("ifd_handling",
               // jb::Projection<&T::ifd_handling>(jb::Enum<...>(...))),
               // jb::Member("use_ome", jb::Projection<&T::use_ome_metadata>())
               ))

// --- ResolveMetadata Implementation ---
Result<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
    const TiffParseResult& source, const TiffSpecOptions& options,
    const Schema& schema) {
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Resolving TIFF metadata for IFD: " << options.ifd_index;

  // 1. Select and Validate IFD
  if (options.ifd_index >= source.image_directories.size()) {
    return absl::NotFoundError(
        tensorstore::StrCat("Requested IFD index ", options.ifd_index,
                            " not found in TIFF file (found ",
                            source.image_directories.size(), " IFDs)"));
  }
  // Get the relevant ImageDirectory directly from the TiffParseResult
  const ImageDirectory& img_dir = source.image_directories[options.ifd_index];

  // 2. Initial Interpretation (Basic Properties)
  auto metadata = std::make_shared<TiffMetadata>();
  metadata->ifd_index = options.ifd_index;
  metadata->num_ifds = 1;  // Stacking not implemented
  metadata->endian = source.endian;

  // Validate Planar Configuration and Compression early
  metadata->planar_config =
      static_cast<PlanarConfigType>(img_dir.planar_config);
  if (metadata->planar_config != PlanarConfigType::kChunky) {
    return absl::UnimplementedError(
        tensorstore::StrCat("PlanarConfiguration=", img_dir.planar_config,
                            " is not supported yet (only Chunky=1)"));
  }

  metadata->compression_type =
      static_cast<CompressionType>(img_dir.compression);

  // Determine rank, shape, dtype
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->shape, GetShapeAndRankFromTiff(img_dir, metadata->rank));

  if (metadata->rank == dynamic_rank) {
    return absl::InvalidArgumentError("Could not determine rank from TIFF IFD");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(metadata->dtype, GetDataTypeFromTiff(img_dir));
  metadata->samples_per_pixel = img_dir.samples_per_pixel;

  // 3. Initial Chunk Layout
  ChunkLayout& layout = metadata->chunk_layout;
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(RankConstraint{metadata->rank}));

  bool planar_lead = (metadata->planar_config != PlanarConfigType::kChunky);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto chunk_shape,
      GetChunkShapeFromTiff(img_dir, metadata->rank, planar_lead));

  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::ChunkShape(chunk_shape)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(metadata->rank))));
  TENSORSTORE_ASSIGN_OR_RETURN(auto default_inner_order,
                               GetInnerOrderFromTiff(metadata->rank));

  // 4. Initial Codec Spec
  TENSORSTORE_ASSIGN_OR_RETURN(
      std::string_view type_id,
      CompressionTypeToStringId(metadata->compression_type));

  // Use the tiff::Compressor binder to get the instance.
  // We pass a dummy JSON object containing only the "type" field.
  ::nlohmann::json compressor_json = {{"type", type_id}};
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->compressor,
      Compressor::FromJson(
          std::move(compressor_json),
          internal::JsonSpecifiedCompressor::FromJsonOptions{}));

  // Check if the factory returned an unimplemented error (for unsupported
  // types)
  if (!metadata->compressor &&
      metadata->compression_type != CompressionType::kNone) {
    // This case should ideally be caught by CompressionTypeToStringId,
    // but double-check based on registry content.
    return absl::UnimplementedError(tensorstore::StrCat(
        "TIFF compression type ", static_cast<int>(metadata->compression_type),
        " (", type_id,
        ") is registered but not supported by this driver yet."));
  }

  // 5. Initial Dimension Units (Default: Unknown)
  metadata->dimension_units.resize(metadata->rank);

  // --- OME-XML Interpretation Placeholder ---
  // if (options.use_ome_metadata && source.ome_xml_string) {
  //    TENSORSTORE_ASSIGN_OR_RETURN(OmeXmlData ome_data,
  //    ParseOmeXml(*source.ome_xml_string));
  //    // Apply OME data: potentially override rank, shape, dtype, units,
  //    inner_order
  //    // This requires mapping between OME concepts and TensorStore
  //    schema ApplyOmeDataToMetadata(*metadata, ome_data);
  // }

  // 6. Merge Schema Constraints
  // Data Type: Check for compatibility (schema.dtype() vs metadata->dtype)
  if (schema.dtype().valid() &&
      !IsPossiblySameDataType(metadata->dtype, schema.dtype())) {
    return absl::FailedPreconditionError(
        StrCat("Schema dtype ", schema.dtype(),
               " is incompatible with TIFF dtype ", metadata->dtype));
  }

  // Chunk Layout: Merge schema constraints *component-wise*.
  const ChunkLayout& schema_layout = schema.chunk_layout();
  if (schema_layout.rank() != dynamic_rank) {
    // Rank constraint from schema is checked against metadata rank
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(RankConstraint{schema_layout.rank()}));
  }
  // Apply schema constraints for individual components. This will respect
  // existing hard constraints (like chunk_shape from TIFF tags).
  if (!schema_layout.inner_order().empty()) {
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.inner_order()));
  }
  if (!schema_layout.grid_origin().empty()) {
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.grid_origin()));
  }
  // Setting write/read/codec components handles hard/soft constraint merging.
  // This should now correctly fail if schema tries to set a conflicting hard
  // shape.
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.write_chunk()));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.read_chunk()));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.codec_chunk()));

  // *After* merging schema, apply TIFF defaults *if still unspecified*,
  // setting them as SOFT constraints to allow schema to override.
  if (layout.inner_order().empty()) {
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::InnerOrder(
        default_inner_order, /*hard_constraint=*/false)));
  }

  // Codec Spec Validation
  if (schema.codec().valid()) {
    // Create a temporary TiffCodecSpec representing the file's compression
    auto file_codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
    file_codec_spec->compression_type = metadata->compression_type;

    // Attempt to merge the user's schema codec into the file's codec spec.
    // This validates compatibility.
    TENSORSTORE_RETURN_IF_ERROR(
        file_codec_spec->MergeFrom(schema.codec()),
        tensorstore::MaybeAnnotateStatus(
            _, "Schema codec is incompatible with TIFF file compression"));
  }

  // Dimension Units: Merge schema constraints *only if* schema units are valid.
  if (schema.dimension_units().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(metadata->dimension_units,
                                                    schema.dimension_units()));
  }

  if (schema.fill_value().valid()) {
    return absl::InvalidArgumentError(
        "fill_value not supported by TIFF format");
  }

  // 7. Finalize Layout
  TENSORSTORE_RETURN_IF_ERROR(metadata->chunk_layout.Finalize());

  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->layout_order,
      GetLayoutOrderFromInnerOrder(metadata->chunk_layout.inner_order()));

  // 8. Final consistency: chunk_shape must divide shape
  // NB: Not a given apparently...
  // const auto& cs = metadata->chunk_layout.read_chunk().shape();
  // for (DimensionIndex d = 0; d < metadata->rank; ++d) {
  //   if (metadata->shape[d] % cs[d] != 0) {
  //     return absl::FailedPreconditionError(
  //         StrCat("Chunk shape ", cs, " does not evenly divide image shape ",
  //                metadata->shape));
  //   }
  // }

  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Resolved TiffMetadata: rank=" << metadata->rank
      << ", shape=" << tensorstore::span(metadata->shape)
      << ", dtype=" << metadata->dtype
      << ", chunk_shape=" << metadata->chunk_layout.read_chunk().shape()
      << ", compression=" << static_cast<int>(metadata->compression_type)
      << ", layout_enum=" << metadata->layout_order << ", endian="
      << (metadata->endian == internal_tiff_kvstore::Endian::kLittle ? "little"
                                                                     : "big");

  // Return the final immutable metadata object
  return std::const_pointer_cast<const TiffMetadata>(metadata);
}

// --- ValidateResolvedMetadata Implementation ---
absl::Status ValidateResolvedMetadata(
    const TiffMetadata& resolved_metadata,
    const TiffMetadataConstraints& user_constraints) {
  // Validate Rank
  if (!RankConstraint::EqualOrUnspecified(resolved_metadata.rank,
                                          user_constraints.rank)) {
    return absl::FailedPreconditionError(StrCat(
        "Resolved TIFF rank (", resolved_metadata.rank,
        ") does not match user constraint rank (", user_constraints.rank, ")"));
  }

  // Validate Data Type
  if (user_constraints.dtype.has_value() &&
      resolved_metadata.dtype != *user_constraints.dtype) {
    return absl::FailedPreconditionError(
        StrCat("Resolved TIFF dtype (", resolved_metadata.dtype,
               ") does not match user constraint dtype (",
               *user_constraints.dtype, ")"));
  }

  // Validate Shape
  if (user_constraints.shape.has_value()) {
    if (resolved_metadata.rank != user_constraints.shape->size()) {
      return absl::FailedPreconditionError(
          StrCat("Rank of resolved TIFF shape (", resolved_metadata.rank,
                 ") does not match rank of user constraint shape (",
                 user_constraints.shape->size(), ")"));
    }
    if (!std::equal(resolved_metadata.shape.begin(),
                    resolved_metadata.shape.end(),
                    user_constraints.shape->begin())) {
      return absl::FailedPreconditionError(StrCat(
          "Resolved TIFF shape ", tensorstore::span(resolved_metadata.shape),
          " does not match user constraint shape ",
          tensorstore::span(*user_constraints.shape)));
    }
  }

  // Validate Axes (if added to constraints)
  // if (user_constraints.axes.has_value()) { ... }

  // Validate Chunk Shape (if added to constraints)
  // if (user_constraints.chunk_shape.has_value()) { ... }

  return absl::OkStatus();
}

Result<DataType> GetEffectiveDataType(
    const TiffMetadataConstraints& constraints, const Schema& schema) {
  DataType dtype = schema.dtype();
  if (constraints.dtype.has_value()) {
    if (dtype.valid() && dtype != *constraints.dtype) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "dtype specified in schema (", dtype,
          ") conflicts with dtype specified in metadata constraints (",
          *constraints.dtype, ")"));
    }
    dtype = *constraints.dtype;
  }
  return dtype;  // May still be invalid if neither specified
}

Result<IndexDomain<>> GetEffectiveDomain(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema) {
  // 1. Determine Rank
  DimensionIndex rank = dynamic_rank;
  if (constraints.rank != dynamic_rank) {
    rank = constraints.rank;
  }
  if (schema.rank() != dynamic_rank) {
    if (rank != dynamic_rank && rank != schema.rank()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank specified by metadata constraints (", rank,
          ") conflicts with rank specified by schema (", schema.rank(), ")"));
    }
    rank = schema.rank();
  }
  if (constraints.shape.has_value()) {
    if (rank != dynamic_rank && rank != constraints.shape->size()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank specified by metadata constraints (", rank,
          ") conflicts with rank of shape specified in metadata constraints (",
          constraints.shape->size(), ")"));
    }
    rank = constraints.shape->size();
  }

  if (rank == dynamic_rank) {
    // If rank is still unknown, return default unknown domain
    return IndexDomain<>();
  }

  // 2. Create initial domain based *only* on constraints.shape if specified
  IndexDomain domain_from_constraints;
  if (constraints.shape.has_value()) {
    IndexDomainBuilder builder(rank);
    builder.shape(*constraints.shape);  // Sets origin 0, explicit shape
    TENSORSTORE_ASSIGN_OR_RETURN(domain_from_constraints, builder.Finalize());
  } else {
    // If no shape constraint, start with an unknown domain of correct rank
    domain_from_constraints = IndexDomain(rank);
  }

  // 3. Merge with schema domain
  // MergeIndexDomains handles compatibility checks (rank, bounds, etc.)
  TENSORSTORE_ASSIGN_OR_RETURN(
      IndexDomain<> effective_domain,
      MergeIndexDomains(domain_from_constraints, schema.domain()));

  return effective_domain;
}

Result<ChunkLayout> GetEffectiveChunkLayout(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema) {
  // Determine rank first
  DimensionIndex rank = dynamic_rank;
  if (constraints.rank != dynamic_rank) rank = constraints.rank;
  if (schema.rank() != dynamic_rank) {
    if (rank != dynamic_rank && rank != schema.rank()) {
      return absl::InvalidArgumentError("Rank conflict for chunk layout");
    }
    rank = schema.rank();
  }
  if (constraints.shape.has_value()) {
    if (rank != dynamic_rank && rank != constraints.shape->size()) {
      return absl::InvalidArgumentError(
          "Rank conflict for chunk layout (shape)");
    }
    rank = constraints.shape->size();
  }
  // Cannot determine layout without rank
  if (rank == dynamic_rank) return ChunkLayout{};

  ChunkLayout layout;
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(RankConstraint{rank}));

  // Apply TIFF defaults (inner order and grid origin) as SOFT constraints
  // first.
  TENSORSTORE_ASSIGN_OR_RETURN(auto default_inner_order,
                               GetInnerOrderFromTiff(rank));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(
      ChunkLayout::InnerOrder(default_inner_order, /*hard_constraint=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::GridOrigin(
      GetConstantVector<Index, 0>(rank), /*hard_constraint=*/false)));

  // Apply schema constraints using component-wise Set, potentially overriding
  // soft defaults.
  const ChunkLayout& schema_layout = schema.chunk_layout();
  if (schema_layout.rank() != dynamic_rank) {
    // Re-check rank compatibility if schema specifies rank
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(RankConstraint{schema_layout.rank()}));
  }
  if (!schema_layout.inner_order().empty()) {
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.inner_order()));
  }
  if (!schema_layout.grid_origin().empty()) {
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.grid_origin()));
  }
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.write_chunk()));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.read_chunk()));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.codec_chunk()));

  // Apply constraints from TiffMetadataConstraints (if chunk_shape is added)
  // if (constraints.chunk_shape.has_value()) {
  //     TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::ChunkShape(*constraints.chunk_shape)));
  // }

  // Don't finalize here, let the caller finalize if needed.
  return layout;
}

Result<internal::CodecDriverSpec::PtrT<TiffCodecSpec>> GetEffectiveCodec(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema) {
  auto codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
  // Apply constraints from TiffMetadataConstraints (if compression_type is
  // added). if (constraints.compression_type.has_value()) {
  //     codec_spec->compression_type = *constraints.compression_type;
  // }
  if (schema.codec().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
    if (!dynamic_cast<const TiffCodecSpec*>(codec_spec.get())) {
      return absl::InvalidArgumentError(
          StrCat("Schema codec spec ", schema.codec(),
                 " results in an invalid codec type for the TIFF driver"));
    }
  }
  return codec_spec;
}

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const TiffSpecOptions& options, const TiffMetadataConstraints& constraints,
    const Schema& schema) {
  // Determine rank first
  DimensionIndex rank = dynamic_rank;
  if (constraints.rank != dynamic_rank) rank = constraints.rank;
  if (schema.rank() != dynamic_rank) {
    if (rank != dynamic_rank && rank != schema.rank()) {
      return absl::InvalidArgumentError("Rank conflict for dimension units");
    }
    rank = schema.rank();
  }
  if (constraints.shape.has_value()) {
    if (rank != dynamic_rank && rank != constraints.shape->size()) {
      return absl::InvalidArgumentError(
          "Rank conflict for dimension units (shape)");
    }
    rank = constraints.shape->size();
  }

  DimensionUnitsVector units(
      rank == dynamic_rank ? 0 : rank);  // Initialize with unknown units

  // Merge schema units
  if (schema.dimension_units().valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        MergeDimensionUnits(units, schema.dimension_units()));
  }

  // Apply constraints (if units/resolution are added to
  // TiffMetadataConstraints)
  // TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(units,
  // constraints.dimension_units));

  return units;
}

Result<SharedArray<const void>> DecodeChunk(const TiffMetadata& metadata,
                                            absl::Cord buffer) {
  // 1. Setup Riegeli reader for the input buffer
  riegeli::CordReader<> base_reader(&buffer);
  riegeli::Reader* data_reader = &base_reader;  // Start with base reader

  // 2. Apply Decompression if needed
  std::unique_ptr<riegeli::Reader> decompressor_reader;
  if (metadata.compressor) {
    // Get the appropriate decompressor reader from the Compressor instance
    // The compressor instance was resolved based on metadata.compression_type
    // during ResolveMetadata.
    decompressor_reader =
        metadata.compressor->GetReader(base_reader, metadata.dtype.size());
    if (!decompressor_reader) {
      return absl::InvalidArgumentError(StrCat(
          "Failed to create decompressor reader for TIFF compression type: ",
          static_cast<int>(metadata.compression_type)));
    }
    data_reader = decompressor_reader.get();  // Use the decompressing reader
    ABSL_LOG_IF(INFO, tiff_metadata_logging)
        << "Applied decompressor for type "
        << static_cast<int>(metadata.compression_type);
  } else {
    ABSL_LOG_IF(INFO, tiff_metadata_logging)
        << "No decompression needed (raw).";
    // data_reader remains &base_reader
  }

  // 3. Determine target array properties
  // Use read_chunk_shape() for the expected shape of this chunk
  span<const Index> chunk_shape = metadata.chunk_layout.read_chunk_shape();
  DataType dtype = metadata.dtype;

  // 4. Allocate destination array
  SharedArray<void> dest_array =
      AllocateArray(chunk_shape, metadata.layout_order, value_init, dtype);
  if (!dest_array.valid()) {
    return absl::ResourceExhaustedError("Failed to allocate memory for chunk");
  }

  // 5. Determine Endianness for decoding
  endian source_endian =
      (metadata.endian == internal_tiff_kvstore::Endian::kLittle)
          ? endian::little
          : endian::big;

  // 6. Decode data from the reader into the array, handling endianness
  // internal::DecodeArrayEndian handles reading from the Riegeli reader.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto decoded_array,
      internal::DecodeArrayEndian(*data_reader, metadata.dtype, chunk_shape,
                                  source_endian, metadata.layout_order));

  // 7. Verify reader reached end (important for compressed streams)
  if (!data_reader->VerifyEndAndClose()) {
    // Note: Closing the decompressor_reader also closes the base_reader.
    // If no decompressor was used, this closes base_reader directly.
    return absl::DataLossError(
        StrCat("Error reading chunk data: ", data_reader->status().message()));
  }

  // 8. Return the decoded array (cast to const void)
  return decoded_array;
}

// Validates that dtype is supported by the TIFF driver implementation.
absl::Status ValidateDataType(DataType dtype) {
  ABSL_CHECK(dtype.valid());  // Ensure dtype is valid before checking ID
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported TIFF data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

TiffGridMappingInfo GetTiffGridMappingInfo(const TiffMetadata& metadata) {
  TiffGridMappingInfo info;
  const DimensionIndex metadata_rank = metadata.rank;

  if (metadata_rank == 0) {
    // Rank 0 has no dimensions or tiling.
    return info;
  }

  // --- Determine logical Y and X dimensions in the TensorStore array ---
  const auto& inner_order = metadata.chunk_layout.inner_order();

  // Check if inner_order is valid and fully specified
  bool known_order =
      !inner_order.empty() && inner_order.size() == metadata_rank;
  // TODO(user): Add IsValidPermutation check if needed, though ResolveMetadata
  // should ensure it.

  if (known_order) {
    // Find dimensions corresponding to the last two values in the permutation
    // Assumes C-order like interpretation where last is fastest (X), second
    // last is second fastest (Y)
    DimensionIndex x_perm_val = metadata_rank - 1;
    DimensionIndex y_perm_val = metadata_rank - 2;  // Only valid if rank >= 2
    for (DimensionIndex i = 0; i < metadata_rank; ++i) {
      if (inner_order[i] == x_perm_val) info.ts_x_dim = i;
      if (metadata_rank >= 2 && inner_order[i] == y_perm_val) info.ts_y_dim = i;
    }
  } else {
    // Fallback: Assume standard C order if inner_order is missing or invalid
    // size Log a warning? ResolvedMetadata should ideally always set it.
    if (metadata_rank >= 2) {
      info.ts_y_dim = metadata_rank - 2;
      info.ts_x_dim = metadata_rank - 1;
    } else if (metadata_rank == 1) {
      info.ts_x_dim = 0;  // Rank 1 only has an X dimension conceptually
    }
  }
  ABSL_CHECK(info.ts_x_dim != -1)
      << "Could not determine X dimension index from metadata";
  ABSL_CHECK(metadata_rank < 2 || info.ts_y_dim != -1)
      << "Could not determine Y dimension index from metadata";

  // --- Determine logical IFD/Z dimension ---
  if (metadata.num_ifds > 1) {
    // Assume the IFD/Z dimension is the one *not* identified as X or Y.
    // This requires rank >= 3 for a ZYX or ZXY layout.
    // TODO: Enhance this logic based on actual OME-TIFF dimension order parsing
    // later.
    ABSL_CHECK(metadata_rank >= 3) << "Multi-IFD requires metadata rank >= 3";
    for (DimensionIndex i = 0; i < metadata_rank; ++i) {
      if (i != info.ts_x_dim && i != info.ts_y_dim) {
        // Assume the first dimension found that isn't X or Y is IFD/Z
        info.ts_ifd_dim = i;
        break;
      }
    }
    ABSL_CHECK(info.ts_ifd_dim != -1)
        << "Could not determine IFD/Z dimension index";
  }

  // --- Determine if Tiled or Stripped ---
  const auto& read_chunk_shape = metadata.chunk_layout.read_chunk_shape();
  // If rank is < 2, ts_y_dim is -1, but it behaves like strips
  // (width=image_width). Check only if X dimension exists.
  if (info.ts_x_dim != -1) {
    const Index chunk_width = read_chunk_shape[info.ts_x_dim];
    const Index image_width = metadata.shape[info.ts_x_dim];
    // Consider it tiled if chunk width is less than image width.
    info.is_tiled = (chunk_width < image_width);

    // Sanity check for strips: chunk width should equal image width
    if (!info.is_tiled) {
      ABSL_CHECK(chunk_width == image_width)
          << "Chunk width does not match image width for inferred stripped "
             "layout.";
      // Also check Y dimension if it exists
      if (info.ts_y_dim != -1) {
        const Index chunk_height = read_chunk_shape[info.ts_y_dim];
        const Index image_height = metadata.shape[info.ts_y_dim];
        ABSL_CHECK(chunk_height > 0 && chunk_height <= image_height)
            << "Invalid chunk height for stripped layout.";
      }
    } else {
      // Sanity check for tiles: chunk height should also be less than image
      // height (if Y exists)
      if (info.ts_y_dim != -1) {
        const Index chunk_height = read_chunk_shape[info.ts_y_dim];
        const Index image_height = metadata.shape[info.ts_y_dim];
        ABSL_CHECK(chunk_height < image_height)
            << "Chunk height equals image height for inferred tiled layout.";
      }
    }
  } else {
    // Rank 1 case is considered not tiled (like a single column strip)
    info.is_tiled = false;
  }

  return info;
}

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffMetadataConstraints>())
