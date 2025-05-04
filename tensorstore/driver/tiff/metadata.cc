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

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
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
#include "tensorstore/internal/json_binding/std_optional.h"
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
using ::tensorstore::GetConstantVector;
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
  return a.compression_type == b.compression_type;
}

namespace {
const internal::CodecSpecRegistration<TiffCodecSpec> registration;

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
    case static_cast<uint16_t>(SampleFormatType::kUndefined):
      break;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      StrCat("Unsupported TIFF data type: bits=", uniform_bits,
             ", format=", uniform_format));
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

// Helper to check IFD uniformity for multi-IFD stacking
absl::Status CheckIfdUniformity(const ImageDirectory& base_ifd,
                                const ImageDirectory& other_ifd,
                                size_t ifd_index) {
  // Compare essential properties needed for consistent stacking
  if (other_ifd.width != base_ifd.width ||
      other_ifd.height != base_ifd.height) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "IFD %d dimensions (%d x %d) do not match IFD 0 dimensions (%d x %d)",
        ifd_index, other_ifd.width, other_ifd.height, base_ifd.width,
        base_ifd.height));
  }
  if (other_ifd.chunk_width != base_ifd.chunk_width ||
      other_ifd.chunk_height != base_ifd.chunk_height) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "IFD %d chunk dimensions (%d x %d) do not match IFD 0 chunk dimensions "
        "(%d x %d)",
        ifd_index, other_ifd.chunk_width, other_ifd.chunk_height,
        base_ifd.chunk_width, base_ifd.chunk_height));
  }
  if (other_ifd.samples_per_pixel != base_ifd.samples_per_pixel) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "IFD %d SamplesPerPixel (%d) does not match IFD 0 (%d)", ifd_index,
        other_ifd.samples_per_pixel, base_ifd.samples_per_pixel));
  }
  if (other_ifd.bits_per_sample != base_ifd.bits_per_sample) {
    return absl::InvalidArgumentError(
        absl::StrFormat("IFD %d BitsPerSample does not match IFD 0"));
  }
  if (other_ifd.sample_format != base_ifd.sample_format) {
    return absl::InvalidArgumentError(
        absl::StrFormat("IFD %d SampleFormat does not match IFD 0"));
  }
  if (other_ifd.compression != base_ifd.compression) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "IFD %d Compression (%d) does not match IFD 0 (%d)", ifd_index,
        other_ifd.compression, base_ifd.compression));
  }
  if (other_ifd.planar_config != base_ifd.planar_config) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "IFD %d PlanarConfiguration (%d) does not match IFD 0 (%d)", ifd_index,
        other_ifd.planar_config, base_ifd.planar_config));
  }
  return absl::OkStatus();
}

// Helper to build the dimension mapping struct
TiffDimensionMapping BuildDimensionMapping(
    tensorstore::span<const std::string> final_labels,
    const std::optional<TiffSpecOptions::IfdStackingOptions>& stacking_info,
    const std::optional<std::string>& options_sample_label,
    tensorstore::span<const std::string> initial_conceptual_labels,
    uint16_t samples_per_pixel) {
  TiffDimensionMapping mapping;
  const DimensionIndex final_rank = final_labels.size();
  if (final_rank == 0) return mapping;

  mapping.labels_by_ts_dim.resize(final_rank);
  absl::flat_hash_map<std::string_view, DimensionIndex> final_label_to_index;
  for (DimensionIndex i = 0; i < final_rank; ++i) {
    final_label_to_index[final_labels[i]] = i;
  }

  const std::string default_sample_label = "c";
  const std::string& conceptual_sample_label =
      options_sample_label.value_or(default_sample_label);

  std::set<std::string_view> conceptual_stack_labels;
  if (stacking_info) {
    for (const auto& label : stacking_info->dimensions) {
      conceptual_stack_labels.insert(label);
    }
  }

  const std::string conceptual_y_label = "y";
  const std::string conceptual_x_label = "x";

  // Assume initial_conceptual_labels rank == final_rank after merge
  assert(initial_conceptual_labels.size() == final_rank);

  // Map FINAL indices back to INITIAL conceptual labels and identify roles.
  for (DimensionIndex final_idx = 0; final_idx < final_rank; ++final_idx) {
    DimensionIndex initial_idx = final_idx;

    if (initial_idx >= 0 && initial_idx < initial_conceptual_labels.size()) {
      const std::string& conceptual_label =
          initial_conceptual_labels[initial_idx];
      mapping.labels_by_ts_dim[final_idx] = conceptual_label;

      if (conceptual_label == conceptual_y_label) {
        mapping.ts_y_dim = final_idx;
      } else if (conceptual_label == conceptual_x_label) {
        mapping.ts_x_dim = final_idx;
      } else if (samples_per_pixel > 1 &&
                 conceptual_label == conceptual_sample_label) {
        mapping.ts_sample_dim = final_idx;
      } else if (conceptual_stack_labels.count(conceptual_label)) {
        mapping.ts_stacked_dims[conceptual_label] = final_idx;
      }
    } else {
      // Should not happen if rank remains consistent
      mapping.labels_by_ts_dim[final_idx] = "";
    }
  }
  return mapping;
}

// Helper to apply TIFF-derived layout constraints onto an existing layout
// object.
absl::Status SetChunkLayoutFromTiffMetadata(DimensionIndex rank,
                                            ChunkLayout initial_layout,
                                            ChunkLayout& merged_layout) {
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(RankConstraint{rank}));
  if (merged_layout.rank() == dynamic_rank) {
    return absl::OkStatus();
  }
  assert(merged_layout.rank() == rank);

  // - Chunk Shape (TIFF tile/strip size is a hard constraint)
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::ChunkShape(
      initial_layout.read_chunk_shape(), /*hard_constraint=*/true)));

  // - Grid Origin (TIFF grid origin is implicitly 0, a hard constraint)
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::GridOrigin(
      initial_layout.grid_origin(), /*hard_constraint=*/true)));

  // - Inner Order (TIFF doesn't mandate an order, use C as soft default)
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::InnerOrder(
      initial_layout.inner_order(), /*hard_constraint=*/false)));

  // Apply other constraints from initial_layout as. soft constraints.
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::WriteChunkElements(
      initial_layout.write_chunk_elements().value, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::ReadChunkElements(
      initial_layout.read_chunk_elements().value, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(merged_layout.Set(ChunkLayout::CodecChunkElements(
      initial_layout.codec_chunk_elements().value, /*hard=*/false)));

  // Aspect ratios are typically preferences, apply as soft constraints
  TENSORSTORE_RETURN_IF_ERROR(
      merged_layout.Set(ChunkLayout::WriteChunkAspectRatio(
          initial_layout.write_chunk_aspect_ratio(), /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(
      merged_layout.Set(ChunkLayout::ReadChunkAspectRatio(
          initial_layout.read_chunk_aspect_ratio(), /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(
      merged_layout.Set(ChunkLayout::CodecChunkAspectRatio(
          initial_layout.codec_chunk_aspect_ratio(), /*hard=*/false)));

  return absl::OkStatus();
}

auto ifd_stacking_options_binder = jb::Validate(
    [](const auto& options, auto* obj) -> absl::Status {
      if (obj->dimensions.empty()) {
        return absl::InvalidArgumentError(
            "\"dimensions\" must not be empty in \"ifd_stacking\"");
      }

      std::set<std::string_view> dim_set;
      for (const auto& dim : obj->dimensions) {
        if (!dim_set.insert(dim).second) {
          return absl::InvalidArgumentError(
              tensorstore::StrCat("Duplicate dimension label \"", dim,
                                  "\" in \"ifd_stacking.dimensions\""));
        }
      }

      if (obj->dimension_sizes) {
        if (obj->dimension_sizes->size() != obj->dimensions.size()) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "\"dimension_sizes\" length (", obj->dimension_sizes->size(),
              ") must match \"dimensions\" length (", obj->dimensions.size(),
              ")"));
        }
      }

      if (obj->dimensions.size() == 1) {
        if (!obj->dimension_sizes && !obj->ifd_count) {
          return absl::InvalidArgumentError(
              "Either \"dimension_sizes\" or \"ifd_count\" must be specified "
              "when \"ifd_stacking.dimensions\" has length 1");
        }
        if (obj->dimension_sizes && obj->ifd_count &&
            static_cast<uint64_t>((*obj->dimension_sizes)[0]) !=
                *obj->ifd_count) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "\"dimension_sizes\" ([", (*obj->dimension_sizes)[0],
              "]) conflicts with \"ifd_count\" (", *obj->ifd_count, ")"));
        }
      } else {  // dimensions.size() > 1
        if (!obj->dimension_sizes) {
          return absl::InvalidArgumentError(
              "\"dimension_sizes\" must be specified when "
              "\"ifd_stacking.dimensions\" has length > 1");
        }
        if (obj->ifd_count) {
          uint64_t product = 1;
          uint64_t max_val = std::numeric_limits<uint64_t>::max();
          for (Index size : *obj->dimension_sizes) {
            uint64_t u_size = static_cast<uint64_t>(size);
            if (size <= 0) {
              return absl::InvalidArgumentError(
                  "\"dimension_sizes\" must be positive");
            }
            if (product > max_val / u_size) {
              return absl::InvalidArgumentError(
                  "Product of \"dimension_sizes\" overflows uint64_t");
            }
            product *= u_size;
          }
          if (product != *obj->ifd_count) {
            return absl::InvalidArgumentError(tensorstore::StrCat(
                "Product of \"dimension_sizes\" (", product,
                ") does not match specified \"ifd_count\" (", *obj->ifd_count,
                ")"));
          }
        }
      }

      if (obj->ifd_sequence_order) {
        if (obj->ifd_sequence_order->size() != obj->dimensions.size()) {
          return absl::InvalidArgumentError(
              tensorstore::StrCat("\"ifd_sequence_order\" length (",
                                  obj->ifd_sequence_order->size(),
                                  ") must match \"dimensions\" length (",
                                  obj->dimensions.size(), ")"));
        }
        std::set<std::string_view> order_set(obj->ifd_sequence_order->begin(),
                                             obj->ifd_sequence_order->end());
        if (order_set != dim_set) {
          return absl::InvalidArgumentError(
              "\"ifd_sequence_order\" must be a permutation of \"dimensions\"");
        }
      }
      return absl::OkStatus();
    },
    jb::Object(
        jb::Member(
            "dimensions",
            jb::Projection<&TiffSpecOptions::IfdStackingOptions::dimensions>(
                jb::DefaultBinder<>)),
        jb::Member("dimension_sizes",
                   jb::Projection<
                       &TiffSpecOptions::IfdStackingOptions::dimension_sizes>(
                       jb::Optional(jb::DefaultBinder<>))),
        jb::Member(
            "ifd_count",
            jb::Projection<&TiffSpecOptions::IfdStackingOptions::ifd_count>(
                jb::Optional(jb::Integer<uint64_t>(1)))),
        jb::Member(
            "ifd_sequence_order",
            jb::Projection<
                &TiffSpecOptions::IfdStackingOptions::ifd_sequence_order>(
                jb::Optional(jb::DefaultBinder<>)))));
}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TiffMetadataConstraints,
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      using T = absl::remove_cvref_t<decltype(*obj)>;
      DimensionIndex* rank = nullptr;
      if constexpr (is_loading.value) {
        rank = &obj->rank;
      }
      return jb::Object(
          jb::Member("dtype", jb::Projection<&T::dtype>(
                                  jb::Optional(jb::DataTypeJsonBinder))),
          jb::Member("shape", jb::Projection<&T::shape>(
                                  jb::Optional(jb::ShapeVector(rank)))))(
          is_loading, options, obj, j);
    })

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(TiffSpecOptions::IfdStackingOptions,
                                       ifd_stacking_options_binder);

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TiffSpecOptions,
    jb::Object(
        jb::Member("ifd",
                   jb::Projection<&TiffSpecOptions::ifd_index>(jb::DefaultValue(
                       [](auto* v) { *v = 0; }, jb::Integer<uint32_t>(0)))),
        jb::Member(
            "ifd_stacking",
            jb::Projection<&TiffSpecOptions::ifd_stacking>(jb::Optional(
                jb::DefaultBinder<TiffSpecOptions::IfdStackingOptions>))),
        jb::Member("sample_dimension_label",
                   jb::Projection<&TiffSpecOptions::sample_dimension_label>(
                       jb::Optional(jb::NonEmptyStringBinder)))))

Result<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
    const internal_tiff_kvstore::TiffParseResult& source,
    const TiffSpecOptions& options, const Schema& schema) {
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Resolving TIFF metadata. Options: "
      << jb::ToJson(options).value_or(::nlohmann::json::object());

  // 1. Initial Setup & IFD Selection/Validation
  const ImageDirectory* base_ifd_ptr = nullptr;
  uint32_t base_ifd_index = 0;
  uint32_t num_ifds_read = 0;
  std::optional<TiffSpecOptions::IfdStackingOptions> validated_stacking_info;
  std::vector<Index> stack_sizes_vec;

  if (options.ifd_stacking) {
    validated_stacking_info = *options.ifd_stacking;
    const auto& stacking = *validated_stacking_info;
    size_t num_stack_dims = stacking.dimensions.size();
    if (num_stack_dims == 0)
      return absl::InvalidArgumentError(
          "ifd_stacking.dimensions cannot be empty");

    uint64_t total_ifds_needed = 0;
    if (stacking.dimension_sizes) {
      if (stacking.dimension_sizes->size() != num_stack_dims) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"dimension_sizes\" length (", stacking.dimension_sizes->size(),
            ") must match \"dimensions\" length (", num_stack_dims, ")"));
      }
      stack_sizes_vec = *stacking.dimension_sizes;
      total_ifds_needed = 1;
      uint64_t max_val = std::numeric_limits<uint64_t>::max();
      for (Index size : stack_sizes_vec) {
        if (size <= 0)
          return absl::InvalidArgumentError(
              "\"dimension_sizes\" must be positive");
        uint64_t u_size = static_cast<uint64_t>(size);
        if (total_ifds_needed > max_val / u_size) {
          return absl::InvalidArgumentError(
              "Product of dimension_sizes overflows uint64_t");
        }
        total_ifds_needed *= u_size;
      }
      if (stacking.ifd_count && total_ifds_needed != *stacking.ifd_count) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Product of \"dimension_sizes\" (", total_ifds_needed,
            ") does not match specified \"ifd_count\" (", *stacking.ifd_count,
            ")"));
      }
    } else {
      if (num_stack_dims > 1) {
        return absl::InvalidArgumentError(
            "\"dimension_sizes\" is required when more than one stacking "
            "dimension is specified");
      }
      if (!stacking.ifd_count) {
        return absl::InvalidArgumentError(
            "Either \"dimension_sizes\" or \"ifd_count\" must be specified for "
            "stacking");
      }
      if (*stacking.ifd_count <= 0) {
        return absl::InvalidArgumentError("\"ifd_count\" must be positive");
      }
      total_ifds_needed = *stacking.ifd_count;
      stack_sizes_vec.push_back(static_cast<Index>(total_ifds_needed));
      validated_stacking_info->dimension_sizes = stack_sizes_vec;
    }

    num_ifds_read = total_ifds_needed;
    base_ifd_index = 0;

    if (num_ifds_read == 0 || num_ifds_read > source.image_directories.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Required %d IFDs for stacking, but only %d available/parsed",
          num_ifds_read, source.image_directories.size()));
    }
    base_ifd_ptr = &source.image_directories[0];

    for (size_t i = 1; i < num_ifds_read; ++i) {
      TENSORSTORE_RETURN_IF_ERROR(
          CheckIfdUniformity(*base_ifd_ptr, source.image_directories[i], i));
    }

  } else {
    // Single IFD Mode Logic
    base_ifd_index = options.ifd_index;
    num_ifds_read = 1;
    validated_stacking_info = std::nullopt;

    if (base_ifd_index >= source.image_directories.size()) {
      return absl::NotFoundError(
          absl::StrFormat("Requested IFD index %d not found (found %d IFDs)",
                          base_ifd_index, source.image_directories.size()));
    }
    base_ifd_ptr = &source.image_directories[base_ifd_index];
  }
  const ImageDirectory& base_ifd = *base_ifd_ptr;

  // 2. Determine Initial Structure
  DimensionIndex initial_rank = dynamic_rank;
  std::vector<Index> initial_shape;
  std::vector<std::string> initial_labels;
  PlanarConfigType initial_planar_config =
      static_cast<PlanarConfigType>(base_ifd.planar_config);
  uint16_t initial_samples_per_pixel = base_ifd.samples_per_pixel;

  const std::string implicit_y_label = "y";
  const std::string implicit_x_label = "x";
  const std::string default_sample_label = "c";
  const std::string& sample_label =
      options.sample_dimension_label.value_or(default_sample_label);

  initial_shape.clear();
  initial_labels.clear();

  if (initial_planar_config != PlanarConfigType::kChunky) {
    if (initial_samples_per_pixel <= 1) {
      // Treat Planar with SPP=1 as Chunky for layout purposes.
      ABSL_LOG_IF(WARNING, tiff_metadata_logging)
          << "PlanarConfiguration=2 with SamplesPerPixel<=1; treating as "
             "Chunky.";
      initial_planar_config = PlanarConfigType::kChunky;
    } else if (validated_stacking_info) {
      // Stacking + Planar is not supported (yet).
      return absl::UnimplementedError(
          "PlanarConfiguration=2 is not supported with ifd_stacking.");
    } else {
      // Single IFD Planar: Use {Sample, Y, X} initial order
      initial_shape.push_back(static_cast<Index>(initial_samples_per_pixel));
      initial_labels.push_back(sample_label);
      initial_shape.push_back(static_cast<Index>(base_ifd.height));
      initial_labels.push_back(implicit_y_label);
      initial_shape.push_back(static_cast<Index>(base_ifd.width));
      initial_labels.push_back(implicit_x_label);
      initial_rank = 3;
    }
  }

  if (initial_planar_config == PlanarConfigType::kChunky) {
    // Add stacked dimensions first
    if (validated_stacking_info) {
      initial_shape.insert(initial_shape.end(), stack_sizes_vec.begin(),
                           stack_sizes_vec.end());
      initial_labels.insert(initial_labels.end(),
                            validated_stacking_info->dimensions.begin(),
                            validated_stacking_info->dimensions.end());
    }
    initial_shape.push_back(static_cast<Index>(base_ifd.height));
    initial_labels.push_back(implicit_y_label);
    initial_shape.push_back(static_cast<Index>(base_ifd.width));
    initial_labels.push_back(implicit_x_label);
    // Add Sample dimension last if Chunky and spp > 1
    if (initial_samples_per_pixel > 1) {
      initial_shape.push_back(static_cast<Index>(initial_samples_per_pixel));
      initial_labels.push_back(sample_label);
    }
    initial_rank = initial_shape.size();
  }

  std::set<std::string_view> label_set;
  for (const auto& label : initial_labels) {
    if (!label_set.insert(label).second) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Duplicate dimension label detected in initial structure: \"", label,
          "\""));
    }
  }

  // 3. Determine Initial Properties
  TENSORSTORE_ASSIGN_OR_RETURN(DataType initial_dtype,
                               GetDataTypeFromTiff(base_ifd));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(initial_dtype));
  CompressionType initial_compression_type =
      static_cast<CompressionType>(base_ifd.compression);
  PlanarConfigType ifd_planar_config =
      static_cast<PlanarConfigType>(base_ifd.planar_config);
  TENSORSTORE_ASSIGN_OR_RETURN(
      ChunkLayout initial_layout,
      GetInitialChunkLayout(base_ifd, initial_rank, initial_labels,
                            ifd_planar_config, initial_samples_per_pixel,
                            sample_label));

  // 4. Merge with Schema
  Schema merged_schema = schema;

  TENSORSTORE_ASSIGN_OR_RETURN(
      DataType effective_dtype,
      GetEffectiveDataType(TiffMetadataConstraints{/*.dtype=*/initial_dtype},
                           merged_schema));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(effective_dtype));

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto final_domain_pair,
      GetEffectiveDomain(initial_rank, initial_shape, initial_labels,
                         merged_schema));
  const IndexDomain<>& final_domain = final_domain_pair.first;
  const std::vector<std::string>& final_labels = final_domain_pair.second;
  const DimensionIndex final_rank = final_domain.rank();

  TENSORSTORE_ASSIGN_OR_RETURN(
      ChunkLayout final_layout,
      GetEffectiveChunkLayout(initial_layout, merged_schema));
  TENSORSTORE_RETURN_IF_ERROR(final_layout.Finalize());

  TENSORSTORE_ASSIGN_OR_RETURN(
      Compressor final_compressor,
      GetEffectiveCompressor(initial_compression_type, merged_schema.codec()));

  TENSORSTORE_ASSIGN_OR_RETURN(
      DimensionUnitsVector final_units,
      GetEffectiveDimensionUnits(final_rank, merged_schema));

  if (merged_schema.fill_value().valid()) {
    return absl::InvalidArgumentError(
        "fill_value not supported by TIFF format");
  }

  // 5. Build Final TiffMetadata
  auto metadata = std::make_shared<TiffMetadata>();
  metadata->base_ifd_index = base_ifd_index;
  metadata->num_ifds_read = num_ifds_read;
  metadata->stacking_info = validated_stacking_info;
  metadata->endian = source.endian;
  metadata->is_tiled = base_ifd.is_tiled;
  metadata->planar_config =
      static_cast<PlanarConfigType>(base_ifd.planar_config);
  metadata->samples_per_pixel = initial_samples_per_pixel;
  metadata->ifd0_chunk_width = base_ifd.chunk_width;
  metadata->ifd0_chunk_height = base_ifd.chunk_height;
  metadata->compressor = std::move(final_compressor);
  metadata->compression_type =
      metadata->compressor ? initial_compression_type : CompressionType::kNone;
  metadata->rank = final_rank;
  metadata->shape.assign(final_domain.shape().begin(),
                         final_domain.shape().end());
  metadata->dtype = effective_dtype;
  metadata->chunk_layout = std::move(final_layout);
  metadata->fill_value = SharedArray<const void>();
  metadata->dimension_units = std::move(final_units);
  metadata->dimension_labels = final_labels;

  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->layout_order,
      GetLayoutOrderFromInnerOrder(metadata->chunk_layout.inner_order()));

  metadata->dimension_mapping =
      BuildDimensionMapping(metadata->dimension_labels, metadata->stacking_info,
                            options.sample_dimension_label, initial_labels,
                            metadata->samples_per_pixel);

  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Resolved TiffMetadata: rank=" << metadata->rank
      << ", shape=" << tensorstore::span(metadata->shape)
      << ", labels=" << tensorstore::span(metadata->dimension_labels)
      << ", dtype=" << metadata->dtype
      << ", chunk_layout=" << metadata->chunk_layout
      << ", compression=" << static_cast<int>(metadata->compression_type)
      << ", planar_config=" << static_cast<int>(metadata->planar_config);

  return metadata;
}

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

  if (user_constraints.dtype.has_value() &&
      resolved_metadata.dtype != *user_constraints.dtype) {
    return absl::FailedPreconditionError(
        StrCat("Resolved TIFF dtype (", resolved_metadata.dtype,
               ") does not match user constraint dtype (",
               *user_constraints.dtype, ")"));
  }

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
  // TODO: Implement axis validation

  // Validate Chunk Shape (if added to constraints)
  // TODO: Implement chunk shape validation

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
  if (dtype.valid()) TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(dtype));
  return dtype;
}

// Helper to get the effective compressor based on type and codec spec options
Result<Compressor> GetEffectiveCompressor(CompressionType compression_type,
                                          const CodecSpec& schema_codec) {
  // Determine initial compressor type from TIFF tag
  // TENSORSTORE_ASSIGN_OR_RETURN(std::string_view type_id,
  //                              CompressionTypeToStringId(compression_type));

  auto initial_codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
  initial_codec_spec->compression_type = compression_type;

  // Merge with schema codec spec
  if (schema_codec.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        initial_codec_spec->MergeFrom(schema_codec),
        tensorstore::MaybeAnnotateStatus(
            _, "Schema codec is incompatible with TIFF file compression"));
    // If schema specified options for the *same* compression type, they would
    // be merged here (currently only type is stored).
  }

  auto final_compression_type =
      initial_codec_spec->compression_type.value_or(CompressionType::kNone);

  if (final_compression_type == CompressionType::kNone) {
    return Compressor{nullptr};
  }

  // Re-lookup the type ID in case merging changed the type
  TENSORSTORE_ASSIGN_OR_RETURN(
      std::string_view final_type_id,
      CompressionTypeToStringId(final_compression_type));

  // Create the JSON spec for the final compressor type
  ::nlohmann::json final_compressor_json = {{"type", final_type_id}};
  // TODO: Incorporate options from the potentially merged schema_codec if
  // drivers support it. E.g., if schema_codec was {"driver":"tiff",
  // "compression":"deflate", "level": 9} and final_compression_type is Deflate,
  // we'd want to add {"level": 9} to final_compressor_json. This requires
  // parsing the schema_codec.

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto final_compressor,
      Compressor::FromJson(
          std::move(final_compressor_json),
          internal::JsonSpecifiedCompressor::FromJsonOptions{}));

  if (!final_compressor && final_compression_type != CompressionType::kNone) {
    return absl::UnimplementedError(tensorstore::StrCat(
        "TIFF compression type ", static_cast<int>(final_compression_type),
        " (", final_type_id, ") is not supported by this driver build."));
  }

  return final_compressor;
}

Result<std::pair<IndexDomain<>, std::vector<std::string>>> GetEffectiveDomain(
    DimensionIndex initial_rank, span<const Index> initial_shape,
    span<const std::string> initial_labels, const Schema& schema) {
  // 1. Validate Rank Compatibility & Determine Final Rank
  if (!RankConstraint::EqualOrUnspecified(initial_rank, schema.rank())) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Schema rank constraint ", schema.rank(),
                            " is incompatible with TIFF rank ", initial_rank));
  }
  const DimensionIndex rank =
      schema.rank().rank == dynamic_rank ? initial_rank : schema.rank().rank;
  if (rank == dynamic_rank) {
    return std::make_pair(IndexDomain<>(dynamic_rank),
                          std::vector<std::string>{});
  }
  if (initial_rank != dynamic_rank && initial_rank != rank) {
    return absl::InternalError(
        "Rank mismatch after effective rank determination");
  }

  // 2. Determine Final Labels
  std::vector<std::string> final_labels;
  bool schema_has_labels =
      schema.domain().valid() && !schema.domain().labels().empty();
  if (schema_has_labels) {
    if (static_cast<DimensionIndex>(schema.domain().labels().size()) != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Schema domain labels rank (", schema.domain().labels().size(),
          ") does not match effective rank (", rank, ")"));
    }
    final_labels.assign(schema.domain().labels().begin(),
                        schema.domain().labels().end());
  } else {
    if (initial_labels.size() != rank) {
      return absl::InternalError(
          tensorstore::StrCat("Initial labels rank (", initial_labels.size(),
                              ") does not match effective rank (", rank, ")"));
    }
    final_labels.assign(initial_labels.begin(), initial_labels.end());
  }

  // 3. Build Initial Domain (with final labels for merge compatibility)
  IndexDomainBuilder initial_builder(rank);
  initial_builder.shape(initial_shape);
  initial_builder.labels(final_labels);
  initial_builder.implicit_lower_bounds(false);
  initial_builder.implicit_upper_bounds(false);
  TENSORSTORE_ASSIGN_OR_RETURN(auto initial_domain, initial_builder.Finalize());

  //  4. Build Effective Schema Domain (with final labels)
  IndexDomain<> effective_schema_domain;
  if (schema.domain().valid()) {
    IndexDomainBuilder schema_builder(rank);
    schema_builder.origin(schema.domain().origin());
    schema_builder.shape(schema.domain().shape());
    schema_builder.labels(final_labels);
    schema_builder.implicit_lower_bounds(
        schema.domain().implicit_lower_bounds());
    schema_builder.implicit_upper_bounds(
        schema.domain().implicit_upper_bounds());
    TENSORSTORE_ASSIGN_OR_RETURN(effective_schema_domain,
                                 schema_builder.Finalize());
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        effective_schema_domain,
        IndexDomainBuilder(rank).labels(final_labels).Finalize());
  }

  // 5. Merge Domains
  TENSORSTORE_ASSIGN_OR_RETURN(
      IndexDomain<> merged_domain_bounds_only,
      MergeIndexDomains(effective_schema_domain, initial_domain),
      tensorstore::MaybeAnnotateStatus(_,
                                       "Mismatch between TIFF-derived domain "
                                       "and schema domain bounds/shape"));

  return std::make_pair(std::move(merged_domain_bounds_only),
                        std::move(final_labels));
}

Result<IndexDomain<>> GetEffectiveDomain(
    const TiffMetadataConstraints& constraints, const Schema& schema) {
  DimensionIndex rank = schema.rank().rank;
  if (constraints.rank != dynamic_rank) {
    if (rank != dynamic_rank && rank != constraints.rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Rank specified in schema (", rank,
          ") conflicts with rank specified in metadata constraints (",
          constraints.rank, ")"));
    }
    rank = constraints.rank;
  }
  if (rank == dynamic_rank && constraints.shape.has_value()) {
    rank = constraints.shape->size();
  }
  if (rank == dynamic_rank && schema.domain().valid()) {
    rank = schema.domain().rank();
  }
  // If rank is still dynamic after checking all available sources in the spec
  // and constraints, return a dynamic_rank domain.
  if (rank == dynamic_rank) {
    return IndexDomain<>();
  }

  IndexDomainBuilder builder(rank);
  if (constraints.shape) {
    if (constraints.shape->size() != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Internal error: Metadata constraints shape rank (",
          constraints.shape->size(), ") conflicts with effective rank (", rank,
          ")"));
    }
    builder.shape(*constraints.shape);
    builder.implicit_lower_bounds(false);
    builder.implicit_upper_bounds(false);
  } else {
    builder.implicit_lower_bounds(true);
    builder.implicit_upper_bounds(true);
  }

  if (schema.domain().valid() && !schema.domain().labels().empty()) {
    if (static_cast<DimensionIndex>(schema.domain().labels().size()) != rank) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Schema domain labels rank (", schema.domain().labels().size(),
          ") does not match effective rank (", rank, ")"));
    }
    builder.labels(schema.domain().labels());
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_constraints,
                               builder.Finalize());

  TENSORSTORE_ASSIGN_OR_RETURN(
      IndexDomain<> merged_domain,
      MergeIndexDomains(schema.domain(), domain_from_constraints),
      tensorstore::MaybeAnnotateStatus(
          _, "Conflict between schema domain and metadata constraints"));

  return merged_domain;
}

Result<ChunkLayout> GetEffectiveChunkLayout(ChunkLayout initial_layout,
                                            const Schema& schema) {
  ChunkLayout merged_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromTiffMetadata(
      initial_layout.rank(), initial_layout, merged_layout));
  return merged_layout;
}

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    DimensionIndex rank, /* const DimensionUnitsVector& initial_units, */
    const Schema& schema) {
  // Currently, no initial_units are derived from standard TIFF.
  // Start with schema units.
  DimensionUnitsVector final_units(schema.dimension_units());

  if (final_units.empty() && rank != dynamic_rank) {
    final_units.resize(rank);
  } else if (!final_units.empty() &&
             static_cast<DimensionIndex>(final_units.size()) != rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Schema dimension_units rank (", final_units.size(),
                            ") conflicts with TIFF-derived rank (", rank, ")"));
  }

  // TODO: When OME-XML or other sources provide initial_units, merge here:
  // TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(final_units,
  // initial_units));

  return final_units;
}

Result<ChunkLayout> GetInitialChunkLayout(
    const internal_tiff_kvstore::ImageDirectory& base_ifd,
    DimensionIndex initial_rank, span<const std::string> initial_labels,
    internal_tiff_kvstore::PlanarConfigType initial_planar_config,
    uint16_t initial_samples_per_pixel, std::string_view sample_label) {
  ChunkLayout layout;
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(RankConstraint{initial_rank}));
  if (initial_rank == dynamic_rank || initial_rank == 0) {
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::ChunkShape({}, /*hard=*/true)));
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::CodecChunkShape({}, /*hard=*/true)));
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::GridOrigin({}, /*hard=*/true)));
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::InnerOrder({}, /*hard=*/false)));
    return layout;
  }

  // 1. Set Grid Origin (Hard Constraint)
  DimensionSet all_dims_hard = DimensionSet::UpTo(initial_rank);
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::GridOrigin(
      GetConstantVector<Index, 0>(initial_rank), all_dims_hard)));

  // 2. Set Default Inner Order (Soft Constraint)
  std::vector<DimensionIndex> default_inner_order(initial_rank);
  std::iota(default_inner_order.begin(), default_inner_order.end(), 0);
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(
      ChunkLayout::InnerOrder(default_inner_order, /*hard_constraint=*/false)));

  // 3. Determine Initial Chunk Shape (Hard Constraint)
  std::vector<Index> initial_chunk_shape(initial_rank);

  absl::flat_hash_map<std::string_view, DimensionIndex> label_to_index;
  for (DimensionIndex i = 0; i < initial_rank; ++i) {
    label_to_index[initial_labels[i]] = i;
  }

  // Find indices corresponding to conceptual Y, X, and sample dimensions
  DimensionIndex y_dim_idx = -1;
  DimensionIndex x_dim_idx = -1;
  DimensionIndex sample_dim_idx = -1;

  if (auto it = label_to_index.find("y"); it != label_to_index.end()) {
    y_dim_idx = it->second;
  } else if (initial_rank >= 2) {
    return absl::InternalError(
        "Conceptual 'y' dimension label not found in initial labels");
  }

  if (auto it = label_to_index.find("x"); it != label_to_index.end()) {
    x_dim_idx = it->second;
  } else if (initial_rank >= 1) {
    return absl::InternalError(
        "Conceptual 'x' dimension label not found in initial labels");
  }

  if (initial_samples_per_pixel > 1) {
    if (auto it = label_to_index.find(sample_label);
        it != label_to_index.end()) {
      sample_dim_idx = it->second;
    } else {
      return absl::InternalError(tensorstore::StrCat(
          "Sample dimension label '", sample_label,
          "' not found in initial labels, but SamplesPerPixel=",
          initial_samples_per_pixel));
    }
  }

  // Assign chunk sizes based on dimension type
  for (DimensionIndex i = 0; i < initial_rank; ++i) {
    if (i == y_dim_idx) {
      initial_chunk_shape[i] = base_ifd.chunk_height;
      if (initial_chunk_shape[i] <= 0)
        return absl::InvalidArgumentError(
            "TIFF TileLength/RowsPerStrip must be positive");
    } else if (i == x_dim_idx) {
      initial_chunk_shape[i] = base_ifd.chunk_width;
      if (initial_chunk_shape[i] <= 0)
        return absl::InvalidArgumentError(
            "TIFF TileWidth must be positive (or image width for strips)");
    } else if (i == sample_dim_idx) {
      if (initial_planar_config ==
          internal_tiff_kvstore::PlanarConfigType::kChunky) {
        initial_chunk_shape[i] = initial_samples_per_pixel;
      } else {  // Planar
        initial_chunk_shape[i] = 1;
      }
      if (initial_chunk_shape[i] <= 0)
        return absl::InvalidArgumentError("SamplesPerPixel must be positive");
    } else {
      initial_chunk_shape[i] = 1;  // Assume stacked dims are chunked at size 1
    }
  }

  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::ChunkShape(initial_chunk_shape, all_dims_hard)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(
      ChunkLayout::CodecChunkShape(initial_chunk_shape, all_dims_hard)));

  // 4. Set Other Defaults (Soft Constraints)
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::WriteChunkElements(
      ChunkLayout::kDefaultShapeValue, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::ReadChunkElements(
      ChunkLayout::kDefaultShapeValue, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::CodecChunkElements(
      ChunkLayout::kDefaultShapeValue, /*hard=*/false)));

  std::vector<double> default_aspect_ratio(
      initial_rank, ChunkLayout::kDefaultAspectRatioValue);
  tensorstore::span<const double> default_aspect_ratio_span =
      default_aspect_ratio;

  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::WriteChunkAspectRatio(
      default_aspect_ratio_span, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::ReadChunkAspectRatio(
      default_aspect_ratio_span, /*hard=*/false)));
  TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::CodecChunkAspectRatio(
      default_aspect_ratio_span, /*hard=*/false)));

  return layout;
}

Result<SharedArray<const void>> DecodeChunk(const TiffMetadata& metadata,
                                            absl::Cord buffer) {
  riegeli::CordReader<> base_reader(&buffer);
  riegeli::Reader* data_reader = &base_reader;

  std::unique_ptr<riegeli::Reader> decompressor_reader;
  if (metadata.compressor) {
    decompressor_reader =
        metadata.compressor->GetReader(base_reader, metadata.dtype.size());
    if (!decompressor_reader) {
      return absl::InvalidArgumentError(StrCat(
          "Failed to create decompressor reader for TIFF compression type: ",
          static_cast<int>(metadata.compression_type)));
    }
    data_reader = decompressor_reader.get();
    ABSL_LOG_IF(INFO, tiff_metadata_logging)
        << "Applied decompressor for type "
        << static_cast<int>(metadata.compression_type);
  } else {
    ABSL_LOG_IF(INFO, tiff_metadata_logging)
        << "No decompression needed (raw).";
  }

  tensorstore::span<const Index> chunk_shape =
      metadata.chunk_layout.read_chunk_shape();

  std::vector<Index> buffer_data_shape_vec;
  buffer_data_shape_vec.reserve(metadata.rank);
  if (metadata.planar_config == PlanarConfigType::kPlanar) {
    // Find sample dimension index from mapping
    DimensionIndex sample_dim =
        metadata.dimension_mapping.ts_sample_dim.value_or(-1);
    if (sample_dim == -1 && metadata.samples_per_pixel > 1)
      return absl::InternalError(
          "Planar config with spp > 1 requires a sample dimension in mapping");
    // Assume chunk shape from layout reflects the grid {1, stack..., h, w}
    buffer_data_shape_vec.assign(chunk_shape.begin(), chunk_shape.end());
  } else {  // Chunky or single sample
    // Grid chunk shape is {stack..., h, w}. Component shape has spp at the end.
    buffer_data_shape_vec.assign(chunk_shape.begin(), chunk_shape.end());
    if (static_cast<DimensionIndex>(buffer_data_shape_vec.size()) !=
        metadata.rank) {
      return absl::InternalError(StrCat(
          "Internal consistency error: Buffer data shape rank (",
          buffer_data_shape_vec.size(), ") does not match component rank (",
          metadata.rank, ") in chunky mode"));
    }
  }
  tensorstore::span<const Index> buffer_data_shape = buffer_data_shape_vec;

  endian source_endian =
      (metadata.endian == internal_tiff_kvstore::Endian::kLittle)
          ? endian::little
          : endian::big;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto decoded_array, internal::DecodeArrayEndian(
                              *data_reader, metadata.dtype, buffer_data_shape,
                              source_endian, metadata.layout_order));

  if (!data_reader->VerifyEndAndClose()) {
    return absl::DataLossError(
        StrCat("Error reading chunk data: ", data_reader->status().message()));
  }

  return decoded_array;
}

// Validates that dtype is supported by the TIFF driver implementation.
absl::Status ValidateDataType(DataType dtype) {
  ABSL_CHECK(dtype.valid());
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported TIFF data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffSpecOptions::IfdStackingOptions,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffSpecOptions::IfdStackingOptions>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffSpecOptions,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffSpecOptions>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffMetadataConstraints>())
