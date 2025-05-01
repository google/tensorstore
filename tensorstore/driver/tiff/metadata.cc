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
    const std::vector<std::string>& final_labels,
    const std::optional<TiffSpecOptions::IfdStackingOptions>& stacking_info,
    const std::optional<std::string>& sample_dimension_label,
    std::string_view implicit_y_label, std::string_view implicit_x_label,
    std::string_view default_sample_label, PlanarConfigType planar_config,
    uint16_t samples_per_pixel) {
  TiffDimensionMapping mapping;
  const DimensionIndex final_rank = final_labels.size();
  mapping.labels_by_ts_dim.resize(final_rank);

  // Create a map from final label -> final index for quick lookup
  absl::flat_hash_map<std::string_view, DimensionIndex> label_to_final_idx;
  for (DimensionIndex i = 0; i < final_rank; ++i) {
    label_to_final_idx[final_labels[i]] = i;
  }

  // Map Y and X
  if (auto it = label_to_final_idx.find(implicit_y_label);
      it != label_to_final_idx.end()) {
    mapping.ts_y_dim = it->second;
    mapping.labels_by_ts_dim[it->second] = std::string(implicit_y_label);
  }
  if (auto it = label_to_final_idx.find(implicit_x_label);
      it != label_to_final_idx.end()) {
    mapping.ts_x_dim = it->second;
    mapping.labels_by_ts_dim[it->second] = std::string(implicit_x_label);
  }

  // Map Sample dimension (only if spp > 1)
  if (samples_per_pixel > 1) {
    std::string_view actual_sample_label =
        sample_dimension_label ? std::string_view(*sample_dimension_label)
                               : default_sample_label;
    if (auto it = label_to_final_idx.find(actual_sample_label);
        it != label_to_final_idx.end()) {
      mapping.ts_sample_dim = it->second;
      mapping.labels_by_ts_dim[it->second] = std::string(actual_sample_label);
    }
    // It's possible the user filtered out the sample dim via schema, so absence
    // isn't necessarily an error here.
  }

  // Map Stacked dimensions
  if (stacking_info) {
    for (const auto& stack_label : stacking_info->dimensions) {
      if (auto it = label_to_final_idx.find(stack_label);
          it != label_to_final_idx.end()) {
        mapping.ts_stacked_dims[stack_label] = it->second;
        mapping.labels_by_ts_dim[it->second] = stack_label;
      } else {
        // This dimension might have been filtered out by schema. Log if needed.
        ABSL_LOG_IF(INFO, tiff_metadata_logging)
            << "Stacked dimension label '" << stack_label
            << "' specified in options but not found in final dimension "
               "labels.";
      }
    }
  }

  return mapping;
}

auto IfdStackingOptionsBinder = jb::Validate(
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

      // Validate relationship between dimension_sizes and ifd_count
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

      // Validate ifd_sequence_order
      if (obj->ifd_sequence_order) {
        if (obj->ifd_sequence_order->size() != obj->dimensions.size()) {
          return absl::InvalidArgumentError(
              tensorstore::StrCat("\"ifd_sequence_order\" length (",
                                  obj->ifd_sequence_order->size(),
                                  ") must match \"dimensions\" length (",
                                  obj->dimensions.size(), ")"));
        }
        // Check if it's a permutation of dimensions
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

// Implement JSON binder for TiffMetadataConstraints here
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

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TiffSpecOptions,
    jb::Object(
        jb::Member("ifd",
                   jb::Projection<&TiffSpecOptions::ifd_index>(jb::DefaultValue(
                       [](auto* v) { *v = 0; }, jb::Integer<uint32_t>(0)))),
        jb::Member("ifd_stacking",
                   jb::Projection<&TiffSpecOptions::ifd_stacking>(
                       jb::Optional(IfdStackingOptionsBinder))),
        jb::Member("sample_dimension_label",
                   jb::Projection<&TiffSpecOptions::sample_dimension_label>(
                       jb::Optional(jb::NonEmptyStringBinder)))))

// ResolveMetadata Implementation
Result<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
    const internal_tiff_kvstore::TiffParseResult& source,
    const TiffSpecOptions& options, const Schema& schema) {
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Resolving TIFF metadata. Options: "
      << jb::ToJson(options).value_or(::nlohmann::json::object());

  auto metadata = std::make_shared<TiffMetadata>();
  metadata->endian = source.endian;

  // --- Initial Interpretation based on TiffSpecOptions ---
  DimensionIndex initial_rank;
  std::vector<Index> initial_shape;
  std::vector<std::string> initial_labels;
  const internal_tiff_kvstore::ImageDirectory* base_ifd_ptr = nullptr;
  size_t num_stack_dims = 0;           // Number of dimensions added by stacking
  std::vector<Index> stack_sizes_vec;  // Store stack sizes if applicable

  const std::string implicit_y_label = "y";
  const std::string implicit_x_label = "x";
  const std::string default_sample_label = "c";
  const std::string& sample_label =
      options.sample_dimension_label.value_or(default_sample_label);

  if (options.ifd_stacking) {
    // --- Multi-IFD Stacking Mode ---
    metadata->stacking_info = *options.ifd_stacking;
    const auto& stacking = *metadata->stacking_info;
    num_stack_dims = stacking.dimensions.size();

    uint64_t total_ifds_needed = 0;
    if (stacking.dimension_sizes) {
      stack_sizes_vec = *stacking.dimension_sizes;
      total_ifds_needed = 1;
      uint64_t max_val = std::numeric_limits<uint64_t>::max();
      for (Index size : stack_sizes_vec) {
        uint64_t u_size = static_cast<uint64_t>(size);
        if (size <= 0)
          return absl::InternalError(
              "Non-positive dimension_size found after validation");
        if (total_ifds_needed > max_val / u_size) {
          return absl::InvalidArgumentError(
              "Product of dimension_sizes overflows uint64_t");
        }
        total_ifds_needed *= u_size;
      }
    } else {  // dimension_sizes was absent, use ifd_count
      total_ifds_needed =
          *stacking.ifd_count;  // Already validated to exist and be positive
      stack_sizes_vec.push_back(static_cast<Index>(total_ifds_needed));
      // Update the stored stacking_info to include the inferred dimension_sizes
      metadata->stacking_info->dimension_sizes = stack_sizes_vec;
    }

    metadata->num_ifds_read = total_ifds_needed;
    metadata->base_ifd_index = 0;  // Stacking starts from IFD 0

    if (total_ifds_needed == 0 ||
        total_ifds_needed > source.image_directories.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Required %d IFDs for stacking, but only %d available/parsed",
          total_ifds_needed, source.image_directories.size()));
    }

    base_ifd_ptr = &source.image_directories[0];

    for (size_t i = 1; i < total_ifds_needed; ++i) {
      TENSORSTORE_RETURN_IF_ERROR(
          CheckIfdUniformity(*base_ifd_ptr, source.image_directories[i], i));
    }

  } else {
    // --- Single IFD Mode ---
    metadata->base_ifd_index = options.ifd_index;
    metadata->num_ifds_read = 1;
    num_stack_dims = 0;  // Ensure this is 0 for single IFD mode

    if (metadata->base_ifd_index >= source.image_directories.size()) {
      return absl::NotFoundError(absl::StrFormat(
          "Requested IFD index %d not found (found %d IFDs)",
          metadata->base_ifd_index, source.image_directories.size()));
    }
    base_ifd_ptr = &source.image_directories[metadata->base_ifd_index];
  }

  // --- Populate common metadata fields from base IFD ---
  assert(base_ifd_ptr != nullptr);
  const auto& base_ifd = *base_ifd_ptr;
  metadata->compression_type =
      static_cast<CompressionType>(base_ifd.compression);
  metadata->planar_config =
      static_cast<PlanarConfigType>(base_ifd.planar_config);
  metadata->samples_per_pixel = base_ifd.samples_per_pixel;
  metadata->ifd0_chunk_width = base_ifd.chunk_width;
  metadata->ifd0_chunk_height = base_ifd.chunk_height;
  auto planar_config = metadata->planar_config;

  // --- Determine Initial TensorStore Structure based on Planar Config ---
  initial_labels.clear();
  initial_shape.clear();

  if (planar_config == PlanarConfigType::kPlanar) {
    if (metadata->samples_per_pixel <= 1) {
      return absl::InvalidArgumentError(
          "PlanarConfiguration=2 requires SamplesPerPixel > 1");
    }
    initial_rank = 1 + num_stack_dims + 2;
    initial_shape.push_back(static_cast<Index>(metadata->samples_per_pixel));
    initial_labels.push_back(sample_label);
    if (metadata->stacking_info) {
      const auto& stack_dims = metadata->stacking_info->dimensions;
      initial_shape.insert(initial_shape.end(), stack_sizes_vec.begin(),
                           stack_sizes_vec.end());
      initial_labels.insert(initial_labels.end(), stack_dims.begin(),
                            stack_dims.end());
    }
    initial_shape.push_back(static_cast<Index>(base_ifd.height));
    initial_labels.push_back(implicit_y_label);
    initial_shape.push_back(static_cast<Index>(base_ifd.width));
    initial_labels.push_back(implicit_x_label);

  } else {  // Chunky (or single sample)
    initial_rank =
        num_stack_dims + 2 + (metadata->samples_per_pixel > 1 ? 1 : 0);
    if (metadata->stacking_info) {
      initial_shape = stack_sizes_vec;
      initial_labels = metadata->stacking_info->dimensions;
    }
    initial_shape.push_back(static_cast<Index>(base_ifd.height));
    initial_labels.push_back(implicit_y_label);
    initial_shape.push_back(static_cast<Index>(base_ifd.width));
    initial_labels.push_back(implicit_x_label);
    if (metadata->samples_per_pixel > 1) {
      initial_shape.push_back(static_cast<Index>(metadata->samples_per_pixel));
      initial_labels.push_back(sample_label);
    }
  }

  // --- Get Initial Properties ---
  TENSORSTORE_ASSIGN_OR_RETURN(DataType initial_dtype,
                               GetDataTypeFromTiff(base_ifd));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(initial_dtype));

  // Determine Grid Rank and Dimensions relative to the *initial* layout
  DimensionIndex grid_rank;
  std::vector<DimensionIndex> grid_dims_in_initial_rank;
  std::vector<Index> grid_chunk_shape_vec;
  if (planar_config == PlanarConfigType::kPlanar) {
    grid_rank = 1 + num_stack_dims + 2;
    grid_dims_in_initial_rank.resize(grid_rank);
    grid_chunk_shape_vec.resize(grid_rank);
    size_t current_grid_dim = 0;
    grid_dims_in_initial_rank[current_grid_dim] = 0;  // Sample dim
    grid_chunk_shape_vec[current_grid_dim] = 1;
    current_grid_dim++;
    for (size_t i = 0; i < num_stack_dims; ++i) {
      grid_dims_in_initial_rank[current_grid_dim] = 1 + i;  // Stacked dim index
      grid_chunk_shape_vec[current_grid_dim] = 1;
      current_grid_dim++;
    }
    grid_dims_in_initial_rank[current_grid_dim] =
        1 + num_stack_dims;  // Y dim index
    grid_chunk_shape_vec[current_grid_dim] =
        static_cast<Index>(base_ifd.chunk_height);
    current_grid_dim++;
    grid_dims_in_initial_rank[current_grid_dim] =
        1 + num_stack_dims + 1;  // X dim index
    grid_chunk_shape_vec[current_grid_dim] =
        static_cast<Index>(base_ifd.chunk_width);
  } else {  // Chunky
    grid_rank = num_stack_dims + 2;
    grid_dims_in_initial_rank.resize(grid_rank);
    grid_chunk_shape_vec.resize(grid_rank);
    size_t current_grid_dim = 0;
    for (size_t i = 0; i < num_stack_dims; ++i) {
      grid_dims_in_initial_rank[current_grid_dim] = i;  // Stacked dim index
      grid_chunk_shape_vec[current_grid_dim] = 1;
      current_grid_dim++;
    }
    grid_dims_in_initial_rank[current_grid_dim] =
        num_stack_dims;  // Y dim index
    grid_chunk_shape_vec[current_grid_dim] =
        static_cast<Index>(base_ifd.chunk_height);
    current_grid_dim++;
    grid_dims_in_initial_rank[current_grid_dim] =
        num_stack_dims + 1;  // X dim index
    grid_chunk_shape_vec[current_grid_dim] =
        static_cast<Index>(base_ifd.chunk_width);
  }
  ABSL_CHECK(static_cast<DimensionIndex>(grid_chunk_shape_vec.size()) ==
             grid_rank);

  // Create initial CodecSpec
  auto initial_codec_spec_ptr =
      internal::CodecDriverSpec::Make<TiffCodecSpec>();
  initial_codec_spec_ptr->compression_type = metadata->compression_type;
  CodecSpec initial_codec(std::move(initial_codec_spec_ptr));

  // Initial Dimension Units (default unspecified)
  DimensionUnitsVector initial_units(initial_rank);

  // --- Reconcile with Schema ---
  Schema merged_schema = schema;  // Start with user-provided schema

  // Merge dtype
  if (merged_schema.dtype().valid() &&
      !IsPossiblySameDataType(merged_schema.dtype(), initial_dtype)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Schema dtype ", merged_schema.dtype(),
        " is incompatible with TIFF dtype ", initial_dtype));
  }
  TENSORSTORE_RETURN_IF_ERROR(merged_schema.Set(initial_dtype));

  // Merge rank
  TENSORSTORE_RETURN_IF_ERROR(merged_schema.Set(RankConstraint{initial_rank}));

  // Build initial domain
  TENSORSTORE_ASSIGN_OR_RETURN(IndexDomain<> initial_domain,
                               IndexDomainBuilder(initial_rank)
                                   .shape(initial_shape)
                                   .labels(initial_labels)
                                   .Finalize());
  // Merge domain constraints
  TENSORSTORE_ASSIGN_OR_RETURN(
      IndexDomain<> final_domain,
      MergeIndexDomains(merged_schema.domain(), initial_domain));
  TENSORSTORE_RETURN_IF_ERROR(merged_schema.Set(std::move(final_domain)));

  // Merge chunk layout constraints
  ChunkLayout final_layout = merged_schema.chunk_layout();
  // Ensure rank matches before merging
  if (final_layout.rank() == dynamic_rank &&
      merged_schema.rank() != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(
        final_layout.Set(RankConstraint{merged_schema.rank()}));
  } else if (final_layout.rank() != dynamic_rank &&
             final_layout.rank() != merged_schema.rank()) {
    return absl::InvalidArgumentError("Schema chunk_layout rank mismatch");
  }
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state BEFORE applying any TIFF constraints: " << final_layout;

  // Apply TIFF Hard Constraints Directly to the final_layout
  // 1. Grid Shape Hard Constraint (only for grid dims)
  std::vector<Index> full_rank_chunk_shape(initial_rank, 0);
  DimensionSet shape_hard_constraint_dims;
  for (DimensionIndex i = 0; i < grid_rank; ++i) {
    DimensionIndex final_dim_idx = grid_dims_in_initial_rank[i];
    if (final_dim_idx >= initial_rank)
      return absl::InternalError("Grid dimension index out of bounds");
    full_rank_chunk_shape[final_dim_idx] = grid_chunk_shape_vec[i];
    shape_hard_constraint_dims[final_dim_idx] = true;
  }
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Applying TIFF Shape Constraint: shape="
      << tensorstore::span<const Index>(
             full_rank_chunk_shape)  // Variable from your code
      << " hard_dims="
      << shape_hard_constraint_dims;  // Variable from your code

  TENSORSTORE_RETURN_IF_ERROR(final_layout.Set(ChunkLayout::ChunkShape(
      full_rank_chunk_shape, shape_hard_constraint_dims)));

  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state AFTER applying Shape constraint: " << final_layout;

  // 2. Grid Origin Hard Constraint (only for grid dims)
    // --- CORRECTION START ---
    // Get existing origins and hardness from the layout (after schema merge)
    std::vector<Index> current_origin(initial_rank);
    // Use accessor that returns span<const Index> or equivalent
    span<const Index> layout_origin_span = final_layout.grid_origin();
    std::copy(layout_origin_span.begin(), layout_origin_span.end(), current_origin.begin());
    DimensionSet current_hard_origin_dims = final_layout.grid_origin().hard_constraint;

    // Prepare the new constraints from TIFF grid
    std::vector<Index> tiff_origin_values(initial_rank, kImplicit);
    DimensionSet tiff_origin_hard_dims; // Define the DimensionSet for TIFF constraints
     for (DimensionIndex i = 0; i < grid_rank; ++i) {
        DimensionIndex final_dim_idx = grid_dims_in_initial_rank[i];
        if (final_dim_idx >= initial_rank) return absl::InternalError("Grid dimension index out of bounds");
        tiff_origin_values[final_dim_idx] = 0; // TIFF grid origin is 0
        tiff_origin_hard_dims[final_dim_idx] = true; // Mark this grid dim as hard
    }

    // Apply the TIFF constraints.
    TENSORSTORE_RETURN_IF_ERROR(final_layout.Set(
        ChunkLayout::GridOrigin(tiff_origin_values, tiff_origin_hard_dims)));

    // NOW, ensure ALL dimensions have a hard origin constraint IF any were set hard.
    // Check the combined hardness after applying TIFF constraints.
    DimensionSet combined_hard_dims = final_layout.grid_origin().hard_constraint;
    if (combined_hard_dims.any()) {
        std::vector<Index> final_origin_values(initial_rank);
        DimensionSet final_origin_hard_dims; // This will mark ALL dimensions hard
        span<const Index> origin_after_tiff_set = final_layout.grid_origin(); // Get current state

        for(DimensionIndex i = 0; i < initial_rank; ++i) {
            // Default to 0 if still implicit after schema and TIFF merge
            final_origin_values[i] = (origin_after_tiff_set[i] != kImplicit) ? origin_after_tiff_set[i] : 0;
            final_origin_hard_dims[i] = true; // Mark ALL dimensions as hard
        }
         // Re-apply the origin with *all* dimensions marked hard
         TENSORSTORE_RETURN_IF_ERROR(final_layout.Set(
             ChunkLayout::GridOrigin(final_origin_values, final_origin_hard_dims)));
    }
    // --- CORRECTION END ---

  // 3. Apply Default Inner Order (Soft Constraint for full rank)
  std::vector<DimensionIndex> default_inner_order(initial_rank);
  std::iota(default_inner_order.begin(), default_inner_order.end(), 0);
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Applying TIFF InnerOrder (Soft) Constraint: order="
      << tensorstore::span<const DimensionIndex>(
             default_inner_order);  // Variable from your code

  TENSORSTORE_RETURN_IF_ERROR(final_layout.Set(
      ChunkLayout::InnerOrder(default_inner_order, /*hard=*/false)));
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state AFTER applying InnerOrder constraint: " << final_layout;

  // Update the schema with the layout containing merged constraints
  TENSORSTORE_RETURN_IF_ERROR(merged_schema.Set(final_layout));

  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state AFTER merged_schema.Set(final_layout): "
      << merged_schema.chunk_layout();  // Log directly from schema

  // Merge codec spec
  CodecSpec schema_codec = merged_schema.codec();
  if (schema_codec.valid()) {
    // Use MergeFrom on the initial CodecSpec pointer
    TENSORSTORE_RETURN_IF_ERROR(
        initial_codec.MergeFrom(schema_codec),
        tensorstore::MaybeAnnotateStatus(
            _, "Schema codec is incompatible with TIFF file compression"));
  }
  TENSORSTORE_RETURN_IF_ERROR(
      merged_schema.Set(initial_codec));  // Set merged spec back

  // Merge dimension units
  DimensionUnitsVector final_units(merged_schema.dimension_units());
  if (final_units.empty() && merged_schema.rank() != dynamic_rank) {
    final_units.resize(merged_schema.rank());
  } else if (!final_units.empty() &&
             static_cast<DimensionIndex>(final_units.size()) !=
                 merged_schema.rank()) {
    return absl::InvalidArgumentError("Schema dimension_units rank mismatch");
  }
  TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(final_units, initial_units));
  TENSORSTORE_RETURN_IF_ERROR(
      merged_schema.Set(Schema::DimensionUnits(final_units)));

  // Check fill value
  if (merged_schema.fill_value().valid()) {
    return absl::InvalidArgumentError(
        "fill_value not supported by TIFF format");
  }

  // --- Finalize Resolved Metadata ---
  metadata->chunk_layout = merged_schema.chunk_layout();
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state BEFORE Finalize(): " << metadata->chunk_layout;

  // Finalize the layout AFTER retrieving it from the schema
  TENSORSTORE_RETURN_IF_ERROR(metadata->chunk_layout.Finalize());
  ABSL_LOG_IF(INFO, tiff_metadata_logging)
      << "Layout state AFTER Finalize(): " << metadata->chunk_layout;

  // Populate the TiffMetadata struct from the finalized merged_schema
  metadata->rank = merged_schema.rank();
  metadata->shape.assign(merged_schema.domain().shape().begin(),
                         merged_schema.domain().shape().end());
  metadata->dtype = merged_schema.dtype();
  metadata->dimension_units = std::move(final_units);
  metadata->dimension_labels.assign(merged_schema.domain().labels().begin(),
                                    merged_schema.domain().labels().end());
  metadata->fill_value = SharedArray<const void>();

  // Get the final compression type from the merged codec spec *within the
  // schema*
  const TiffCodecSpec* final_codec_spec_ptr = nullptr;
  if (merged_schema.codec().valid()) {
    final_codec_spec_ptr =
        dynamic_cast<const TiffCodecSpec*>(merged_schema.codec().get());
  }
  CompressionType final_compression_type =
      final_codec_spec_ptr && final_codec_spec_ptr->compression_type
          ? *final_codec_spec_ptr->compression_type
          : CompressionType::kNone;

  // Use the helper to instantiate the compressor based on the final type and
  // schema codec
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->compressor,
      GetEffectiveCompressor(final_compression_type, merged_schema.codec()));
  // Update metadata->compression_type to reflect the final resolved type
  metadata->compression_type = final_compression_type;

  // Finalize layout order enum
  TENSORSTORE_ASSIGN_OR_RETURN(
      metadata->layout_order,
      GetLayoutOrderFromInnerOrder(metadata->chunk_layout.inner_order()));

  // Build the final dimension mapping
  metadata->dimension_mapping = BuildDimensionMapping(
      metadata->dimension_labels, metadata->stacking_info,
      options.sample_dimension_label, implicit_y_label, implicit_x_label,
      default_sample_label, planar_config, metadata->samples_per_pixel);

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

// --- ResolveMetadata Implementation ---
// Result<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
//     const TiffParseResult& source, const TiffSpecOptions& options,
//     const Schema& schema) {
//   ABSL_LOG_IF(INFO, tiff_metadata_logging)
//       << "Resolving TIFF metadata for IFD: " << options.ifd_index;

//   // 1. Select and Validate IFD
//   if (options.ifd_index >= source.image_directories.size()) {
//     return absl::NotFoundError(
//         tensorstore::StrCat("Requested IFD index ", options.ifd_index,
//                             " not found in TIFF file (found ",
//                             source.image_directories.size(), " IFDs)"));
//   }
//   // Get the relevant ImageDirectory directly from the TiffParseResult
//   const ImageDirectory& img_dir =
//   source.image_directories[options.ifd_index];

//   // 2. Initial Interpretation (Basic Properties)
//   auto metadata = std::make_shared<TiffMetadata>();
//   metadata->ifd_index = options.ifd_index;
//   metadata->num_ifds = 1;  // Stacking not implemented
//   metadata->endian = source.endian;

//   // Validate Planar Configuration and Compression early
//   metadata->planar_config =
//       static_cast<PlanarConfigType>(img_dir.planar_config);
//   if (metadata->planar_config != PlanarConfigType::kChunky) {
//     return absl::UnimplementedError(
//         tensorstore::StrCat("PlanarConfiguration=", img_dir.planar_config,
//                             " is not supported yet (only Chunky=1)"));
//   }

//   metadata->compression_type =
//       static_cast<CompressionType>(img_dir.compression);

//   // Determine rank, shape, dtype
//   TENSORSTORE_ASSIGN_OR_RETURN(
//       metadata->shape, GetShapeAndRankFromTiff(img_dir, metadata->rank));

//   if (metadata->rank == dynamic_rank) {
//     return absl::InvalidArgumentError("Could not determine rank from TIFF
//     IFD");
//   }

//   TENSORSTORE_ASSIGN_OR_RETURN(metadata->dtype,
//   GetDataTypeFromTiff(img_dir)); metadata->samples_per_pixel =
//   img_dir.samples_per_pixel;

//   // 3. Initial Chunk Layout
//   ChunkLayout& layout = metadata->chunk_layout;
//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(RankConstraint{metadata->rank}));

//   bool planar_lead = (metadata->planar_config != PlanarConfigType::kChunky);
//   TENSORSTORE_ASSIGN_OR_RETURN(
//       auto chunk_shape,
//       GetChunkShapeFromTiff(img_dir, metadata->rank, planar_lead));

//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::ChunkShape(chunk_shape)));
//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(
//       ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(metadata->rank))));
//   TENSORSTORE_ASSIGN_OR_RETURN(auto default_inner_order,
//                                GetInnerOrderFromTiff(metadata->rank));

//   // 4. Initial Codec Spec
//   TENSORSTORE_ASSIGN_OR_RETURN(
//       std::string_view type_id,
//       CompressionTypeToStringId(metadata->compression_type));

//   // Use the tiff::Compressor binder to get the instance.
//   // We pass a dummy JSON object containing only the "type" field.
//   ::nlohmann::json compressor_json = {{"type", type_id}};
//   TENSORSTORE_ASSIGN_OR_RETURN(
//       metadata->compressor,
//       Compressor::FromJson(
//           std::move(compressor_json),
//           internal::JsonSpecifiedCompressor::FromJsonOptions{}));

//   // Check if the factory returned an unimplemented error (for unsupported
//   // types)
//   if (!metadata->compressor &&
//       metadata->compression_type != CompressionType::kNone) {
//     // This case should ideally be caught by CompressionTypeToStringId,
//     // but double-check based on registry content.
//     return absl::UnimplementedError(tensorstore::StrCat(
//         "TIFF compression type ",
//         static_cast<int>(metadata->compression_type), " (", type_id,
//         ") is registered but not supported by this driver yet."));
//   }

//   // 5. Initial Dimension Units (Default: Unknown)
//   metadata->dimension_units.resize(metadata->rank);

//   // --- OME-XML Interpretation Placeholder ---
//   // if (options.use_ome_metadata && source.ome_xml_string) {
//   //    TENSORSTORE_ASSIGN_OR_RETURN(OmeXmlData ome_data,
//   //    ParseOmeXml(*source.ome_xml_string));
//   //    // Apply OME data: potentially override rank, shape, dtype, units,
//   //    inner_order
//   //    // This requires mapping between OME concepts and TensorStore
//   //    schema ApplyOmeDataToMetadata(*metadata, ome_data);
//   // }

//   // 6. Merge Schema Constraints
//   // Data Type: Check for compatibility (schema.dtype() vs metadata->dtype)
//   if (schema.dtype().valid() &&
//       !IsPossiblySameDataType(metadata->dtype, schema.dtype())) {
//     return absl::FailedPreconditionError(
//         StrCat("Schema dtype ", schema.dtype(),
//                " is incompatible with TIFF dtype ", metadata->dtype));
//   }

//   // Chunk Layout: Merge schema constraints *component-wise*.
//   const ChunkLayout& schema_layout = schema.chunk_layout();
//   if (schema_layout.rank() != dynamic_rank) {
//     // Rank constraint from schema is checked against metadata rank
//     TENSORSTORE_RETURN_IF_ERROR(
//         layout.Set(RankConstraint{schema_layout.rank()}));
//   }
//   // Apply schema constraints for individual components. This will respect
//   // existing hard constraints (like chunk_shape from TIFF tags).
//   if (!schema_layout.inner_order().empty()) {
//     TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.inner_order()));
//   }
//   if (!schema_layout.grid_origin().empty()) {
//     TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.grid_origin()));
//   }
//   // Setting write/read/codec components handles hard/soft constraint
//   merging.
//   // This should now correctly fail if schema tries to set a conflicting hard
//   // shape.
//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.write_chunk()));
//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.read_chunk()));
//   TENSORSTORE_RETURN_IF_ERROR(layout.Set(schema_layout.codec_chunk()));

//   // *After* merging schema, apply TIFF defaults *if still unspecified*,
//   // setting them as SOFT constraints to allow schema to override.
//   if (layout.inner_order().empty()) {
//     TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::InnerOrder(
//         default_inner_order, /*hard_constraint=*/false)));
//   }

//   // Codec Spec Validation
//   if (schema.codec().valid()) {
//     // Create a temporary TiffCodecSpec representing the file's compression
//     auto file_codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
//     file_codec_spec->compression_type = metadata->compression_type;

//     // Attempt to merge the user's schema codec into the file's codec spec.
//     // This validates compatibility.
//     TENSORSTORE_RETURN_IF_ERROR(
//         file_codec_spec->MergeFrom(schema.codec()),
//         tensorstore::MaybeAnnotateStatus(
//             _, "Schema codec is incompatible with TIFF file compression"));
//   }

//   // Dimension Units: Merge schema constraints *only if* schema units are
//   valid. if (schema.dimension_units().valid()) {
//     TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(metadata->dimension_units,
//                                                     schema.dimension_units()));
//   }

//   if (schema.fill_value().valid()) {
//     return absl::InvalidArgumentError(
//         "fill_value not supported by TIFF format");
//   }

//   // 7. Finalize Layout
//   TENSORSTORE_RETURN_IF_ERROR(metadata->chunk_layout.Finalize());

//   TENSORSTORE_ASSIGN_OR_RETURN(
//       metadata->layout_order,
//       GetLayoutOrderFromInnerOrder(metadata->chunk_layout.inner_order()));

//   // 8. Final consistency: chunk_shape must divide shape
//   // NB: Not a given apparently...
//   // const auto& cs = metadata->chunk_layout.read_chunk().shape();
//   // for (DimensionIndex d = 0; d < metadata->rank; ++d) {
//   //   if (metadata->shape[d] % cs[d] != 0) {
//   //     return absl::FailedPreconditionError(
//   //         StrCat("Chunk shape ", cs, " does not evenly divide image shape
//   ",
//   //                metadata->shape));
//   //   }
//   // }

//   ABSL_LOG_IF(INFO, tiff_metadata_logging)
//       << "Resolved TiffMetadata: rank=" << metadata->rank
//       << ", shape=" << tensorstore::span(metadata->shape)
//       << ", dtype=" << metadata->dtype
//       << ", chunk_shape=" << metadata->chunk_layout.read_chunk().shape()
//       << ", compression=" << static_cast<int>(metadata->compression_type)
//       << ", layout_enum=" << metadata->layout_order << ", endian="
//       << (metadata->endian == internal_tiff_kvstore::Endian::kLittle ?
//       "little"
//                                                                      :
//                                                                      "big");

//   return std::const_pointer_cast<const TiffMetadata>(metadata);
// }

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
  // TODO: Implement axis validation
  // if (user_constraints.axes.has_value()) { ... }

  // Validate Chunk Shape (if added to constraints)
  // TODO: Implement chunk shape validation
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
  if (dtype.valid()) TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(dtype));
  return dtype;
}

// Helper to get the effective compressor based on type and codec spec options
Result<Compressor> GetEffectiveCompressor(CompressionType compression_type,
                                          const CodecSpec& schema_codec) {
  // Determine initial compressor type from TIFF tag
  TENSORSTORE_ASSIGN_OR_RETURN(std::string_view type_id,
                               CompressionTypeToStringId(compression_type));

  // Create a TiffCodecSpec representing the TIFF file's compression
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

  // Get the final compression type after merging
  auto final_compression_type =
      initial_codec_spec->compression_type.value_or(CompressionType::kNone);

  if (final_compression_type == CompressionType::kNone) {
    return Compressor{nullptr};  // Explicitly return null pointer for raw
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

  // Check if the factory actually supports this type
  if (!final_compressor && final_compression_type != CompressionType::kNone) {
    return absl::UnimplementedError(tensorstore::StrCat(
        "TIFF compression type ", static_cast<int>(final_compression_type),
        " (", final_type_id, ") is not supported by this driver build."));
  }

  return final_compressor;
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
  tensorstore::span<const Index> chunk_shape =
      metadata.chunk_layout.read_chunk_shape();

  // DecodeArrayEndian needs the shape of the data *as laid out in
  // the buffer.
  // For chunky: This is {stack..., h, w, spp} potentially permuted by
  // layout_order. For planar: This is {1, stack..., h, w} potentially permuted
  // by layout_order.
  std::vector<Index> buffer_data_shape_vec;
  buffer_data_shape_vec.reserve(metadata.rank);
  if (metadata.planar_config == PlanarConfigType::kPlanar) {
    // Find sample dimension index from mapping
    DimensionIndex sample_dim =
        metadata.dimension_mapping.ts_sample_dim.value_or(-1);
    if (sample_dim == -1)
      return absl::InternalError(
          "Planar config without sample dimension in mapping");
    // Assume chunk shape from layout reflects the grid {1, stack..., h, w}
    buffer_data_shape_vec.assign(chunk_shape.begin(), chunk_shape.end());

  } else {  // Chunky or single sample
    // Find sample dimension index (if exists)
    DimensionIndex sample_dim =
        metadata.dimension_mapping.ts_sample_dim.value_or(-1);
    // Grid chunk shape is {stack..., h, w}. Component shape has spp at the end.
    buffer_data_shape_vec.assign(chunk_shape.begin(), chunk_shape.end());
    if (sample_dim != -1) {
      // Ensure rank matches
      if (static_cast<DimensionIndex>(buffer_data_shape_vec.size()) !=
          metadata.rank - 1) {
        return absl::InternalError(
            "Rank mismatch constructing chunky buffer shape");
      }
      buffer_data_shape_vec.push_back(
          static_cast<Index>(metadata.samples_per_pixel));
    } else {
      if (static_cast<DimensionIndex>(buffer_data_shape_vec.size()) !=
          metadata.rank) {
        return absl::InternalError(
            "Rank mismatch constructing single sample buffer shape");
      }
    }
  }
  tensorstore::span<const Index> buffer_data_shape = buffer_data_shape_vec;

  // 5. Determine Endianness for decoding
  endian source_endian =
      (metadata.endian == internal_tiff_kvstore::Endian::kLittle)
          ? endian::little
          : endian::big;

  // 6. Decode data from the reader into the array, handling endianness
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto decoded_array, internal::DecodeArrayEndian(
                              *data_reader, metadata.dtype, buffer_data_shape,
                              source_endian, metadata.layout_order));

  // 7. Verify reader reached end (important for compressed streams)
  if (!data_reader->VerifyEndAndClose()) {
    return absl::DataLossError(
        StrCat("Error reading chunk data: ", data_reader->status().message()));
  }

  // 8. Return the decoded array
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
    tensorstore::internal_tiff::TiffSpecOptions,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffSpecOptions>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_tiff::TiffMetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_tiff::TiffMetadataConstraints>())
