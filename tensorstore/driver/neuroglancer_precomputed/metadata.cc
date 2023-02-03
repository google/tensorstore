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

#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/bit_operations.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/enum.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/serialization/std_map.h"  // IWYU pragma: keep
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

namespace jb = ::tensorstore::internal_json_binding;
using ::tensorstore::internal::MetadataMismatchError;

void to_json(::nlohmann::json& out,  // NOLINT
             const std::variant<NoShardingSpec, ShardingSpec>& s) {
  std::visit([&](const auto& x) { out = ::nlohmann::json(x); }, s);
}

std::string_view to_string(ScaleMetadata::Encoding e) {
  using E = ScaleMetadata::Encoding;
  switch (e) {
    case E::raw:
      return "raw";
    case E::jpeg:
      return "jpeg";
    case E::compressed_segmentation:
      return "compressed_segmentation";
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

namespace {

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,  DataTypeId::uint16_t,  DataTypeId::uint32_t,
    DataTypeId::int8_t,   DataTypeId::int16_t,   DataTypeId::int32_t,
    DataTypeId::uint64_t, DataTypeId::float32_t,
};

constexpr char kMultiscaleVolumeTypeId[] = "neuroglancer_multiscale_volume";

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

absl::Status ValidateEncodingDataType(ScaleMetadata::Encoding encoding,
                                      DataType dtype,
                                      std::optional<Index> num_channels) {
  switch (encoding) {
    case ScaleMetadata::Encoding::raw:
      break;
    case ScaleMetadata::Encoding::compressed_segmentation:
      if (!dtype.valid()) break;
      if (dtype.id() != DataTypeId::uint32_t &&
          dtype.id() != DataTypeId::uint64_t) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "compressed_segmentation encoding only supported for "
            "uint32 and uint64, not for ",
            dtype));
      }
      break;
    case ScaleMetadata::Encoding::jpeg:
      if (dtype.valid() && dtype.id() != DataTypeId::uint8_t) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"jpeg\" encoding only supported for uint8, not for ", dtype));
      }
      if (num_channels && *num_channels != 1 && *num_channels != 3) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"jpeg\" encoding only supports 1 or 3 channels, not ",
            *num_channels));
      }
      break;
  }
  return absl::OkStatus();
}

absl::Status CheckScaleBounds(BoxView<3> box) {
  for (int i = 0; i < 3; ++i) {
    if (!IndexInterval::ValidSized(box.origin()[i], box.shape()[i]) ||
        !IsFinite(box[i])) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "\"size\" of ", ::nlohmann::json(box.shape()).dump(),
          " and \"voxel_offset\" of ", ::nlohmann::json(box.origin()).dump(),
          " do not specify a valid region"));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateChunkSize(
    span<const Index, 3> chunk_size, span<const Index, 3> shape,
    const std::variant<NoShardingSpec, ShardingSpec>& sharding) {
  if (std::holds_alternative<NoShardingSpec>(sharding)) {
    // No constraints for unsharded format.
    return absl::OkStatus();
  }
  const auto bits = GetCompressedZIndexBits(shape, chunk_size);
  if (bits[0] + bits[1] + bits[2] > 64) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "\"size\" of ", ::nlohmann::json(shape).dump(),
        " with \"chunk_size\" of ", ::nlohmann::json(chunk_size).dump(),
        " is not compatible with sharded format because the chunk keys "
        "would exceed 64 bits"));
  }
  return absl::OkStatus();
}

/// Binder for std::variant<NoShardingSpec, ShardingSpec>, maps discarded
/// (missing) and nullptr values to NoShardingSpec{}.
constexpr static auto ShardingBinder = [](auto is_loading, const auto& options,
                                          auto* obj, auto* j) -> absl::Status {
  if constexpr (is_loading) {
    if (j->is_discarded() || j->is_null()) {
      *obj = NoShardingSpec{};
      return absl::OkStatus();
    }
    obj->template emplace<ShardingSpec>();
  } else {
    if (std::holds_alternative<NoShardingSpec>(*obj)) {
      *j = nullptr;
      return absl::OkStatus();
    }
  }
  return jb::DefaultBinder<ShardingSpec>(is_loading, options,
                                         std::get_if<ShardingSpec>(obj), j);
};

/// Binder for ScaleMetadata::Encoding
constexpr static auto ScaleMetatadaEncodingBinder() {
  return jb::Enum<ScaleMetadata::Encoding, std::string_view>({
      {ScaleMetadata::Encoding::raw, "raw"},
      {ScaleMetadata::Encoding::jpeg, "jpeg"},
      {ScaleMetadata::Encoding::compressed_segmentation,
       "compressed_segmentation"},
  });
}

constexpr static auto EncodingJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    using T = internal::remove_cvref_t<decltype(*obj)>;
    return jb::Sequence(
        jb::Member(
            "encoding",
            jb::Projection(&T::encoding,
                           maybe_optional(ScaleMetatadaEncodingBinder()))),
        jb::Member("jpeg_quality",
                   [maybe_optional](auto is_loading, const auto& options,
                                    auto* obj, auto* j) -> absl::Status {
                     if constexpr (is_loading) {
                       if (j->is_discarded()) return absl::OkStatus();
                       if (obj->encoding != ScaleMetadata::Encoding::jpeg) {
                         return absl::InvalidArgumentError(
                             "Only valid for \"jpeg\" encoding");
                       }
                     } else {
                       if (obj->encoding != ScaleMetadata::Encoding::jpeg) {
                         *j = ::nlohmann::json(
                             ::nlohmann::json::value_t::discarded);
                         return absl::OkStatus();
                       }
                     }
                     return jb::Projection(&T::jpeg_quality,
                                           maybe_optional(jb::Integer(0, 100)))(
                         is_loading, options, obj, j);
                   }))(is_loading, options, obj, j);
  };
};

constexpr static auto WrapInOptional = [](auto binder) {
  return jb::Optional(binder);
};

/// Common attributes for NeuroglancerPrecomputedCodecSpec,
/// ScaleMetadata and ScaleMetadataConstraints.
template <typename MaybeOptional>
constexpr auto ScaleMetadataCommon(MaybeOptional maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    using T = internal::remove_cvref_t<decltype(*obj)>;
    return jb::Sequence(
        jb::Member("key", jb::Projection(&T::key)),
        jb::Member(
            "resolution",
            jb::Projection(&T::resolution, maybe_optional(jb::FixedSizeArray(
                                               jb::LooseFloatBinder)))),
        jb::Member(
            "sharding",
            jb::Projection(
                &T::sharding,
                [=](auto is_loading, const auto& options, auto* obj, auto* j) {
                  if constexpr (!is_loading &&
                                std::is_same_v<T, ScaleMetadata>) {
                    if (std::holds_alternative<NoShardingSpec>(*obj))
                      return absl::OkStatus();
                  }
                  return maybe_optional(ShardingBinder)(is_loading, options,
                                                        obj, j);
                })),
        EncodingJsonBinder(maybe_optional),
        jb::Member(
            "compressed_segmentation_block_size",
            [maybe_optional](auto is_loading, const auto& options, auto* obj,
                             auto* j) -> absl::Status {
              if constexpr (is_loading) {
                if (obj->encoding !=
                    ScaleMetadata::Encoding::compressed_segmentation) {
                  if (j->is_discarded()) return absl::OkStatus();
                  return absl::InvalidArgumentError(
                      "Only valid for \"compressed_segmentation\" encoding");
                }
              } else {
                if (obj->encoding !=
                    ScaleMetadata::Encoding::compressed_segmentation) {
                  return absl::OkStatus();
                }
              }
              return jb::Projection(
                  &T::compressed_segmentation_block_size,
                  maybe_optional(jb::FixedSizeArray(jb::Integer(1))))(
                  is_loading, options, obj, j);
            })
        /**/)(is_loading, options, obj, j);
  };
}

constexpr static auto ScaleMetadataBinder = jb::Object(
    ScaleMetadataCommon(internal::identity{}),
    jb::Initialize([](ScaleMetadata* x) {
      x->box.Fill(IndexInterval::UncheckedSized(0, 0));
    }),
    jb::Member("size", jb::Projection([](auto& x) { return x.box.shape(); })),
    jb::OptionalMember("voxel_offset",
                       jb::Projection([](auto& x) { return x.box.origin(); })),
    jb::Member("chunk_sizes",
               jb::Projection(&ScaleMetadata::chunk_sizes,
                              jb::Array(jb::FixedSizeArray(
                                  jb::Integer<Index>(1, kInfSize - 1))))),
    jb::Projection(&ScaleMetadata::extra_attributes),
    jb::Initialize([](ScaleMetadata* obj) {
      if (obj->chunk_sizes.empty()) {
        return absl::InvalidArgumentError(
            "At least one chunk size must be specified");
      }
      if (std::holds_alternative<ShardingSpec>(obj->sharding) &&
          obj->chunk_sizes.size() != 1) {
        return absl::InvalidArgumentError(
            "Sharded format does not support more than one chunk size");
      }
      for (const auto& x : obj->chunk_sizes) {
        TENSORSTORE_RETURN_IF_ERROR(
            ValidateChunkSize(x, obj->box.shape(), obj->sharding));
      }
      return CheckScaleBounds(obj->box);
    }));

constexpr static auto ScaleMetadataConstraintsBinder = jb::Object(
    ScaleMetadataCommon(WrapInOptional),
    jb::Member("size",
               jb::Projection(
                   &ScaleMetadataConstraints::box,
                   jb::Optional(jb::Sequence(
                       jb::Initialize([](Box<3>* box) {
                         std::fill(box->origin().begin(), box->origin().end(),
                                   Index(0));
                       }),
                       jb::Projection([](auto& x) { return x.shape(); }))))),
    jb::OptionalMember(
        "voxel_offset",
        jb::Projection(
            &ScaleMetadataConstraints::box,
            [](auto is_loading, const auto& options, auto* obj, auto* j) {
              if constexpr (is_loading) {
                if (!obj->has_value()) {
                  return absl::InvalidArgumentError(
                      "cannot be specified without \"size\"");
                }
              } else {
                if (!obj->has_value()) return absl::OkStatus();
              }
              // span<Index, 4> or span<const Index, 4>
              auto origin = (*obj)->origin();
              return jb::DefaultBinder<>(is_loading, options, &origin, j);
            })),
    jb::Member("chunk_size",
               jb::Projection(&ScaleMetadataConstraints::chunk_size,
                              jb::Optional(jb::FixedSizeArray(
                                  jb::Integer<Index>(1, kInfSize - 1))))),
    jb::Initialize([](ScaleMetadataConstraints* obj) {
      if (obj->chunk_size.has_value() && obj->sharding.has_value() &&
          obj->box.has_value()) {
        TENSORSTORE_RETURN_IF_ERROR(ValidateChunkSize(
            *obj->chunk_size, obj->box->shape(), *obj->sharding));
      }
      if (obj->box.has_value()) {
        return CheckScaleBounds(*obj->box);
      }
      return absl ::OkStatus();
    }));

constexpr static auto MultiscaleMetadataBinder = jb::Object(
    jb::OptionalMember("@type",
                       jb::Constant([] { return kMultiscaleVolumeTypeId; })),
    jb::Member("type", jb::Projection(&MultiscaleMetadata::type)),
    jb::Member("data_type",
               jb::Projection(&MultiscaleMetadata::dtype,
                              jb::Validate(
                                  [](const auto& options, auto* obj) {
                                    return ValidateDataType(*obj);
                                  },
                                  jb::DataTypeJsonBinder))),
    jb::Member("num_channels", jb::Projection(&MultiscaleMetadata::num_channels,
                                              jb::Integer(1))),
    jb::Member("scales", jb::Projection(&MultiscaleMetadata::scales,
                                        jb::Array(ScaleMetadataBinder))),
    jb::Projection(&MultiscaleMetadata::extra_attributes),
    jb::Initialize([](MultiscaleMetadata* obj) {
      for (const auto& s : obj->scales) {
        TENSORSTORE_RETURN_IF_ERROR(ValidateEncodingDataType(
            s.encoding, obj->dtype, obj->num_channels));
      }
      return absl::OkStatus();
    }));

constexpr static auto MultiscaleMetadataConstraintsBinder = jb::Object(
    jb::Member("type", jb::Projection(&MultiscaleMetadataConstraints::type)),
    jb::Member("data_type",
               jb::Projection(&MultiscaleMetadataConstraints::dtype,
                              jb::ConstrainedDataTypeJsonBinder)),
    jb::Member("num_channels",
               jb::Projection(&MultiscaleMetadataConstraints::num_channels,
                              jb::Optional(jb::Integer(1)))));

constexpr static auto OpenConstraintsBinder = jb::Object(
    jb::Member("scale_index", jb::Projection(&OpenConstraints::scale_index)),
    jb::Projection(
        &OpenConstraints::multiscale,
        jb::Validate(
            [](const auto& options, auto* obj) {
              auto& dtype = obj->dtype;
              if (!dtype.valid()) {
                dtype = options.dtype();
              }
              if (!dtype.valid()) return absl::OkStatus();
              return ValidateDataType(dtype);
            },
            jb::Member(
                "multiscale_metadata",
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()))),
    jb::Member(
        "scale_metadata",
        jb::Validate(
            [](const auto& options, OpenConstraints* obj) {
              if (obj->scale.encoding) {
                return ValidateEncodingDataType(obj->scale.encoding.value(),
                                                obj->multiscale.dtype,
                                                obj->multiscale.num_channels);
              }
              return absl::OkStatus();
            },
            jb::Projection(
                &OpenConstraints::scale,
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()))));

absl::Status ValidateScaleConstraintsForOpen(
    const ScaleMetadataConstraints& constraints,
    const ScaleMetadata& metadata) {
  if (constraints.key && *constraints.key != metadata.key) {
    return MetadataMismatchError(kKeyId, *constraints.key, metadata.key);
  }
  if (constraints.encoding && *constraints.encoding != metadata.encoding) {
    return MetadataMismatchError(kEncodingId, *constraints.encoding,
                                 metadata.encoding);
  }
  if (metadata.encoding == ScaleMetadata::Encoding::jpeg &&
      constraints.jpeg_quality &&
      *constraints.jpeg_quality != metadata.jpeg_quality) {
    return MetadataMismatchError(kJpegQualityId, *constraints.jpeg_quality,
                                 metadata.jpeg_quality);
  }
  if (metadata.encoding == ScaleMetadata::Encoding::compressed_segmentation &&
      constraints.compressed_segmentation_block_size &&
      *constraints.compressed_segmentation_block_size !=
          metadata.compressed_segmentation_block_size) {
    return MetadataMismatchError(
        kCompressedSegmentationBlockSizeId,
        *constraints.compressed_segmentation_block_size,
        metadata.compressed_segmentation_block_size);
  }
  if (constraints.resolution &&
      *constraints.resolution != metadata.resolution) {
    return MetadataMismatchError(kResolutionId, *constraints.resolution,
                                 metadata.resolution);
  }
  if (constraints.sharding && *constraints.sharding != metadata.sharding) {
    return MetadataMismatchError(kShardingId, *constraints.sharding,
                                 metadata.sharding);
  }
  if (constraints.box) {
    if (!absl::c_equal(constraints.box->shape(), metadata.box.shape())) {
      return MetadataMismatchError(kSizeId, constraints.box->shape(),
                                   metadata.box.shape());
    }
    if (!absl::c_equal(constraints.box->origin(), metadata.box.origin())) {
      return MetadataMismatchError(kVoxelOffsetId, constraints.box->origin(),
                                   metadata.box.origin());
    }
  }
  if (constraints.chunk_size &&
      !absl::c_linear_search(metadata.chunk_sizes, *constraints.chunk_size)) {
    return MetadataMismatchError(kChunkSizeId, *constraints.chunk_size,
                                 metadata.chunk_sizes);
  }
  return internal::ValidateMetadataSubset(constraints.extra_attributes,
                                          metadata.extra_attributes);
}

absl::Status ValidateMultiscaleConstraintsForOpen(
    const MultiscaleMetadataConstraints& constraints,
    const MultiscaleMetadata& metadata) {
  if (constraints.dtype.valid() && constraints.dtype != metadata.dtype) {
    return MetadataMismatchError(kDataTypeId, constraints.dtype.name(),
                                 metadata.dtype.name());
  }
  if (constraints.num_channels &&
      *constraints.num_channels != metadata.num_channels) {
    return MetadataMismatchError(kNumChannelsId, *constraints.num_channels,
                                 metadata.num_channels);
  }
  if (constraints.type && *constraints.type != metadata.type) {
    return MetadataMismatchError(kTypeId, *constraints.type, metadata.type);
  }
  return internal::ValidateMetadataSubset(constraints.extra_attributes,
                                          metadata.extra_attributes);
}

std::string GetScaleKeyFromResolution(span<const double, 3> resolution) {
  return tensorstore::StrCat(resolution[0], "_", resolution[1], "_",
                             resolution[2]);
}

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ScaleMetadata, ScaleMetadataBinder)

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(MultiscaleMetadata,
                                       MultiscaleMetadataBinder)

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(MultiscaleMetadataConstraints,
                                       MultiscaleMetadataConstraintsBinder)

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ScaleMetadataConstraints,
                                       ScaleMetadataConstraintsBinder)

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(OpenConstraints, OpenConstraintsBinder)

absl::Status ValidateMetadataCompatibility(
    const MultiscaleMetadata& existing_metadata,
    const MultiscaleMetadata& new_metadata, std::size_t scale_index,
    const std::array<Index, 3>& chunk_size) {
  if (new_metadata.num_channels != existing_metadata.num_channels) {
    return MetadataMismatchError(kNumChannelsId, existing_metadata.num_channels,
                                 new_metadata.num_channels);
  }
  if (new_metadata.dtype != existing_metadata.dtype) {
    return MetadataMismatchError(kDataTypeId, existing_metadata.dtype.name(),
                                 new_metadata.dtype.name());
  }
  if (new_metadata.scales.size() <= scale_index) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Updated metadata is missing scale ", scale_index));
  }
  const auto& existing_scale = existing_metadata.scales[scale_index];
  const auto& new_scale = new_metadata.scales[scale_index];
  if (existing_scale.key != new_scale.key) {
    return MetadataMismatchError(kKeyId, existing_scale.key, new_scale.key);
  }
  if (!absl::c_linear_search(new_scale.chunk_sizes, chunk_size)) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Updated metadata is missing chunk size ",
        ::nlohmann::json(chunk_size).dump(), " for scale ", scale_index));
  }
  if (!absl::c_equal(existing_scale.box.shape(), new_scale.box.shape())) {
    return MetadataMismatchError(kSizeId, existing_scale.box.shape(),
                                 new_scale.box.shape());
  }
  if (!absl::c_equal(existing_scale.box.origin(), new_scale.box.origin())) {
    return MetadataMismatchError(kVoxelOffsetId, existing_scale.box.origin(),
                                 new_scale.box.origin());
  }
  if (existing_scale.encoding != new_scale.encoding) {
    return MetadataMismatchError(kEncodingId, existing_scale.encoding,
                                 new_scale.encoding);
  }
  // jpeg_quality not checked because it does not affect compatibility.
  if (existing_scale.encoding ==
          ScaleMetadata::Encoding::compressed_segmentation &&
      existing_scale.compressed_segmentation_block_size !=
          new_scale.compressed_segmentation_block_size) {
    return MetadataMismatchError(
        kCompressedSegmentationBlockSizeId,
        existing_scale.compressed_segmentation_block_size,
        new_scale.compressed_segmentation_block_size);
  }
  if (existing_scale.sharding != new_scale.sharding) {
    return MetadataMismatchError(kShardingId, existing_scale.sharding,
                                 new_scale.sharding);
  }
  return absl::OkStatus();
}

std::string GetMetadataCompatibilityKey(
    const MultiscaleMetadata& metadata, std::size_t scale_index,
    const std::array<Index, 3>& chunk_size) {
  const auto& scale_metadata = metadata.scales[scale_index];
  ::nlohmann::json obj;
  obj.emplace(kDataTypeId, metadata.dtype.name());
  obj.emplace(kNumChannelsId, metadata.num_channels);
  obj.emplace(kScaleIndexId, scale_index);
  obj.emplace(kKeyId, scale_metadata.key);
  obj.emplace(kVoxelOffsetId, scale_metadata.box.origin());
  obj.emplace(kSizeId, scale_metadata.box.shape());
  obj.emplace(kEncodingId, scale_metadata.encoding);
  // jpeg_quality excluded does not affect compatibility.
  if (scale_metadata.encoding ==
      ScaleMetadata::Encoding::compressed_segmentation) {
    obj.emplace(kCompressedSegmentationBlockSizeId,
                scale_metadata.compressed_segmentation_block_size);
  }
  obj.emplace(kShardingId, scale_metadata.sharding);
  obj.emplace(kChunkSizeId, chunk_size);
  return obj.dump();
}

namespace {
absl::Status ChooseShardingSpec(ChunkLayout::ChunkShapeBase shape_constraints,
                                Index target_elements,
                                span<const Index, 3> shape,
                                span<const Index, 3> chunk_size,
                                ShardingSpec& sharding_spec) {
  std::array<int, 3> max_bits_per_dim =
      GetCompressedZIndexBits(shape, chunk_size);
  // Maximum value for the sum of the total shard bits per dimension.  The total
  // shard bits is `preshift_bits + minishard_bits + shard_bits`.
  const int max_total_bits =
      max_bits_per_dim[0] + max_bits_per_dim[1] + max_bits_per_dim[2];

  // Convert `target_elements` into `target_bits`, the value of
  // `total_within_shard_bits` that makes the total number of elements per shard
  // closest to `target_elements`.
  int target_bits = 0;
  {
    const Index chunk_elements = ProductOfExtents(chunk_size);
    const Index target_chunks =
        (target_elements + (chunk_elements / 2)) / chunk_elements;
    while (target_bits < max_total_bits &&
           (Index(1) << target_bits) < target_chunks) {
      ++target_bits;
    }
    if (target_bits > 0 &&
        (Index(1) << target_bits) - target_chunks >
            target_chunks - (Index(1) << (target_bits - 1))) {
      --target_bits;
    }
  }

  const auto get_cost =
      [&](span<const int, 3> within_shard_bits_per_dim) -> double {
    double cost = 0;
    if (shape_constraints.valid()) {
      for (int dim = 0; dim < 3; ++dim) {
        Index desired_size = shape_constraints[dim];
        if (desired_size == 0) continue;
        if (desired_size == -1 || desired_size == kInfSize) {
          desired_size = chunk_size[dim] << max_bits_per_dim[dim];
        }
        const Index cur_size = chunk_size[dim]
                               << within_shard_bits_per_dim[dim];
        if (shape_constraints.hard_constraint[dim] &&
            cur_size != desired_size) {
          return INFINITY;
        }
        cost += std::abs(cur_size - desired_size) /
                static_cast<double>(desired_size);
      }
    }
    return cost;
  };

  // Specifies the total within-shard bits (preshift_bits + minishard_bits) per
  // dimension, for the current write chunk shape being evaluated:
  // `write_chunk_shape[dim] = chunk_size[dim] <<
  // within_shard_bits_per_dim[dim]`
  std::array<int, 3> within_shard_bits_per_dim;
  within_shard_bits_per_dim.fill(0);

  // Computes the cost of the current write chunk shape specified by
  // `within_shard_bits_per_dim`.  The cost is INFINITY if a hard constraint is
  // violated.  Otherwise, it is the sum of
  // `|cur_size - desired_size| / desired_size` for each dimension with a soft
  // constraint on the size.  The cost is defined such that it is zero if all
  // hard and soft constraints are perfectly satisfied, and increases as
  // constraints become more violated.  Note that the choice of sum for
  // combining the costs is somewhat arbitrary, but given that we just have a
  // single degree of freedom (`total_within_shard_bits`), it shouldn't make
  // that much difference how we combine the costs.

  // Iterate through all possible values of `total_within_shard_bits` to find
  // the one that minimizes cost, and as a secondary objective is as close to
  // `target_bits` as possible.
  int best_total_within_shard_bits = 0;
  double best_cost = get_cost(within_shard_bits_per_dim);
  for (int i = 0, total_within_shard_bits = 0;
       total_within_shard_bits < max_total_bits; ++i) {
    const int dim = i % 3;
    if (within_shard_bits_per_dim[dim] == max_bits_per_dim[dim]) {
      // This dimension is already at its maximum size.
      continue;
    }
    ++total_within_shard_bits;
    ++within_shard_bits_per_dim[dim];

    const double cost = get_cost(within_shard_bits_per_dim);
    if (cost < best_cost ||
        (cost == best_cost && total_within_shard_bits <= target_bits)) {
      best_cost = cost;
      best_total_within_shard_bits = total_within_shard_bits;
    }
  }

  if (best_cost == INFINITY) {
    return absl::InvalidArgumentError(
        "Cannot satisfy write chunk shape constraint");
  }

  sharding_spec.preshift_bits = std::min(best_total_within_shard_bits, 9);
  sharding_spec.minishard_bits =
      best_total_within_shard_bits - sharding_spec.preshift_bits;
  sharding_spec.shard_bits = max_total_bits - sharding_spec.preshift_bits -
                             sharding_spec.minishard_bits;
  sharding_spec.hash_function = ShardingSpec::HashFunction::identity;
  sharding_spec.minishard_index_encoding = ShardingSpec::DataEncoding::gzip;
  return absl::OkStatus();
}
}  // namespace

Result<IndexDomain<>> GetDomainFromMetadata(const MultiscaleMetadata& metadata,
                                            size_t scale_index) {
  const auto& scale = metadata.scales[scale_index];
  IndexDomainBuilder domain_builder(4);
  domain_builder.labels({"x", "y", "z", "channel"});
  auto origin = domain_builder.origin();
  auto shape = domain_builder.shape();
  origin[3] = 0;
  shape[3] = metadata.num_channels;
  std::copy_n(scale.box.origin().begin(), 3, origin.begin());
  std::copy_n(scale.box.shape().begin(), 3, shape.begin());
  return domain_builder.Finalize();
}

Result<IndexDomain<>> GetEffectiveDomain(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints, const Schema& schema) {
  IndexDomainBuilder domain_builder(4);
  domain_builder.labels({"x", "y", "z", "channel"});
  auto domain_inclusive_min = domain_builder.origin();
  auto domain_shape = domain_builder.shape();
  std::fill_n(domain_inclusive_min.begin(), 3, -kInfIndex);
  std::fill_n(domain_shape.begin(), 4, kInfSize);
  auto& domain_implicit_lower_bounds = domain_builder.implicit_lower_bounds();
  auto& domain_implicit_upper_bounds = domain_builder.implicit_upper_bounds();
  domain_inclusive_min[3] = 0;
  domain_implicit_lower_bounds[3] = false;
  domain_implicit_upper_bounds[3] = true;
  if (existing_metadata) {
    // Set constraints from existing_metadata.
    TENSORSTORE_RETURN_IF_ERROR(ValidateMultiscaleConstraintsForOpen(
        constraints.multiscale, *existing_metadata));
    domain_shape[3] = existing_metadata->num_channels;
    domain_implicit_upper_bounds[3] = false;
  }
  if (constraints.multiscale.num_channels) {
    domain_shape[3] = *constraints.multiscale.num_channels;
    domain_implicit_upper_bounds[3] = false;
  }
  if (constraints.scale.box) {
    for (int i = 0; i < 3; ++i) {
      domain_inclusive_min[i] = constraints.scale.box->origin()[i];
      domain_shape[i] = constraints.scale.box->shape()[i];
      domain_implicit_lower_bounds[i] = false;
      domain_implicit_upper_bounds[i] = false;
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      domain_implicit_lower_bounds[i] = true;
      domain_implicit_upper_bounds[i] = true;
    }
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_constraint,
                               domain_builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, MergeIndexDomains(schema.domain(), domain_constraint),
      tensorstore::MaybeAnnotateStatus(
          _,
          "Error applying domain constraints from \"multiscale_metadata\" and "
          "\"scale_metadata\""));
  return domain;
}

namespace {
/// Updates the `write_chunk_shape` of `chunk_layout` based on the
/// `sharding_spec`.
absl::Status SetShardedWriteChunkConstraints(
    IndexDomainView<> domain, ChunkLayout& chunk_layout,
    const ShardingSpec& sharding_spec) {
  auto read_chunk_shape = chunk_layout.read_chunk_shape();
  if (!read_chunk_shape.valid()) return absl::OkStatus();
  if (!domain.valid() ||
      !IsFinite(BoxView<>(3, domain.origin().data(), domain.shape().data()))) {
    // Non-channel dimensions of domain are not fully specified.
    return absl::OkStatus();
  }
  for (DimensionIndex i = 0; i < 3; ++i) {
    if (read_chunk_shape[i] == 0 || !read_chunk_shape.hard_constraint[i]) {
      // read_chunk_shape is not fully specified.
      return absl::OkStatus();
    }
  }
  ShardChunkHierarchy hierarchy;
  if (!GetShardChunkHierarchy(sharding_spec, domain.shape().first<3>(),
                              read_chunk_shape.first<3>(), hierarchy)) {
    // Shards are non-rectangular.
    return absl::OkStatus();
  }
  Index write_chunk_shape[4];
  write_chunk_shape[3] = IsFinite(domain[3]) ? domain.shape()[3] : 0;
  for (DimensionIndex dim = 0; dim < 3; ++dim) {
    const Index chunk_size = read_chunk_shape[dim];
    const Index volume_size = domain.shape()[dim];
    write_chunk_shape[dim] = std::min(
        hierarchy.shard_shape_in_chunks[dim] * chunk_size, volume_size);
  }
  return chunk_layout.Set(ChunkLayout::WriteChunkShape(write_chunk_shape));
}
}  // namespace

absl::Status SetChunkLayoutFromMetadata(
    IndexDomainView<> domain, std::optional<span<const Index, 3>> chunk_size,
    const std::variant<NoShardingSpec, ShardingSpec>* sharding,
    std::optional<ScaleMetadata::Encoding> encoding,
    std::optional<span<const Index, 3>> compressed_segmentation_block_size,
    ChunkLayout& chunk_layout) {
  {
    Index origin[4];
    origin[3] = 0;
    if (domain.valid()) {
      for (DimensionIndex i = 0; i < 4; ++i) {
        const Index origin_value = domain.origin()[i];
        origin[i] =
            (!domain.implicit_lower_bounds()[i] || origin_value != -kInfIndex)
                ? origin_value
                : kImplicit;
      }
    } else {
      std::fill_n(origin, 3, kImplicit);
    }
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::GridOrigin(origin)),
        tensorstore::MaybeAnnotateStatus(
            _, "Chunk grid origin must match domain origin"));
  }
  TENSORSTORE_RETURN_IF_ERROR(
      chunk_layout.Set(ChunkLayout::InnerOrder({3, 2, 1, 0})),
      tensorstore::MaybeAnnotateStatus(
          _, "Only lexicographic {channel, z, y, x} inner order is supported"));

  if (domain.valid() && IsFinite(domain[3])) {
    Index csize[4] = {0, 0, 0, domain.shape()[3]};
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::ChunkShape(csize)),
        tensorstore::MaybeAnnotateStatus(
            _, "Chunking of channel dimension is not supported"));
  }

  if (chunk_size) {
    Index csize[4];
    csize[3] = 0;
    std::copy_n(chunk_size->begin(), 3, csize);
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::ReadChunkShape(csize)));
  }

  if (sharding) {
    if (auto* sharding_spec = std::get_if<ShardingSpec>(sharding)) {
      // Sharded format.
      TENSORSTORE_RETURN_IF_ERROR(SetShardedWriteChunkConstraints(
          domain, chunk_layout, *sharding_spec));
    } else {
      // Unsharded format.  Write chunk shape must match read chunk shape.
      TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
          ChunkLayout::WriteChunk(chunk_layout.read_chunk_shape())));
    }
  }

  if (encoding == ScaleMetadata::Encoding::compressed_segmentation) {
    Index codec_block_size[4];
    codec_block_size[3] = 1;
    if (compressed_segmentation_block_size) {
      std::copy_n(compressed_segmentation_block_size->begin(), 3,
                  codec_block_size);
    } else {
      std::fill_n(codec_block_size, 3, 0);
    }
    TENSORSTORE_RETURN_IF_ERROR(
        chunk_layout.Set(ChunkLayout::CodecChunkShape(codec_block_size)));
  }

  return absl::OkStatus();
}

Result<std::pair<IndexDomain<>, ChunkLayout>> GetEffectiveDomainAndChunkLayout(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints, const Schema& schema) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetEffectiveDomain(existing_metadata, constraints, schema));
  auto chunk_layout = schema.chunk_layout();
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      domain, constraints.scale.chunk_size,
      constraints.scale.sharding ? &*constraints.scale.sharding : nullptr,
      constraints.scale.encoding,
      constraints.scale.compressed_segmentation_block_size, chunk_layout));
  return {std::in_place, std::move(domain), std::move(chunk_layout)};
}

Result<internal::CodecDriverSpec::PtrT<NeuroglancerPrecomputedCodecSpec>>
GetEffectiveCodec(const OpenConstraints& constraints, const Schema& schema) {
  auto codec_spec =
      internal::CodecDriverSpec::Make<NeuroglancerPrecomputedCodecSpec>();
  codec_spec->encoding = constraints.scale.encoding;
  codec_spec->jpeg_quality = constraints.scale.jpeg_quality;

  if (constraints.scale.sharding) {
    if (auto* sharding =
            std::get_if<ShardingSpec>(&*constraints.scale.sharding)) {
      codec_spec->shard_data_encoding = sharding->data_encoding;
    }
  }
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  return codec_spec;
}

namespace {

absl::Status ValidateDimensionUnits(span<const std::optional<Unit>> units) {
  if (!units.empty()) {
    assert(units.size() == 4);
    if (units[3]) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Invalid dimension units ", DimensionUnitsToString(units),
          ": neuroglancer_precomputed format does not allow units to be "
          "specified for channel dimension"));
    }
    for (int i = 0; i < 3; ++i) {
      const auto& unit = units[i];
      if (!unit) continue;
      if (unit->base_unit != "nm") {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Invalid dimension units ", DimensionUnitsToString(units),
            ": neuroglancer_precomputed format requires a base unit of \"nm\" "
            "for the \"x\", \"y\", and \"z\" dimensions"));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDimensionUnitsForResolution(
    span<const double, 3> xyz_resolution,
    span<const std::optional<Unit>> units) {
  if (!units.empty()) {
    assert(units.size() == 4);
    for (int i = 0; i < 3; ++i) {
      const auto& unit = units[i];
      if (!unit) continue;
      if (unit->multiplier != xyz_resolution[i]) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Dimension units ", DimensionUnitsToString(units),
            " do not match \"resolution\" in metadata: ", xyz_resolution));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
    const OpenConstraints& constraints, const Schema& schema) {
  DimensionUnitsVector units(4);
  if (auto schema_units = schema.dimension_units(); schema_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(MergeDimensionUnits(units, schema_units));
    TENSORSTORE_RETURN_IF_ERROR(ValidateDimensionUnits(units));
  }
  if (constraints.scale.resolution) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateDimensionUnitsForResolution(
        *constraints.scale.resolution, units));
    for (int i = 0; i < 3; ++i) {
      units[i] = Unit((*constraints.scale.resolution)[i], "nm");
    }
  }
  return units;
}

Result<std::pair<std::shared_ptr<MultiscaleMetadata>, std::size_t>> CreateScale(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& orig_constraints, const Schema& orig_schema) {
  auto schema = orig_schema;
  auto constraints = orig_constraints;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain_and_chunk_layout,
      GetEffectiveDomainAndChunkLayout(existing_metadata, constraints, schema));
  auto domain = std::move(domain_and_chunk_layout.first);
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }

  auto chunk_layout = std::move(domain_and_chunk_layout.second);
  if (existing_metadata) {
    // Set constraints from existing_metadata.
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(existing_metadata->dtype));
  }
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(constraints.multiscale.dtype));
  if (!schema.dtype().valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(schema.dtype()));

  constraints.multiscale.dtype = schema.dtype();
  constraints.scale.box.emplace();
  for (int i = 0; i < 3; ++i) {
    constraints.scale.box->origin()[i] = domain.origin()[i];
    constraints.scale.box->shape()[i] = domain.shape()[i];
  }
  constraints.multiscale.num_channels = domain.shape()[3];
  if (!constraints.multiscale.type) {
    // Choose default "type".
    constraints.multiscale.type = (schema.dtype() == dtype_v<uint64_t> ||
                                   schema.dtype() == dtype_v<uint32_t>)
                                      ? "segmentation"
                                      : "image";
  }

  // Set the resolution.
  {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto dimension_units, GetEffectiveDimensionUnits(constraints, schema));
    auto& resolution = constraints.scale.resolution.emplace();
    for (int i = 0; i < 3; ++i) {
      const auto& unit = dimension_units[i];
      resolution[i] = unit ? unit->multiplier : 1.0;
    }
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto codec_spec,
                               GetEffectiveCodec(constraints, schema));
  constraints.scale.encoding =
      codec_spec->encoding.value_or(ScaleMetadata::Encoding::raw);
  constraints.scale.jpeg_quality = codec_spec->jpeg_quality;

  TENSORSTORE_RETURN_IF_ERROR(
      schema.Set(ChunkLayout::GridOrigin(domain.origin())));

  // Compute read chunk shape.
  Box<4> read_chunk_box;
  TENSORSTORE_RETURN_IF_ERROR(internal::ChooseChunkGrid(
      chunk_layout.grid_origin(), chunk_layout.read_chunk(), domain.box(),
      read_chunk_box));
  std::copy_n(read_chunk_box.shape().begin(), 3,
              constraints.scale.chunk_size.emplace().begin());

  if (constraints.scale.encoding ==
      ScaleMetadata::Encoding::compressed_segmentation) {
    chunk_layout
        .Set(ChunkLayout::CodecChunkElements(512, /*hard_constraint=*/false))
        // Setting a soft constraint cannot fail.
        .IgnoreError();

    {
      Index codec_chunk_shape[4] = {0, 0, 0, 1};
      // Constrain the channel dimension to have a chunk size of 1.
      TENSORSTORE_RETURN_IF_ERROR(
          chunk_layout.Set(ChunkLayout::CodecChunkShape(codec_chunk_shape)));
    }

    // Compute codec chunk shape.
    {
      Index codec_chunk_shape[4];
      TENSORSTORE_RETURN_IF_ERROR(internal::ChooseChunkShape(
          chunk_layout.codec_chunk(), read_chunk_box, codec_chunk_shape));
      std::copy_n(codec_chunk_shape, 3,
                  constraints.scale.compressed_segmentation_block_size.emplace()
                      .begin());
    }
  }

  if (!constraints.scale.sharding) {
    // Determine write chunk shape as multiple of read chunk shape.
    auto& sharding =
        constraints.scale.sharding.emplace().emplace<ShardingSpec>();
    TENSORSTORE_RETURN_IF_ERROR(ChooseShardingSpec(
        schema.chunk_layout().write_chunk_shape(),
        schema.chunk_layout().write_chunk_elements(), domain.shape().first<3>(),
        read_chunk_box.shape().first<3>(), sharding));
    if (sharding.preshift_bits == 0 && sharding.minishard_bits == 0 &&
        !codec_spec->shard_data_encoding) {
      // Use unsharded format since there would just be a single chunk per
      // shard.
      constraints.scale.sharding->emplace<NoShardingSpec>();
    } else {
      if (!codec_spec->shard_data_encoding) {
        codec_spec->shard_data_encoding =
            constraints.scale.encoding != ScaleMetadata::Encoding::jpeg
                ? ShardingSpec::DataEncoding::gzip
                : ShardingSpec::DataEncoding::raw;
      }
      sharding.data_encoding = *codec_spec->shard_data_encoding;
    }
  }

  std::string scale_key =
      constraints.scale.key
          ? *constraints.scale.key
          : GetScaleKeyFromResolution(*constraints.scale.resolution);
  std::shared_ptr<MultiscaleMetadata> new_metadata;
  if (!existing_metadata) {
    if (constraints.scale_index && *constraints.scale_index != 0) {
      return absl::FailedPreconditionError(
          tensorstore::StrCat("Cannot create scale ", *constraints.scale_index,
                              " in new multiscale volume"));
    }
    new_metadata = std::make_shared<MultiscaleMetadata>();
    new_metadata->type = *constraints.multiscale.type;
    new_metadata->num_channels = *constraints.multiscale.num_channels;
    new_metadata->dtype = constraints.multiscale.dtype;
    new_metadata->extra_attributes = constraints.multiscale.extra_attributes;
  } else {
    TENSORSTORE_RETURN_IF_ERROR(ValidateMultiscaleConstraintsForOpen(
        constraints.multiscale, *existing_metadata));
    if (constraints.scale_index) {
      if (*constraints.scale_index < existing_metadata->scales.size()) {
        // Scale index already exists
        return absl::AlreadyExistsError(tensorstore::StrCat(
            "Scale index ", *constraints.scale_index, " already exists"));
      }
      if (*constraints.scale_index != existing_metadata->scales.size()) {
        return absl::FailedPreconditionError(tensorstore::StrCat(
            "Scale index to create (", *constraints.scale_index,
            ") must equal the existing number of scales (",
            existing_metadata->scales.size(), ")"));
      }
    } else {
      // Check if any existing scale has matching key
      for (const auto& scale : existing_metadata->scales) {
        if (scale.key == scale_key) {
          return absl::AlreadyExistsError(tensorstore::StrCat(
              "Scale with key ", QuoteString(scale_key), " already exists"));
        }
      }
      if (!constraints.scale.key) {
        // Check if any existing scale has matching resolution, to avoid
        // ambiguity.
        for (const auto& scale : existing_metadata->scales) {
          if (scale.resolution == *constraints.scale.resolution) {
            return absl::AlreadyExistsError(tensorstore::StrCat(
                "Scale with resolution ",
                ::nlohmann::json(scale.resolution).dump(), " already exists"));
          }
        }
      }
    }
    new_metadata = std::make_shared<MultiscaleMetadata>(*existing_metadata);
  }
  if (auto status = ValidateEncodingDataType(*constraints.scale.encoding,
                                             new_metadata->dtype,
                                             new_metadata->num_channels);
      !status.ok()) {
    return absl::FailedPreconditionError(status.message());
  }
  auto& scale = new_metadata->scales.emplace_back();
  scale.extra_attributes = constraints.scale.extra_attributes;
  scale.key = scale_key;
  scale.box = *constraints.scale.box;
  scale.chunk_sizes = {*constraints.scale.chunk_size};
  scale.encoding = *constraints.scale.encoding;
  scale.resolution = *constraints.scale.resolution;
  if (constraints.scale.sharding) {
    scale.sharding = *constraints.scale.sharding;
  }
  if (constraints.scale.jpeg_quality) {
    scale.jpeg_quality = *constraints.scale.jpeg_quality;
  }
  if (constraints.scale.compressed_segmentation_block_size) {
    scale.compressed_segmentation_block_size =
        *constraints.scale.compressed_segmentation_block_size;
  }
  if (auto status = ValidateChunkSize(scale.chunk_sizes[0], scale.box.shape(),
                                      scale.sharding);
      !status.ok()) {
    return absl::FailedPreconditionError(status.message());
  }
  const size_t scale_index = new_metadata->scales.size() - 1;
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(
      *new_metadata, scale_index, scale.chunk_sizes[0], schema));
  return std::pair(new_metadata, scale_index);
}

CodecSpec GetCodecFromMetadata(const MultiscaleMetadata& metadata,
                               size_t scale_index) {
  const auto& scale = metadata.scales[scale_index];
  auto codec =
      internal::CodecDriverSpec::Make<NeuroglancerPrecomputedCodecSpec>();
  codec->encoding = scale.encoding;
  if (scale.encoding == ScaleMetadata::Encoding::jpeg) {
    codec->jpeg_quality = scale.jpeg_quality;
  }
  if (auto* sharding = std::get_if<ShardingSpec>(&scale.sharding)) {
    codec->shard_data_encoding = sharding->data_encoding;
  }
  return CodecSpec(std::move(codec));
}

absl::Status ValidateMetadataSchema(const MultiscaleMetadata& metadata,
                                    size_t scale_index,
                                    span<const Index, 3> chunk_size_xyz,
                                    const Schema& schema) {
  const auto& scale = metadata.scales[scale_index];

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(dtype, metadata.dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("data_type from metadata (", metadata.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecFromMetadata(metadata, scale_index);
    TENSORSTORE_RETURN_IF_ERROR(
        codec.MergeFrom(schema_codec),
        tensorstore::MaybeAnnotateStatus(
            _, "codec from metadata does not match codec in schema"));
    if (static_cast<const NeuroglancerPrecomputedCodecSpec&>(*codec)
            .shard_data_encoding &&
        std::holds_alternative<NoShardingSpec>(scale.sharding)) {
      return absl::InvalidArgumentError(
          "shard_data_encoding requires sharded format");
    }
  }

  IndexDomain<> domain;
  auto schema_domain = schema.domain();
  auto chunk_layout = schema.chunk_layout();
  if (schema_domain.valid() || chunk_layout.rank() != dynamic_rank) {
    TENSORSTORE_ASSIGN_OR_RETURN(domain,
                                 GetDomainFromMetadata(metadata, scale_index));
  }
  if (schema_domain.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(
        MergeIndexDomains(domain, schema_domain),
        tensorstore::MaybeAnnotateStatus(
            _, "domain from metadata does not match domain in schema"));
  }

  if (chunk_layout.rank() != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(
        SetChunkLayoutFromMetadata(
            domain, chunk_size_xyz, &scale.sharding, scale.encoding,
            scale.compressed_segmentation_block_size, chunk_layout),
        tensorstore::MaybeAnnotateStatus(_,
                                         "chunk layout from metadata does not "
                                         "match chunk layout in schema"));
    if (scale.encoding != ScaleMetadata::Encoding::compressed_segmentation &&
        chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "codec_chunk_shape not supported by ", scale.encoding, " encoding"));
    }
  }

  if (schema.fill_value().valid()) {
    return absl::InvalidArgumentError(
        "fill_value not supported by neuroglancer_precomputed format");
  }

  if (auto dimension_units = schema.dimension_units();
      dimension_units.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateDimensionUnits(dimension_units));
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateDimensionUnitsForResolution(scale.resolution, dimension_units));
  }

  return absl::OkStatus();
}

Result<std::size_t> OpenScale(const MultiscaleMetadata& metadata,
                              const OpenConstraints& constraints,
                              const Schema& schema) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateMultiscaleConstraintsForOpen(constraints.multiscale, metadata));
  std::size_t scale_index;
  TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                               GetEffectiveDimensionUnits(constraints, schema));
  if (constraints.scale_index) {
    scale_index = *constraints.scale_index;
    if (scale_index >= metadata.scales.size()) {
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "Scale ", scale_index, " does not exist, number of scales is ",
          metadata.scales.size()));
    }
  } else {
    for (scale_index = 0; scale_index < metadata.scales.size(); ++scale_index) {
      const auto& scale = metadata.scales[scale_index];
      if (constraints.scale.key && scale.key != *constraints.scale.key) {
        continue;
      }
      bool resolution_mismatch = false;
      for (int i = 0; i < 3; ++i) {
        const auto& unit = dimension_units[i];
        if (unit && scale.resolution[i] != unit->multiplier) {
          resolution_mismatch = true;
          break;
        }
      }
      if (resolution_mismatch) continue;
      break;
    }
    if (scale_index == metadata.scales.size()) {
      std::string explanation = "No scale found matching ";
      std::string_view sep = "";
      if (std::any_of(dimension_units.begin(), dimension_units.end(),
                      [](const auto& unit) { return unit.has_value(); })) {
        tensorstore::StrAppend(
            &explanation, "dimension_units=",
            tensorstore::DimensionUnitsToString(dimension_units));
        sep = ", ";
      }
      if (constraints.scale.key) {
        tensorstore::StrAppend(
            &explanation, sep, kKeyId, "=",
            tensorstore::QuoteString(*constraints.scale.key));
      }
      return absl::NotFoundError(explanation);
    }
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateScaleConstraintsForOpen(
      constraints.scale, metadata.scales[scale_index]));
  return scale_index;
}

std::string ResolveScaleKey(std::string_view key_prefix,
                            std::string_view scale_key) {
  if (!key_prefix.empty() && key_prefix.back() == '/') {
    key_prefix.remove_suffix(1);
  }
  if (key_prefix.empty()) return std::string(scale_key);
  std::vector<std::string_view> output_parts = absl::StrSplit(key_prefix, '/');
  for (std::string_view part : absl::StrSplit(scale_key, '/')) {
    if (part == ".." && !output_parts.empty()) {
      output_parts.resize(output_parts.size() - 1);
    } else {
      output_parts.push_back(part);
    }
  }
  return absl::StrJoin(output_parts, "/");
}

absl::Status ValidateDataType(DataType dtype) {
  assert(dtype.valid());
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

std::array<int, 3> GetCompressedZIndexBits(span<const Index, 3> shape,
                                           span<const Index, 3> chunk_size) {
  std::array<int, 3> bits;
  for (int i = 0; i < 3; ++i) {
    bits[i] = internal::bit_width(
        std::max(Index(0), CeilOfRatio(shape[i], chunk_size[i]) - 1));
  }
  return bits;
}

std::uint64_t EncodeCompressedZIndex(span<const Index, 3> indices,
                                     std::array<int, 3> bits) {
  const int max_bit = std::max(bits[0], std::max(bits[1], bits[2]));
  int out_bit = 0;
  std::uint64_t x = 0;
  for (int bit = 0; bit < max_bit; ++bit) {
    for (int i = 0; i < 3; ++i) {
      if (bit < bits[i]) {
        x |= ((static_cast<std::uint64_t>(indices[i]) >> bit) & 1)
             << (out_bit++);
      }
    }
  }
  return x;
}

namespace {
struct CompressedMortonBitIterator {
  explicit CompressedMortonBitIterator(span<const int, 3> z_index_bits)
      : z_index_bits(z_index_bits) {
    cur_bit_for_dim.fill(0);
  }

  int GetNextDim() {
    while (cur_bit_for_dim[dim_i] == z_index_bits[dim_i]) {
      dim_i = (dim_i + 1) % 3;
    }
    return dim_i;
  }

  void Next() {
    ++cur_bit_for_dim[dim_i];
    dim_i = (dim_i + 1) % 3;
  }
  void Next(int n) {
    for (int i = 0; i < n; ++i) {
      GetNextDim();
      Next();
    }
  }
  std::array<Index, 3> GetCurrentCellShape(
      span<const Index, 3> grid_shape_in_chunks) const {
    std::array<Index, 3> shape;
    for (int i = 0; i < 3; ++i) {
      shape[i] =
          std::min(grid_shape_in_chunks[i], Index(1) << cur_bit_for_dim[i]);
    }
    return shape;
  }

  int dim_i = 0;
  std::array<Index, 3> cur_bit_for_dim;
  span<const int, 3> z_index_bits;
};
}  // namespace

bool GetShardChunkHierarchy(const ShardingSpec& sharding_spec,
                            span<const Index, 3> volume_shape,
                            span<const Index, 3> chunk_shape,
                            ShardChunkHierarchy& hierarchy) {
  if (sharding_spec.hash_function != ShardingSpec::HashFunction::identity) {
    // For non-identity hash functions, the number of chunks per shard is not
    // predicable and the shard doesn't correspond to a rectangular region
    // anyway.
    return false;
  }

  const auto& z_index_bits = hierarchy.z_index_bits =
      GetCompressedZIndexBits(volume_shape, chunk_shape);
  const int total_z_index_bits =
      z_index_bits[0] + z_index_bits[1] + z_index_bits[2];
  if (total_z_index_bits >
      (sharding_spec.preshift_bits + sharding_spec.minishard_bits +
       sharding_spec.shard_bits)) {
    // A shard doesn't correspond to a rectangular region.
    return false;
  }

  for (int i = 0; i < 3; ++i) {
    hierarchy.grid_shape_in_chunks[i] =
        CeilOfRatio(volume_shape[i], chunk_shape[i]);
  }

  const int within_minishard_bits =
      std::min(sharding_spec.preshift_bits, total_z_index_bits);

  // Any additional non-shard bits beyond `total_z_index_bits` are irrelevant
  // because they will always be 0.  Constraining `non_shard_bits` here allows
  // us to avoid checking later.
  const int non_shard_bits = hierarchy.non_shard_bits =
      std::min(sharding_spec.minishard_bits + sharding_spec.preshift_bits,
               total_z_index_bits);

  // Any additional shard bits beyond `total_z_index_bits - non_shard_bits` are
  // irrelevant because they will always be 0.
  hierarchy.shard_bits =
      std::min(sharding_spec.shard_bits, total_z_index_bits - non_shard_bits);

  CompressedMortonBitIterator bit_it(z_index_bits);
  // Determine minishard shape.
  bit_it.Next(within_minishard_bits);
  hierarchy.minishard_shape_in_chunks =
      bit_it.GetCurrentCellShape(hierarchy.grid_shape_in_chunks);

  // Determine shard shape.
  bit_it.Next(non_shard_bits - within_minishard_bits);
  hierarchy.shard_shape_in_chunks =
      bit_it.GetCurrentCellShape(hierarchy.grid_shape_in_chunks);
  return true;
}

std::function<std::uint64_t(std::uint64_t shard)>
GetChunksPerVolumeShardFunction(const ShardingSpec& sharding_spec,
                                span<const Index, 3> volume_shape,
                                span<const Index, 3> chunk_shape) {
  ShardChunkHierarchy hierarchy;
  if (!GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                              hierarchy)) {
    return {};
  }
  return [hierarchy](std::uint64_t shard) -> std::uint64_t {
    if ((shard >> hierarchy.shard_bits) != 0) {
      // Invalid shard number.
      return 0;
    }

    CompressedMortonBitIterator bit_it(hierarchy.z_index_bits);
    bit_it.Next(hierarchy.non_shard_bits);
    auto cell_shape =
        bit_it.GetCurrentCellShape(hierarchy.grid_shape_in_chunks);
    std::array<Index, 3> cell_origin;
    cell_origin.fill(0);
    for (int bit_i = 0; bit_i < hierarchy.shard_bits; ++bit_i) {
      int dim_i = bit_it.GetNextDim();
      if ((shard >> bit_i) & 1) {
        cell_origin[dim_i] |= Index(1) << bit_it.cur_bit_for_dim[dim_i];
      }
      bit_it.Next();
    }

    std::uint64_t num_chunks = 1;
    for (int dim_i = 0; dim_i < 3; ++dim_i) {
      num_chunks *= static_cast<std::uint64_t>(
          std::min(hierarchy.grid_shape_in_chunks[dim_i] - cell_origin[dim_i],
                   cell_shape[dim_i]));
    }
    assert(((hierarchy.non_shard_bits == 0)
                ? num_chunks
                : (num_chunks >> hierarchy.non_shard_bits)) <= 1);
    return num_chunks;
  };
}

CodecSpec NeuroglancerPrecomputedCodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<NeuroglancerPrecomputedCodecSpec>(
      *this);
}

absl::Status NeuroglancerPrecomputedCodecSpec::DoMergeFrom(
    const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(NeuroglancerPrecomputedCodecSpec)) {
    return absl::InvalidArgumentError("");
  }
  auto& other =
      static_cast<const NeuroglancerPrecomputedCodecSpec&>(other_base);
  if (other.encoding) {
    if (!encoding) {
      encoding = other.encoding;
    } else if (*encoding != *other.encoding) {
      return absl::InvalidArgumentError("\"encoding\" mismatch");
    }
  }

  if (other.jpeg_quality) {
    if (!jpeg_quality) {
      jpeg_quality = other.jpeg_quality;
    } else if (*jpeg_quality != *other.jpeg_quality) {
      return absl::InvalidArgumentError("\"jpeg_quality\" mismatch");
    }
  }

  if (other.shard_data_encoding) {
    if (!shard_data_encoding) {
      shard_data_encoding = other.shard_data_encoding;
    } else if (*shard_data_encoding != *other.shard_data_encoding) {
      return absl::InvalidArgumentError("\"shard_data_encoding\" mismatch");
    }
  }

  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    NeuroglancerPrecomputedCodecSpec,
    jb::Sequence(
        EncodingJsonBinder(WrapInOptional),
        jb::Member(
            "shard_data_encoding",
            jb::Projection(
                &NeuroglancerPrecomputedCodecSpec::shard_data_encoding,
                jb::Optional(
                    neuroglancer_uint64_sharded::DataEncodingJsonBinder)))))

namespace {
const internal::CodecSpecRegistration<NeuroglancerPrecomputedCodecSpec>
    encoding_registration;
}  // namespace

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_neuroglancer_precomputed::OpenConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_neuroglancer_precomputed::OpenConstraints>())
