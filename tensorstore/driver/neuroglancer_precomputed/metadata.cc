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
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/bit_operations.h"
#include "tensorstore/internal/data_type_json_binder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span_json.h"

namespace jb = ::tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

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
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

namespace {

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,  DataTypeId::uint16_t,  DataTypeId::uint32_t,
    DataTypeId::uint64_t, DataTypeId::float32_t,
};

constexpr char kMultiscaleVolumeTypeId[] = "neuroglancer_multiscale_volume";

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

Status ValidateEncodingDataType(ScaleMetadata::Encoding encoding,
                                DataType dtype,
                                std::optional<Index> num_channels) {
  switch (encoding) {
    case ScaleMetadata::Encoding::raw:
      break;
    case ScaleMetadata::Encoding::compressed_segmentation:
      if (!dtype.valid()) break;
      if (dtype.id() != DataTypeId::uint32_t &&
          dtype.id() != DataTypeId::uint64_t) {
        return absl::InvalidArgumentError(
            StrCat("compressed_segmentation encoding only supported for "
                   "uint32 and uint64, not for ",
                   dtype));
      }
      break;
    case ScaleMetadata::Encoding::jpeg:
      if (dtype.valid() && dtype.id() != DataTypeId::uint8_t) {
        return absl::InvalidArgumentError(StrCat(
            "\"jpeg\" encoding only supported for uint8, not for ", dtype));
      }
      if (num_channels && *num_channels != 1 && *num_channels != 3) {
        return absl::InvalidArgumentError(
            StrCat("\"jpeg\" encoding only supports 1 or 3 channels, not ",
                   *num_channels));
      }
      break;
  }
  return absl::OkStatus();
}

Status CheckScaleBounds(BoxView<3> box) {
  for (int i = 0; i < 3; ++i) {
    if (!IndexInterval::ValidSized(box.origin()[i], box.shape()[i]) ||
        !IsFinite(box[i])) {
      return absl::InvalidArgumentError(StrCat(
          "\"size\" of ", ::nlohmann::json(box.shape()).dump(),
          " and \"voxel_offset\" of ", ::nlohmann::json(box.origin()).dump(),
          " do not specify a valid region"));
    }
  }
  return absl::OkStatus();
}

Status ValidateChunkSize(
    span<const Index, 3> chunk_size, span<const Index, 3> shape,
    const std::variant<NoShardingSpec, ShardingSpec>& sharding) {
  if (std::holds_alternative<NoShardingSpec>(sharding)) {
    // No constraints for unsharded format.
    return absl::OkStatus();
  }
  const auto bits = GetCompressedZIndexBits(shape, chunk_size);
  if (bits[0] + bits[1] + bits[2] > 64) {
    return absl::InvalidArgumentError(
        StrCat("\"size\" of ", ::nlohmann::json(shape).dump(),
               " with \"chunk_size\" of ", ::nlohmann::json(chunk_size).dump(),
               " is not compatible with sharded format because the chunk keys "
               "would exceed 64 bits"));
  }
  return absl::OkStatus();
}


/// The default json object copy moves the attributes; instead copy before
/// anything else is done.
constexpr static auto CopyAttributesBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json::object_t* j_obj) -> Status {
  if constexpr (is_loading) {
    obj->attributes = *j_obj;
  }
  return absl::OkStatus();
};

/// Binder for std::variant<NoShardingSpec, ShardingSpec>, maps discarded
/// (missing) and nullptr values to NoShardingSpec{}.
constexpr static auto ShardingBinder =
    [](auto is_loading, const auto& options,
       std::variant<NoShardingSpec, ShardingSpec>* obj,
       auto* j) -> absl::Status {
  if constexpr (is_loading) {
    if (j->is_discarded() || nullptr == *j) {
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

/// Common attributes for ScaleMetadata and ScaleMetadataConstraints.
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
        jb::Member("sharding", jb::Projection(&T::sharding,
                                              maybe_optional(ShardingBinder))),
        jb::Member(
            "encoding",
            jb::Projection(&T::encoding,
                           maybe_optional(ScaleMetatadaEncodingBinder()))),
        jb::Member("jpeg_quality",
                   [maybe_optional](auto is_loading, const auto& options,
                                    auto* obj, auto* j) -> Status {
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
                   }),
        jb::Member(
            "compressed_segmentation_block_size",
            [maybe_optional](auto is_loading, const auto& options, auto* obj,
                             auto* j) -> Status {
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
                  *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
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
    CopyAttributesBinder, ScaleMetadataCommon(internal::identity{}),
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
    jb::DiscardExtraMembers, jb::Initialize([](ScaleMetadata* obj) {
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
    ScaleMetadataCommon([](auto binder) { return jb::Optional(binder); }),
    jb::OptionalMember(
        "size", jb::Sequence(jb::Initialize([](ScaleMetadataConstraints* x) {
                               x->box.emplace().Fill(
                                   IndexInterval::UncheckedSized(0, 0));
                             }),
                             jb::Projection([](ScaleMetadataConstraints& x) {
                               return x.box->shape();
                             }))),
    jb::OptionalMember(
        "voxel_offset",
        jb::Sequence(jb::Initialize([](ScaleMetadataConstraints* x) {
                       if (!x->box) {
                         return absl::InvalidArgumentError(
                             "cannot be specified without \"size\"");
                       }
                       return absl::OkStatus();
                     }),
                     jb::Projection([](ScaleMetadataConstraints& x) {
                       return x.box->origin();
                     }))),
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
    CopyAttributesBinder, jb::OptionalMember("@type", jb::Constant([] {
                                               return kMultiscaleVolumeTypeId;
                                             })),
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
    jb::DiscardExtraMembers, jb::Initialize([](MultiscaleMetadata* obj) {
      for (const auto& s : obj->scales) {
        TENSORSTORE_RETURN_IF_ERROR(ValidateEncodingDataType(
            s.encoding, obj->dtype, obj->num_channels));
      }
      return absl::OkStatus();
    }));

constexpr static auto MultiscaleMetadataConstraintsBinder = jb::Object(
    jb::Member("type", jb::Projection(&MultiscaleMetadataConstraints::type)),
    jb::Member(
        "data_type",
        [](auto is_loading, const auto& options, auto* obj, auto* j) -> Status {
          if constexpr (is_loading) {
            if (j->is_discarded()) return absl::OkStatus();
          }
          return jb::Projection(
              &MultiscaleMetadataConstraints::dtype,
              jb::Validate([](const auto& options,
                              auto* obj) { return ValidateDataType(*obj); },
                           jb::DataTypeJsonBinder))(is_loading, options, obj,
                                                    j);
        }),
    jb::Member("num_channels",
               jb::Projection(&MultiscaleMetadataConstraints::num_channels,
                              jb::Optional(jb::Integer(1)))));

constexpr static auto OpenConstraintsBinder = jb::Object(
    jb::Member("scale_index", jb::Projection(&OpenConstraints::scale_index)),
    jb::OptionalMember("multiscale_metadata",
                       jb::Projection(&OpenConstraints::multiscale,
                                      MultiscaleMetadataConstraintsBinder)),
    jb::OptionalMember(
        "scale_metadata",
        jb::Sequence(jb::Projection(&OpenConstraints::scale,
                                    ScaleMetadataConstraintsBinder),
                     jb::Initialize([](OpenConstraints* obj) {
                       if (obj->scale.encoding &&
                           obj->multiscale.num_channels) {
                         return ValidateEncodingDataType(
                             obj->scale.encoding.value(), obj->multiscale.dtype,
                             obj->multiscale.num_channels);
                       }
                       return absl::OkStatus();
                     }))));

Status ValidateScaleConstraintsForCreate(const ScaleMetadataConstraints& m) {
  const auto Error = [](const char* property) {
    return absl::InvalidArgumentError(StrCat(
        QuoteString(property), " must be specified in \"scale_metadata\""));
  };
  if (!m.box) return Error(kSizeId);
  if (!m.resolution) return Error(kResolutionId);
  if (!m.chunk_size) return Error(kChunkSizeId);
  if (!m.encoding) return Error(kEncodingId);
  if (*m.encoding == ScaleMetadata::Encoding::compressed_segmentation &&
      !m.compressed_segmentation_block_size) {
    return Error(kCompressedSegmentationBlockSizeId);
  }
  return absl::OkStatus();
}

Status ValidateScaleConstraintsForOpen(
    const ScaleMetadataConstraints& constraints,
    const ScaleMetadata& metadata) {
  const auto Error = [](const char* name, const auto& expected,
                        const auto& actual) -> Status {
    return absl::FailedPreconditionError(
        StrCat("Expected ", QuoteString(name), " of ",
               ::nlohmann::json(expected).dump(),
               " but received: ", ::nlohmann::json(actual).dump()));
  };
  if (constraints.key && *constraints.key != metadata.key) {
    return Error(kKeyId, *constraints.key, metadata.key);
  }
  if (constraints.encoding && *constraints.encoding != metadata.encoding) {
    return Error(kEncodingId, *constraints.encoding, metadata.encoding);
  }
  if (metadata.encoding == ScaleMetadata::Encoding::jpeg &&
      constraints.jpeg_quality &&
      *constraints.jpeg_quality != metadata.jpeg_quality) {
    return Error(kJpegQualityId, *constraints.jpeg_quality,
                 metadata.jpeg_quality);
  }
  if (metadata.encoding == ScaleMetadata::Encoding::compressed_segmentation &&
      constraints.compressed_segmentation_block_size &&
      *constraints.compressed_segmentation_block_size !=
          metadata.compressed_segmentation_block_size) {
    return Error(kCompressedSegmentationBlockSizeId,
                 *constraints.compressed_segmentation_block_size,
                 metadata.compressed_segmentation_block_size);
  }
  if (constraints.resolution &&
      *constraints.resolution != metadata.resolution) {
    return Error(kResolutionId, *constraints.resolution, metadata.resolution);
  }
  if (constraints.sharding && *constraints.sharding != metadata.sharding) {
    return Error(kShardingId, *constraints.sharding, metadata.sharding);
  }
  if (constraints.box) {
    if (!absl::c_equal(constraints.box->shape(), metadata.box.shape())) {
      return Error(kSizeId, constraints.box->shape(), metadata.box.shape());
    }
    if (!absl::c_equal(constraints.box->origin(), metadata.box.origin())) {
      return Error(kVoxelOffsetId, constraints.box->origin(),
                   metadata.box.origin());
    }
  }
  if (constraints.chunk_size &&
      !absl::c_linear_search(metadata.chunk_sizes, *constraints.chunk_size)) {
    return Error(kChunkSizeId, *constraints.chunk_size, metadata.chunk_sizes);
  }
  return absl::OkStatus();
}

Status ValidateMultiscaleConstraintsForCreate(
    const MultiscaleMetadataConstraints& m) {
  const auto Error = [](const char* property) {
    return absl::InvalidArgumentError(
        StrCat(QuoteString(property),
               " must be specified in \"multiscale_metadata\""));
  };
  if (!m.dtype.valid()) return Error(kDataTypeId);
  if (!m.num_channels) return Error(kNumChannelsId);
  if (!m.type) return Error(kTypeId);
  return absl::OkStatus();
}

Status ValidateMultiscaleConstraintsForOpen(
    const MultiscaleMetadataConstraints& constraints,
    const MultiscaleMetadata& metadata) {
  const auto Error = [](const char* name, const auto& expected,
                        const auto& actual) -> Status {
    return absl::FailedPreconditionError(
        StrCat("Expected ", QuoteString(name), " of ",
               ::nlohmann::json(expected).dump(),
               " but received: ", ::nlohmann::json(actual).dump()));
  };
  if (constraints.dtype.valid() && constraints.dtype != metadata.dtype) {
    return Error(kDataTypeId, constraints.dtype.name(), metadata.dtype.name());
  }
  if (constraints.num_channels &&
      *constraints.num_channels != metadata.num_channels) {
    return Error(kNumChannelsId, *constraints.num_channels,
                 metadata.num_channels);
  }
  if (constraints.type && *constraints.type != metadata.type) {
    return Error(kTypeId, *constraints.type, metadata.type);
  }
  return absl::OkStatus();
}

std::string GetScaleKeyFromResolution(span<const double, 3> resolution) {
  return absl::StrCat(resolution[0], "_", resolution[1], "_", resolution[2]);
}

Result<std::shared_ptr<MultiscaleMetadata>> InitializeNewMultiscaleMetadata(
    const MultiscaleMetadataConstraints& m) {
  if (auto status = ValidateMultiscaleConstraintsForCreate(m); !status.ok()) {
    return status;
  }
  auto new_metadata = std::make_shared<MultiscaleMetadata>();
  new_metadata->type = *m.type;
  new_metadata->num_channels = *m.num_channels;
  new_metadata->dtype = m.dtype;
  new_metadata->attributes = {{kTypeId, new_metadata->type},
                              {kNumChannelsId, new_metadata->num_channels},
                              {kDataTypeId, new_metadata->dtype.name()},
                              {kAtSignTypeId, kMultiscaleVolumeTypeId},
                              {kScalesId, ::nlohmann::json::array_t{}}};
  return new_metadata;
}

}  // namespace

Result<MultiscaleMetadata> MultiscaleMetadata::Parse(::nlohmann::json j) {
  auto result = jb::FromJson<MultiscaleMetadata>(j, MultiscaleMetadataBinder);
  if (!result.ok()) {
    return MaybeAnnotateStatus(result.status(),
                               StrCat("While parsing ", j.dump()));
  }
  return result;
}

Result<MultiscaleMetadataConstraints> MultiscaleMetadataConstraints::Parse(
    const ::nlohmann::json& j) {
  return jb::FromJson<MultiscaleMetadataConstraints>(
      j, MultiscaleMetadataConstraintsBinder);
}

Result<ScaleMetadataConstraints> ScaleMetadataConstraints::Parse(
    const ::nlohmann::json& j, DataType dtype,
    std::optional<Index> num_channels) {
  TENSORSTORE_ASSIGN_OR_RETURN(ScaleMetadataConstraints metadata,
                               jb::FromJson<ScaleMetadataConstraints>(
                                   j, ScaleMetadataConstraintsBinder));
  if (metadata.encoding) {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateEncodingDataType(*metadata.encoding, dtype, num_channels));
  }
  return metadata;
}

Result<OpenConstraints> OpenConstraints::Parse(const ::nlohmann::json& j,
                                               DataType data_type_constraint) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      OpenConstraints constraints,
      jb::FromJson<OpenConstraints>(j, OpenConstraintsBinder));

  if (data_type_constraint.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(data_type_constraint));
    if (constraints.multiscale.dtype.valid() &&
        constraints.multiscale.dtype != data_type_constraint) {
      return absl::InvalidArgumentError(
          StrCat("Mismatch between data type in TensorStore Spec (",
                 data_type_constraint, ") and in \"multiscale_metadata\" (",
                 constraints.multiscale.dtype, ")"));
    }
    constraints.multiscale.dtype = data_type_constraint;
  }

  return constraints;
}

Status ValidateMetadataCompatibility(
    const MultiscaleMetadata& existing_metadata,
    const MultiscaleMetadata& new_metadata, std::size_t scale_index,
    const std::array<Index, 3>& chunk_size) {
  const auto MismatchError = [](const char* member_name,
                                const auto& expected_value,
                                const auto& actual_value) {
    return absl::FailedPreconditionError(
        StrCat("Mismatch in ", QuoteString(member_name), ": expected ",
               ::nlohmann::json(expected_value).dump(), ", received ",
               ::nlohmann::json(actual_value).dump()));
  };
  if (new_metadata.num_channels != existing_metadata.num_channels) {
    return MismatchError(kNumChannelsId, existing_metadata.num_channels,
                         new_metadata.num_channels);
  }
  if (new_metadata.dtype != existing_metadata.dtype) {
    return MismatchError(kDataTypeId, existing_metadata.dtype.name(),
                         new_metadata.dtype.name());
  }
  if (new_metadata.scales.size() <= scale_index) {
    return absl::FailedPreconditionError(
        StrCat("Updated metadata is missing scale ", scale_index));
  }
  const auto& existing_scale = existing_metadata.scales[scale_index];
  const auto& new_scale = new_metadata.scales[scale_index];
  if (existing_scale.key != new_scale.key) {
    return MismatchError(kKeyId, existing_scale.key, new_scale.key);
  }
  if (!absl::c_linear_search(new_scale.chunk_sizes, chunk_size)) {
    return absl::FailedPreconditionError(StrCat(
        "Updated metadata is missing chunk size ",
        ::nlohmann::json(chunk_size).dump(), " for scale ", scale_index));
  }
  if (!absl::c_equal(existing_scale.box.shape(), new_scale.box.shape())) {
    return MismatchError(kSizeId, existing_scale.box.shape(),
                         new_scale.box.shape());
  }
  if (!absl::c_equal(existing_scale.box.origin(), new_scale.box.origin())) {
    return MismatchError(kVoxelOffsetId, existing_scale.box.origin(),
                         new_scale.box.origin());
  }
  if (existing_scale.encoding != new_scale.encoding) {
    return MismatchError(kEncodingId, existing_scale.encoding,
                         new_scale.encoding);
  }
  // jpeg_quality not checked because it does not affect compatibility.
  if (existing_scale.encoding ==
          ScaleMetadata::Encoding::compressed_segmentation &&
      existing_scale.compressed_segmentation_block_size !=
          new_scale.compressed_segmentation_block_size) {
    return MismatchError(kCompressedSegmentationBlockSizeId,
                         existing_scale.compressed_segmentation_block_size,
                         new_scale.compressed_segmentation_block_size);
  }
  if (existing_scale.sharding != new_scale.sharding) {
    return MismatchError(kShardingId, existing_scale.sharding,
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

Result<std::pair<std::shared_ptr<MultiscaleMetadata>, std::size_t>> CreateScale(
    const MultiscaleMetadata* existing_metadata,
    const OpenConstraints& constraints) {
  if (auto status = ValidateScaleConstraintsForCreate(constraints.scale);
      !status.ok()) {
    return status;
  }
  std::string scale_key =
      constraints.scale.key
          ? *constraints.scale.key
          : GetScaleKeyFromResolution(*constraints.scale.resolution);
  std::shared_ptr<MultiscaleMetadata> new_metadata;
  if (!existing_metadata) {
    if (constraints.scale_index && *constraints.scale_index != 0) {
      return absl::FailedPreconditionError(StrCat("Cannot create scale ",
                                                  *constraints.scale_index,
                                                  " in new multiscale volume"));
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        new_metadata, InitializeNewMultiscaleMetadata(constraints.multiscale));
  } else {
    if (auto status = ValidateMultiscaleConstraintsForOpen(
            constraints.multiscale, *existing_metadata);
        !status.ok()) {
      return status;
    }
    if (constraints.scale_index) {
      if (*constraints.scale_index < existing_metadata->scales.size()) {
        // Scale index already exists
        return absl::AlreadyExistsError(StrCat(
            "Scale index ", *constraints.scale_index, " already exists"));
      }
      if (*constraints.scale_index != existing_metadata->scales.size()) {
        return absl::FailedPreconditionError(
            StrCat("Scale index to create (", *constraints.scale_index,
                   ") must equal the existing number of scales (",
                   existing_metadata->scales.size(), ")"));
      }
    } else {
      // Check if any existing scale has matching key
      for (const auto& scale : existing_metadata->scales) {
        if (scale.key == scale_key) {
          return absl::AlreadyExistsError(StrCat(
              "Scale with key ", QuoteString(scale_key), " already exists"));
        }
      }
      if (!constraints.scale.key) {
        // Check if any existing scale has matching resolution, to avoid
        // ambiguity.
        for (const auto& scale : existing_metadata->scales) {
          if (scale.resolution == *constraints.scale.resolution) {
            return absl::AlreadyExistsError(StrCat(
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
  scale.key = scale_key;
  scale.box = *constraints.scale.box;
  scale.chunk_sizes = {*constraints.scale.chunk_size};
  scale.encoding = *constraints.scale.encoding;
  scale.resolution = *constraints.scale.resolution;
  if (constraints.scale.sharding) {
    scale.sharding = *constraints.scale.sharding;
  }
  if (auto status = ValidateChunkSize(scale.chunk_sizes[0], scale.box.shape(),
                                      scale.sharding);
      !status.ok()) {
    return absl::FailedPreconditionError(status.message());
  }
  scale.attributes = {{kKeyId, scale_key},
                      {kSizeId, scale.box.shape()},
                      {kVoxelOffsetId, scale.box.origin()},
                      {kResolutionId, scale.resolution},
                      {kChunkSizesId, scale.chunk_sizes},
                      {kEncodingId, scale.encoding}};
  if (std::holds_alternative<ShardingSpec>(scale.sharding)) {
    scale.attributes[kShardingId] = scale.sharding;
  }
  if (scale.encoding == ScaleMetadata::Encoding::jpeg) {
    scale.jpeg_quality =
        constraints.scale.jpeg_quality.value_or(kDefaultJpegQuality);
    scale.attributes[kJpegQualityId] = scale.jpeg_quality;
  }
  if (scale.encoding == ScaleMetadata::Encoding::compressed_segmentation) {
    scale.compressed_segmentation_block_size =
        *constraints.scale.compressed_segmentation_block_size;
    scale.attributes[kCompressedSegmentationBlockSizeId] =
        scale.compressed_segmentation_block_size;
  }
  new_metadata->attributes[kScalesId].push_back(scale.attributes);
  return std::pair(new_metadata, new_metadata->scales.size() - 1);
}

Result<std::size_t> OpenScale(const MultiscaleMetadata& metadata,
                              const OpenConstraints& constraints) {
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateMultiscaleConstraintsForOpen(constraints.multiscale, metadata));
  std::size_t scale_index;
  if (constraints.scale_index) {
    scale_index = *constraints.scale_index;
    if (scale_index >= metadata.scales.size()) {
      return absl::FailedPreconditionError(
          StrCat("Scale ", scale_index, " does not exist, number of scales is ",
                 metadata.scales.size()));
    }
  } else {
    for (scale_index = 0; scale_index < metadata.scales.size(); ++scale_index) {
      const auto& scale = metadata.scales[scale_index];
      if (constraints.scale.key && scale.key != *constraints.scale.key) {
        continue;
      }
      if (constraints.scale.resolution &&
          scale.resolution != *constraints.scale.resolution) {
        continue;
      }
      break;
    }
    if (scale_index == metadata.scales.size()) {
      ::nlohmann::json c;
      if (constraints.scale.resolution) {
        c[kResolutionId] = *constraints.scale.resolution;
      }
      if (constraints.scale.key) {
        c[kKeyId] = *constraints.scale.key;
      }
      return absl::NotFoundError(StrCat("No scale found matching ", c.dump()));
    }
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateScaleConstraintsForOpen(
      constraints.scale, metadata.scales[scale_index]));
  return scale_index;
}

std::string ResolveScaleKey(std::string_view key_prefix,
                            std::string_view scale_key) {
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

Status ValidateDataType(DataType dtype) {
  assert(dtype.valid());
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(
        StrCat(dtype, " data type is not one of the supported data types: ",
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

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore
