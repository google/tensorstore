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
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/bit_operations.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span_json.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

void to_json(::nlohmann::json& out,  // NOLINT
             const std::variant<NoShardingSpec, ShardingSpec>& s) {
  std::visit([&](const auto& x) { out = ::nlohmann::json(x); }, s);
}

absl::string_view to_string(ScaleMetadata::Encoding e) {
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

Status ParseDataType(const ::nlohmann::json& value, DataType* data_type) {
  std::string s;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireValueAs(value, &s));
  absl::AsciiStrToLower(&s);
  auto x = GetDataType(s);
  if (!x.valid() || !absl::c_linear_search(kSupportedDataTypes, x.id())) {
    return absl::InvalidArgumentError(StrCat(
        QuoteString(s),
        " is not one of the supported data types: ", GetSupportedDataTypes()));
  }
  *data_type = x;
  return absl::OkStatus();
}

Status ParseNumChannels(const ::nlohmann::json& value, Index* num_channels) {
  return internal::JsonRequireInteger(value, num_channels,
                                      /*strict=*/true,
                                      /*min_value=*/1, /*max_value=*/kInfIndex);
}

Status ParseSize(const ::nlohmann::json& value, span<Index, 3> size) {
  return internal::JsonParseArray(
      value,
      [](std::ptrdiff_t n) { return internal::JsonValidateArrayLength(n, 3); },
      [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
        return internal::JsonRequireInteger(v, &size[i],
                                            /*strict=*/false, 0, kInfIndex);
      });
}

Status ParseVoxelOffset(const ::nlohmann::json& value,
                        span<Index, 3> voxel_offset) {
  return internal::JsonParseArray(
      value,
      [](std::ptrdiff_t n) { return internal::JsonValidateArrayLength(n, 3); },
      [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
        return internal::JsonRequireInteger(v, &voxel_offset[i],
                                            /*strict=*/false, -kInfIndex + 1,
                                            kInfIndex - 1);
      });
}

Status ParseChunkSize(const ::nlohmann::json& value,
                      std::array<Index, 3>* chunk_size) {
  return internal::JsonParseArray(
      value,
      [](std::ptrdiff_t size) {
        return internal::JsonValidateArrayLength(size, 3);
      },
      [&](const ::nlohmann::json& value, std::ptrdiff_t i) {
        return internal::JsonRequireInteger(value, &(*chunk_size)[i],
                                            /*strict=*/false, 1, kInfIndex);
      });
}

Status ParseEncoding(const ::nlohmann::json& value,
                     ScaleMetadata::Encoding* encoding) {
  std::string s;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireValueAs(value, &s));
  absl::AsciiStrToLower(&s);
  if (s == "raw") {
    *encoding = ScaleMetadata::Encoding::raw;
  } else if (s == "jpeg") {
    *encoding = ScaleMetadata::Encoding::jpeg;
  } else if (s == "compressed_segmentation") {
    *encoding = ScaleMetadata::Encoding::compressed_segmentation;
  } else {
    return absl::InvalidArgumentError(
        StrCat("Encoding not supported: ", value.dump()));
  }
  return absl::OkStatus();
}

Status ParseJpegQuality(const ::nlohmann::json& value, int* jpeg_quality) {
  return internal::JsonRequireInteger<int>(value, jpeg_quality, /*strict=*/true,
                                           0, 100);
}

Status ParseCompressedSegmentationBlockSize(
    const ::nlohmann::json& value,
    std::array<Index, 3>* compressed_segmentation_block_size) {
  return internal::JsonParseArray(
      value,
      [](std::ptrdiff_t size) {
        return internal::JsonValidateArrayLength(size, 3);
      },
      [&](const ::nlohmann::json& value, std::ptrdiff_t i) {
        return internal::JsonRequireInteger(
            value, &(*compressed_segmentation_block_size)[i],
            /*strict=*/false, 1, kInfIndex);
      });
}

Status ParseResolution(const ::nlohmann::json& value,
                       span<double, 3> resolution) {
  return internal::JsonParseArray(
      value,
      [](std::ptrdiff_t size) {
        return internal::JsonValidateArrayLength(size, 3);
      },
      [&](const ::nlohmann::json& value, std::ptrdiff_t i) {
        return internal::JsonRequireValueAs(value, &resolution[i],
                                            /*strict=*/false);
      });
}

Status ValidateEncodingDataType(ScaleMetadata::Encoding encoding,
                                DataType data_type,
                                std::optional<Index> num_channels) {
  switch (encoding) {
    case ScaleMetadata::Encoding::raw:
      break;
    case ScaleMetadata::Encoding::compressed_segmentation:
      if (!data_type.valid()) break;
      if (data_type.id() != DataTypeId::uint32_t &&
          data_type.id() != DataTypeId::uint64_t) {
        return absl::InvalidArgumentError(
            StrCat("compressed_segmentation encoding only supported for "
                   "uint32 and uint64, not for ",
                   data_type));
      }
      break;
    case ScaleMetadata::Encoding::jpeg:
      if (data_type.valid() && data_type.id() != DataTypeId::uint8_t) {
        return absl::InvalidArgumentError(StrCat(
            "\"jpeg\" encoding only supported for uint8, not for ", data_type));
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

Result<ScaleMetadata> ParseScaleMetadata(const ::nlohmann::json& j,
                                         DataType data_type,
                                         Index num_channels) {
  ScaleMetadata metadata;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, kKeyId, [&](const ::nlohmann::json& value) {
        return internal::JsonRequireValueAs(value, &metadata.key,
                                            /*strict=*/true);
      }));
  metadata.attributes = *j.get_ptr<const ::nlohmann::json::object_t*>();
  metadata.box.Fill(IndexInterval::UncheckedSized(0, 0));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, kSizeId, [&](const ::nlohmann::json& value) {
        return ParseSize(value, metadata.box.shape());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kVoxelOffsetId, [&](const ::nlohmann::json& value) {
        return ParseVoxelOffset(value, metadata.box.origin());
      }));
  TENSORSTORE_RETURN_IF_ERROR(CheckScaleBounds(metadata.box));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, kResolutionId, [&](const ::nlohmann::json& value) {
        return ParseResolution(value, metadata.resolution);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kShardingId, [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(metadata.sharding,
                                     ShardingSpec::FromJson(value));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, kChunkSizesId, [&](const ::nlohmann::json& value) {
        return internal::JsonParseArray(
            value,
            [&](std::ptrdiff_t size) {
              if (size == 0) {
                return absl::InvalidArgumentError(
                    "At least one chunk size must be specified");
              }
              if (std::holds_alternative<ShardingSpec>(metadata.sharding) &&
                  size != 1) {
                return absl::InvalidArgumentError(
                    "Sharded format does not support more than one chunk size");
              }
              metadata.chunk_sizes.resize(size);
              return absl::OkStatus();
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              TENSORSTORE_RETURN_IF_ERROR(
                  ParseChunkSize(v, &metadata.chunk_sizes[i]));
              TENSORSTORE_RETURN_IF_ERROR(
                  ValidateChunkSize(metadata.chunk_sizes[i],
                                    metadata.box.shape(), metadata.sharding));
              return absl::OkStatus();
            });
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      j, kEncodingId, [&](const ::nlohmann::json& value) {
        return ParseEncoding(value, &metadata.encoding);
      }));
  if (metadata.encoding == ScaleMetadata::Encoding::jpeg) {
    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
        j, kJpegQualityId, [&](const ::nlohmann::json& value) {
          return ParseJpegQuality(value, &metadata.jpeg_quality);
        }));
  } else {
    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
        j, kJpegQualityId, [&](const ::nlohmann::json& value) {
          return absl::InvalidArgumentError("Only valid for \"jpeg\" encoding");
        }));
  }
  if (metadata.encoding == ScaleMetadata::Encoding::compressed_segmentation) {
    TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
        j, kCompressedSegmentationBlockSizeId,
        [&](const ::nlohmann::json& value) {
          return ParseCompressedSegmentationBlockSize(
              value, &metadata.compressed_segmentation_block_size);
        }));
  } else {
    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
        j, kCompressedSegmentationBlockSizeId,
        [&](const ::nlohmann::json& value) {
          return absl::InvalidArgumentError(
              "Only valid for \"compressed_segmentation\" encoding");
        }));
  }
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateEncodingDataType(metadata.encoding, data_type, num_channels));
  return metadata;
}

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
  if (!m.data_type.valid()) return Error(kDataTypeId);
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
  if (constraints.data_type.valid() &&
      constraints.data_type != metadata.data_type) {
    return Error(kDataTypeId, constraints.data_type.name(),
                 metadata.data_type.name());
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
  new_metadata->data_type = m.data_type;
  new_metadata->attributes = {{kTypeId, new_metadata->type},
                              {kNumChannelsId, new_metadata->num_channels},
                              {kDataTypeId, new_metadata->data_type.name()},
                              {kAtSignTypeId, kMultiscaleVolumeTypeId},
                              {kScalesId, ::nlohmann::json::array_t{}}};
  return new_metadata;
}

}  // namespace

Result<MultiscaleMetadata> MultiscaleMetadata::Parse(::nlohmann::json j) {
  MultiscaleMetadata metadata;
  if (auto* obj = j.get_ptr<::nlohmann::json::object_t*>()) {
    metadata.attributes = std::move(*obj);
  } else {
    return internal_json::ExpectedError(j, "object");
  }
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      metadata.attributes, kAtSignTypeId, [](const ::nlohmann::json& value) {
        if (value != kMultiscaleVolumeTypeId) {
          return absl::InvalidArgumentError(
              StrCat("Expected ", QuoteString(kMultiscaleVolumeTypeId),
                     " but received: ", value.dump()));
        }
        return absl::OkStatus();
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, kTypeId, [&](const ::nlohmann::json& value) {
        return internal::JsonRequireValueAs(value, &metadata.type,
                                            /*strict=*/true);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, kDataTypeId, [&](const ::nlohmann::json& value) {
        return ParseDataType(value, &metadata.data_type);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, kNumChannelsId, [&](const ::nlohmann::json& value) {
        return ParseNumChannels(value, &metadata.num_channels);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonRequireObjectMember(
      metadata.attributes, kScalesId, [&](const ::nlohmann::json& value) {
        return internal::JsonParseArray(
            value,
            [&](std::ptrdiff_t size) {
              metadata.scales.resize(size);
              return absl::OkStatus();
            },
            [&](const ::nlohmann::json& v, std::ptrdiff_t i) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  metadata.scales[i],
                  ParseScaleMetadata(v, metadata.data_type,
                                     metadata.num_channels));
              return absl::OkStatus();
            });
      }));
  return metadata;
}

Result<MultiscaleMetadataConstraints> MultiscaleMetadataConstraints::Parse(
    const ::nlohmann::json& j) {
  MultiscaleMetadataConstraints metadata;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
      j, {kTypeId, kDataTypeId, kNumChannelsId}));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kTypeId, [&](const ::nlohmann::json& value) {
        return internal::JsonRequireValueAs(value, &metadata.type.emplace(),
                                            /*strict=*/true);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kDataTypeId, [&](const ::nlohmann::json& value) {
        return ParseDataType(value, &metadata.data_type);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kNumChannelsId, [&](const ::nlohmann::json& value) {
        return ParseNumChannels(value, &metadata.num_channels.emplace());
      }));
  return metadata;
}

Result<ScaleMetadataConstraints> ScaleMetadataConstraints::Parse(
    const ::nlohmann::json& j, DataType data_type,
    std::optional<Index> num_channels) {
  ScaleMetadataConstraints metadata;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
      j, {kKeyId, kSizeId, kChunkSizeId, kVoxelOffsetId, kResolutionId,
          kEncodingId, kJpegQualityId, kCompressedSegmentationBlockSizeId,
          kShardingId}));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kKeyId, [&](const ::nlohmann::json& value) {
        return internal::JsonRequireValueAs(value, &metadata.key.emplace(),
                                            /*strict=*/true);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kSizeId, [&](const ::nlohmann::json& value) {
        metadata.box.emplace().Fill(IndexInterval::UncheckedSized(0, 0));
        return ParseSize(value, metadata.box->shape());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kVoxelOffsetId, [&](const ::nlohmann::json& value) {
        if (!metadata.box) {
          return absl::InvalidArgumentError(
              "cannot be specified without \"size\"");
        }
        return ParseVoxelOffset(value, metadata.box->origin());
      }));
  if (metadata.box) {
    TENSORSTORE_RETURN_IF_ERROR(CheckScaleBounds(*metadata.box));
  }
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kResolutionId, [&](const ::nlohmann::json& value) {
        return ParseResolution(value, metadata.resolution.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kChunkSizeId, [&](const ::nlohmann::json& value) {
        return ParseChunkSize(value, &metadata.chunk_size.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kShardingId, [&](const ::nlohmann::json& value) {
        if (value.is_null()) {
          metadata.sharding = NoShardingSpec{};
        } else {
          TENSORSTORE_ASSIGN_OR_RETURN(metadata.sharding,
                                       ShardingSpec::FromJson(value));
        }
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kEncodingId, [&](const ::nlohmann::json& value) {
        return ParseEncoding(value, &metadata.encoding.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kJpegQualityId, [&](const ::nlohmann::json& value) {
        if (metadata.encoding != ScaleMetadata::Encoding::jpeg) {
          return absl::InvalidArgumentError("Only valid for \"jpeg\" encoding");
        }
        return ParseJpegQuality(value, &metadata.jpeg_quality.emplace());
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kCompressedSegmentationBlockSizeId,
      [&](const ::nlohmann::json& value) {
        if (metadata.encoding !=
            ScaleMetadata::Encoding::compressed_segmentation) {
          return absl::InvalidArgumentError(
              "Only valid for \"compressed_segmentation\" encoding");
        }
        return ParseCompressedSegmentationBlockSize(
            value, &metadata.compressed_segmentation_block_size.emplace());
      }));
  if (metadata.encoding) {
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateEncodingDataType(*metadata.encoding, data_type, num_channels));
  }
  if (metadata.box && metadata.chunk_size && metadata.sharding) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateChunkSize(
        *metadata.chunk_size, metadata.box->shape(), *metadata.sharding));
  }
  return metadata;
}

Result<OpenConstraints> OpenConstraints::Parse(const ::nlohmann::json& j,
                                               DataType data_type_constraint) {
  OpenConstraints constraints;
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, kScaleIndexId, [&](const ::nlohmann::json& value) {
        return internal::JsonRequireInteger(value,
                                            &constraints.scale_index.emplace(),
                                            /*strict=*/true, /*min_value=*/0);
      }));
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "multiscale_metadata", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            constraints.multiscale,
            MultiscaleMetadataConstraints::Parse(value));
        return absl::OkStatus();
      }));
  if (data_type_constraint.valid()) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(data_type_constraint));
    if (constraints.multiscale.data_type.valid() &&
        constraints.multiscale.data_type != data_type_constraint) {
      return absl::InvalidArgumentError(
          StrCat("Mismatch between data type in TensorStore Spec (",
                 data_type_constraint, ") and in \"multiscale_metadata\" (",
                 constraints.multiscale.data_type, ")"));
    }
    constraints.multiscale.data_type = data_type_constraint;
  }
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(
      j, "scale_metadata", [&](const ::nlohmann::json& value) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            constraints.scale, ScaleMetadataConstraints::Parse(
                                   value, constraints.multiscale.data_type,
                                   constraints.multiscale.num_channels));
        return absl::OkStatus();
      }));
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
  if (new_metadata.data_type != existing_metadata.data_type) {
    return MismatchError(kDataTypeId, existing_metadata.data_type.name(),
                         new_metadata.data_type.name());
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
  obj.emplace(kDataTypeId, metadata.data_type.name());
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
                                             new_metadata->data_type,
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
                              const OpenConstraints& constraints,
                              OpenMode open_mode) {
  if (!(open_mode & OpenMode::allow_option_mismatch)) {
    if (auto status = ValidateMultiscaleConstraintsForOpen(
            constraints.multiscale, metadata);
        !status.ok()) {
      return status;
    }
  }
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
  if (!(open_mode & OpenMode::allow_option_mismatch)) {
    if (auto status = ValidateScaleConstraintsForOpen(
            constraints.scale, metadata.scales[scale_index]);
        !status.ok()) {
      return status;
    }
  }
  return scale_index;
}

std::string ResolveScaleKey(absl::string_view key_prefix,
                            absl::string_view scale_key) {
  std::vector<absl::string_view> output_parts = absl::StrSplit(key_prefix, '/');
  for (absl::string_view part : absl::StrSplit(scale_key, '/')) {
    if (part == ".." && !output_parts.empty()) {
      output_parts.resize(output_parts.size() - 1);
    } else {
      output_parts.push_back(part);
    }
  }
  return absl::StrJoin(output_parts, "/");
}

Status ValidateDataType(DataType data_type) {
  assert(data_type.valid());
  if (!absl::c_linear_search(kSupportedDataTypes, data_type.id())) {
    return absl::InvalidArgumentError(
        StrCat(data_type, " data type is not one of the supported data types: ",
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

std::function<std::uint64_t(std::uint64_t shard)>
GetChunksPerVolumeShardFunction(const ShardingSpec& sharding_spec,
                                span<const Index, 3> volume_shape,
                                span<const Index, 3> chunk_shape) {
  if (sharding_spec.hash_function != ShardingSpec::HashFunction::identity) {
    // For non-identity hash functions, the number of chunks per shard is not
    // predicable and the shard doesn't correspond to a rectangular region
    // anyway.
    return {};
  }

  const std::array<int, 3> z_index_bits =
      GetCompressedZIndexBits(volume_shape, chunk_shape);
  const int total_z_index_bits =
      z_index_bits[0] + z_index_bits[1] + z_index_bits[2];
  if (total_z_index_bits >
      (sharding_spec.preshift_bits + sharding_spec.minishard_bits +
       sharding_spec.shard_bits)) {
    // A shard doesn't correspond to a rectangular region.
    return {};
  }

  std::array<Index, 3> grid_shape;
  for (int i = 0; i < 3; ++i) {
    grid_shape[i] = CeilOfRatio(volume_shape[i], chunk_shape[i]);
  }

  // Any additional non-shard bits beyond `total_z_index_bits` are irrelevant
  // because they will always be 0.  Constraining `non_shard_bits` here allows
  // us to avoid checking later.
  const int non_shard_bits =
      std::min(sharding_spec.minishard_bits + sharding_spec.preshift_bits,
               total_z_index_bits);

  // Any additional shard bits beyond `total_z_index_bits - non_shard_bits` are
  // irrelevant because they will always be 0.
  const int shard_bits =
      std::min(sharding_spec.shard_bits, total_z_index_bits - non_shard_bits);

  return [grid_shape, shard_bits, non_shard_bits,
          z_index_bits](std::uint64_t shard) -> std::uint64_t {
    if ((shard >> shard_bits) != 0) {
      // Invalid shard number.
      return 0;
    }

    std::array<Index, 3> cell_shape;
    cell_shape.fill(1);
    std::array<Index, 3> cur_bit_for_dim;
    cur_bit_for_dim.fill(0);

    const auto ForEachBit = [&](int num_bits, auto func) {
      int dim_i = 0;
      for (int bit_i = 0; bit_i < num_bits; ++bit_i) {
        while (cur_bit_for_dim[dim_i] == z_index_bits[dim_i]) {
          dim_i = (dim_i + 1) % 3;
        }
        func(bit_i, dim_i);
        ++cur_bit_for_dim[dim_i];
        dim_i = (dim_i + 1) % 3;
      }
    };

    ForEachBit(non_shard_bits, [](int bit_i, int dim_i) {});

    for (int dim_i = 0; dim_i < 3; ++dim_i) {
      cell_shape[dim_i] = Index(1) << cur_bit_for_dim[dim_i];
    }

    std::array<Index, 3> cell_origin;
    cell_origin.fill(0);
    ForEachBit(shard_bits, [&](int bit_i, int dim_i) {
      if ((shard >> bit_i) & 1) {
        cell_origin[dim_i] |= Index(1) << cur_bit_for_dim[dim_i];
      }
    });

    std::uint64_t num_chunks = 1;
    for (int dim_i = 0; dim_i < 3; ++dim_i) {
      num_chunks *= static_cast<std::uint64_t>(
          std::min(grid_shape[dim_i] - cell_origin[dim_i], cell_shape[dim_i]));
    }
    assert(((non_shard_bits == 0) ? num_chunks
                                  : (num_chunks >> non_shard_bits)) <= 1);
    return num_chunks;
  };
}

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore
