// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/ometiff/ometiff_spec.h"

#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/serialization/json_bindable.h"

// Keep at the very end please.
#include <tiffio.h>

#include <tiffio.hxx>

namespace tensorstore {
namespace ometiff {
namespace {

namespace jb = tensorstore::internal_json_binding;

Result<DataType> SetDType(uint16_t sample_format, uint16_t bits_per_sample) {
  const char* sample_format_str = "";
  /// Validate sample format
  switch (sample_format) {
    case SAMPLEFORMAT_INT:
      sample_format_str = " INT";
      // TODO: Support bits_per_sample < 8.
      if (bits_per_sample == 8) {
        return dtype_v<int8_t>;
      } else if (bits_per_sample == 16) {
        return dtype_v<int16_t>;
      } else if (bits_per_sample == 32) {
        return dtype_v<int32_t>;
      }
      break;
    case SAMPLEFORMAT_UINT:
      sample_format_str = " UINT";
      if (bits_per_sample == 1) {
        return dtype_v<bool>;
      } else if (bits_per_sample == 2 || bits_per_sample == 4 ||
                 bits_per_sample == 8) {
        return dtype_v<uint8_t>;
      } else if (bits_per_sample == 16) {
        return dtype_v<uint16_t>;
      } else if (bits_per_sample == 32) {
        return dtype_v<uint32_t>;
      }
      break;
    case SAMPLEFORMAT_IEEEFP:
      sample_format_str = " IEEE FP";
      if (bits_per_sample == 16) {
        return dtype_v<tensorstore::float16_t>;
      } else if (bits_per_sample == 32) {
        return dtype_v<float>;
      } else if (bits_per_sample == 64) {
        return dtype_v<double>;
      }
      break;
    case SAMPLEFORMAT_COMPLEXIEEEFP:
      sample_format_str = " COMPLEX IEEE FP";
      if (bits_per_sample == 64) {
        return dtype_v<tensorstore::complex64_t>;
      } else if (bits_per_sample == 128) {
        return dtype_v<tensorstore::complex128_t>;
      }
      break;
    case SAMPLEFORMAT_COMPLEXINT:
      sample_format_str = " COMPLEX INT";
      // tensorstore does not have a complex<int> type.
      break;
    case SAMPLEFORMAT_VOID:
      sample_format_str = " VOID";
      // maybe this should just be uint_t[n]?
      break;
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "TIFF read failed: sampleformat%s / bitspersample (%d) not supported",
      sample_format_str, bits_per_sample));
}
}  // namespace

std::ostream& operator<<(std::ostream& os, const OMETiffImageInfo& x) {
  // `ToJson` is guaranteed not to fail for this type.
  return os << jb::ToJson(x).value();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(OMETiffImageInfo, [](auto is_loading,
                                                            const auto& options,
                                                            auto* obj,
                                                            auto* j) {
  return jb::Object(
      jb::Member("width", jb::Projection(&OMETiffImageInfo::width)),
      jb::Member("height", jb::Projection(&OMETiffImageInfo::height)),
      jb::Member("bits_per_sample",
                 jb::Projection(&OMETiffImageInfo::bits_per_sample)),
      jb::Member("tile_width", jb::Projection(&OMETiffImageInfo::tile_width)),
      jb::Member("tile_height", jb::Projection(&OMETiffImageInfo::tile_height)),
      jb::Member("rows_per_strip",
                 jb::Projection(&OMETiffImageInfo::rows_per_strip)),
      jb::Member("sample_format",
                 jb::Projection(&OMETiffImageInfo::sample_format)),
      jb::Member("samples_per_pixel",
                 jb::Projection(&OMETiffImageInfo::samples_per_pixel)),
      jb::Member("is_tiled", jb::Projection(&OMETiffImageInfo::is_tiled)),
      jb::Member("chunk_offset",
                 jb::Projection(&OMETiffImageInfo::chunk_offset)),
      jb::Member("chunk_size", jb::Projection(&OMETiffImageInfo::chunk_size)),
      jb::Member("num_chunks", jb::Projection(&OMETiffImageInfo::num_chunks)),
      jb::Member("compression", jb::Projection(&OMETiffImageInfo::compression)),
      jb::Member("dtype", jb::Projection(&OMETiffImageInfo::dtype,
                                         jb::ConstrainedDataTypeJsonBinder)))(
      is_loading, options, obj, j);
});

Result<::nlohmann::json> GetOMETiffImageInfo(std::istream& istream) {
  OMETiffImageInfo image_info;

  ABSL_LOG(INFO) << "Opening TIFF";
  TIFF* tiff = TIFFStreamOpen("ts", &istream);
  if (tiff == nullptr) {
    return absl::NotFoundError("Unable to open TIFF file");
  }

  ABSL_LOG(INFO) << "Reading image width and height";
  if (!TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &image_info.width) ||
      !TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &image_info.height)) {
    return absl::InvalidArgumentError("TIFF read failed: invalid image");
  }

  ABSL_LOG(INFO) << "Checking to see if image is tiled";
  image_info.is_tiled = TIFFIsTiled(tiff);

  if (image_info.is_tiled) {
    ABSL_LOG(INFO) << "Reading tile width and height";
    if (!TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &image_info.tile_width) ||
        !TIFFGetField(tiff, TIFFTAG_TILELENGTH, &image_info.tile_height)) {
      return absl::InvalidArgumentError("TIFF read failed: invalid tile");
    }
    image_info.chunk_size = TIFFTileSize64(tiff);
    image_info.num_chunks = TIFFNumberOfTiles(tiff);
  } else {
    ABSL_LOG(INFO) << "Reading rows per strip";
    TIFFGetFieldDefaulted(tiff, TIFFTAG_ROWSPERSTRIP,
                          &image_info.rows_per_strip);
    image_info.chunk_size = TIFFStripSize64(tiff);
    image_info.num_chunks = TIFFNumberOfStrips(tiff);
  }

  // These call TIFFSetField to update the in-memory structure so that
  // subsequent calls get appropriate defaults.
  ABSL_LOG(INFO) << "Reading bits per sample";
  if (!TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &image_info.bits_per_sample)) {
    image_info.bits_per_sample = 1;
    ABSL_LOG(INFO) << "Setting bits per sample";
    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, image_info.bits_per_sample);
  }

  ABSL_LOG(INFO) << "Reading samples per pixel";
  if (!TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL,
                    &image_info.samples_per_pixel)) {
    image_info.samples_per_pixel = 1;
    ABSL_LOG(INFO) << "Setting samples per pixel";
    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, image_info.samples_per_pixel);
  }

  ABSL_LOG(INFO) << "Reading sample format";
  TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &image_info.sample_format);

  ABSL_LOG(INFO) << "Computing data type";
  TENSORSTORE_ASSIGN_OR_RETURN(
      image_info.dtype,
      SetDType(image_info.sample_format, image_info.bits_per_sample));

  ABSL_LOG(INFO) << "Data type: " << image_info.dtype;

  ABSL_LOG(INFO) << "Reading compression";
  TIFFGetFieldDefaulted(tiff, TIFFTAG_COMPRESSION, &image_info.compression);
  if (image_info.compression != COMPRESSION_NONE)
    return absl::InternalError(
        "Cannot read TIFF; compression format not supported");

  ABSL_LOG(INFO) << "Getting strile offset";
  // Get offset of first strile and we can calculate the rest.
  image_info.chunk_offset = TIFFGetStrileOffset(tiff, 0);

  return jb::ToJson(image_info);
}
}  // namespace ometiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::ometiff::OMETiffImageInfo,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::ometiff::OMETiffImageInfo>())
