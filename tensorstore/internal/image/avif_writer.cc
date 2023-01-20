// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/image/avif_writer.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/avif_common.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// Include libavif last
#include <avif/avif.h>

namespace tensorstore {
namespace internal_image {
namespace {

template <size_t NumBytes, size_t NumChannels>
void strided_scatter(const unsigned char* src,
                     std::array<unsigned char*, NumChannels> dest,
                     size_t n_elems) {
  for (size_t i = 0; i < n_elems; i++) {
    for (size_t j = 0; j < NumChannels; j++) {
      memcpy(dest[j], src, NumBytes);
      dest[j] += NumBytes;
      src += NumBytes;
    }
  }
}

void FillYUVImage(const ImageInfo& info,
                  tensorstore::span<const unsigned char> source,
                  avifImage* image) {
  ImageView source_view = MakeWriteImageView(info, source);

  if (info.num_components == 1) {
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV400;
    image->yuvPlanes[AVIF_CHAN_Y] = source_view.data().data();
    image->yuvRowBytes[AVIF_CHAN_Y] = info.width;
    image->imageOwnsYUVPlanes = AVIF_FALSE;
    return;
  }

  if (info.num_components == 2) {
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV400;
    avifImageAllocatePlanes(image, AVIF_PLANES_ALL);
    assert(image->alphaRowBytes > 0);
  } else if (info.num_components == 3) {
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV444;
    avifImageAllocatePlanes(image, AVIF_PLANES_YUV);
  } else if (info.num_components == 4) {
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV444;
    avifImageAllocatePlanes(image, AVIF_PLANES_ALL);
    assert(image->alphaRowBytes > 0);
  } else {
    ABSL_LOG(FATAL) << "Wrong num_channels for FillYUVImage";
  }

  // Copy source image into avif image.
  unsigned char* ptrY = image->yuvPlanes[AVIF_CHAN_Y];
  unsigned char* ptrU = image->yuvPlanes[AVIF_CHAN_U];
  unsigned char* ptrV = image->yuvPlanes[AVIF_CHAN_V];
  unsigned char* ptrA = image->alphaPlane;

  if (avifImageUsesU16(image)) {
    if (info.num_components == 2) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<2, 2>(source_view.data_row(y).data(), {ptrY, ptrA},
                              info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrA += image->alphaRowBytes;
      }
    } else if (info.num_components == 3) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<2, 3>(source_view.data_row(y).data(),
                              {ptrY, ptrU, ptrV}, info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrU += image->yuvRowBytes[AVIF_CHAN_U];
        ptrV += image->yuvRowBytes[AVIF_CHAN_V];
      }

    } else if (info.num_components == 4) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<2, 4>(source_view.data_row(y).data(),
                              {ptrY, ptrU, ptrV, ptrA}, info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrU += image->yuvRowBytes[AVIF_CHAN_U];
        ptrV += image->yuvRowBytes[AVIF_CHAN_V];
        ptrA += image->alphaRowBytes;
      }
    }
  } else {
    if (info.num_components == 2) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<1, 2>(source_view.data_row(y).data(), {ptrY, ptrA},
                              info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrA += image->alphaRowBytes;
      }
    } else if (info.num_components == 3) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<1, 3>(source_view.data_row(y).data(),
                              {ptrY, ptrU, ptrV}, info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrU += image->yuvRowBytes[AVIF_CHAN_U];
        ptrV += image->yuvRowBytes[AVIF_CHAN_V];
      }
    } else if (info.num_components == 4) {
      for (uint32_t y = 0; y < image->height; y++) {
        strided_scatter<1, 4>(source_view.data_row(y).data(),
                              {ptrY, ptrU, ptrV, ptrA}, info.width);
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrU += image->yuvRowBytes[AVIF_CHAN_U];
        ptrV += image->yuvRowBytes[AVIF_CHAN_V];
        ptrA += image->alphaRowBytes;
      }
    }
  }
}

absl::Status AvifAddImage(avifEncoder* encoder,
                          const AvifWriterOptions& options,
                          const ImageInfo& info,
                          tensorstore::span<const unsigned char> source) {
  if (info.dtype != dtype_v<uint8_t> && info.dtype != dtype_v<uint16_t>) {
    return absl::InvalidArgumentError(
        "AVIF encoding requires uint8_t or uint16_t");
  }
  if (info.num_components == 0 || info.num_components > 4) {
    return absl::InvalidArgumentError("AVIF encoding invalid num_components");
  }

  // NOTE: Maybe allow setting pixel format to something other than yuv444?
  std::unique_ptr<avifImage, AvifDeleter> image(avifImageCreateEmpty());

  image->height = info.height;
  image->width = info.width;
  image->depth = info.dtype.size() * 8;
  // Default values to assume lossless encoding.
  // image->alphaPremultiplied = AVIF_FALSE;
  image->yuvFormat = AVIF_PIXEL_FORMAT_YUV444;
  image->yuvRange = AVIF_RANGE_FULL;
  // CICP (CP/TC/MC): https://github.com/AOMediaCodec/libavif/wiki/CICP
  image->colorPrimaries = AVIF_COLOR_PRIMARIES_UNSPECIFIED;
  image->transferCharacteristics = AVIF_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
  image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_IDENTITY;

  const bool lossless = (options.quantizer == 0);
  if (info.num_components >= 3 && options.input_is_rgb) {
    if (!lossless) {
      image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT601;
    }

    avifRGBImage rgb_image;
    avifRGBImageSetDefaults(&rgb_image, image.get());
    rgb_image.format =
        info.num_components == 3 ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
    rgb_image.pixels = const_cast<uint8_t*>(source.data());
    rgb_image.rowBytes = info.num_components * info.width * info.dtype.size();
    avifResult result = avifImageRGBToYUV(image.get(), &rgb_image);
    if (result != AVIF_RESULT_OK) {
      return absl::InternalError(tensorstore::StrCat(
          "Failed to convert RGB to YUV ", avifResultToString(result)));
    }
  } else {
    FillYUVImage(info, source, image.get());
  }

#if 0
  ABSL_LOG(INFO) <<"avifImageUsesU16 "<< avifImageUsesU16(image.get());
  ABSL_LOG(INFO) <<"yuvFormat "<< image->yuvFormat;
  ABSL_LOG(INFO) <<"yuvRange "<< image->yuvRange;
  ABSL_LOG(INFO) <<"colorPrimaries "<< image->colorPrimaries;
  ABSL_LOG(INFO) <<"transferCharacteristics "<< image->transferCharacteristics;
  ABSL_LOG(INFO) <<"matrixCoefficients "<< image->matrixCoefficients;
  ABSL_LOG(INFO) <<"alphaPremultiplied "<< image->alphaPremultiplied;
  ABSL_LOG(INFO) <<"alphaRowBytes "<< image->alphaRowBytes;
#endif

  /// For more options, see libavif/src/codec_aom.c
  avifResult result =
      avifEncoderAddImage(encoder, image.get(), 1, AVIF_ADD_IMAGE_FLAG_SINGLE);
  if (result != AVIF_RESULT_OK) {
    return absl::InternalError(tensorstore::StrCat("Failed to encode image ",
                                                   avifResultToString(result)));
  }
  return absl::OkStatus();
}

absl::Status AvifFinish(avifEncoder* encoder, riegeli::Writer* writer) {
  avifRWData avif_output = AVIF_DATA_EMPTY;
  avifResult result = avifEncoderFinish(encoder, &avif_output);
  if (result != AVIF_RESULT_OK) {
    return absl::DataLossError(tensorstore::StrCat("Failed to finish encode ",
                                                   avifResultToString(result)));
  }
  absl::string_view buffer(reinterpret_cast<const char*>(avif_output.data),
                           avif_output.size);
  bool ok;

  if (buffer.size() <= riegeli::kMaxBytesToCopy || writer->PrefersCopying()) {
    ok = writer->Write(buffer);
    avifRWDataFree(&avif_output);
  } else {
    ok = writer->Write(absl::MakeCordFromExternal(
        buffer, [data = avif_output]() mutable { avifRWDataFree(&data); }));
  }
  if (!ok) {
    if (!writer->ok()) {
      return MaybeAnnotateStatus(writer->status(), "Encoding AVIF");
    }
    return absl::InternalError("Encoding AVIF");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status AvifWriter::InitializeImpl(riegeli::Writer* writer,
                                        const AvifWriterOptions& options) {
  ABSL_CHECK(writer != nullptr);
  if (encoder_) {
    return absl::InternalError("Initialize() already called");
  }
  writer_ = std::move(writer);
  options_ = options;

  if (options.quantizer < AVIF_QUANTIZER_BEST_QUALITY ||
      options.quantizer > AVIF_QUANTIZER_WORST_QUALITY) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "AVIF quantizer option must be in the range [",
        AVIF_QUANTIZER_BEST_QUALITY, "..", AVIF_QUANTIZER_WORST_QUALITY, "] "));
  }
  if (options.speed > AVIF_SPEED_FASTEST ||
      options.speed < AVIF_SPEED_SLOWEST) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "AVIF speed must be in the range [", AVIF_SPEED_SLOWEST, "..",
        AVIF_SPEED_FASTEST, "] "));
  }

  /// Test for the AOM codec for lossless encoding.
  const bool lossless = (options.quantizer == 0);
  if (lossless) {
    if (avifCodecName(AVIF_CODEC_CHOICE_AOM, AVIF_CODEC_FLAG_CAN_ENCODE) ==
        nullptr) {
      return absl::InvalidArgumentError(
          "AVIF codec AOM missing for lossless encoding");
    }
  }

  std::unique_ptr<avifEncoder, AvifDeleter> encoder(avifEncoderCreate());
  encoder->speed = options.speed;
  if (lossless) {
    /// Use the AOM reference codec. While others may be available, the aom
    /// codec is the only codec which supports lossless encoding.
    ///   https://github.com/xiph/rav1e/issues/151
    ///   https://gitlab.com/AOMediaCodec/SVT-AV1/-/issues/1636
    encoder->codecChoice = AVIF_CODEC_CHOICE_AOM;
    encoder->minQuantizer = AVIF_QUANTIZER_LOSSLESS;
    encoder->maxQuantizer = AVIF_QUANTIZER_LOSSLESS;
    encoder->minQuantizerAlpha = AVIF_QUANTIZER_LOSSLESS;
    encoder->maxQuantizerAlpha = AVIF_QUANTIZER_LOSSLESS;
  } else {
    /// AVIF encodes the alpha channel as another image pane, so the
    /// quantizer settings for alpha and main channels should be the same.
    encoder->minQuantizer = AVIF_QUANTIZER_BEST_QUALITY;
    encoder->maxQuantizer = AVIF_QUANTIZER_WORST_QUALITY;
    encoder->minQuantizerAlpha = AVIF_QUANTIZER_BEST_QUALITY;
    encoder->maxQuantizerAlpha = AVIF_QUANTIZER_WORST_QUALITY;
  }

  /// Use the codec specific cq-level option rather than the global
  /// quantizer setting for quality.
  std::string quantizer = tensorstore::StrCat(options.quantizer);
  avifEncoderSetCodecSpecificOption(encoder.get(), "cq-level",
                                    quantizer.c_str());

  /// Set the rate control to constant-quality mode.
  avifEncoderSetCodecSpecificOption(encoder.get(), "end-usage", "q");

  /// TODO: Experiment with the tune parameter: ssim vs psnr
#if 0
  if (cq_level <= 32) {
    avifEncoderSetCodecSpecificOption(encoder.get(), "tune", "ssim");
  }
#endif
  encoder_ = std::move(encoder);
  return absl::OkStatus();
}

absl::Status AvifWriter::Encode(const ImageInfo& info,
                                tensorstore::span<const unsigned char> source) {
  if (!encoder_) {
    return absl::InternalError("AVIF writer not initialized");
  }
  ABSL_CHECK_EQ(source.size(), ImageRequiredBytes(info));
  return AvifAddImage(encoder_.get(), options_, info, source);
}

absl::Status AvifWriter::Done() {
  if (!encoder_) {
    return absl::InternalError("No data written");
  }
  std::unique_ptr<avifEncoder, AvifDeleter> encoder = std::move(encoder_);
  TENSORSTORE_RETURN_IF_ERROR(AvifFinish(encoder.get(), writer_));
  if (!writer_->Close()) {
    return writer_->status();
  }
  return absl::OkStatus();
}

}  // namespace internal_image
}  // namespace tensorstore
