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

#include "tensorstore/internal/compression/avif.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include <avif/avif.h>
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace avif {
namespace {

struct AvifEncoderDeleter {
  void operator()(avifEncoder* encoder) { avifEncoderDestroy(encoder); }
};

struct AvifDecoderDeleter {
  void operator()(avifDecoder* decoder) { avifDecoderDestroy(decoder); }
};

struct AvifImageDeleter {
  void operator()(avifImage* image) { avifImageDestroy(image); }
};

struct ReadState {
  absl::Cord input;
  std::string buffer;  // temporary buffer.
};

avifResult AvifRead(avifIO* io, uint32_t readFlags, uint64_t offset,
                    size_t size, avifROData* out) {
  if (readFlags != 0) {
    return AVIF_RESULT_IO_ERROR;
  }
  // io->data cannot be nullptr.
  auto* state = static_cast<ReadState*>(io->data);
  if (offset > state->input.size()) {
    return AVIF_RESULT_IO_ERROR;
  }

  uint64_t availableSize = state->input.size() - offset;
  if (size > availableSize) {
    size = (size_t)availableSize;
  }
  out->size = size;
  out->data = nullptr;
  if (size > 0) {
    auto subcord = state->input.Subcord(offset, size);
    if (auto range = subcord.TryFlat(); range.has_value()) {
      out->data = reinterpret_cast<const uint8_t*>(range->data());
    } else {
      absl::CopyCordToString(subcord, &state->buffer);
      out->data = reinterpret_cast<const uint8_t*>(state->buffer.data());
    }
  }
  return AVIF_RESULT_OK;
}

}  // namespace

absl::Status Encode(const unsigned char* source, size_t width, size_t height,
                    size_t num_components, const EncodeOptions& options,
                    absl::Cord* output) {
  if (num_components < 1 || num_components > 4) {
    return absl::InvalidArgumentError(
        "AVIF encoding requires between 1 and 4 components");
  }
  if (options.quantizer < AVIF_QUANTIZER_BEST_QUALITY ||
      options.quantizer > AVIF_QUANTIZER_WORST_QUALITY) {
    return absl::InvalidArgumentError(StrCat(
        "AVIF quantizer option must be in the range [",
        AVIF_QUANTIZER_BEST_QUALITY, "..", AVIF_QUANTIZER_WORST_QUALITY, "] "));
  }
  if (options.speed > AVIF_SPEED_FASTEST ||
      options.speed < AVIF_SPEED_SLOWEST) {
    return absl::InvalidArgumentError(
        StrCat("AVIF speed must be in the range [", AVIF_SPEED_SLOWEST, "..",
               AVIF_SPEED_FASTEST, "] "));
  }
  const bool lossless = options.quantizer == 0;

  // See avifenc.c to see how to set parameters.
  std::unique_ptr<avifImage, AvifImageDeleter> image(
      avifImageCreate(width, height, /*depth=*/8, AVIF_PIXEL_FORMAT_NONE));
  if (lossless) {
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV444;
    image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_IDENTITY;
  } else {
    // TODO: Check how chroma subsampling affects results (YUV422, etc.)
    image->yuvFormat = AVIF_PIXEL_FORMAT_YUV422;
    image->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT601;
  }

  avifResult result = AVIF_RESULT_OK;
  switch (num_components) {
    case 1:
      image->yuvFormat = AVIF_PIXEL_FORMAT_YUV400;
      image->yuvPlanes[AVIF_CHAN_Y] = const_cast<uint8_t*>(source);
      image->yuvRowBytes[AVIF_CHAN_Y] = width;
      image->imageOwnsYUVPlanes = AVIF_FALSE;
      break;

    case 2: {
      image->yuvFormat = AVIF_PIXEL_FORMAT_YUV400;
      avifImageAllocatePlanes(image.get(), AVIF_PLANES_ALL);
      // Copy source image into avif image.
      const uint8_t* data = source;
      uint8_t* ptrY = image->yuvPlanes[AVIF_CHAN_Y];
      uint8_t* ptrA = image->alphaPlane;

      for (uint32_t j = 0; j < image->height; j++) {
        for (uint32_t i = 0; i < image->width; i++) {
          ptrY[i] = *data++;
          ptrA[i] = *data++;
        }
        ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        ptrA += image->alphaRowBytes;
      }
      break;
    }

    case 3: {
      /// Should we avoid transcoding from RGB to YUV?
      avifRGBImage rgb_image;
      avifRGBImageSetDefaults(&rgb_image, image.get());
      rgb_image.format = AVIF_RGB_FORMAT_RGB;
      rgb_image.pixels = const_cast<uint8_t*>(source);
      rgb_image.rowBytes = width * 3;
      result = avifImageRGBToYUV(image.get(), &rgb_image);
      break;
    }

    case 4: {
      avifRGBImage rgb_image;
      avifRGBImageSetDefaults(&rgb_image, image.get());
      rgb_image.format = AVIF_RGB_FORMAT_RGBA;
      rgb_image.pixels = const_cast<uint8_t*>(source);
      rgb_image.rowBytes = width * 4;
      result = avifImageRGBToYUV(image.get(), &rgb_image);
      break;
    }
    default:
      return absl::InternalError("Unexpected num_components");
  }

  if (result != AVIF_RESULT_OK) {
    return absl::InternalError(
        StrCat("Failed to convert RGB to YUV ", avifResultToString(result)));
  }

  std::unique_ptr<avifEncoder, AvifEncoderDeleter> encoder(avifEncoderCreate());

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
  std::string quantizer = absl::StrCat(options.quantizer);
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

  /// For more options, see libavif/src/codec_aom.c
  result = avifEncoderAddImage(encoder.get(), image.get(), 1,
                               AVIF_ADD_IMAGE_FLAG_SINGLE);
  if (result != AVIF_RESULT_OK) {
    return absl::InternalError(
        StrCat("Failed to encode image ", avifResultToString(result)));
  }

  avifRWData avif_output = AVIF_DATA_EMPTY;
  result = avifEncoderFinish(encoder.get(), &avif_output);
  if (result != AVIF_RESULT_OK) {
    return absl::UnknownError(
        StrCat("Failed to finish encode ", avifResultToString(result)));
  }

  *output = absl::MakeCordFromExternal(
      absl::string_view(reinterpret_cast<const char*>(avif_output.data),
                        avif_output.size),
      [data = avif_output]() mutable { avifRWDataFree(&data); });

  return absl::OkStatus();
}

absl::Status Decode(const absl::Cord& input,
                    absl::FunctionRef<Result<unsigned char*>(
                        size_t width, size_t height, size_t num_components)>
                        allocate_buffer) {
  if (input.empty()) {
    return absl::InvalidArgumentError("Cannot decode an AVIF from empty data");
  }
  ReadState state{input};

  avifIO io;
  io.destroy = nullptr;
  io.read = AvifRead;
  io.write = nullptr;
  io.sizeHint = input.size();
  io.persistent = AVIF_FALSE;
  io.data = &state;

  // decoding defaults to AVIF_CODEC_CHOICE_AUTO; if we want more control
  // over the decoder, add that here.
  std::unique_ptr<avifDecoder, AvifDecoderDeleter> decoder(avifDecoderCreate());

  avifDecoderSetIO(decoder.get(), &io);

  avifResult result = avifDecoderParse(decoder.get());
  if (result != AVIF_RESULT_OK) {
    return absl::InvalidArgumentError(
        StrCat("Failed to parse AVIF stream: ", avifResultToString(result)));
  }
  if (decoder->imageCount != 1) {
    return absl::InvalidArgumentError(
        "AVIF contains more than one image (not supported)");
  }

  // Only read the first image even if there are multiple.
  result = avifDecoderNextImage(decoder.get());
  if (result != AVIF_RESULT_OK) {
    return absl::InvalidArgumentError(
        StrCat("Failed to decode AVIF image: ", avifResultToString(result)));
  }
  const size_t height = decoder->image->height;
  const size_t width = decoder->image->width;
  const size_t num_channels =
      ((decoder->image->yuvFormat == AVIF_PIXEL_FORMAT_YUV400) ? 1 : 3) +
      (decoder->alphaPresent ? 1 : 0);

  if (decoder->image->depth != 8 && num_channels <= 2) {
    // When num_channels > 2 we call avifImageYUVToRGB(), which allows the
    // image to be encoded with >8 bits.
    return absl::InvalidArgumentError(
        "AVIF depth not 8-bit (not supported for <3 channels)");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(unsigned char* buffer,
                               allocate_buffer(width, height, num_channels));

  if (num_channels == 1) {
    const uint8_t* src = decoder->image->yuvPlanes[AVIF_CHAN_Y];
    uint8_t* dst = buffer;
    for (size_t j = 0; j < height; j++) {
      memcpy(dst, src, width);
      src += decoder->image->yuvRowBytes[AVIF_CHAN_Y];
      dst += width;
    }
    return absl::OkStatus();
  }
  if (num_channels == 2) {
    const uint8_t* ptrY = decoder->image->yuvPlanes[AVIF_CHAN_Y];
    const uint8_t* ptrA = decoder->image->alphaPlane;
    uint8_t* dest = buffer;
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        *dest++ = ptrY[i];
        *dest++ = ptrA[i];
      }
      ptrY += decoder->image->yuvRowBytes[AVIF_CHAN_Y];
      ptrA += decoder->image->alphaRowBytes;
    }
    return absl::OkStatus();
  }

  avifRGBImage rgb_image;
  avifRGBImageSetDefaults(&rgb_image, decoder->image);

  // Bit depth transformation happens here within libavif, including both RGB
  // channels and alpha channel (if present).
  rgb_image.depth = 8;
  rgb_image.format = (num_channels == 4) ? avifRGBFormat::AVIF_RGB_FORMAT_RGBA
                                         : avifRGBFormat::AVIF_RGB_FORMAT_RGB;
  rgb_image.rowBytes = width * avifRGBImagePixelSize(&rgb_image);
  rgb_image.pixels = buffer;

  auto transform_result = avifImageYUVToRGB(decoder->image, &rgb_image);
  if (transform_result != AVIF_RESULT_OK) {
    return absl::InvalidArgumentError(
        StrCat("Failed to convert AVIF YUV to RGB image: ",
               avifResultToString(result)));
  }
  return absl::OkStatus();
}

}  // namespace avif
}  // namespace tensorstore
