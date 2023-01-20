// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/image/avif_reader.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/avif_common.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

// Include libavif last
#include <avif/avif.h>

namespace tensorstore {
namespace internal_image {
namespace {

template <size_t NumBytes, size_t NumChannels>
void strided_gather(unsigned char* dest,
                    std::array<const unsigned char*, NumChannels> src,
                    size_t n_elems) {
  for (size_t i = 0; i < n_elems; i++) {
    for (size_t j = 0; j < NumChannels; j++) {
      memcpy(dest, src[j], NumBytes);
      dest += NumBytes;
      src[j] += NumBytes;
    }
  }
}

// avifIO for reading from a riegeli::Reader.
struct AvifRiegeli : public avifIO {
  riegeli::Reader* reader = nullptr;

  static avifResult Read(avifIO* io, uint32_t readFlags, uint64_t offset,
                         size_t size, avifROData* out);

  static void Destroy(avifIO* io);

  static avifIO* Create(riegeli::Reader* reader);
};

// This function should return a block of memory that *must* remain valid until
// another read call to this avifIO struct is made (reusing a read buffer is
// acceptable/expected).
//
// * If offset exceeds the size of the content (past EOF), return
//   AVIF_RESULT_IO_ERROR.
// * If offset is *exactly* at EOF, provide a 0-byte buffer and return
//   AVIF_RESULT_OK.
// * If (offset+size) exceeds the contents' size, it must truncate the range to
//   provide all bytes from the offset to EOF.
// * If the range is unavailable yet (due to network conditions or any other
//   reason), return AVIF_RESULT_WAITING_ON_IO.
// * Otherwise, provide the range and return AVIF_RESULT_OK.
//
avifResult AvifRiegeli::Read(avifIO* io, uint32_t readFlags, uint64_t offset,
                             size_t size, avifROData* out) {
  auto& self = *reinterpret_cast<AvifRiegeli*>(io);
  if (readFlags != 0) {
    return AVIF_RESULT_IO_ERROR;
  }
  out->data = nullptr;
  out->size = 0;

  self.reader->Seek(offset);
  self.reader->Pull(size);
  if (!self.reader->ok()) {
    return AVIF_RESULT_IO_ERROR;
  }

  size = std::min(size, self.reader->available());
  if (size) {
    out->data = reinterpret_cast<const uint8_t*>(self.reader->cursor());
    out->size = size;
    self.reader->move_cursor(size);
  }
  return AVIF_RESULT_OK;
}

void AvifRiegeli::Destroy(avifIO* io) {
  delete reinterpret_cast<AvifRiegeli*>(io);
}

avifIO* AvifRiegeli::Create(riegeli::Reader* reader) {
  reader->SetReadAllHint(true);

  AvifRiegeli* io = new AvifRiegeli();
  io->destroy = AvifRiegeli::Destroy;
  io->read = AvifRiegeli::Read;
  io->write = nullptr;
  io->sizeHint = 0;
  if (reader->SupportsSize()) {
    absl::optional<size_t> reader_size = reader->Size();
    io->sizeHint = reader_size.value_or(0);
  }
  io->reader = reader;
  io->persistent = AVIF_FALSE;  // buffer contents may change
  return io;
}

ImageInfo AvifGetImageInfo(avifDecoder* decoder) {
  ImageInfo info;
  info.width = decoder->image->width;
  info.height = decoder->image->height;

  switch (decoder->image->yuvFormat) {
    case AVIF_PIXEL_FORMAT_YUV444:
    case AVIF_PIXEL_FORMAT_YUV422:
    case AVIF_PIXEL_FORMAT_YUV420:
      info.num_components = 3;
      break;
    case AVIF_PIXEL_FORMAT_YUV400:
      info.num_components = 1;
      break;
    default:
      info.num_components = 0;
      break;
  }
  info.num_components += (decoder->alphaPresent ? 1 : 0);

  if (avifImageUsesU16(decoder->image)) {
    info.dtype = dtype_v<uint16_t>;
  } else {
    info.dtype = dtype_v<uint8_t>;
  }
  return info;
}

absl::Status AvifDefaultDecodeYUV(avifImage* image, const ImageInfo& info,
                                  tensorstore::span<unsigned char> dest) {
  ImageView dest_view(info, dest);

  const unsigned char* ptrY = image->yuvPlanes[AVIF_CHAN_Y];
  const unsigned char* ptrU = image->yuvPlanes[AVIF_CHAN_U];
  const unsigned char* ptrV = image->yuvPlanes[AVIF_CHAN_V];
  const unsigned char* ptrA = image->alphaPlane;

  switch (image->yuvFormat) {
    case AVIF_PIXEL_FORMAT_YUV444:
      if (image->alphaRowBytes > 0) {
        if (avifImageUsesU16(image)) {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<2, 4>(row.data(), {ptrY, ptrU, ptrV, ptrA},
                                 info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrU += image->yuvRowBytes[AVIF_CHAN_U];
            ptrV += image->yuvRowBytes[AVIF_CHAN_V];
            ptrA += image->alphaRowBytes;
          }
        } else {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<1, 4>(row.data(), {ptrY, ptrU, ptrV, ptrA},
                                 info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrU += image->yuvRowBytes[AVIF_CHAN_U];
            ptrV += image->yuvRowBytes[AVIF_CHAN_V];
            ptrA += image->alphaRowBytes;
          }
        }
      } else {
        if (avifImageUsesU16(image)) {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<2, 3>(row.data(), {ptrY, ptrU, ptrV}, info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrU += image->yuvRowBytes[AVIF_CHAN_U];
            ptrV += image->yuvRowBytes[AVIF_CHAN_V];
          }
        } else {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<1, 3>(row.data(), {ptrY, ptrU, ptrV}, info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrU += image->yuvRowBytes[AVIF_CHAN_U];
            ptrV += image->yuvRowBytes[AVIF_CHAN_V];
          }
        }
      }
      break;

    case AVIF_PIXEL_FORMAT_YUV400:
      if (image->alphaRowBytes > 0) {
        if (avifImageUsesU16(image)) {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<2, 2>(row.data(), {ptrY, ptrA}, info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrA += image->alphaRowBytes;
          }
        } else {
          for (size_t y = 0; y < info.height; y++) {
            auto row = dest_view.data_row(y);
            strided_gather<1, 2>(row.data(), {ptrY, ptrA}, info.width);
            ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
            ptrA += image->alphaRowBytes;
          }
        }
      } else {
        // No alpha, only 1 channel.
        for (size_t y = 0; y < info.height; y++) {
          auto row = dest_view.data_row(y);
          memcpy(row.data(), ptrY, row.size());
          ptrY += image->yuvRowBytes[AVIF_CHAN_Y];
        }
      }
      break;

    case AVIF_PIXEL_FORMAT_NONE:
      if (image->alphaRowBytes > 0) {
        for (size_t y = 0; y < info.height; y++) {
          auto row = dest_view.data_row(y);
          memcpy(row.data(), ptrA, row.size());
          ptrA += image->alphaRowBytes;
        }
      }
      break;

    default:
      return absl::InternalError("Failed to decode AVIF.");
  }

  return absl::OkStatus();
}

absl::Status AvifDefaultDecodeRGB(avifImage* image, const ImageInfo& info,
                                  tensorstore::span<unsigned char> dest) {
  avifRGBImage rgb_image;
  avifRGBImageSetDefaults(&rgb_image, image);

  // Bit depth transformation happens here within libavif, including both RGB
  // channels and alpha channel (if present) when rgb_image.depth is not set
  // to decoder->image.depth.
  rgb_image.format = (info.num_components == 4)
                         ? avifRGBFormat::AVIF_RGB_FORMAT_RGBA
                         : avifRGBFormat::AVIF_RGB_FORMAT_RGB;
  rgb_image.rowBytes = info.width * avifRGBImagePixelSize(&rgb_image);
  rgb_image.pixels = reinterpret_cast<uint8_t*>(dest.data());

  auto transform_result = avifImageYUVToRGB(image, &rgb_image);
  if (transform_result != AVIF_RESULT_OK) {
    return absl::DataLossError(
        tensorstore::StrCat("Failed to convert AVIF YUV to RGB image: ",
                            avifResultToString(transform_result)));
  }
  return absl::OkStatus();
}

absl::Status AvifDefaultDecode(avifDecoder* decoder,
                               tensorstore::span<unsigned char> dest,
                               const AvifReaderOptions& options) {
  auto info = AvifGetImageInfo(decoder);
  ABSL_CHECK(dest.size() == ImageRequiredBytes(info));

  avifImage* image = decoder->image;

#if 0
  ABSL_LOG(INFO) <<"avifImageUsesU16 "<< avifImageUsesU16(image.get());
  ABSL_LOG(INFO) <<"yuvFormat "<< image->yuvFormat;
  ABSL_LOG(INFO) <<"yuvRange " <<image->yuvRange;
  ABSL_LOG(INFO) <<"colorPrimaries "<< image->colorPrimaries;
  ABSL_LOG(INFO) <<"transferCharacteristics "<< image->transferCharacteristics;
  ABSL_LOG(INFO) <<"matrixCoefficients "<< image->matrixCoefficients;
  ABSL_LOG(INFO) <<"alphaPremultiplied "<< image->alphaPremultiplied;
  ABSL_LOG(INFO) <<"alphaRowBytes "<< image->alphaRowBytes;
#endif

  if (image->yuvFormat == AVIF_PIXEL_FORMAT_YUV400 ||
      image->yuvFormat == AVIF_PIXEL_FORMAT_NONE ||
      (image->yuvFormat == AVIF_PIXEL_FORMAT_YUV444 &&
       !options.convert_to_rgb)) {
    return AvifDefaultDecodeYUV(image, info, dest);
  } else {
    return AvifDefaultDecodeRGB(image, info, dest);
  }
}

}  // namespace

absl::Status AvifReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);

  decoder_ = nullptr;

  /// Check the signature. (offset 4) "ftypavif", etc.
  if (!reader->Pull(12) ||
      (  //
          memcmp("ftypavif", reader->cursor() + 4, 8) == 0 &&
          memcmp("ftypheic", reader->cursor() + 4, 8) == 0 &&
          memcmp("ftypheix", reader->cursor() + 4, 8) == 0 &&
          memcmp("ftypmif1", reader->cursor() + 4, 8) == 0 &&
          memcmp("ftypmsf1", reader->cursor() + 4, 8) == 0)) {
    return absl::InvalidArgumentError(
        "Failed to decode AVIF: missing AVIF signature");
  }

  auto* io = AvifRiegeli::Create(reader);
  ABSL_CHECK(io != nullptr);

  // decoding defaults to AVIF_CODEC_CHOICE_AUTO; if we want more control
  // over the decoder, add that here.
  std::unique_ptr<avifDecoder, AvifDeleter> decoder(avifDecoderCreate());
  avifDecoderSetIO(decoder.get(), io);

  avifResult result = avifDecoderParse(decoder.get());
  if (result != AVIF_RESULT_OK) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Failed to parse AVIF stream: ", avifResultToString(result)));
  }
  if (decoder->imageCount != 1) {
    return absl::InvalidArgumentError(
        "AVIF contains more than one image (not supported)");
  }

  // Only read the first image even if there are multiple.
  result = avifDecoderNextImage(decoder.get());
  if (result != AVIF_RESULT_OK) {
    return absl::DataLossError(tensorstore::StrCat(
        "Failed to decode AVIF image: ", avifResultToString(result)));
  }

  decoder_ = std::move(decoder);
  return absl::OkStatus();
}

ImageInfo AvifReader::GetImageInfo() {
  if (!decoder_) return {};
  return AvifGetImageInfo(decoder_.get());
}

absl::Status AvifReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                    const AvifReaderOptions& options) {
  if (!decoder_) {
    return absl::InternalError("No AVIF file to decode");
  }
  std::unique_ptr<avifDecoder, AvifDeleter> decoder = std::move(decoder_);
  return AvifDefaultDecode(decoder.get(), dest, options);
}

}  // namespace internal_image
}  // namespace tensorstore
