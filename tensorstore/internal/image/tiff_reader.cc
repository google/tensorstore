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

#include "tensorstore/internal/image/tiff_reader.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

// Include libtiff last.
// See: http://www.libtiff.org/man/index.html
#include "tensorstore/internal/image/tiff_common.h"
#include <tiff.h>
#include <tiffio.h>

namespace tensorstore {
namespace internal_image {

struct TiffReader::Context : public LibTiffErrorBase {
  riegeli::Reader* reader_ = nullptr;  // unowned
  TIFF* tiff_ = nullptr;

  Context(riegeli::Reader* reader);
  ~Context();
  absl::Status ExtractErrors(absl::Status in);

  absl::Status Open();
  absl::Status DefaultDecode(tensorstore::span<unsigned char> data);
};

namespace {

/// libtiff has static globals for the warning handler; best to lock and
/// only allow a single thread.
tmsize_t ReadProc(thandle_t data, void* buf, tmsize_t len) {
  assert(data != nullptr);
  auto* reader = static_cast<TiffReader::Context*>(data)->reader_;
  assert(reader != nullptr);
  size_t read;
  // ABSL_LOG(INFO) << "tiff read " << reader->pos() << " " << len;
  if (!reader->Read(len, static_cast<char*>(buf), &read) && !reader->ok()) {
    errno = EBADF;
    return -1;
  }
  return read;
}

tmsize_t NoWriteProc(thandle_t data, void* buf, tmsize_t len) {
  errno = EBADF;
  return -1;
}

toff_t SeekProc(thandle_t data, toff_t pos, int whence) {
  assert(data != nullptr);
  auto* reader = static_cast<TiffReader::Context*>(data)->reader_;
  assert(reader != nullptr);

  switch (whence) {
    case SEEK_SET:
      // ABSL_LOG(INFO) << "tiff seek " << pos;
      reader->Seek(pos);
      break;
    case SEEK_CUR:
      // ABSL_LOG(INFO) << "tiff skip "<< reader->pos()<< " "<< pos;
      reader->Skip(pos);
      break;
    case SEEK_END:
      assert(pos <= 0);
      // ABSL_LOG(INFO) << "tiff seek_end "<< pos;
      if (auto size = reader->Size(); size) {
        reader->Seek(*size - static_cast<uint64_t>(-pos));
      } else {
        // Error getting size.
        return -1;
      }
      break;
    default:
      return -1;
  }
  return reader->ok() ? static_cast<toff_t>(reader->pos()) : -1;
}

int CloseProc(thandle_t data) {
  assert(data != nullptr);
  return 0;
}

toff_t SizeProc(thandle_t data) {
  assert(data != nullptr);
  auto* reader = static_cast<TiffReader::Context*>(data)->reader_;
  assert(reader != nullptr);
  auto size = reader->Size();
  return size ? static_cast<toff_t>(*size) : -1;
}

/// Mapping function to convert bits-per-sample to byte arrays.
template <size_t NBITS>
const unsigned char* TranslateBits(ptrdiff_t& stride) {
  static constexpr int kMask = (1 << NBITS) - 1;
  static unsigned char mapping[256 * (8 / NBITS)];
  [[maybe_unused]] static bool done = [&] {
    unsigned char* dest = mapping;
    for (int j = 0; j < 256; j++)
      for (int i = NBITS - 1; i >= 0; i--) {
        *(dest++) = static_cast<unsigned char>((j >> (NBITS * i)) & kMask);
      }
    return true;
  }();

  stride = 8 / NBITS;
  return mapping;
}

struct TiffImageInfo : public ImageInfo {
  uint16_t extra_samples_ = 0;
  uint16_t extra_types_ = 0;
  uint16_t bits_per_sample_ = 0;
};

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

absl::Status GetTIFFImageInfo(TIFF* tiff, TiffImageInfo& info) {
  uint32_t width, height, depth;
  if (!TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) ||
      !TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height)) {
    return absl::InvalidArgumentError("TIFF read failed: invalid image");
  }

  if (TIFFGetField(tiff, TIFFTAG_IMAGEDEPTH, &depth) && depth > 1) {
    return absl::InvalidArgumentError("TIFF read failed: depth not supported");
  }

  // These call TIFFSetField to update the in-memory structure so that
  // subsequent calls get appropriate defaults.
  if (!TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &info.bits_per_sample_)) {
    info.bits_per_sample_ = 1;
    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, info.bits_per_sample_);
  }
  uint32_t samples_per_pixel = 0;
  if (!TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel)) {
    samples_per_pixel = 1;
    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
  }
  uint16_t* sample_info;
  if (TIFFGetField(tiff, TIFFTAG_EXTRASAMPLES, &info.extra_samples_,
                   &sample_info) != 1) {
    info.extra_samples_ = 0;
  } else {
    if (info.extra_samples_ == 1) {
      info.extra_types_ = *sample_info;
      if (info.extra_types_ == EXTRASAMPLE_UNSPECIFIED) {
        if (samples_per_pixel > 3) {
          info.extra_types_ = EXTRASAMPLE_ASSOCALPHA;
        }
      }
    }
  }

  uint16_t sample_format = 0;
  TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);

  info.width = width;
  info.height = height;
  info.num_components = samples_per_pixel + info.extra_samples_;
  TENSORSTORE_ASSIGN_OR_RETURN(info.dtype,
                               SetDType(sample_format, info.bits_per_sample_));
  return absl::OkStatus();
}

absl::Status ReadStripImpl(TIFF* tiff, TiffImageInfo& info,
                           tensorstore::span<unsigned char> data) {
  ImageView dest_view(info, data);

  // Translate 1,2,4 bits per sample to 8-bpp images.
  const unsigned char* mapping = nullptr;
  ptrdiff_t trstride = 1;
  if (info.bits_per_sample_ == 1) {
    mapping = TranslateBits<1>(trstride);
  } else if (info.bits_per_sample_ == 2) {
    mapping = TranslateBits<2>(trstride);
  } else if (info.bits_per_sample_ == 4) {
    mapping = TranslateBits<4>(trstride);
  }

  const int strip_bytes = TIFFStripSize(tiff);
  uint32_t rows_per_strip = 1;
  TIFFGetFieldDefaulted(tiff, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);

  if (!mapping &&
      strip_bytes == rows_per_strip * dest_view.row_stride_bytes()) {
    /// No extra data && no mapping means that the TIFF can be read directly
    /// into the output buffer.
    for (size_t y = 0; y < info.height; y += rows_per_strip) {
      // Read the strip.
      if (TIFFReadEncodedStrip(tiff, TIFFComputeStrip(tiff, y, 0),
                               dest_view.data_row(y).data(),
                               strip_bytes) == -1) {
        return absl::DataLossError("TIFF read strip failed");
      }
    }
    return absl::OkStatus();
  }

  std::unique_ptr<unsigned char[]> buffer(new unsigned char[strip_bytes]);
  uint32_t line_bytes =
      info.width * (info.bits_per_sample_ / 8) * info.num_components;

  for (size_t y = 0; y < info.height; y += rows_per_strip) {
    // Read the strip.
    if (TIFFReadEncodedStrip(tiff, TIFFComputeStrip(tiff, y, 0), buffer.get(),
                             strip_bytes) == -1) {
      return absl::DataLossError("TIFF read strip failed");
    }

    unsigned char* source_row = buffer.get();
    for (size_t r = 0; r < rows_per_strip; r++) {
      if (y + r >= info.height) break;
      auto target_row = dest_view.data_row(y + r);

      if (info.bits_per_sample_ >= 8) {
        assert(line_bytes == dest_view.row_stride_bytes());
        memcpy(target_row.data(), source_row, line_bytes);
        source_row += line_bytes;
      } else {
        while (!target_row.empty()) {
          size_t n = std::min(trstride, target_row.size());
          memcpy(target_row.data(), mapping + (*source_row * trstride), n);
          target_row = target_row.subspan(n);
          source_row++;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ReadTiledImpl(TIFF* tiff, TiffImageInfo& info,
                           tensorstore::span<unsigned char> data) {
  ImageView dest_view(info, data);

  // Translate 1,2,4 bits per sample to 8-bpp images.
  const unsigned char* mapping = nullptr;
  ptrdiff_t trstride = 1;
  if (info.bits_per_sample_ == 1) {
    mapping = TranslateBits<1>(trstride);
  } else if (info.bits_per_sample_ == 2) {
    mapping = TranslateBits<2>(trstride);
  } else if (info.bits_per_sample_ == 4) {
    mapping = TranslateBits<4>(trstride);
  }

  uint32_t tile_width, tile_height;
  TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &tile_width);
  TIFFGetField(tiff, TIFFTAG_TILELENGTH, &tile_height);

  const int tile_bytes = TIFFTileSize(tiff);
  std::unique_ptr<unsigned char[]> tile_buffer(new unsigned char[tile_bytes]);
  ImageView tile_view(ImageInfo{/*.height=*/static_cast<int32_t>(tile_height),
                                /*.width=*/static_cast<int32_t>(tile_width),
                                /*.num_components=*/info.num_components},
                      {tile_buffer.get(), tile_bytes});

  for (size_t y = 0; y < info.height; y += tile_height) {
    for (size_t x = 0; x < info.width; x += tile_width) {
      if (TIFFReadTile(tiff, tile_buffer.get(), x, y, 0, 0) == -1) {
        return absl::DataLossError("TIFF read tile failed");
      }
      for (size_t y1 = 0; y1 < tile_height; y1++) {
        if ((y + y1) >= info.height) break;
        auto target_row = dest_view.data_row(y + y1, x);
        auto tile_row = tile_view.data_row(y1);
        if (info.bits_per_sample_ >= 8) {
          memcpy(target_row.data(), tile_row.data(),
                 std::min(target_row.size(), tile_row.size()));
        } else {
          unsigned char* packed = tile_row.data();
          while (!target_row.empty()) {
            size_t n = std::min(trstride, target_row.size());
            memcpy(target_row.data(), mapping + (*packed * trstride), n);
            target_row = target_row.subspan(n);
            packed++;
          }
          assert(packed <= tile_row.data() + tile_row.size());
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

TiffReader::Context::Context(riegeli::Reader* reader) : reader_(reader) {}

TiffReader::Context::~Context() {
  if (tiff_ != nullptr) {
    TIFFClose(tiff_);
  }
}

absl::Status TiffReader::Context::ExtractErrors(absl::Status in) {
  if (error_.ok()) return in;
  return std::exchange(error_, absl::OkStatus());
}

absl::Status TiffReader::Context::Open() {
  tiff_ = TIFFClientOpen("tensorstore_tiff_reader", "rmh",
                         static_cast<thandle_t>(this), &ReadProc, &NoWriteProc,
                         &SeekProc, &CloseProc, &SizeProc,
                         /*mmap*/ nullptr, /*munmap*/ nullptr);
  if (!tiff_) {
    return absl::InvalidArgumentError("Not a TIFF file");
  }

  // NOTE: If https://gitlab.com/libtiff/libtiff/-/merge_requests/390 is
  // submitted, then we can setup per-file error handling and skip the initial
  // header read.
  if (!TIFFReadDirectory(tiff_)) {
    error_.Update(absl::InvalidArgumentError("Failed to read TIFF directory"));
  }

#if 0
  // TODO: If this tiff has a thumbnail, then we probably want to skip it
  // by default.
  uint32_t file_type;
  if (TIFFGetField(tiff_, TIFFTAG_SUBFILETYPE, &file_type) &&
      file_type == FILETYPE_REDUCEDIMAGE) {
    // It's a thumbnail
  }
#endif

  return ExtractErrors(absl::OkStatus());
}

absl::Status TiffReader::Context::DefaultDecode(
    tensorstore::span<unsigned char> data) {
  TiffImageInfo info;
  TENSORSTORE_RETURN_IF_ERROR(GetTIFFImageInfo(tiff_, info));
  ABSL_CHECK_EQ(data.size(), ImageRequiredBytes(info));

  // Additional fields checks (beyond the info)
  uint32_t compress_tag = 0;
  uint32_t photometric = 0;
  TIFFGetFieldDefaulted(tiff_, TIFFTAG_PHOTOMETRIC, &photometric);
  TIFFGetFieldDefaulted(tiff_, TIFFTAG_COMPRESSION, &compress_tag);

  // Validate compression; this also initializes libtiff compression settings.
  if (compress_tag != COMPRESSION_NONE &&
      !TIFFIsCODECConfigured(compress_tag)) {
    return absl::InternalError(
        "Cannot read TIFF; compression format not supported");
  }

  // Validate photometric.
  switch (photometric) {
    case PHOTOMETRIC_YCBCR:
      if (compress_tag == COMPRESSION_JPEG) {
        // Let libjpeg handle the ycbcr -> rgb conversion.
        TIFFSetField(tiff_, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
        break;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case PHOTOMETRIC_SEPARATED:
    case PHOTOMETRIC_CIELAB:
    case PHOTOMETRIC_ICCLAB:
    case PHOTOMETRIC_ITULAB:
    case PHOTOMETRIC_CFA:
    case PHOTOMETRIC_LOGL:
    case PHOTOMETRIC_LOGLUV:
    case PHOTOMETRIC_PALETTE:
      return absl::InvalidArgumentError(
          "Cannot read TIFF: photometric not supported");
    default:
      break;
  }

  if (info.num_components > 1) {
    uint32_t planarconfig = PLANARCONFIG_CONTIG;
    TIFFGetFieldDefaulted(tiff_, TIFFTAG_PLANARCONFIG, &planarconfig);
    if (planarconfig == PLANARCONFIG_SEPARATE) {
      return absl::InvalidArgumentError(
          "TIFF read failed: separate planes not supported");
    }
  }

  absl::Status status;
  if (TIFFIsTiled(tiff_)) {
    status = ReadTiledImpl(tiff_, info, data);
  } else {
    status = ReadStripImpl(tiff_, info, data);
  }

  return ExtractErrors(status);
}

TiffReader::TiffReader() = default;
TiffReader::~TiffReader() = default;
TiffReader::TiffReader(TiffReader&& src) = default;
TiffReader& TiffReader::operator=(TiffReader&& src) = default;

absl::Status TiffReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);
  context_ = nullptr;

  auto context = std::make_unique<TiffReader::Context>(reader);
  TENSORSTORE_RETURN_IF_ERROR(context->Open());
  context_ = std::move(context);
  return absl::OkStatus();
}

// Returns the number of frames that can be decoded.
int TiffReader::GetFrameCount() const {
  if (!context_) return 0;
  auto n = TIFFNumberOfDirectories(context_->tiff_);
  return n ? n : 1;
}

absl::Status TiffReader::SeekFrame(int frame_number) {
  if (!context_) {
    return absl::UnknownError("No TIFF file opened.");
  }
  context_->error_ = absl::OkStatus();
  if (TIFFSetDirectory(context_->tiff_, frame_number) != 1) {
    return context_->ExtractErrors(absl::InvalidArgumentError(
        "TIFF Initialize failed: failed to set directory"));
  }
  return context_->ExtractErrors(absl::OkStatus());
}

ImageInfo TiffReader::GetImageInfo() {
  if (!context_) {
    return {};
  }
  TiffImageInfo info;
  if (auto status = GetTIFFImageInfo(context_->tiff_, info); !status.ok()) {
    return {};
  }
  return info;
}

absl::Status TiffReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                    const TiffReaderOptions& options) {
  if (!context_) {
    return absl::InternalError("No TIFF file to decode");
  }
  return context_->DefaultDecode(dest);
}

}  // namespace internal_image
}  // namespace tensorstore
