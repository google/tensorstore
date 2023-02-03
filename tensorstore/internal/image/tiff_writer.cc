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

#include "tensorstore/internal/image/tiff_writer.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>

#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
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

struct TiffWriter::Context : public LibTiffErrorBase {
  riegeli::Writer* writer_ = nullptr;
  TIFF* tiff_ = nullptr;
  int image_number = -1;
  TiffWriterOptions options_;

  Context(riegeli::Writer* writer, const TiffWriterOptions& options);
  ~Context();
  absl::Status ExtractErrors();

  absl::Status Open();
  absl::Status WriteImage(const ImageInfo& info,
                          tensorstore::span<const unsigned char> source);
  absl::Status Close();
};

namespace {

tmsize_t NoReadProc(thandle_t data, void* buf, tmsize_t len) {
  errno = EBADF;
  return -1;
}

tmsize_t WriteProc(thandle_t data, void* buf, tmsize_t len) {
  assert(data != nullptr);

  auto* writer = static_cast<TiffWriter::Context*>(data)->writer_;
  if (!writer->Write(std::string_view(static_cast<char*>(buf), len))) {
    errno = EBADF;
    return -1;
  }
  return len;
}

toff_t SeekProc(thandle_t data, toff_t pos, int whence) {
  assert(data != nullptr);
  auto* writer = static_cast<TiffWriter::Context*>(data)->writer_;
  assert(writer != nullptr);
  auto writer_size = writer->Size();
  uint64_t target_pos = 0;

  switch (whence) {
    case SEEK_SET:
      // ABSL_LOG(INFO) << "tiff seek "<< pos;
      target_pos = pos;
      break;
    case SEEK_CUR:
      // ABSL_LOG(INFO) <<"tiff skip "<< writer->pos()<< " "<< pos;
      target_pos = writer->pos() + pos;
      break;
    case SEEK_END:
      assert(pos <= 0);
      // ABSL_LOG(INFO) <<"tiff seek_end "<< writer->pos()<< " "<< pos;
      if (writer_size) {
        target_pos = *writer_size - static_cast<uint64_t>(-pos);
      } else {
        // Error getting size.
        return -1;
      }
      break;
    default:
      return -1;
  }

  // libtiff assumes that seek works like a file; so if the target_pos is beyond
  // EOF, the writer needs to be extended and the simplest way is WriteZeros.
  if (target_pos > writer_size.value_or(0)) {
    uint64_t zeros = target_pos - writer_size.value_or(0);
    writer->Seek(writer_size.value_or(0));
    writer->WriteZeros(zeros);
  } else {
    writer->Seek(target_pos);
  }
  return writer->ok() ? static_cast<toff_t>(writer->pos()) : -1;
}

int CloseProc(thandle_t data) {
  assert(data != nullptr);
  return 0;
}

toff_t SizeProc(thandle_t data) {
  assert(data != nullptr);
  auto* writer = static_cast<TiffWriter::Context*>(data)->writer_;
  return writer->Size().value_or(-1);
}

}  // namespace

TiffWriter::Context::Context(riegeli::Writer* writer,
                             const TiffWriterOptions& options)
    : writer_(writer), options_(options) {}

TiffWriter::Context::~Context() {
  if (tiff_ != nullptr) {
    TIFFFlush(tiff_);
    TIFFClose(tiff_);
  }
}

absl::Status TiffWriter::Context::Close() {
  if (tiff_ != nullptr) {
    TIFFFlush(tiff_);
    TIFFClose(tiff_);
    tiff_ = nullptr;
  }
  if (!writer_->Close()) {
    return writer_->status();
  }
  return ExtractErrors();
}

absl::Status TiffWriter::Context::ExtractErrors() {
  return std::exchange(error_, absl::OkStatus());
}

absl::Status TiffWriter::Context::Open() {
  tiff_ = TIFFClientOpen("tensorstore_tiff_writer", "w",
                         static_cast<thandle_t>(this), &NoReadProc, &WriteProc,
                         &SeekProc, &CloseProc, &SizeProc, nullptr, nullptr);
  if (!tiff_) {
    error_.Update(absl::InvalidArgumentError("Failed to open directory"));
    return ExtractErrors();
  }
  return absl::OkStatus();
}

absl::Status TiffWriter::Context::WriteImage(
    const ImageInfo& info, tensorstore::span<const unsigned char> source) {
  image_number++;
  if (image_number > 0) {
    return absl::UnknownError(
        "Failed to write TIFF file; multi-page write support incomplete");
  }

  TIFFSetField(tiff_, TIFFTAG_IMAGEWIDTH, info.width);
  TIFFSetField(tiff_, TIFFTAG_IMAGELENGTH, info.height);
  TIFFSetField(tiff_, TIFFTAG_BITSPERSAMPLE, info.dtype.size() * 8);
  TIFFSetField(tiff_, TIFFTAG_SAMPLESPERPIXEL, info.num_components);

  if (info.num_components == 3 || info.num_components == 4) {
    TIFFSetField(tiff_, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  } else {
    // Grayscale, black is 0
    TIFFSetField(tiff_, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  }

  // TODO: extra samples?
  // TODO: Compression config.
  // TODO: Orientation
  TIFFSetField(tiff_, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tiff_, TIFFTAG_ROWSPERSTRIP, 1);
  TIFFSetField(tiff_, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff_, TIFFTAG_SOFTWARE, "tensorstore");

  // This assumes that no conversion is required.
  ImageView view = MakeWriteImageView(info, source);
  for (int row = 0; row < info.height; ++row) {
    if (!TIFFWriteScanline(tiff_, view.data_row(row).data(), row, 0)) {
      error_.Update(absl::InvalidArgumentError("Failed to write scanline"));
      return ExtractErrors();
    }
  }

  if (TIFFWriteDirectory(tiff_) == 0) {
    error_.Update(absl::InvalidArgumentError("Failed to write directory"));
  }
  return ExtractErrors();
}

TiffWriter::TiffWriter() = default;
TiffWriter::~TiffWriter() = default;
TiffWriter::TiffWriter(TiffWriter&& src) = default;
TiffWriter& TiffWriter::operator=(TiffWriter&& src) = default;

absl::Status TiffWriter::InitializeImpl(riegeli::Writer* writer,
                                        const TiffWriterOptions& options) {
  ABSL_CHECK(writer != nullptr);
  if (context_) {
    return absl::InternalError("Initialize() already called");
  }
  if (!writer->SupportsRandomAccess()) {
    return absl::InternalError("TiffWriter requires seekable riegeli::Writer");
  }
  auto context = std::make_unique<TiffWriter::Context>(writer, options);
  TENSORSTORE_RETURN_IF_ERROR(context->Open());
  context_ = std::move(context);
  return absl::OkStatus();
}

absl::Status TiffWriter::Encode(const ImageInfo& info,
                                tensorstore::span<const unsigned char> source) {
  if (!context_) {
    return absl::InternalError("TIFF writer not initialized");
  }
  ABSL_CHECK_EQ(source.size(), ImageRequiredBytes(info));
  return context_->WriteImage(info, source);
}

absl::Status TiffWriter::Done() {
  if (!context_) {
    return absl::InternalError("TIFF writer not initialized");
  }
  std::unique_ptr<Context> context = std::move(context_);
  return context->Close();
}

}  // namespace internal_image
}  // namespace tensorstore
