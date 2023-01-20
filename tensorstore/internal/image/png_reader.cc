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

#include "tensorstore/internal/image/png_reader.h"

#include <assert.h>
#include <stddef.h>

#include <csetjmp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// Include libpng last
#include <png.h>

namespace tensorstore {
namespace internal_image {
namespace {

static const char* kRiegeliError = "Riegeli error";

void ReadFunction(png_structp png_ptr, png_bytep data, png_size_t size) {
  if (!static_cast<riegeli::Reader*>(png_get_io_ptr(png_ptr))
           ->Read(size, reinterpret_cast<char*>(data))) {
    png_error(png_ptr, kRiegeliError);
  }
}

void ErrorFunction(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    *static_cast<absl::Status*>(png_get_error_ptr(png_ptr)) =
        absl::InternalError(error_message);
  }
  longjmp(png_jmpbuf(png_ptr), 1);
}

void WarningFunction(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    *static_cast<absl::Status*>(png_get_error_ptr(png_ptr)) =
        absl::InternalError(error_message);
    // png_warning doesn't longjmp.
  }
}

}  // namespace

struct PngReader::Context {
  png_structp png_ptr_ = nullptr;
  png_infop info_ptr_ = nullptr;
  png_infop end_info_ptr_ = nullptr;

  riegeli::Reader* reader_;
  absl::Status last_error_;

  ~Context();
  Context(riegeli::Reader* reader) : reader_(reader) {}

  absl::Status Initialize();
  ImageInfo GetImageInfo();
  absl::Status Decode(tensorstore::span<unsigned char> dest);
};

PngReader::Context::~Context() {
  if (png_ptr_) {
    png_destroy_read_struct(&png_ptr_, &info_ptr_, &end_info_ptr_);
  }
}

absl::Status PngReader::Context::Initialize() {
  // Create the PNG struct and PNG info objects.
  png_ptr_ =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  ABSL_CHECK(png_ptr_ != nullptr);

  png_set_error_fn(png_ptr_, &last_error_, &ErrorFunction, &WarningFunction);
  info_ptr_ = png_create_info_struct(png_ptr_);
  ABSL_CHECK(info_ptr_ != nullptr);

  end_info_ptr_ = png_create_info_struct(png_ptr_);
  ABSL_CHECK(end_info_ptr_ != nullptr);

  png_set_read_fn(png_ptr_, reader_, &ReadFunction);
  // png_set_sig_bytes(png_ptr_, kPngBytesToCheck);

  [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(png_jmpbuf(png_ptr_))) {
      return;
    }

    png_read_info(png_ptr_, info_ptr_);
  }();

  if (!reader_->ok() || !last_error_.ok()) {
    return internal::MaybeConvertStatusTo(
        !reader_->ok() ? reader_->status() : last_error_,
        absl::StatusCode::kInvalidArgument);
  }

  /// png_get... functions don't require longjmp
  uint32_t width = png_get_image_width(png_ptr_, info_ptr_);
  uint32_t height = png_get_image_height(png_ptr_, info_ptr_);
  uint32_t bit_depth = png_get_bit_depth(png_ptr_, info_ptr_);

  if (width == 0 || height == 0) {
    return absl::InvalidArgumentError("Failed to decode PNG: zero sized image");
  }
  assert(width < std::numeric_limits<int32_t>::max());
  assert(height < std::numeric_limits<int32_t>::max());
  // num_components is between [1,4] inclusive.
  if (bit_depth > 16) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Failed to decode PNG: bit_depth (", bit_depth, ")"));
  }
  return absl::OkStatus();
}

ImageInfo PngReader::Context::GetImageInfo() {
  ImageInfo info;
  /// png_get... functions don't require longjmp
  info.width = png_get_image_width(png_ptr_, info_ptr_);
  info.height = png_get_image_height(png_ptr_, info_ptr_);
  info.num_components = png_get_channels(png_ptr_, info_ptr_);
  const int bit_depth = png_get_bit_depth(png_ptr_, info_ptr_);

  if (bit_depth == 1) {
    info.dtype = dtype_v<bool>;
  } else if (bit_depth <= 8) {
    info.dtype = dtype_v<uint8_t>;
  } else if (bit_depth <= 16) {
    info.dtype = dtype_v<uint16_t>;
  }
  return info;
}

absl::Status PngReader::Context::Decode(tensorstore::span<unsigned char> dest) {
  auto info = GetImageInfo();
  if (auto required = ImageRequiredBytes(info); required > dest.size()) {
    return absl::InternalError(
        absl::StrFormat("Cannot read PNG; required buffer size %d, got %d",
                        required, dest.size()));
  }

  ImageView dest_view(info, dest);
  std::vector<uint8_t*> row_ptrs;
  absl::Status status;

  bool ok = [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(png_jmpbuf(png_ptr_))) {
      return false;
    }

    // http://www.libpng.org/pub/png/libpng-1.2.5-manual.html
    if (png_get_bit_depth(png_ptr_, info_ptr_) > 8) {
      png_set_expand_16(png_ptr_);
      if constexpr (tensorstore::endian::native ==
                    tensorstore::endian::little) {
        // png is big endian.
        png_set_swap(png_ptr_);
      }
    } else {
      png_set_packing(png_ptr_);
    }

    // Update info after we have set decoding options.
    png_read_update_info(png_ptr_, info_ptr_);

    int64_t height = png_get_image_height(png_ptr_, info_ptr_);
    assert(png_get_rowbytes(png_ptr_, info_ptr_) ==
           dest_view.row_stride_bytes());

    if (png_get_interlace_type(png_ptr_, info_ptr_) != PNG_INTERLACE_NONE) {
      std::vector<uint8_t*> row_ptrs;
      // Interleaved images require that allocation of an image-sized buffer.
      row_ptrs.resize(height);
      for (int y = 0; y < height; ++y) {
        row_ptrs[y] = dest_view.data_row(y).data();
      }
      png_read_image(png_ptr_, &row_ptrs[0]);
    } else {
      // Decode and stream row by row.
      for (int y = 0; y < height; ++y) {
        png_read_row(png_ptr_, dest_view.data_row(y).data(), nullptr);
      }
    }
    return true;
  }();

  if (!ok || !reader_->ok() || !last_error_.ok()) {
    absl::Status status = internal::MaybeConvertStatusTo(
        !reader_->ok() ? reader_->status() : last_error_,
        absl::StatusCode::kDataLoss);
    if (status.ok()) {
      return absl::DataLossError("Failed to decode PNG");
    }
    return MaybeAnnotateStatus(status, "Failed to decode PNG");
  }
  return absl::OkStatus();
}

PngReader::PngReader() = default;
PngReader::~PngReader() = default;
PngReader::PngReader(PngReader&& src) = default;
PngReader& PngReader::operator=(PngReader&& src) = default;

absl::Status PngReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);

  /// Check the signature.  TODO: Move to a static method somewhere.
  constexpr const unsigned char kSignature[] = {0x89, 0x50, 0x4E, 0x47,
                                                0x0D, 0x0A, 0x1A, 0x0A};
  if (!reader->Pull(sizeof(kSignature)) ||
      memcmp(kSignature, reader->cursor(), sizeof(kSignature)) != 0) {
    return absl::InvalidArgumentError(
        "Failed to decode PNG: missing PNG signature");
  }

  reader_ = reader;

  auto context = std::make_unique<PngReader::Context>(reader_);
  TENSORSTORE_RETURN_IF_ERROR(context->Initialize());
  context_ = std::move(context);
  return absl::OkStatus();
}

ImageInfo PngReader::GetImageInfo() {
  if (!context_) return {};
  return context_->GetImageInfo();
}

absl::Status PngReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                   const PngReaderOptions& options) {
  if (!context_) {
    return absl::InternalError("No PNG file to decode");
  }
  auto context = std::move(context_);
  return context->Decode(dest);
}

}  // namespace internal_image
}  // namespace tensorstore
