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

#include "tensorstore/internal/image/png_writer.h"

#include <csetjmp>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

// Include libpng last
#include <png.h>

namespace tensorstore {
namespace internal_image {
namespace {

static const char* kRiegeliError = "Riegeli error";

void WriteFunction(png_structp png_ptr, png_bytep data, png_size_t size) {
  if (!static_cast<riegeli::Writer*>(png_get_io_ptr(png_ptr))
           ->Write(std::string_view(reinterpret_cast<char*>(data), size))) {
    png_error(png_ptr, kRiegeliError);
  }
}

void FlushFunction(png_structp png_ptr) {}

void ErrorFunction(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    *static_cast<absl::Status*>(png_get_error_ptr(png_ptr)) =
        absl::DataLossError(error_message);
  }
  longjmp(png_jmpbuf(png_ptr), 1);
}

void WarningFunction(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    *static_cast<absl::Status*>(png_get_error_ptr(png_ptr)) =
        absl::DataLossError(error_message);
    // png_warning doesn't longjmp.
  }
}

}  // namespace

struct PngWriter::Context {
  png_structp png_ptr_ = nullptr;
  png_infop info_ptr_ = nullptr;
  riegeli::Writer* writer_ = nullptr;
  PngWriterOptions options_;
  absl::Status last_error_;
  bool written_ = false;

  Context(riegeli::Writer* writer) : writer_(writer) {}
  ~Context() {
    if (png_ptr_) {
      png_destroy_write_struct(&png_ptr_, &info_ptr_);
    }
  }

  void Initialize(const PngWriterOptions& options);
  absl::Status Encode(const ImageInfo& info,
                      tensorstore::span<const unsigned char> source);
  absl::Status Finish();
};

void PngWriter::Context::Initialize(const PngWriterOptions& options) {
  options_ = options;
  png_ptr_ =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  ABSL_CHECK(png_ptr_ != nullptr);

  // Redirect error and warning messages to the error state.
  png_set_error_fn(png_ptr_, &last_error_, &ErrorFunction, &WarningFunction);

  info_ptr_ = png_create_info_struct(png_ptr_);
  ABSL_CHECK(info_ptr_ != nullptr);

  png_set_write_fn(png_ptr_, writer_, &WriteFunction, &FlushFunction);
}

absl::Status PngWriter::Context::Encode(
    const ImageInfo& info, tensorstore::span<const unsigned char> source) {
  if (written_) {
    return absl::InternalError("Cannot write multiple images to PNG.");
  }
  std::vector<uint8_t*> row_ptrs;

  if (info.dtype != dtype_v<uint8_t>) {
    return absl::DataLossError("PNG encoding failed");
  }
  if (info.num_components == 0 || info.num_components > 4) {
    return absl::DataLossError("PNG encoding failed");
  }

  int png_color_type = PNG_COLOR_TYPE_GRAY;
  if (info.num_components == 2) {
    png_color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
  } else if (info.num_components == 3) {
    png_color_type = PNG_COLOR_TYPE_RGB;
  } else if (info.num_components == 4) {
    png_color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  }

  ImageView view = MakeWriteImageView(info, source);
  [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(png_jmpbuf(png_ptr_))) {
      return;
    }
    written_ = true;
    png_set_IHDR(png_ptr_, info_ptr_, info.width, info.height,
                 8 * info.dtype.size(), png_color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr_, info_ptr_);

    // These options must be set after png_write_info(), otherwise they are
    // effectively ignored due to png_ptr_->bit_depth not yet being set.
    if (info.dtype == dtype_v<uint8_t>) {
      png_set_packing(png_ptr_);
    }
    if (info.dtype == dtype_v<uint16_t>) {
      png_set_swap(png_ptr_);
    }
    if (options_.compression_level >= 0 && options_.compression_level <= 9) {
      png_set_compression_level(png_ptr_, options_.compression_level);
      if (options_.compression_level < 3) {
        png_set_filter(png_ptr_, PNG_FILTER_TYPE_BASE, PNG_NO_FILTERS);
      }
    }

    // Encode directly from the input image.
    row_ptrs.resize(info.height);
    for (int y = 0; y < info.height; ++y) {
      row_ptrs[y] = view.data_row(y).data();
    }
    png_write_rows(png_ptr_, &row_ptrs[0], row_ptrs.size());
    png_write_end(png_ptr_, info_ptr_);
  }();

  if (!writer_->ok() || !last_error_.ok()) {
    return internal::MaybeConvertStatusTo(
        !writer_->ok() ? writer_->status() : last_error_,
        absl::StatusCode::kDataLoss);
  }
  return absl::OkStatus();
}

absl::Status PngWriter::Context::Finish() {
  if (!writer_->Close()) {
    return writer_->status();
  }
  return absl::OkStatus();
}

PngWriter::PngWriter() = default;
PngWriter::~PngWriter() = default;
PngWriter::PngWriter(PngWriter&& src) = default;
PngWriter& PngWriter::operator=(PngWriter&& src) = default;

absl::Status PngWriter::InitializeImpl(riegeli::Writer* writer,
                                       const PngWriterOptions& options) {
  ABSL_CHECK(writer != nullptr);
  if (context_) {
    return absl::InternalError("Initialize() already called");
  }
  writer_ = std::move(writer);
  context_ = std::make_unique<PngWriter::Context>(writer_);
  context_->Initialize(options);
  return absl::OkStatus();
}

absl::Status PngWriter::Encode(const ImageInfo& info,
                               tensorstore::span<const unsigned char> source) {
  if (!context_) {
    return absl::InternalError("AVIF reader not initialized");
  }
  ABSL_CHECK_EQ(source.size(), ImageRequiredBytes(info));
  return context_->Encode(info, source);
}

absl::Status PngWriter::Done() {
  if (!context_) {
    return absl::InternalError("No data written");
  }
  auto context = std::move(context_);
  return context->Finish();
}

}  // namespace internal_image
}  // namespace tensorstore
