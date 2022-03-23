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

#include "tensorstore/internal/compression/png.h"

#include <stddef.h>

#include <csetjmp>
#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_join.h"
#include <png.h>
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace png {
namespace {

static const char* kRiegeliError = "Riegeli error";

void write_row_callback_empty(png_structp png_ptr, png_uint_32 row, int pass) {}

void write_callback(png_structp png_ptr, png_bytep data, png_size_t size) {
  if (!static_cast<riegeli::Writer*>(png_get_io_ptr(png_ptr))
           ->Write(reinterpret_cast<char*>(data), size)) {
    png_error(png_ptr, kRiegeliError);
  }
}

void read_callback(png_structp png_ptr, png_bytep data, png_size_t size) {
  if (!static_cast<riegeli::Reader*>(png_get_io_ptr(png_ptr))
           ->Read(size, reinterpret_cast<char*>(data))) {
    png_error(png_ptr, kRiegeliError);
  }
}

void set_png_error(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    static_cast<std::vector<std::string>*>(png_get_error_ptr(png_ptr))
        ->push_back(error_message);
  }
  longjmp(png_jmpbuf(png_ptr), 1);
}

void set_png_warning(png_structp png_ptr, png_const_charp error_message) {
  if (error_message != kRiegeliError) {
    /// This treats warnings as errors; consider logging them instead.
    static_cast<std::vector<std::string>*>(png_get_error_ptr(png_ptr))
        ->push_back(error_message);
  }
}

}  // namespace

absl::Status Encode(const unsigned char* source, size_t width, size_t height,
                    size_t num_components, const EncodeOptions& options,
                    absl::Cord* output) {
  if (num_components < 1 || num_components > 4) {
    return absl::InvalidArgumentError(
        "PNG encoding requires between 1 and 4 components");
  }

  riegeli::CordWriter writer(output);
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)0, 0, 0);
  if (!png_ptr) {
    return absl::DataLossError("PNG encoding failed");
  }

  std::vector<std::string> png_errors;
  png_set_error_fn(png_ptr, &png_errors, &set_png_error, &set_png_warning);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    return absl::DataLossError("PNG encoding failed");
  }

  int png_color_type = PNG_COLOR_TYPE_GRAY;
  if (num_components == 2) {
    png_color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
  } else if (num_components == 3) {
    png_color_type = PNG_COLOR_TYPE_RGB;
  } else if (num_components == 4) {
    png_color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  }

  const std::unique_ptr<png_byte*[]> rows(new png_byte*[height]);
  for (int i = 0; i < height; ++i) {
    rows[i] = const_cast<png_byte*>(&source[i * width * num_components]);
  }

  [&]() {
    if (setjmp(png_jmpbuf(png_ptr))) {
      /// Actually the return from a longjmp(), and either the reader or
      /// the png_errors will include error content.
      return;
    }

    png_set_write_fn(png_ptr, &writer, write_callback, NULL);
    png_set_write_status_fn(png_ptr, write_row_callback_empty);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);

    if (options.compression_level >= 0 && options.compression_level <= 9) {
      png_set_compression_level(png_ptr, options.compression_level);
    }

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, png_color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    png_set_rows(png_ptr, info_ptr, rows.get());
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
  }();

  png_destroy_write_struct(&png_ptr, &info_ptr);

  if (!png_errors.empty()) {
    return absl::DataLossError(
        StrCat("PNG Encoding failed: ", absl::StrJoin(png_errors, " ")));
  }
  if (!writer.Close()) return writer.status();
  return absl::OkStatus();
}

absl::Status Decode(const absl::Cord& input,
                    absl::FunctionRef<Result<unsigned char*>(
                        size_t width, size_t height, size_t num_components)>
                        validate_size) {
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    return absl::DataLossError("PNG decoding failed");
  }

  std::vector<std::string> png_errors;
  png_set_error_fn(png_ptr, &png_errors, &set_png_error, &set_png_warning);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    return absl::DataLossError("PNG decoding failed");
  }

  png_infop end_info = png_create_info_struct(png_ptr);
  if (!end_info) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return absl::DataLossError("PNG decoding failed");
  }

  riegeli::CordReader reader(input);
  std::unique_ptr<png_byte*[]> rows;

  auto status = [&]() -> absl::Status {
    if (setjmp(png_jmpbuf(png_ptr))) {
      /// Actually the return from a longjmp(), and either the reader or
      /// the png_errors will include error content.
      return absl::OkStatus();
    }

    png_set_read_fn(png_ptr, &reader, read_callback);

    png_read_info(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
      return absl::UnimplementedError(
          "PNG decoding of palettized image failed");
    }
    if (png_get_bit_depth(png_ptr, info_ptr) != 8) {
      return absl::UnimplementedError(
          "PNG decoding failed with non 8-bit depth");
    }

    size_t width = png_get_image_width(png_ptr, info_ptr);
    size_t height = png_get_image_height(png_ptr, info_ptr);
    size_t num_channels = png_get_channels(png_ptr, info_ptr);

    TENSORSTORE_ASSIGN_OR_RETURN(unsigned char* buffer,
                                 validate_size(width, height, num_channels));

    rows.reset(new png_byte*[height]);
    for (int i = 0; i < height; ++i) {
      rows[i] = const_cast<png_byte*>(&buffer[i * width * num_channels]);
    }

    png_set_rows(png_ptr, info_ptr, rows.get());
    png_read_image(png_ptr, rows.get());
    png_read_end(png_ptr, end_info);

    return absl::OkStatus();
  }();

  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

  if (!status.ok()) return status;
  if (!png_errors.empty()) {
    return absl::DataLossError(
        StrCat("PNG decoding failed: ", absl::StrJoin(png_errors, " ")));
  }
  if (!reader.VerifyEndAndClose()) return reader.status();
  return absl::OkStatus();
}

}  // namespace png
}  // namespace tensorstore
