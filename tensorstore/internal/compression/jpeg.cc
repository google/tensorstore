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

#include "tensorstore/internal/compression/jpeg.h"

#include <csetjmp>

#include <jpeglib.h>

namespace tensorstore {
namespace jpeg {

namespace {

/// Wrapper for a `::jpeg_compress_struct` or `::jpeg_decompress_struct` that
/// provides error handling.
template <typename T>
struct JpegStateWrapper {
  JpegStateWrapper() {
    ::jpeg_std_error(&jerr);
    // Only `emit_message` and `error_exit` are called by the rest of the
    // library.
    jerr.emit_message = &EmitMessage;
    jerr.error_exit = &ErrorExit;

    // Minimal initialization required to safely call `jpeg_destroy`.
    cinfo.mem = nullptr;

    // Set up error handler.
    cinfo.client_data = this;
    cinfo.err = &jerr;
  }
  ~JpegStateWrapper() {
    // Because `jpeg_common_struct` and `jpeg_{de,}compress_struct` are standard
    // layout with a common initial sequence, we can safely `reinterpret_cast`
    // between them.
    ::jpeg_destroy(reinterpret_cast<::jpeg_common_struct*>(&cinfo));
  }

  [[noreturn]] static void ErrorExit(::jpeg_common_struct* cinfo) {
    OutputMessage(cinfo);
    auto* self = static_cast<JpegStateWrapper*>(cinfo->client_data);
    std::longjmp(self->jmpbuf, 1);
  }

  static void OutputMessage(::jpeg_common_struct* cinfo) {
    auto* self = static_cast<JpegStateWrapper*>(cinfo->client_data);
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    self->status = absl::InvalidArgumentError(buffer);
  }

  static void EmitMessage(::jpeg_common_struct* cinfo, int msg_level) {
    if (msg_level > 0) {
      // Ignore trace messages.
      return;
    }
    cinfo->err->num_warnings++;
    // Warnings (which can indicate corrupt data) are treated as errors.
    ErrorExit(cinfo);
  }

  T cinfo;
  std::jmp_buf jmpbuf;
  Status status;
  ::jpeg_error_mgr jerr;
};

struct MemoryDestManager {
  struct ::jpeg_destination_mgr mgr;
  char buffer[32 * 1024];
  std::string* out;

  explicit MemoryDestManager(std::string* out) : out(out) {
    mgr.init_destination = &InitDestination;
    mgr.empty_output_buffer = &EmptyOutputBuffer;
    mgr.term_destination = &TermDestination;
  }

  void Initialize(::jpeg_compress_struct* cinfo) { cinfo->dest = &mgr; }

 private:
  static void InitDestination(::jpeg_compress_struct* cinfo) {
    // Because `MemoryDestLayout` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<MemoryDestManager*>(cinfo->dest);
    self->mgr.next_output_byte = reinterpret_cast<JOCTET*>(self->buffer);
    self->mgr.free_in_buffer = std::size(self->buffer);
  }

  // Note: `boolean` is an unfortunately-named platform-dependent type alias
  // defined by `jpeglib.h`; typically it is `int` but may be a different type
  // on some platforms.
  static boolean EmptyOutputBuffer(::jpeg_compress_struct* cinfo) {
    // Because `MemoryDestLayout` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<MemoryDestManager*>(cinfo->dest);
    self->out->append(self->buffer, std::size(self->buffer));
    InitDestination(cinfo);
    return static_cast<boolean>(true);
  }

  static void TermDestination(::jpeg_compress_struct* cinfo) {
    // Because `MemoryDestLayout` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<MemoryDestManager*>(cinfo->dest);
    self->out->append(self->buffer,
                      std::size(self->buffer) - self->mgr.free_in_buffer);
  }
};

}  // namespace

Status Decode(absl::string_view source,
              FunctionView<Result<unsigned char*>(size_t width, size_t height,
                                                  size_t num_components)>
                  validate_size) {
  // libjpeg_turbo uses `unsigned long` to specify length of memory buffer,
  // which on Windows is 32-bit.
  if (source.size() > std::numeric_limits<unsigned long>::max()) {  // NOLINT
    return absl::InvalidArgumentError("JPEG data exceeds maximum length");
  }
  JpegStateWrapper<::jpeg_decompress_struct> state;
  // Wrap actual logic in lambda to avoid having any non-trivial local variables
  // in function using setjmp.
  [&] {
    if (setjmp(state.jmpbuf)) return;
    ::jpeg_create_decompress(&state.cinfo);
    ::jpeg_mem_src(&state.cinfo,
                   reinterpret_cast<const unsigned char*>(source.data()),
                   source.size());
    {
      [[maybe_unused]] int code =
          ::jpeg_read_header(&state.cinfo, /*require_image=*/1);
      // With `require_image=1` and a memory source, `jpeg_read_header` cannot
      // return except in the case of success.
      assert(code == JPEG_HEADER_OK);
    }
    if (state.cinfo.num_components != 1 && state.cinfo.num_components != 3) {
      state.status = absl::InvalidArgumentError(
          StrCat("Expected 1 or 3 components, but received: ",
                 state.cinfo.num_components));
      return;
    }
    // Cannot return false since memory data source does not suspend.
    ::jpeg_start_decompress(&state.cinfo);
    unsigned char* buf;
    if (auto result =
            validate_size(state.cinfo.image_width, state.cinfo.image_height,
                          state.cinfo.num_components)) {
      buf = *result;
    } else {
      state.status = std::move(result).status();
      return;
    }
    while (state.cinfo.output_scanline < state.cinfo.output_height) {
      auto* output_line = reinterpret_cast<JSAMPLE*>(buf);
      if (::jpeg_read_scanlines(&state.cinfo, &output_line, 1) != 1) {
        state.status = absl::InvalidArgumentError(
            StrCat("JPEG data ended after ", state.cinfo.output_scanline, "/",
                   state.cinfo.output_height, " scan lines"));
        return;
      }
      buf += state.cinfo.output_width * state.cinfo.num_components;
    }
  }();

  return MaybeAnnotateStatus(state.status, "Error decoding JPEG");
}

Status Encode(const unsigned char* source, size_t width, size_t height,
              size_t num_components, const EncodeOptions& options,
              std::string* out) {
  assert(options.quality >= 0 && options.quality <= 100);
  if (width > std::numeric_limits<JDIMENSION>::max() ||
      height > std::numeric_limits<JDIMENSION>::max()) {
    return absl::InvalidArgumentError(StrCat(
        "Image dimensions of (", width, ", ", height, ") exceed maximum size"));
  }
  if (num_components != 1 && num_components != 3) {
    return absl::InvalidArgumentError(
        StrCat("Expected 1 or 3 components, but received: ", num_components));
  }
  JpegStateWrapper<::jpeg_compress_struct> state;
  MemoryDestManager dest(out);
  [&] {
    if (setjmp(state.jmpbuf)) return;
    ::jpeg_create_compress(&state.cinfo);
    dest.Initialize(&state.cinfo);
    state.cinfo.image_width = width;
    state.cinfo.image_height = height;
    state.cinfo.input_components = num_components;
    state.cinfo.in_color_space =
        (num_components == 1) ? JCS_GRAYSCALE : JCS_RGB;
    ::jpeg_set_defaults(&state.cinfo);
    ::jpeg_set_quality(&state.cinfo, options.quality, /*force_baseline=*/1);
    ::jpeg_start_compress(&state.cinfo, /*write_all_tables=*/1);

    const unsigned char* buf = source;

    while (state.cinfo.next_scanline < state.cinfo.image_height) {
      auto* input_line =
          const_cast<JSAMPLE*>(reinterpret_cast<const JSAMPLE*>(buf));
      // Always returns 1 since memory destination does not suspend.
      ::jpeg_write_scanlines(&state.cinfo, &input_line, 1);
      buf += width * num_components;
    }
    ::jpeg_finish_compress(&state.cinfo);
  }();
  TENSORSTORE_CHECK_OK(state.status);
  return absl::OkStatus();
}

}  // namespace jpeg
}  // namespace tensorstore
