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

#include "absl/strings/cord.h"
#include <jerror.h>
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

struct CordSourceManager {
  struct ::jpeg_source_mgr mgr;
  absl::Cord::CharIterator char_it;
  size_t input_remaining;

  explicit CordSourceManager(const absl::Cord& input)
      : char_it(input.char_begin()), input_remaining(input.size()) {
    mgr.init_source = &InitSource;
    mgr.fill_input_buffer = &FillInputBuffer;
    mgr.skip_input_data = &SkipInputData;
    // Use default `resync_to_restart` implementation provided by libjpeg.
    mgr.resync_to_restart = ::jpeg_resync_to_restart;
    mgr.term_source = &TermSource;
    mgr.next_input_byte = nullptr;
    mgr.bytes_in_buffer = 0;
  }

  void Initialize(::jpeg_decompress_struct* cinfo) { cinfo->src = &mgr; }

 private:
  static CordSourceManager& GetSelf(::jpeg_decompress_struct* cinfo) {
    // Because `CordSourceManager` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    return *reinterpret_cast<CordSourceManager*>(cinfo->src);
  }

  static void InitSource(::jpeg_decompress_struct* cinfo) {}

  // Note: `boolean` is an unfortunately-named platform-dependent type alias
  // defined by `jpeglib.h`; typically it is `int` but may be a different type
  // on some platforms.
  static boolean FillInputBuffer(::jpeg_decompress_struct* cinfo) {
    auto& self = GetSelf(cinfo);
    if (self.input_remaining) {
      auto chunk = absl::Cord::ChunkRemaining(self.char_it);
      self.mgr.next_input_byte = reinterpret_cast<const JOCTET*>(chunk.data());
      self.mgr.bytes_in_buffer = chunk.size();
      absl::Cord::Advance(&self.char_it, chunk.size());
      self.input_remaining -= chunk.size();
    } else {
      // libjpeg doesn't provide a clean way to signal EOF.  The documentation
      // recommends emitting a fake EOI marker in the case that all input is
      // consumed (that is what the built-in data sources do).
      static const JOCTET fake_eoi_buffer[2] = {(JOCTET)0xFF, (JOCTET)JPEG_EOI};

      // The warning emitted below will be treated as an error.
      WARNMS(cinfo, JWRN_JPEG_EOF);

      self.mgr.next_input_byte = fake_eoi_buffer;
      self.mgr.bytes_in_buffer = 2;
    }
    return static_cast<boolean>(true);
  }

  static void SkipInputData(::jpeg_decompress_struct* cinfo, long num_bytes) {
    if (num_bytes <= 0) return;
    auto& self = GetSelf(cinfo);
    size_t remove_from_buffer =
        std::min(self.mgr.bytes_in_buffer, static_cast<size_t>(num_bytes));
    self.mgr.bytes_in_buffer -= remove_from_buffer;
    self.mgr.next_input_byte += remove_from_buffer;
    num_bytes -= remove_from_buffer;
    if (num_bytes) {
      size_t num_to_advance =
          std::min(static_cast<size_t>(num_bytes), self.input_remaining);
      absl::Cord::Advance(&self.char_it, num_to_advance);
      self.input_remaining -= num_to_advance;
    }
  }

  static void TermSource(::jpeg_decompress_struct* cinfo) {}
};

struct CordDestManager {
  struct ::jpeg_destination_mgr mgr;
  char buffer[32 * 1024];
  absl::Cord* output;

  explicit CordDestManager(absl::Cord* output) : output(output) {
    mgr.init_destination = &InitDestination;
    mgr.empty_output_buffer = &EmptyOutputBuffer;
    mgr.term_destination = &TermDestination;
  }

  void Initialize(::jpeg_compress_struct* cinfo) { cinfo->dest = &mgr; }

 private:
  static void InitDestination(::jpeg_compress_struct* cinfo) {
    // Because `CordDestManager` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<CordDestManager*>(cinfo->dest);
    self->mgr.next_output_byte = reinterpret_cast<JOCTET*>(self->buffer);
    self->mgr.free_in_buffer = std::size(self->buffer);
  }

  // Note: `boolean` is an unfortunately-named platform-dependent type alias
  // defined by `jpeglib.h`; typically it is `int` but may be a different type
  // on some platforms.
  static boolean EmptyOutputBuffer(::jpeg_compress_struct* cinfo) {
    // Because `CordDestManager` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<CordDestManager*>(cinfo->dest);
    self->output->Append(
        std::string_view(self->buffer, std::size(self->buffer)));
    InitDestination(cinfo);
    return static_cast<boolean>(true);
  }

  static void TermDestination(::jpeg_compress_struct* cinfo) {
    // Because `CordDestManager` is standard layout, we can safely
    // `reinterpret_cast` from pointer to first member.
    auto* self = reinterpret_cast<CordDestManager*>(cinfo->dest);
    self->output->Append(std::string_view(
        self->buffer, std::size(self->buffer) - self->mgr.free_in_buffer));
  }
};

}  // namespace

Status Decode(const absl::Cord& input,
              FunctionView<Result<unsigned char*>(size_t width, size_t height,
                                                  size_t num_components)>
                  validate_size) {
  JpegStateWrapper<::jpeg_decompress_struct> state;
  CordSourceManager source_manager(input);
  // Wrap actual logic in lambda to avoid having any non-trivial local variables
  // in function using setjmp.
  [&] {
    if (setjmp(state.jmpbuf)) return;
    ::jpeg_create_decompress(&state.cinfo);
    source_manager.Initialize(&state.cinfo);
    {
      [[maybe_unused]] int code =
          ::jpeg_read_header(&state.cinfo, /*require_image=*/1);
      // With `require_image=1` and a non-suspending input source,
      // `jpeg_read_header` cannot return except in the case of success.
      assert(code == JPEG_HEADER_OK);
    }
    if (state.cinfo.num_components != 1 && state.cinfo.num_components != 3) {
      state.status = absl::InvalidArgumentError(
          StrCat("Expected 1 or 3 components, but received: ",
                 state.cinfo.num_components));
      return;
    }
    // Cannot return false since cord data source does not suspend.
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
              absl::Cord* output) {
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
  CordDestManager dest(output);
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
