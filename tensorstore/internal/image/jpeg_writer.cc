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

#include "tensorstore/internal/image/jpeg_writer.h"

#include <cassert>
#include <csetjmp>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/status.h"

// Include libjpeg last
#include <jerror.h>
#include <jpeglib.h>
#include "tensorstore/internal/image/jpeg_common.h"
// See: https://libjpeg-turbo.org/Documentation/Documentation

namespace tensorstore {
namespace internal_image {
namespace {

struct EncodeState {
  ::jpeg_compress_struct cinfo_;
  JpegError error_;
  ::jpeg_destination_mgr jdest_;

  std::jmp_buf jmpbuf;
  absl::Status status_;
  riegeli::Writer* writer;
  bool started_ = false;

  EncodeState(riegeli::Writer* writer);
  ~EncodeState();

  absl::Status EncodeImpl();
};

void InitDestination(::jpeg_compress_struct* cinfo) {
  /*
  init_destination (j_compress_ptr cinfo)
          Initialize destination.  This is called by jpeg_start_compress()
          before any data is actually written.  It must initialize
          next_output_byte and free_in_buffer.  free_in_buffer must be
          initialized to a positive value.
  */
  auto* self = static_cast<EncodeState*>(cinfo->client_data);
  self->writer->Push();
  if (!self->writer->ok()) {
    longjmp(self->jmpbuf, 1);
  }

  self->jdest_.next_output_byte =
      reinterpret_cast<JOCTET*>(self->writer->cursor());
  self->jdest_.free_in_buffer = self->writer->available();
}

// Note: `boolean` is an unfortunately-named platform-dependent type alias
// defined by `jpeglib.h`; typically it is `int` but may be a different type
// on some platforms.
boolean EmptyOutputBuffer(::jpeg_compress_struct* cinfo) {
  /*
  empty_output_buffer (j_compress_ptr cinfo)
          This is called whenever the buffer has filled (free_in_buffer
          reaches zero).  In typical applications, it should write out the
          *entire* buffer (use the saved start address and buffer length;
          ignore the current state of next_output_byte and free_in_buffer).
          Then reset the pointer & count to the start of the buffer, and
          return TRUE indicating that the buffer has been dumped.
          free_in_buffer must be set to a positive value when TRUE is
          returned.  A FALSE return should only be used when I/O suspension is
          desired (this operating mode is discussed in the next section).
  */
  auto* self = static_cast<EncodeState*>(cinfo->client_data);
  self->writer->move_cursor(self->writer->available());
  InitDestination(cinfo);
  return static_cast<boolean>(true);
}

void TermDestination(::jpeg_compress_struct* cinfo) {
  /*
  term_destination (j_compress_ptr cinfo)
          Terminate destination --- called by jpeg_finish_compress() after all
          data has been written.  In most applications, this must flush any
          data remaining in the buffer.  Use either next_output_byte or
          free_in_buffer to determine how much data is in the buffer.
  */
  auto* self = static_cast<EncodeState*>(cinfo->client_data);
  self->writer->set_cursor(
      reinterpret_cast<char*>(self->jdest_.next_output_byte));
}

void SetDestHandlers(::jpeg_destination_mgr* jdest) {
  jdest->init_destination = &InitDestination;
  jdest->empty_output_buffer = &EmptyOutputBuffer;
  jdest->term_destination = &TermDestination;
}

EncodeState::EncodeState(riegeli::Writer* writer) : writer(writer) {
  // Set up error handler.
  error_.Construct(reinterpret_cast<::jpeg_common_struct*>(&cinfo_));

  cinfo_.client_data = this;
  cinfo_.mem = nullptr;

  jpeg_create_compress(&cinfo_);

  // Set up source manager.
  SetDestHandlers(&jdest_);
  cinfo_.dest = &jdest_;
}

EncodeState::~EncodeState() {
  if (started_) {
    jpeg_abort_compress(&cinfo_);
  }
  jpeg_destroy_compress(&cinfo_);
}

}  // namespace

JpegWriter::JpegWriter() = default;

JpegWriter::~JpegWriter() = default;

absl::Status JpegWriter::InitializeImpl(riegeli::Writer* writer,
                                        const JpegWriterOptions& options) {
  ABSL_CHECK(writer != nullptr);

  writer_ = std::move(writer);
  options_ = options;

  if (options.quality < 0 || options.quality > 100) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "JPEG options.quality of %d exceeds bounds", options.quality));
  }
  return absl::OkStatus();
}

absl::Status JpegWriter::Encode(const ImageInfo& info,
                                tensorstore::span<const unsigned char> source) {
  if (writer_ == nullptr) {
    return absl::InternalError("JPEG writer not initialized");
  }
  ABSL_CHECK(source.size() == ImageRequiredBytes(info));

  if (info.width > std::numeric_limits<JDIMENSION>::max() ||
      info.height > std::numeric_limits<JDIMENSION>::max()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Image dimensions of (%d, %d) exceed maximum size",
                        info.width, info.height));
  }
  if (info.num_components != 1 && info.num_components != 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected 1 or 3 components, but received: %d", info.num_components));
  }

  EncodeState state(writer_);
  ImageView source_view = MakeWriteImageView(info, source);

  state.cinfo_.image_width = info.width;
  state.cinfo_.image_height = info.height;
  state.cinfo_.input_components = info.num_components;
  state.cinfo_.in_color_space =
      (state.cinfo_.input_components == 1) ? JCS_GRAYSCALE : JCS_RGB;

  bool ok = [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(state.jmpbuf)) {
      return false;
    }

    ::jpeg_set_defaults(&state.cinfo_);
    ::jpeg_set_quality(&state.cinfo_, options_.quality, /*force_baseline=*/1);
    ::jpeg_start_compress(&state.cinfo_, /*write_all_tables=*/1);
    state.started_ = true;

    while (state.cinfo_.next_scanline < state.cinfo_.image_height) {
      auto* input_line = reinterpret_cast<JSAMPLE*>(
          source_view.data_row(state.cinfo_.next_scanline).data());
      // Always returns 1 since memory destination does not suspend.
      ::jpeg_write_scanlines(&state.cinfo_, &input_line, 1);
    }
    ::jpeg_finish_compress(&state.cinfo_);
    return true;
  }();

  // On failure, clear the writer.
  absl::Status status;
  if (!ok) {
    absl::Status status = internal::MaybeConvertStatusTo(
        state.writer->ok() ? state.error_.last_error : state.writer->status(),
        absl::StatusCode::kDataLoss);
    writer_ = nullptr;
    return status;
  }
  return absl::OkStatus();
}

absl::Status JpegWriter::Done() {
  if (writer_ == nullptr) {
    return absl::InternalError("JPEG writer not initialized");
  }
  if (!writer_->Close()) {
    return writer_->status();
  }
  return absl::OkStatus();
}

}  // namespace internal_image
}  // namespace tensorstore
