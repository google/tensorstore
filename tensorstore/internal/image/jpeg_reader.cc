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

#include "tensorstore/internal/image/jpeg_reader.h"

#include <cassert>
#include <csetjmp>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/data_type.h"
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

// jpeg_source_manager implementatgion.
// Because JpegSourceRiegeli is standard layout, we can safely
// `reinterpret_cast` from pointer to first member.
struct JpegSourceRiegeli {
  struct ::jpeg_source_mgr jsrc_;
  riegeli::Reader* reader = nullptr;
  size_t advance_by = 0;

  // ::jpeg_source_mgr handlers
  static void InitSource(::jpeg_decompress_struct* cinfo);
  static boolean FillInputBuffer(::jpeg_decompress_struct* cinfo);
  static void SkipInputData(::jpeg_decompress_struct* cinfo,
                            long num_bytes);  // NOLINT(runtime/int)
  static void TermSource(::jpeg_decompress_struct* cinfo);

  // "Constructor"
  void Construct(::jpeg_decompress_struct* cinfo, riegeli::Reader* input);
};

void JpegSourceRiegeli::InitSource(::jpeg_decompress_struct* cinfo) {
  auto& self = *reinterpret_cast<JpegSourceRiegeli*>(cinfo->src);
  self.jsrc_.next_input_byte = nullptr;
  self.jsrc_.bytes_in_buffer = 0;
  self.advance_by = 0;
}

// Note: `boolean` is an unfortunately-named platform-dependent type alias
// defined by `jpeglib.h`; typically it is `int` but may be a different type
// on some platforms.
boolean JpegSourceRiegeli::FillInputBuffer(::jpeg_decompress_struct* cinfo) {
  /*
  fill_input_buffer (j_decompress_ptr cinfo)
      This is called whenever bytes_in_buffer has reached zero and more
      data is wanted.  In typical applications, it should read fresh data
      into the buffer (ignoring the current state of next_input_byte and
      bytes_in_buffer), reset the pointer & count to the start of the
      buffer, and return TRUE indicating that the buffer has been reloaded.
      It is not necessary to fill the buffer entirely, only to obtain at
      least one more byte.  bytes_in_buffer MUST be set to a positive
      value if TRUE is returned.  A FALSE return should only be used when I/O
      suspension is desired (this mode is discussed in the next section).
  */
  auto& self = *reinterpret_cast<JpegSourceRiegeli*>(cinfo->src);

  self.reader->move_cursor(self.advance_by);
  self.advance_by = 0;

  if (self.reader->Pull()) {
    // There is still more data to read.
    self.jsrc_.next_input_byte =
        reinterpret_cast<const JOCTET*>(self.reader->cursor());
    self.jsrc_.bytes_in_buffer = self.reader->available();
    self.advance_by = self.reader->available();
  } else {
    // The reader appears to be at EOF.
    // libjpeg doesn't provide a clean way to signal EOF.  The documentation
    // recommends emitting a fake EOI marker in the case that all input is
    // consumed (that is what the built-in data sources do).
    static const JOCTET fake_eoi_buffer[2] = {(JOCTET)0xFF, (JOCTET)JPEG_EOI};

    // The warning emitted below will be treated as an error.
    WARNMS(cinfo, JWRN_JPEG_EOF);

    self.jsrc_.next_input_byte = fake_eoi_buffer;
    self.jsrc_.bytes_in_buffer = 2;
  }
  return static_cast<boolean>(true);
}

void JpegSourceRiegeli::SkipInputData(::jpeg_decompress_struct* cinfo,
                                      long num_bytes) {  // NOLINT(runtime/int)
  /*
  skip_input_data (j_decompress_ptr cinfo, long num_bytes)
      Skip num_bytes worth of data.  The buffer pointer and count should
      be advanced over num_bytes input bytes, refilling the buffer as
      needed.  This is used to skip over a potentially large amount of
      uninteresting data (such as an APPn marker).  In some applications
      it may be possible to optimize away the reading of the skipped data,
      but it's not clear that being smart is worth much trouble; large
      skips are uncommon.  bytes_in_buffer may be zero on return.
      A zero or negative skip count should be treated as a no-op.
  */
  if (num_bytes <= 0) return;

  auto& self = *reinterpret_cast<JpegSourceRiegeli*>(cinfo->src);

  // Advance the current position
  size_t locate = self.reader->available() - self.jsrc_.bytes_in_buffer;
  self.reader->move_cursor(locate);

  // And seek to the desired forward location.
  self.reader->Skip(num_bytes);
  self.jsrc_.next_input_byte = nullptr;
  self.jsrc_.bytes_in_buffer = 0;
  self.advance_by = 0;
}

void JpegSourceRiegeli::TermSource(::jpeg_decompress_struct* cinfo) {
  auto& self = *reinterpret_cast<JpegSourceRiegeli*>(cinfo->src);
  self.reader->move_cursor(self.advance_by);
  self.advance_by = 0;
  self.jsrc_.bytes_in_buffer = 0;
}

void JpegSourceRiegeli::Construct(::jpeg_decompress_struct* cinfo,
                                  riegeli::Reader* input) {
  assert(cinfo);
  assert(input);
  /// Check for available data first.
  reader = input;

  jsrc_.init_source = &JpegSourceRiegeli::InitSource;
  jsrc_.fill_input_buffer = &JpegSourceRiegeli::FillInputBuffer;
  jsrc_.skip_input_data = &JpegSourceRiegeli::SkipInputData;
  jsrc_.term_source = &JpegSourceRiegeli::TermSource;

  // Use default `resync_to_restart` implementation provided by libjpeg.
  jsrc_.resync_to_restart = ::jpeg_resync_to_restart;
  jsrc_.next_input_byte = nullptr;
  jsrc_.bytes_in_buffer = 0;

  cinfo->src = &jsrc_;
}

/// Helper method.
ImageInfo GetJpegImageInfo(::jpeg_decompress_struct* cinfo) {
  ImageInfo info;
  info.width = cinfo->image_width;
  info.height = cinfo->image_height;
  info.num_components = cinfo->num_components;
  info.dtype = dtype_v<uint8_t>;
  // TODO: Colorspace stuff.
  return info;
}

}  // namespace

struct JpegReader::Context {
  ::jpeg_decompress_struct cinfo_;
  JpegError error_;
  JpegSourceRiegeli riegeli_src_;
  bool created_ = false;
  bool started_ = false;

  ~Context();

  absl::Status Initialize(riegeli::Reader* reader);
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const JpegReaderOptions& options);
};

JpegReader::Context::~Context() {
  if (started_) {
    // Call abort instead of finish so that we will avoid exiting with an error.
    jpeg_abort_decompress(&cinfo_);
    // ... however seek to the end of the buffered data.
    riegeli_src_.reader->move_cursor(riegeli_src_.advance_by);
    riegeli_src_.advance_by = 0;
  }
  if (created_) {
    jpeg_destroy_decompress(&cinfo_);
  }
}

absl::Status JpegReader::Context::Initialize(riegeli::Reader* reader) {
  // Initialize jpeg library.
  error_.Construct(reinterpret_cast<::jpeg_common_struct*>(&cinfo_));

  cinfo_.mem = nullptr;
  cinfo_.client_data = nullptr;

  jpeg_create_decompress(&cinfo_);
  created_ = true;

  // Set up source manager.
  riegeli_src_.Construct(&cinfo_, reader);

  bool ok = [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(error_.jmpbuf)) {
      return false;
    }

    // Read the JPEG header so that we can do error checking.
    [[maybe_unused]] int code = jpeg_read_header(&cinfo_, /*require_image=*/1);
    // With `require_image=1` and a non-suspending input source,
    // `jpeg_read_header` cannot return except in the case of success.
    assert(code == JPEG_HEADER_OK);
    return true;
  }();

  if (!ok || !riegeli_src_.reader->ok()) {
    return internal::MaybeConvertStatusTo(riegeli_src_.reader->ok()
                                              ? error_.last_error
                                              : riegeli_src_.reader->status(),
                                          absl::StatusCode::kDataLoss);
  }

  if (cinfo_.num_components != 1 && cinfo_.num_components != 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to decode JPEG: Expected 1 or 3 components, but received: %d",
        cinfo_.num_components));
  }
  return absl::OkStatus();
}

absl::Status JpegReader::Context::Decode(tensorstore::span<unsigned char> dest,
                                         const JpegReaderOptions& options) {
  if (started_) {
    return absl::InternalError("");
  }

  // Validate the image is compatible.
  auto info = GetJpegImageInfo(&cinfo_);
  ABSL_CHECK_EQ(dest.size(), ImageRequiredBytes(info));

  ImageView dest_view(info, dest);
  bool ok = [&]() {
    // Setjump is problematic with C++; by convention we put it in a
    // lambda which has no variables requiring cleanup.
    if (setjmp(error_.jmpbuf)) {
      return false;
    }

    // Start decompressing
    ::jpeg_start_decompress(&cinfo_);
    started_ = true;

    // ... then read each scanline
    while (cinfo_.output_scanline < cinfo_.output_height) {
      auto* output_line = reinterpret_cast<JSAMPLE*>(
          dest_view.data_row(cinfo_.output_scanline).data());
      if (::jpeg_read_scanlines(&cinfo_, &output_line, 1) != 1) {
        error_.last_error.Update(absl::DataLossError(absl::StrFormat(
            "Cannot read JPEG; data ended after %d/%d scan lines",
            cinfo_.output_scanline, cinfo_.output_height)));
        return false;
      }
    }
    return true;
  }();

  if (!ok || !riegeli_src_.reader->ok()) {
    return internal::MaybeConvertStatusTo(riegeli_src_.reader->ok()
                                              ? error_.last_error
                                              : riegeli_src_.reader->status(),
                                          absl::StatusCode::kDataLoss);
  }
  return absl::OkStatus();
}

JpegReader::JpegReader() = default;
JpegReader::~JpegReader() = default;
JpegReader::JpegReader(JpegReader&& src) = default;
JpegReader& JpegReader::operator=(JpegReader&& src) = default;

absl::Status JpegReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);
  if (context_) {
    context_ = nullptr;
  }

  /// Check the signature.
  constexpr const unsigned char kSignature[] = {0xFF, 0xD8, 0xFF};
  reader->SetReadAllHint(true);
  if (!reader->Pull(3) ||
      memcmp(kSignature, reader->cursor(), sizeof(kSignature)) != 0) {
    return absl::InvalidArgumentError("Not a JPEG file");
  }

  reader_ = reader;
  auto context = std::make_unique<JpegReader::Context>();
  TENSORSTORE_RETURN_IF_ERROR(context->Initialize(reader_));
  context_ = std::move(context);
  return absl::OkStatus();
}

ImageInfo JpegReader::GetImageInfo() {
  if (!context_) return {};
  return GetJpegImageInfo(&context_->cinfo_);
}

absl::Status JpegReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                    const JpegReaderOptions& options) {
  if (!context_) {
    return absl::InternalError("No JPEG file to decode");
  }
  auto context = std::move(context_);
  return context->Decode(dest, options);
}

}  // namespace internal_image
}  // namespace tensorstore
