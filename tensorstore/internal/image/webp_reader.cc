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

#include "tensorstore/internal/image/webp_reader.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

// Include libwebp last
#include <webp/decode.h>
#include <webp/demux.h>

namespace tensorstore {
namespace internal_image {
namespace {

const char* VP8StatusDesc(VP8StatusCode code) {
  switch (code) {
    case VP8_STATUS_OK:
      return "OK";
    case VP8_STATUS_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case VP8_STATUS_INVALID_PARAM:
      return "INVALID_PARAM";
    case VP8_STATUS_BITSTREAM_ERROR:
      return "BITSTREAM_ERROR";
    case VP8_STATUS_UNSUPPORTED_FEATURE:
      return "UNSUPPORTED_FEATURE";
    case VP8_STATUS_SUSPENDED:
      return "SUSPENDED";
    case VP8_STATUS_USER_ABORT:
      return "USER_ABORT";
    case VP8_STATUS_NOT_ENOUGH_DATA:
      return "NOT_ENOUGH_DATA";
    default:
      return "UNKNOWN";
  }
}

}  // namespace

struct WebPReader::Context {
  Context(riegeli::Reader* reader);

  absl::Status Init();
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const WebPReaderOptions& options);
  ImageInfo GetImageInfo();

  riegeli::Reader* reader_;
  WebPBitstreamFeatures features_;
};

WebPReader::Context::Context(riegeli::Reader* reader) : reader_(reader) {}

absl::Status WebPReader::Context::Init() {
  static constexpr size_t kMinBuffer = 1024;
  static constexpr size_t kMaxBuffer = 64 * 1024;
  while (true) {
    auto vp8_status =
        WebPGetFeatures(reinterpret_cast<const uint8_t*>(reader_->cursor()),
                        reader_->available(), &features_);

    if (vp8_status == VP8_STATUS_NOT_ENOUGH_DATA) {
      // ... As a first approximation, double the current buffer on each read,
      // up to the max buffer size. Generally, though, the feature information
      // should be available in the first 512 bytes of the stream.
      size_t target =
          std::min(kMaxBuffer, std::max(kMinBuffer, reader_->available() * 2));
      reader_->Pull(reader_->available() + 1, target);
      if (!reader_->ok()) {
        return reader_->status();
      }
      continue;
    }

    // Either success or an error parsing the WEBP bitstream.
    return (vp8_status == VP8_STATUS_OK)
               ? absl::OkStatus()
               : absl::InvalidArgumentError(
                     "Failed to read WEBP bitstream features");
  };
}

absl::Status WebPReader::Context::Decode(tensorstore::span<unsigned char> dest,
                                         const WebPReaderOptions& options) {
  WebPDecBuffer buf;
  WebPInitDecBuffer(&buf);
  buf.colorspace = features_.has_alpha ? MODE_RGBA : MODE_RGB;
  buf.u.RGBA.rgba = dest.data();
  buf.u.RGBA.stride = features_.width * (features_.has_alpha ? 4 : 3);
  buf.u.RGBA.size = dest.size();
  buf.is_external_memory = 1;

  WebPIDecoder* idec = WebPINewDecoder(&buf);
  auto status = [&]() -> absl::Status {
    while (reader_->Pull()) {
      auto status =
          WebPIAppend(idec, reinterpret_cast<const uint8_t*>(reader_->cursor()),
                      reader_->available());
      reader_->move_cursor(reader_->available());
      if (status != VP8_STATUS_OK && status != VP8_STATUS_SUSPENDED) {
        return absl::DataLossError(tensorstore::StrCat("Error decoding WEBP: ",
                                                       VP8StatusDesc(status)));
      }
    }
    if (!reader_->ok()) {
      return reader_->status();
    }
    return absl::OkStatus();
  }();

  WebPIDelete(idec);
  WebPFreeDecBuffer(&buf);

  return status;
}

ImageInfo WebPReader::Context::GetImageInfo() {
  ImageInfo info;
  info.width = features_.width;
  info.height = features_.height;
  info.num_components = (features_.has_alpha ? 4 : 3);
  info.dtype = dtype_v<uint8_t>;
  return info;
}

WebPReader::WebPReader() = default;
WebPReader::~WebPReader() = default;
WebPReader::WebPReader(WebPReader&& src) = default;
WebPReader& WebPReader::operator=(WebPReader&& src) = default;

absl::Status WebPReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);

  /// Check the signature. "RIFF....WEBP", etc.
  if (!reader->Pull(12) || (memcmp("RIFF", reader->cursor(), 4) != 0 ||
                            memcmp("WEBP", reader->cursor() + 8, 4) != 0)) {
    return absl::InvalidArgumentError(
        "Failed to decode WEBP: missing WEBP signature");
  }
  reader->SetReadAllHint(true);
  auto context = std::make_unique<Context>(reader);
  TENSORSTORE_RETURN_IF_ERROR(context->Init());
  context_ = std::move(context);
  return absl::OkStatus();
}

ImageInfo WebPReader::GetImageInfo() {
  if (!context_) return {};
  return context_->GetImageInfo();
}

absl::Status WebPReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                    const WebPReaderOptions& options) {
  if (!context_) {
    return absl::InternalError("No WEBP file to decode");
  }
  auto context = std::move(context_);
  return context->Decode(dest, options);
}

}  // namespace internal_image
}  // namespace tensorstore
