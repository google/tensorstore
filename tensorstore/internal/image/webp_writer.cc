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

#include "tensorstore/internal/image/webp_writer.h"

#include <stdint.h>

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

// Include libavif last
#include <webp/encode.h>

namespace tensorstore {
namespace internal_image {
namespace {

// Interface to write encoded WebP data out to a Writer.
static int WebPWriterWrite(const uint8_t* data, size_t data_size,
                           const WebPPicture* const picture) {
  riegeli::Writer& writer = *static_cast<riegeli::Writer*>(picture->custom_ptr);
  return writer.Write(
             std::string_view(reinterpret_cast<const char*>(data), data_size))
             ? 1
             : 0;
}

absl::Status EncodeWebP(riegeli::Writer* writer,
                        const WebPWriterOptions& options, const ImageInfo& info,
                        tensorstore::span<const unsigned char> source) {
  if (info.width > WEBP_MAX_DIMENSION || info.height > WEBP_MAX_DIMENSION) {
    return absl::InvalidArgumentError("WEBP image too large");
  }
  if (info.num_components != 3 && info.num_components != 4) {
    return absl::InvalidArgumentError("WEBP invalid num_components");
  }

  WebPConfig config;
  if (!WebPConfigInit(&config)) {
    return absl::InternalError("WEBP encoder init failed");
  }
  config.lossless = options.lossless ? 1 : 0;
  config.quality = options.quality;
  config.method = 6;
  config.exact = (info.num_components == 4) ? 1 : 0;  // Keep alpha channel.
  ABSL_CHECK(WebPValidateConfig(&config));

  WebPPicture pic;
  WebPPictureInit(&pic);
  pic.width = info.width;
  pic.height = info.height;
  pic.writer = WebPWriterWrite;
  pic.custom_ptr = writer;

  if (options.lossless) {
    pic.use_argb = 1;
  }

  auto status = [&]() -> absl::Status {
    if (info.num_components == 3) {
      if (!WebPPictureImportRGB(&pic, source.data(), info.width * 3)) {
        return absl::InvalidArgumentError("WEBP encoder failed to import");
      }
    } else {
      if (!WebPPictureImportRGBA(&pic, source.data(), info.width * 4)) {
        return absl::InvalidArgumentError("WEBP encoder failed to import");
      }
    }
    if (!WebPEncode(&config, &pic)) {
      return absl::InvalidArgumentError("WEBP encoder failed");
    }
    return absl::OkStatus();
  }();

  WebPPictureFree(&pic);
  return status;
}

}  // namespace

absl::Status WebPWriter::InitializeImpl(riegeli::Writer* writer,
                                        const WebPWriterOptions& options) {
  ABSL_CHECK(writer != nullptr);
  if (writer_) {
    return absl::InternalError("Initialize() already called");
  }
  if (options.quality < 0 || options.quality > 100) {
    return absl::InvalidArgumentError(
        "WEBP quality option must be in the range [0.. 100]");
  }
  writer_ = writer;
  options_ = options;
  return absl::OkStatus();
}

absl::Status WebPWriter::Encode(const ImageInfo& info,
                                tensorstore::span<const unsigned char> source) {
  if (!writer_) {
    return absl::InternalError("WEBP writer not initialized");
  }
  ABSL_CHECK_EQ(source.size(), ImageRequiredBytes(info));
  return EncodeWebP(writer_, options_, info, source);
}

absl::Status WebPWriter::Done() {
  if (!writer_) {
    return absl::InternalError("No data written");
  }
  if (!writer_->Close()) {
    return writer_->status();
  }
  writer_ = nullptr;
  return absl::OkStatus();
}

}  // namespace internal_image
}  // namespace tensorstore
