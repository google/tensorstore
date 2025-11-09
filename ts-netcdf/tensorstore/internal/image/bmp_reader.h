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

#ifndef TENSORSTORE_INTERNAL_IMAGE_BMP_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_BMP_READER_H_

#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_reader.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

struct BmpReaderOptions {};

/// Reads BMP format files. See: https://en.wikipedia.org/wiki/BMP_file_format
/// for more details about BMP files.
///
/// This is a minimal implementation of the windows device independent bitmap
/// format which does not yet support all of the features; however it should
/// allow reading full-color 3-channel and greyscale raw image and rle encoded
/// images.
class BmpReader : public ImageReader {
 public:
  BmpReader();
  ~BmpReader() override;

  // Allow move.
  BmpReader(BmpReader&& src);
  BmpReader& operator=(BmpReader&& src);

  // Initialize the decoder.
  absl::Status Initialize(riegeli::Reader* reader) override;

  // Returns the current ImageInfo.
  ImageInfo GetImageInfo() override;

  // Decodes the next available image into 'dest'.
  absl::Status Decode(tensorstore::span<unsigned char> dest) override {
    return DecodeImpl(dest, {});
  }
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const BmpReaderOptions& options) {
    return DecodeImpl(dest, options);
  }

 private:
  struct BmpHeader;

  absl::Status DecodeImpl(tensorstore::span<unsigned char> dest,
                          const BmpReaderOptions& options);

  riegeli::Reader* reader_ = nullptr;  // unowned
  std::unique_ptr<BmpHeader> header_;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_BMP_READER_H_
