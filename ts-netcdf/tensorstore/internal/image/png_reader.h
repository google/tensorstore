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

#ifndef TENSORSTORE_INTERNAL_IMAGE_PNG_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_PNG_READER_H_

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_reader.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

struct PngReaderOptions {};

class PngReader : public ImageReader {
 public:
  PngReader();
  ~PngReader() override;

  PngReader(PngReader&& src);
  PngReader& operator=(PngReader&& src);

  // Initialize the decoder.
  absl::Status Initialize(riegeli::Reader* reader) override;

  // Returns the current ImageInfo.
  ImageInfo GetImageInfo() override;

  // Decodes the next available image into 'dest'.
  absl::Status Decode(tensorstore::span<unsigned char> dest) override {
    return DecodeImpl(dest, {});
  }
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const PngReaderOptions& options) {
    return DecodeImpl(dest, options);
  }

 private:
  absl::Status DecodeImpl(tensorstore::span<unsigned char> dest,
                          const PngReaderOptions& options);

  riegeli::Reader* reader_ = nullptr;  // unowned

  struct Context;

  std::unique_ptr<Context> context_;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_PNG_READER_H_
