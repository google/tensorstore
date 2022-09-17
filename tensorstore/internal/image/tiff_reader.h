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

#ifndef TENSORSTORE_INTERNAL_IMAGE_TIFF_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_TIFF_READER_H_

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_reader.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

struct TiffReaderOptions {};

class TiffReader : public ImageReader {
 public:
  struct Context;

  TiffReader();
  ~TiffReader() override;

  // Allow move.
  TiffReader(TiffReader&& src);
  TiffReader& operator=(TiffReader&& src);

  // Initialize the decoder.
  absl::Status Initialize(riegeli::Reader* reader) override;

  // Returns the number of TIFF directory entries (or pages).
  int GetFrameCount() const;

  // Sets the state of the decoder so that the requested frame will be the next
  // to be returned through a call to Decode(). The parameter 'frame_number' is
  // 0-based, so that 0 will return the first frame. Will return an error if
  // the 'frame_number' parameter is less than zero or greater or equal to
  // GetFrameCount().
  absl::Status SeekFrame(int frame_number);

  // Returns the current ImageInfo.
  ImageInfo GetImageInfo() override;

  // Decodes the next available image into 'dest'.
  absl::Status Decode(tensorstore::span<unsigned char> dest) override {
    return DecodeImpl(dest, {});
  }
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const TiffReaderOptions& options) {
    return DecodeImpl(dest, options);
  }

 private:
  absl::Status DecodeImpl(tensorstore::span<unsigned char> dest,
                          const TiffReaderOptions& options);

  std::unique_ptr<Context> context_;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_TIFF_READER_H_
