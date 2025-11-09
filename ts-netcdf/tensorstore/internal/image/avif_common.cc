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

#include "tensorstore/internal/image/avif_common.h"

// Include libavif last
#include <avif/avif.h>

namespace tensorstore {
namespace internal_image {

void AvifDeleter::operator()(avifEncoder* encoder) const {
  avifEncoderDestroy(encoder);
}

void AvifDeleter::operator()(avifDecoder* decoder) const {
  avifDecoderDestroy(decoder);
}

void AvifDeleter::operator()(avifImage* image) const {
  avifImageDestroy(image);
}

}  // namespace internal_image
}  // namespace tensorstore
