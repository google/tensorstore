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

#ifndef TENSORSTORE_INTERNAL_IMAGE_AVIF_COMMON_H_
#define TENSORSTORE_INTERNAL_IMAGE_AVIF_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include <cstring>

// Forward declare to avoid include issues.
struct avifDecoder;
struct avifEncoder;
struct avifImage;

namespace tensorstore {
namespace internal_image {

struct AvifDeleter {
  void operator()(avifEncoder* encoder) const;
  void operator()(avifDecoder* decoder) const;
  void operator()(avifImage* image) const;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_AVIF_COMMON_H_
