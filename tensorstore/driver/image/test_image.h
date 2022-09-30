// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_IMAGE_TEST_IMAGE_H_
#define TENSORSTORE_DRIVER_IMAGE_TEST_IMAGE_H_

#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal_image_driver {

// Returns a 256x256 RGB image (see internal/image/image.png) encoded
// in various formats. where the pixel values are { x, y, 0 }.
absl::Cord GetAvif();
absl::Cord GetBmp();
absl::Cord GetJpeg();
absl::Cord GetPng();
absl::Cord GetTiff();
absl::Cord GetWebP();

}  // namespace internal_image_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_IMAGE_TEST_IMAGE_H_
