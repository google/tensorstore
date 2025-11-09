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

#include "tensorstore/internal/image/image_info.h"

#include <stddef.h>

#include <cmath>
#include <ostream>

#include "absl/strings/str_format.h"
#include "tensorstore/data_type.h"

namespace tensorstore {
namespace internal_image {

bool operator==(const ImageInfo& a, const ImageInfo& b) {
  return a.width == b.width && a.height == b.height &&
         a.num_components == b.num_components && a.dtype == b.dtype;
}

std::ostream& operator<<(std::ostream& os, const ImageInfo& info) {
  return os << absl::StrFormat(
             "{.width=%d, .height=%d, .num_components=%d, .dtype=%s}",
             info.width, info.height, info.num_components, info.dtype.name());
}

size_t ImageRequiredBytes(const ImageInfo& a) {
  return std::abs(a.width) * std::abs(a.height) * std::abs(a.num_components) *
         a.dtype.size();
}

}  // namespace internal_image
}  // namespace tensorstore
