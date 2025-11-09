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

#include "tensorstore/internal/image/image_view.h"

#include <assert.h>

#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

ImageView::ImageView(const ImageInfo& info,
                     tensorstore::span<unsigned char> data)
    : data_(data),
      dtype_(info.dtype),
      row_stride_(info.num_components * info.width) {
  assert(data.size() >=
         info.width * info.height * info.num_components * dtype_.size());
}

}  // namespace internal_image
}  // namespace tensorstore
