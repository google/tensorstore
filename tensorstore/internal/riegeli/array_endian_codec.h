// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_ARRAY_ENDIAN_CODEC_H_
#define TENSORSTORE_INTERNAL_RIEGELI_ARRAY_ENDIAN_CODEC_H_

#include <stddef.h>

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

inline absl::Cord MakeCordFromSharedPtr(std::shared_ptr<const void> ptr,
                                        size_t size) {
  std::string_view s(static_cast<const char*>(ptr.get()), size);
  return absl::MakeCordFromExternal(
      s, [ptr = std::move(ptr)](std::string_view s) mutable { ptr.reset(); });
}

// Encodes an array of trivial elements in the specified order.
[[nodiscard]] bool EncodeArrayEndian(SharedArrayView<const void> decoded,
                                     endian encoded_endian,
                                     ContiguousLayoutOrder order,
                                     riegeli::Writer& writer);

// Decodes an array of trivial elements in the specified order.
Result<SharedArray<const void>> DecodeArrayEndian(
    riegeli::Reader& reader, DataType dtype, span<const Index> decoded_shape,
    endian encoded_endian, ContiguousLayoutOrder order);

// Decodes an array of trivial elements in the specified order.
absl::Status DecodeArrayEndian(riegeli::Reader& reader, endian encoded_endian,
                               ContiguousLayoutOrder order,
                               ArrayView<void> decoded);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_ARRAY_ENDIAN_CODEC_H_
