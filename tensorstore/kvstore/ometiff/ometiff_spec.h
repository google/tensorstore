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

#ifndef TENSORSTORE_KVSTORE_OMETIFF_OMETIFF_SPEC_H_
#define TENSORSTORE_KVSTORE_OMETIFF_OMETIFF_SPEC_H_

#include <nlohmann/json.hpp>

#include "tensorstore/chunk_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace ometiff {

class OMETiffMetadata {
 public:
  DimensionIndex rank = dynamic_rank;

  /// Overall shape of array.
  std::vector<Index> shape;

  std::vector<Index> chunk_shape;

  uint16_t bits_per_sample = 0;
  uint16_t sample_format = 0;
  uint16_t samples_per_pixel = 0;

  bool is_tiled = 0;
  uint64_t data_offset = 0;
  uint64_t chunk_size = 0;
  uint32_t num_chunks = 0;
  uint32_t compression = 0;
  DataType dtype;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OMETiffMetadata,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  friend std::ostream& operator<<(std::ostream& os, const OMETiffMetadata& x);
};

Result<::nlohmann::json> GetOMETiffMetadata(std::istream& stream);

/// Sets chunk layout constraints implied by `rank` and `chunk_shape`.
absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout);

}  // namespace ometiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::ometiff::OMETiffMetadata)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::ometiff::OMETiffMetadata)

#endif