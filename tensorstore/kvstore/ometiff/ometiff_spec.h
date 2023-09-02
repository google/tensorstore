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

#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace ometiff {

struct OMETiffImageInfo {
  uint32_t width = 0;
  uint32_t height = 0;
  uint16_t bits_per_sample = 0;
  uint32_t tile_width = 0;
  uint32_t tile_height = 0;
  uint32_t rows_per_strip = 0;
  uint16_t sample_format = 0;
  uint16_t samples_per_pixel = 0;

  bool is_tiled = 0;
  uint64_t chunk_offset = 0;
  uint64_t chunk_size = 0;
  uint32_t num_chunks = 0;
  uint32_t compression = 0;
  DataType dtype;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OMETiffImageInfo,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  friend std::ostream& operator<<(std::ostream& os, const OMETiffImageInfo& x);
};

Result<::nlohmann::json> GetOMETiffImageInfo(std::istream& stream);

}  // namespace ometiff
}  // namespace tensorstore

#endif