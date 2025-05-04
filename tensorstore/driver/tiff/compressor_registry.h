// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_TIFF_COMPRESSOR_REGISTRY_H_
#define TENSORSTORE_DRIVER_TIFF_COMPRESSOR_REGISTRY_H_

#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"

namespace tensorstore {
namespace internal_tiff {

// Returns the global registry instance for TIFF compressors.
// This registry maps string IDs (like "lzw", "deflate") to factories/binders
// capable of creating JsonSpecifiedCompressor instances.
internal::JsonSpecifiedCompressor::Registry& GetTiffCompressorRegistry();

// Returns the map from TIFF Compression tag enum to string ID.
const absl::flat_hash_map<internal_tiff_kvstore::CompressionType,
                          std::string_view>&
GetTiffCompressionMap();

template <typename T, typename Binder>
void RegisterCompressor(std::string_view id, Binder binder) {
  GetTiffCompressorRegistry().Register<T>(id, binder);
}

}  // namespace internal_tiff
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_TIFF_COMPRESSOR_REGISTRY_H_
