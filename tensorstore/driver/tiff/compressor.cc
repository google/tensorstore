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

#include "tensorstore/driver/tiff/compressor.h"  // For Compressor alias declaration

#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "tensorstore/driver/tiff/compressor_registry.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"

namespace tensorstore {
namespace internal_tiff {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal_tiff_kvstore::CompressionType;

internal::JsonSpecifiedCompressor::Registry& GetTiffCompressorRegistry() {
  static absl::NoDestructor<internal::JsonSpecifiedCompressor::Registry>
      registry;
  return *registry;
}

// Defines the mapping from TIFF numeric tag values to the string IDs used
// for compressor registration and CodecSpec JSON representation.
const static auto* const kCompressionTypeToStringIdMap =
    new absl::flat_hash_map<CompressionType, std::string_view>{
        {CompressionType::kNone, "raw"},         // No compression
        {CompressionType::kZStd, "zstd"},        // Zstandard compression
        {CompressionType::kDeflate, "zlib"},  // Deflate/Zlib compression.
        // { CompressionType::kPackBits, "packbits" },
    };

const absl::flat_hash_map<CompressionType, std::string_view>&
GetTiffCompressionMap() {
  return *kCompressionTypeToStringIdMap;
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Compressor, [](auto is_loading,
                                                      const auto& options,
                                                      auto* obj, auto* j) {
  auto& registry = GetTiffCompressorRegistry();
  return jb::Object(
      jb::Member("type",
                 jb::MapValue(
                     registry.KeyBinder(),
                     // Map "raw" to a default-constructed Compressor (nullptr)
                     std::make_pair(Compressor{}, std::string("raw")))),
      registry.RegisteredObjectBinder())(is_loading, options, obj, j);
})

}  // namespace internal_tiff
}  // namespace tensorstore
