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

/// Defines the "zstd" compressor for the Tiff driver.
#include "tensorstore/internal/compression/zstd_compressor.h"

#include "tensorstore/driver/tiff/compressor.h"
#include "tensorstore/driver/tiff/compressor_registry.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_tiff {
namespace {

using ::tensorstore::internal::ZstdCompressor;
namespace jb = ::tensorstore::internal_json_binding;

struct Registration {
  Registration() { RegisterCompressor<ZstdCompressor>("zstd", jb::Object()); }
} registration;

}  // namespace
}  // namespace internal_tiff
}  // namespace tensorstore