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

/// \file
///
/// Defines the "zstd" compressor for zarr.  Linking in this library
/// automatically registers it.

#include "tensorstore/internal/compression/zstd_compressor.h"

#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/driver/zarr/compressor_registry.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_zarr {
namespace {

using ::riegeli::ZstdWriterBase;
using ::tensorstore::internal::ZstdCompressor;
namespace jb = ::tensorstore::internal_json_binding;

struct Registration {
  Registration() {
    RegisterCompressor<ZstdCompressor>(
        "zstd",
        jb::Object(jb::Member(
            "level",
            jb::Projection(
                &ZstdCompressor::level,
                jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                    [](auto* v) { *v = 1; },
                    jb::Integer<int>(
                        ZstdWriterBase::Options::kMinCompressionLevel,
                        ZstdWriterBase::Options::kMaxCompressionLevel))))));
  }
} registration;

}  // namespace
}  // namespace internal_zarr
}  // namespace tensorstore
