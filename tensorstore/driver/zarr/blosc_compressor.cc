// Copyright 2020 The TensorStore Authors
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
/// Defines the "blosc" compressor for zarr.  Linking in this library
/// automatically registers it.

#include "tensorstore/internal/compression/blosc_compressor.h"

#include <blosc.h>
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/driver/zarr/compressor_registry.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_zarr {
namespace {

struct Registration {
  Registration() {
    using internal::BloscCompressor;
    namespace jb = tensorstore::internal_json_binding;
    RegisterCompressor<BloscCompressor>(
        "blosc",
        jb::Object(
            jb::Member("cname",
                       jb::Projection(
                           &BloscCompressor::codec,
                           jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                               [](std::string* v) { *v = BLOSC_LZ4_COMPNAME; },
                               BloscCompressor::CodecBinder()))),
            jb::Member(
                "clevel",
                jb::Projection(
                    &BloscCompressor::level,
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                        [](int* v) { *v = 5; }, jb::Integer<int>(0, 9)))),
            jb::Member(
                "shuffle",
                jb::Projection(
                    &BloscCompressor::shuffle,
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                        [](int* v) { *v = -1; }, jb::Integer<int>(-1, 2)))),
            jb::Member(
                "blocksize",
                jb::Projection(&BloscCompressor::blocksize,
                               jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                                   [](std::size_t* v) { *v = 0; },
                                   jb::Integer<std::size_t>())))));
  }
} registration;

}  // namespace
}  // namespace internal_zarr
}  // namespace tensorstore
