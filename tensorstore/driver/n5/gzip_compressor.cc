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
/// Defines the "gzip" compressor for n5.  Linking in this library automatically
/// registers it.

#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/driver/n5/compressor_registry.h"
#include "tensorstore/internal/compression/zlib_compressor.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_n5 {
namespace {

struct Registration {
  Registration() {
    using internal::ZlibCompressor;
    namespace jb = tensorstore::internal_json_binding;
    RegisterCompressor<ZlibCompressor>(
        "gzip",
        jb::Object(
            jb::Member(
                "level",
                jb::Projection(
                    &ZlibCompressor::level,
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                        [](auto* v) { *v = -1; }, jb::Integer<int>(-1, 9)))),
            jb::Member(
                "useZlib",
                jb::Projection(
                    &ZlibCompressor::use_gzip_header,
                    jb::GetterSetter(
                        [](bool use_gzip) { return !use_gzip; },
                        [](bool& use_gzip, bool use_zlib) {
                          use_gzip = !use_zlib;
                        },
                        jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                            [](bool* use_zlib) { *use_zlib = false; }))))));
  }
} registration;

}  // namespace
}  // namespace internal_n5
}  // namespace tensorstore
