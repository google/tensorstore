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
/// Defines the "blosc" compressor for n5.  Linking in this library
/// automatically registers it.

#include "tensorstore/internal/compression/blosc_compressor.h"

#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/driver/n5/compressor_registry.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_n5 {
namespace {

struct Registration {
  Registration() {
    using internal::BloscCompressor;
    namespace jb = tensorstore::internal_json_binding;
    RegisterCompressor<BloscCompressor>(
        "blosc",
        jb::Object(
            jb::Member("cname", jb::Projection(&BloscCompressor::codec,
                                               BloscCompressor::CodecBinder())),
            jb::Member("clevel", jb::Projection(&BloscCompressor::level,
                                                jb::Integer<int>(0, 9))),
            jb::Member("shuffle", jb::Projection(&BloscCompressor::shuffle,
                                                 jb::Integer<int>(0, 2))),
            jb::Member(
                "blocksize",
                jb::Projection(&BloscCompressor::blocksize,
                               jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                                   [](std::size_t* v) { *v = 0; },
                                   jb::Integer<std::size_t>())))));
  }
} registration;

}  // namespace
}  // namespace internal_n5
}  // namespace tensorstore
