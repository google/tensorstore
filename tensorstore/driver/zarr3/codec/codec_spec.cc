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

#include "tensorstore/driver/zarr3/codec/codec_spec.h"

#include <stddef.h>

#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/internal/no_destructor.h"

namespace tensorstore {
namespace internal_zarr3 {

ZarrCodecSpec::~ZarrCodecSpec() = default;

ZarrCodecKind ZarrArrayToArrayCodecSpec::kind() const {
  return ZarrCodecKind::kArrayToArray;
}
ZarrCodecKind ZarrArrayToBytesCodecSpec::kind() const {
  return ZarrCodecKind::kArrayToBytes;
}
size_t ZarrArrayToBytesCodecSpec::sharding_height() const { return 0; }

ZarrCodecKind ZarrBytesToBytesCodecSpec::kind() const {
  return ZarrCodecKind::kBytesToBytes;
}

CodecRegistry& GetCodecRegistry() {
  static internal::NoDestructor<CodecRegistry> registry;
  return *registry;
}

}  // namespace internal_zarr3
}  // namespace tensorstore
