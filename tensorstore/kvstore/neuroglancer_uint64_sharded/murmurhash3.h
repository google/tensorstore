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

#ifndef TENSORSTORE_KVSTORE_NEUROGLANCER_UINT64_SHARDED_MURMURHASH3_H_
#define TENSORSTORE_KVSTORE_NEUROGLANCER_UINT64_SHARDED_MURMURHASH3_H_

#include <cstdint>

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

/// MurmurHash3_x86_128, specialized for 8 bytes of input.
///
/// Derived from the original code at
/// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp but
/// simplified for the case of just 8 bytes of input.
///
/// \param input The input value, treated as a 64-bit little endian value.
/// \param h[in,out] On input, specifies the seed.  On output, equals the hash.
void MurmurHash3_x86_128Hash64Bits(std::uint64_t input, std::uint32_t h[4]);

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_NEUROGLANCER_UINT64_SHARDED_MURMURHASH3_H_
