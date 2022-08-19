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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/murmurhash3.h"

#include <cstdint>

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

namespace {
constexpr std::uint32_t MurmurHash3_x86_128Mix(std::uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

constexpr std::uint32_t RotateLeft(std::uint32_t x, int r) {
  return (x << r) | (x >> (32 - r));
}

}  // namespace

void MurmurHash3_x86_128Hash64Bits(std::uint64_t input, std::uint32_t h[4]) {
  std::uint64_t h1 = h[0], h2 = h[1], h3 = h[2], h4 = h[3];
  const std::uint32_t c1 = 0x239b961b;
  const std::uint32_t c2 = 0xab0e9789;
  const std::uint32_t c3 = 0x38b34ae5;

  // c4 constant is not needed for only 8 bytes of input.
  // const std::uint32_t c4 = 0xa1e38b93;

  const std::uint32_t low = static_cast<std::uint32_t>(input);
  const std::uint32_t high = input >> 32;

  std::uint32_t k2 = high * c2;
  k2 = RotateLeft(k2, 16);
  k2 *= c3;
  h2 ^= k2;

  std::uint32_t k1 = low * c1;
  k1 = RotateLeft(k1, 15);
  k1 *= c2;
  h1 ^= k1;

  const std::uint32_t len = 8;

  h1 ^= len;
  h2 ^= len;
  h3 ^= len;
  h4 ^= len;

  h1 += h2;
  h1 += h3;
  h1 += h4;
  h2 += h1;
  h3 += h1;
  h4 += h1;

  h1 = MurmurHash3_x86_128Mix(h1);
  h2 = MurmurHash3_x86_128Mix(h2);
  h3 = MurmurHash3_x86_128Mix(h3);
  h4 = MurmurHash3_x86_128Mix(h4);

  h1 += h2;
  h1 += h3;
  h1 += h4;
  h2 += h1;

  h3 += h1;
  h4 += h1;

  h[0] = h1;
  h[1] = h2;
  h[2] = h3;
  h[3] = h4;
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
