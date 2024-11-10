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

#ifndef TENSORSTORE_INTERNAL_INTEGER_TYPES_H_
#define TENSORSTORE_INTERNAL_INTEGER_TYPES_H_

/// \file
/// Defines `int_t<Bits>` and `uint_t<Bits>` type aliases.

#include <stddef.h>
#include <stdint.h>

namespace tensorstore {
namespace internal {

template <size_t Bits>
struct int_type;

template <size_t Bits>
using int_t = typename int_type<Bits>::type;

template <>
struct int_type<8> {
  using type = int8_t;
};

template <>
struct int_type<16> {
  using type = int16_t;
};

template <>
struct int_type<32> {
  using type = int32_t;
};

template <>
struct int_type<64> {
  using type = int64_t;
};

template <size_t Bits>
struct uint_type;

template <size_t Bits>
using uint_t = typename uint_type<Bits>::type;

template <>
struct uint_type<8> {
  using type = uint8_t;
};

template <>
struct uint_type<16> {
  using type = uint16_t;
};

template <>
struct uint_type<32> {
  using type = uint32_t;
};

template <>
struct uint_type<64> {
  using type = uint64_t;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTEGER_TYPES_H_
