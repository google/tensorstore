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

#ifndef TENSORSTORE_DRIVER_ZARR3_DEFAULT_NAN_H_
#define TENSORSTORE_DRIVER_ZARR3_DEFAULT_NAN_H_

#include <stdint.h>

#include <limits>

#include "absl/base/casts.h"
#include "tensorstore/data_type.h"
#include "tensorstore/util/float8.h"

namespace tensorstore {
namespace internal_zarr3 {

template <typename T>
inline T GetDefaultNaN() = delete;

template <>
inline dtypes::float32_t GetDefaultNaN<dtypes::float32_t>() {
  return absl::bit_cast<dtypes::float32_t>(uint32_t{0x7fc00000});
}

template <>
inline dtypes::float64_t GetDefaultNaN<dtypes::float64_t>() {
  return absl::bit_cast<dtypes::float64_t>(uint64_t{0x7ff8000000000000});
}

template <>
inline dtypes::float16_t GetDefaultNaN<dtypes::float16_t>() {
  return absl::bit_cast<dtypes::float16_t>(uint16_t{0x7e00});
}

template <>
inline dtypes::bfloat16_t GetDefaultNaN<dtypes::bfloat16_t>() {
  return absl::bit_cast<dtypes::bfloat16_t>(uint16_t{0x7fc0});
}

template <>
inline dtypes::float8_e4m3fn_t GetDefaultNaN<dtypes::float8_e4m3fn_t>() {
  // only a single Nan representation is supported
  return std::numeric_limits<Float8e4m3fn>::quiet_NaN();
}

template <>
inline dtypes::float8_e4m3fnuz_t GetDefaultNaN<dtypes::float8_e4m3fnuz_t>() {
  // only a single Nan representation is supported
  return std::numeric_limits<Float8e4m3fnuz>::quiet_NaN();
}

template <>
inline dtypes::float8_e4m3b11fnuz_t
GetDefaultNaN<dtypes::float8_e4m3b11fnuz_t>() {
  // only a single Nan representation is supported
  return std::numeric_limits<Float8e4m3b11fnuz>::quiet_NaN();
}

template <>
inline dtypes::float8_e5m2_t GetDefaultNaN<dtypes::float8_e5m2_t>() {
  // support both quiet and signaling nan, returning quiet one
  return std::numeric_limits<Float8e5m2>::quiet_NaN();
}

template <>
inline dtypes::float8_e5m2fnuz_t GetDefaultNaN<dtypes::float8_e5m2fnuz_t>() {
  // only a single Nan representation is supported
  return std::numeric_limits<Float8e5m2fnuz>::quiet_NaN();
}

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_DEFAULT_NAN_H_
