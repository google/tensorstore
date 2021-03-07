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

#ifndef TENSORSTORE_DRIVER_ZARR_COMPRESSOR_REGISTRY_H_
#define TENSORSTORE_DRIVER_ZARR_COMPRESSOR_REGISTRY_H_

#include <string_view>

#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/internal/json_registry.h"

namespace tensorstore {
namespace internal_zarr {

internal::JsonSpecifiedCompressor::Registry& GetCompressorRegistry();

template <typename T, typename Binder>
void RegisterCompressor(std::string_view id, Binder binder) {
  GetCompressorRegistry().Register<T>(id, binder);
}

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_COMPRESSOR_REGISTRY_H_
