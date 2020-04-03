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

#include "tensorstore/driver/zarr/compressor.h"

#include "tensorstore/driver/zarr/compressor_registry.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/internal/no_destructor.h"

namespace tensorstore {
namespace internal_zarr {
internal::JsonSpecifiedCompressor::Registry& GetCompressorRegistry() {
  static internal::NoDestructor<internal::JsonSpecifiedCompressor::Registry>
      registry;
  return *registry;
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Compressor, [](auto is_loading,
                                                      const auto& options,
                                                      auto* obj,
                                                      ::nlohmann::json* j) {
  namespace jb = tensorstore::internal::json_binding;
  return jb::MapValue(
      // JSON value of `null` maps to default-initialized `Compressor`
      // (i.e. nullptr).
      Compressor{}, nullptr,
      jb::Object(GetCompressorRegistry().MemberBinder("id")))(is_loading,
                                                              options, obj, j);
})

}  // namespace internal_zarr
}  // namespace tensorstore
