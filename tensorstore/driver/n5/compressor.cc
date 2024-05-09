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

#include "tensorstore/driver/n5/compressor.h"

#include <utility>

#include "absl/base/no_destructor.h"
#include "tensorstore/driver/n5/compressor_registry.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_registry.h"

namespace tensorstore {
namespace internal_n5 {
using CompressorRegistry = internal::JsonSpecifiedCompressor::Registry;
CompressorRegistry& GetCompressorRegistry() {
  static absl::NoDestructor<CompressorRegistry> registry;
  return *registry;
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Compressor, [](auto is_loading,
                                                      const auto& options,
                                                      auto* obj,
                                                      ::nlohmann::json* j) {
  namespace jb = tensorstore::internal_json_binding;
  auto& registry = GetCompressorRegistry();
  return jb::Object(
      jb::Member("type",
                 jb::MapValue(registry.KeyBinder(),
                              // "type" of "raw" maps to a default-initialized
                              // `Compressor` (i.e. nullptr).
                              std::make_pair(Compressor{}, "raw"))),
      registry.RegisteredObjectBinder())(is_loading, options, obj, j);
})

}  // namespace internal_n5
}  // namespace tensorstore
