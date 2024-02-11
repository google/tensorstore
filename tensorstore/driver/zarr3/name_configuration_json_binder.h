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

#ifndef TENSORSTORE_DRIVER_ZARR3_NAME_CONFIGURATION_JSON_BINDER_H_
#define TENSORSTORE_DRIVER_ZARR3_NAME_CONFIGURATION_JSON_BINDER_H_

#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/optional_object.h"

namespace tensorstore {
namespace internal_zarr3 {

template <typename NameBinder, typename ConfigurationBinder>
constexpr auto NameConfigurationJsonBinder(
    NameBinder name_binder, ConfigurationBinder configuration_binder) {
  namespace jb = ::tensorstore::internal_json_binding;
  return jb::Sequence(
      jb::Member("name", name_binder),
      jb::Member("configuration", jb::OptionalObject(configuration_binder)));
}

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_NAME_CONFIGURATION_JSON_BINDER_H_
