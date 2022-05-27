// Copyright 2021 The TensorStore Authors
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

#include <stddef.h>

#include <array>

#include "absl/memory/memory.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/image/driver_impl.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/compression/png.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_image_driver {
namespace {

namespace jb = tensorstore::internal_json_binding;

struct PngSpecialization : public png::EncodeOptions {
  constexpr static char id[] = "png";
  constexpr static char kTransactionError[] =
      "\"png\" driver does not support transactions";

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.compression_level);
  };

  // MSVC doesn't like &PngSpecialization::compression_level
  constexpr static auto default_json_binder =
      jb::Member("compression_level",
                 jb::Projection(&png::EncodeOptions::compression_level,
                                jb::DefaultValue([](auto* v) { *v = -1; })));

  Result<absl::Cord> EncodeImage(ArrayView<const void, 3> array_xyc) const {
    absl::Cord buffer;
    TENSORSTORE_RETURN_IF_ERROR(
        png::Encode(reinterpret_cast<const unsigned char*>(array_xyc.data()),
                    array_xyc.shape()[0], array_xyc.shape()[1],
                    array_xyc.shape()[2], *this, &buffer));
    return buffer;
  }

  Result<SharedArray<uint8_t, 3>> DecodeImage(absl::Cord value) {
    SharedArray<uint8_t, 3> array_xyc;
    auto allocate_data = [&](size_t width, size_t height,
                             size_t num_components) -> Result<unsigned char*> {
      std::array<Index, 3> shape_xyc = {static_cast<Index>(width),
                                        static_cast<Index>(height),
                                        static_cast<Index>(num_components)};
      array_xyc = AllocateArray<uint8_t>(shape_xyc);
      return reinterpret_cast<unsigned char*>(array_xyc.data());
    };
    TENSORSTORE_RETURN_IF_ERROR(png::Decode(value, allocate_data));
    return array_xyc;
  }
};

const internal::DriverRegistration<ImageDriverSpec<PngSpecialization>>
    png_driver_registration;

}  // namespace
}  // namespace internal_image_driver
}  // namespace tensorstore
