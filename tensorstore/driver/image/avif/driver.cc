// Copyright 2022 The TensorStore Authors
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

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/image/driver_impl.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/image/avif_reader.h"
#include "tensorstore/internal/image/avif_writer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_image_driver {
namespace {

using ::tensorstore::internal_image::AvifReader;
using ::tensorstore::internal_image::AvifWriter;
using ::tensorstore::internal_image::AvifWriterOptions;
using ::tensorstore::internal_image::ImageInfo;

namespace jb = tensorstore::internal_json_binding;

struct AvifSpecialization : public AvifWriterOptions {
  constexpr static char id[] = "avif";
  constexpr static char kTransactionError[] =
      "\"avif\" driver does not support transactions";

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.quantizer, x.speed);
  };

  constexpr static auto default_json_binder = jb::Sequence(
      jb::Member("quantizer",
                 jb::Projection(&AvifWriterOptions::quantizer,
                                jb::DefaultValue([](auto* v) { *v = 0; }))),
      jb::Member(   //
          "speed",  //
          jb::Projection(&AvifWriterOptions::speed,
                         jb::DefaultValue([](auto* v) { *v = 6; }))));

  Result<absl::Cord> EncodeImage(ArrayView<const void, 3> array_yxc) const {
    auto shape_yxc = array_yxc.shape();
    ImageInfo info{/*.height =*/static_cast<int32_t>(shape_yxc[0]),
                   /*.width =*/static_cast<int32_t>(shape_yxc[1]),
                   /*.num_components =*/static_cast<int32_t>(shape_yxc[2])};

    absl::Cord buffer;
    riegeli::CordWriter<> buffer_writer(&buffer);

    AvifWriter writer;
    TENSORSTORE_RETURN_IF_ERROR(writer.Initialize(&buffer_writer, *this));
    TENSORSTORE_RETURN_IF_ERROR(writer.Encode(
        info, tensorstore::span(
                  reinterpret_cast<const unsigned char*>(array_yxc.data()),
                  array_yxc.num_elements() * array_yxc.dtype().size())));
    TENSORSTORE_RETURN_IF_ERROR(writer.Done());
    return buffer;
  }

  Result<SharedArray<uint8_t, 3>> DecodeImage(absl::Cord value) {
    riegeli::CordReader<> buffer_reader(&value);
    AvifReader reader;
    TENSORSTORE_RETURN_IF_ERROR(reader.Initialize(&buffer_reader));
    ImageInfo info = reader.GetImageInfo();
    if (info.dtype != dtype_v<uint8_t>) {
      return absl::UnimplementedError(
          "\"avif\" driver only supports uint8 images");
    }
    std::array<Index, 3> shape_yxc = {static_cast<Index>(info.height),
                                      static_cast<Index>(info.width),
                                      static_cast<Index>(info.num_components)};
    SharedArray<uint8_t, 3> array_yxc = AllocateArray<uint8_t>(shape_yxc);
    TENSORSTORE_RETURN_IF_ERROR(reader.Decode(tensorstore::span(
        reinterpret_cast<unsigned char*>(array_yxc.data()),
        array_yxc.num_elements() * array_yxc.dtype().size())));
    return array_yxc;
  }
};

const internal::DriverRegistration<ImageDriverSpec<AvifSpecialization>>
    avif_driver_registration;

}  // namespace
}  // namespace internal_image_driver
}  // namespace tensorstore
