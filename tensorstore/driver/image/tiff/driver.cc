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
#include "tensorstore/internal/image/tiff_reader.h"
#include "tensorstore/internal/image/tiff_writer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_image_driver {
namespace {

using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::TiffReader;
using ::tensorstore::internal_image::TiffWriter;

namespace jb = tensorstore::internal_json_binding;

// NOTE: There are quite a few improvements to be made to the tiff driver,
// such as:
// * The driver should allow reading "all" image pages, and
//   perhaps even listing them. This would require that they
//   all have the same dimensions and data types.
// * The driver should expose more than just uint8.

struct TiffReadOptions {
  // The TIFF directory to read.
  std::optional<int> page;
};

struct TiffSpecialization : public TiffReadOptions {
  constexpr static char id[] = "tiff";
  constexpr static char kTransactionError[] =
      "\"tiff\" driver does not support transactions";

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.page);
  };

  constexpr static auto default_json_binder =
      jb::Member("page", jb::Projection(&TiffReadOptions::page));

  Result<absl::Cord> EncodeImage(ArrayView<const void, 3> array_yxc) const {
    if (page != 1) {
      return absl::InvalidArgumentError(
          "\"tiff\" driver cannot write to specified page");
    }

    auto shape_yxc = array_yxc.shape();
    ImageInfo info{/*.height =*/static_cast<int32_t>(shape_yxc[0]),
                   /*.width =*/static_cast<int32_t>(shape_yxc[1]),
                   /*.num_components =*/static_cast<int32_t>(shape_yxc[2])};

    absl::Cord output;
    riegeli::CordWriter riegeli_writer(&output);

    TiffWriter writer;
    TENSORSTORE_RETURN_IF_ERROR(writer.Initialize(&riegeli_writer));
    TENSORSTORE_RETURN_IF_ERROR(writer.Encode(
        info, tensorstore::span(
                  reinterpret_cast<const unsigned char*>(array_yxc.data()),
                  array_yxc.num_elements() * array_yxc.dtype().size())));
    TENSORSTORE_RETURN_IF_ERROR(writer.Done());
    return output;
  }

  Result<SharedArray<uint8_t, 3>> DecodeImage(absl::Cord value) {
    SharedArray<uint8_t, 3> array_yxc;
    auto status = [&]() -> absl::Status {
      riegeli::CordReader<> buffer_reader(&value);
      TiffReader reader;

      TENSORSTORE_RETURN_IF_ERROR(reader.Initialize(&buffer_reader));

      if (page.has_value()) {
        TENSORSTORE_RETURN_IF_ERROR(reader.SeekFrame(*page));
      } else if (reader.GetFrameCount() > 1) {
        // TIFF files often have embedded thumbnails, etc. This driver doesn't
        // attempt to guess which pages are the correct one.
        return absl::DataLossError(
            "Multi-page TIFF image encountered without a \"page\" specifier. ");
      }
      ImageInfo info = reader.GetImageInfo();
      if (info.dtype != dtype_v<uint8_t>) {
        return absl::UnimplementedError(
            "\"tiff\" driver only supports uint8 images");
      }
      std::array<Index, 3> shape_yxc = {
          static_cast<Index>(info.height), static_cast<Index>(info.width),
          static_cast<Index>(info.num_components)};
      array_yxc = AllocateArray<uint8_t>(shape_yxc);
      return reader.Decode(tensorstore::span(
          reinterpret_cast<unsigned char*>(array_yxc.data()),
          array_yxc.num_elements() * array_yxc.dtype().size()));
    }();
    if (!status.ok()) {
      if (status.code() == absl::StatusCode::kInvalidArgument) {
        return internal::MaybeConvertStatusTo(std::move(status),
                                              absl::StatusCode::kDataLoss);
      }
      return status;
    }
    return array_yxc;
  }
};

const internal::DriverRegistration<ImageDriverSpec<TiffSpecialization>>
    tiff_driver_registration;

}  // namespace
}  // namespace internal_image_driver
}  // namespace tensorstore
