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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_TRANSPOSE_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_TRANSPOSE_H_

#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr3 {

class TransposeCodecSpec : public ZarrArrayToArrayCodecSpec {
 public:
  using Order =
      std::variant<std::vector<DimensionIndex>, ContiguousLayoutOrder>;
  struct Options {
    // order[encoded_dim] specifies the corresponding decoded dim.  `c_order` is
    // equivalent to specifying `0, 1, ..., rank-1`.  `fortran_order` is
    // equivalent to specifying `rank-1, ..., 1, 0`.
    Order order;
  };
  TransposeCodecSpec() = default;
  explicit TransposeCodecSpec(Options&& options)
      : options(std::move(options)) {}

  absl::Status MergeFrom(const ZarrCodecSpec& other, bool strict) override;
  ZarrCodecSpec::Ptr Clone() const override;

  absl::Status PropagateDataTypeAndShape(
      const ArrayDataTypeAndShapeInfo& decoded,
      ArrayDataTypeAndShapeInfo& encoded) const override;

  absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& encoded_info,
      const ArrayCodecChunkLayoutInfo& encoded,
      const ArrayDataTypeAndShapeInfo& decoded_info,
      ArrayCodecChunkLayoutInfo& decoded) const override;

  Result<ZarrArrayToArrayCodec::Ptr> Resolve(
      ArrayCodecResolveParameters&& decoded,
      ArrayCodecResolveParameters& encoded,
      ZarrArrayToArrayCodecSpec::Ptr* resolved_spec) const override;

  Options options;
};

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_ENDIAN_H_
