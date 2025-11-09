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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_BYTES_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_BYTES_H_

#include <optional>

#include "absl/status/status.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

class BytesCodecSpec : public ZarrArrayToBytesCodecSpec {
 public:
  struct Options {
    std::optional<endian> endianness;
    // Indicates whether this spec should be validated by `Resolve` according to
    // the (looser) requirements for codecs specified as part of metadata
    // constraints, rather than the stricter rules for codecs specified in the
    // actual stored metadata.
    bool constraints = false;
  };
  BytesCodecSpec() = default;
  explicit BytesCodecSpec(const Options& options) : options(options) {}

  absl::Status MergeFrom(const ZarrCodecSpec& other, bool strict) override;
  ZarrCodecSpec::Ptr Clone() const override;

  absl::Status GetDecodedChunkLayout(
      const ArrayDataTypeAndShapeInfo& array_info,
      ArrayCodecChunkLayoutInfo& decoded) const override;

  bool SupportsInnerOrder(
      const ArrayCodecResolveParameters& decoded,
      span<DimensionIndex> preferred_inner_order) const override;

  Result<ZarrArrayToBytesCodec::Ptr> Resolve(
      ArrayCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrArrayToBytesCodecSpec::Ptr* resolved_spec) const override;

  Options options;
};

/// Returns a `BytesCodecSpec` with native endianness.
internal::IntrusivePtr<const BytesCodecSpec> DefaultBytesCodec();

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_BYTES_H_
