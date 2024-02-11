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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_ZSTD_CODEC_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_ZSTD_CODEC_H_

#include <optional>

#include "absl/status/status.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr3 {

class ZstdCodecSpec : public ZarrBytesToBytesCodecSpec {
 public:
  struct Options {
    std::optional<int> level;
    std::optional<bool> checksum;
  };
  ZstdCodecSpec() = default;
  explicit ZstdCodecSpec(const Options& options) : options(options) {}
  absl::Status MergeFrom(const ZarrCodecSpec& other, bool strict) override;
  ZarrCodecSpec::Ptr Clone() const override;
  Result<ZarrBytesToBytesCodec::Ptr> Resolve(
      BytesCodecResolveParameters&& decoded,
      BytesCodecResolveParameters& encoded,
      ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const final;

  Options options;
};

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_ZSTD_CODEC_H_
