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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_TEST_UTIL_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_TEST_UTIL_H_

#include <stdint.h>

#include <vector>

#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr3 {

struct CodecSpecRoundTripTestParams {
  ::nlohmann::json orig_spec;
  ::nlohmann::json expected_spec = ::nlohmann::json::value_t::discarded;
  ZarrCodecChainSpec::FromJsonOptions from_json_options{/*.constraints=*/true};
  ZarrCodecChainSpec::ToJsonOptions to_json_options;
  ArrayCodecResolveParameters resolve_params = {
      dtype_v<uint16_t>,
      /*.rank=*/0,
  };
};

void TestCodecSpecRoundTrip(const CodecSpecRoundTripTestParams& params);

Result<::nlohmann::json> TestCodecSpecResolve(
    ::nlohmann::json json_spec, ArrayCodecResolveParameters resolve_params,
    bool constraints = true);

struct CodecRoundTripTestParams {
  ::nlohmann::json spec;
  std::vector<Index> shape{30, 40, 50};
  DataType dtype = dtype_v<uint16_t>;
};

void TestCodecRoundTrip(const CodecRoundTripTestParams& params);

Result<::nlohmann::json> TestCodecMerge(::nlohmann::json a, ::nlohmann::json b,
                                        bool strict);

::nlohmann::json GetDefaultBytesCodecJson();

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_CODEC_TEST_UTIL_H_
