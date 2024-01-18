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

#include "tensorstore/driver/zarr3/codec/codec_test_util.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal_zarr3 {

void TestCodecSpecRoundTrip(const CodecSpecRoundTripTestParams& params) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec_chain_spec,
      ZarrCodecChainSpec::FromJson(params.orig_spec, params.from_json_options));

  // Test that self-merging is a no-op.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto orig_json, codec_chain_spec.ToJson());
    auto codec_chain_spec_copy = codec_chain_spec;
    TENSORSTORE_ASSERT_OK(
        codec_chain_spec_copy.MergeFrom(codec_chain_spec, /*strict=*/true));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto merged_json,
                                     codec_chain_spec_copy.ToJson());
    EXPECT_THAT(merged_json, MatchesJson(orig_json));
  }

  ArrayCodecResolveParameters decoded_params = params.resolve_params;
  if (!decoded_params.fill_value.valid()) {
    decoded_params.fill_value = AllocateArray(span<const Index>{}, c_order,
                                              value_init, decoded_params.dtype);
  }
  BytesCodecResolveParameters encoded_params;

  ZarrCodecChainSpec roundtrip_codec_chain_spec;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec_chain,
      codec_chain_spec.Resolve(std::move(decoded_params), encoded_params,
                               &roundtrip_codec_chain_spec));
  EXPECT_THAT(roundtrip_codec_chain_spec.ToJson(params.to_json_options),
              ::testing::Optional(MatchesJson(
                  params.expected_spec.is_discarded() ? params.orig_spec
                                                      : params.expected_spec)));
}

Result<::nlohmann::json> TestCodecSpecResolve(
    ::nlohmann::json json_spec, ArrayCodecResolveParameters resolve_params,
    bool constraints) {
  ZarrCodecChainSpec::FromJsonOptions from_json_options{constraints};
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto codec_chain_spec,
      ZarrCodecChainSpec::FromJson(json_spec, from_json_options));
  if (!resolve_params.fill_value.valid()) {
    resolve_params.fill_value = AllocateArray(span<const Index>{}, c_order,
                                              value_init, resolve_params.dtype);
  }
  BytesCodecResolveParameters encoded_params;
  ZarrCodecChainSpec resolved_chain_spec;
  TENSORSTORE_RETURN_IF_ERROR(codec_chain_spec.Resolve(
      std::move(resolve_params), encoded_params, &resolved_chain_spec));
  return resolved_chain_spec.ToJson();
}

void TestCodecRoundTrip(const CodecRoundTripTestParams& params) {
  ZarrCodecChainSpec::FromJsonOptions from_json_options{/*.constraints=*/true};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec_chain_spec,
      ZarrCodecChainSpec::FromJson(params.spec, from_json_options));
  ArrayCodecResolveParameters decoded_params;
  decoded_params.rank = params.shape.size();
  decoded_params.dtype = params.dtype;
  decoded_params.fill_value = AllocateArray(span<const Index>{}, c_order,
                                            value_init, decoded_params.dtype);
  BytesCodecResolveParameters encoded_params;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec_chain,
      codec_chain_spec.Resolve(std::move(decoded_params), encoded_params));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto prepared_state,
                                   codec_chain->Prepare(params.shape));
  absl::BitGen gen;
  auto data =
      internal::MakeRandomArray(gen, params.shape, params.dtype, c_order);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   prepared_state->EncodeArray(data));
  EXPECT_THAT(prepared_state->DecodeArray(params.shape, encoded),
              ::testing::Optional(MatchesArrayIdentically(data)))
      << "data=" << data;
}

Result<::nlohmann::json> TestCodecMerge(::nlohmann::json a, ::nlohmann::json b,
                                        bool strict) {
  ZarrCodecChainSpec::FromJsonOptions from_json_options{/*.constraints=*/true};
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto a_spec, ZarrCodecChainSpec::FromJson(a, from_json_options));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto b_spec, ZarrCodecChainSpec::FromJson(b, from_json_options));
  auto a_spec_copy = a_spec;
  auto b_spec_copy = b_spec;
  TENSORSTORE_RETURN_IF_ERROR(a_spec_copy.MergeFrom(b_spec_copy, strict));

  // Try reverse merge.
  TENSORSTORE_RETURN_IF_ERROR(b_spec.MergeFrom(a_spec, strict));

  TENSORSTORE_ASSIGN_OR_RETURN(auto merged_json1, a_spec_copy.ToJson());
  TENSORSTORE_ASSIGN_OR_RETURN(auto merged_json2, b_spec.ToJson());

  EXPECT_THAT(merged_json1, MatchesJson(merged_json2));

  return merged_json1;
}

::nlohmann::json GetDefaultBytesCodecJson() {
  return {{"name", "bytes"},
          {"configuration",
           {{"endian", endian::native == endian::little ? "little" : "big"}}}};
}

}  // namespace internal_zarr3
}  // namespace tensorstore
