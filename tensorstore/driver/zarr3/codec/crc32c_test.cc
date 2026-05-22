// Copyright 2026 The TensorStore Authors
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

#include <gtest/gtest.h>
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"

namespace {

using ::tensorstore::internal_zarr3::CodecRoundTripTestParams;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;

TEST(Crc32cTest, SpecRoundTrip) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "crc32c"}},
  };
  p.expected_spec = {
      tensorstore::internal_zarr3::GetDefaultBytesCodecJson(),
      {{"name", "crc32c"}},
  };
  TestCodecSpecRoundTrip(p);
}

TEST(Crc32cTest, RoundTrip) {
  CodecRoundTripTestParams p;
  p.spec = {"crc32c"};
  TestCodecRoundTrip(p);
}

}  // namespace
