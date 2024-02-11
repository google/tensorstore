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

#include <gtest/gtest.h>
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"

namespace {

using ::tensorstore::internal_zarr3::CodecRoundTripTestParams;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;

TEST(GzipTest, EndianInferred) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "gzip"}, {"configuration", {{"level", 7}}}},
  };
  p.expected_spec = {
      GetDefaultBytesCodecJson(),
      {{"name", "gzip"}, {"configuration", {{"level", 7}}}},
  };
  TestCodecSpecRoundTrip(p);
}

TEST(GzipTest, DefaultLevel) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "gzip"}},
  };
  p.expected_spec = {
      GetDefaultBytesCodecJson(),
      {{"name", "gzip"}, {"configuration", {{"level", 6}}}},
  };
  TestCodecSpecRoundTrip(p);
}

TEST(GzipTest, RoundTrip) {
  CodecRoundTripTestParams p;
  p.spec = {"gzip"};
  TestCodecRoundTrip(p);
}

}  // namespace
