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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;

using ::tensorstore::MatchesStatus;
using tensorstore::internal::JoinPath;

class TestData : public tensorstore::internal::ScopedTemporaryDirectory {
 public:
  std::string OffsetTileTiff() {
    static constexpr unsigned char data[] = {
        0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x01, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x01,
        0x02, 0x00, 0x17, 0x00, 0x00, 0x00, 0xc2, 0x00, 0x00, 0x00, 0x15, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1a, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xe8, 0x00, 0x00, 0x00, 0x1b, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x28, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x31, 0x01,
        0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00, 0x42, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x44, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x45, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x7b, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x20,
        0x5b, 0x31, 0x30, 0x2c, 0x20, 0x31, 0x35, 0x2c, 0x20, 0x31, 0x5d, 0x7d,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x74, 0x69, 0x66, 0x66,
        0x66, 0x69, 0x6c, 0x65, 0x2e, 0x70, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x00,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
        0x1b, 0x1c, 0x1d, 0x00, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
        0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x00, 0x2d, 0x2e, 0x2f, 0x30,
        0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x00,
        0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
        0x48, 0x49, 0x4a, 0x00, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52,
        0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x00, 0x5a, 0x5b, 0x5c, 0x5d,
        0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x00,
        0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74,
        0x75, 0x76, 0x77, 0x00, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x00, 0x87, 0x88, 0x89, 0x8a,
        0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    auto p = JoinPath(path(), "tiled.tiff");
    std::ofstream ofs(p);
    ofs.write(reinterpret_cast<const char*>(data), sizeof(data));
    return p;
  }

  std::string OffsetStripTiff() {
    static constexpr unsigned char data[] = {
        0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x01, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x01,
        0x02, 0x00, 0x17, 0x00, 0x00, 0x00, 0xb6, 0x00, 0x00, 0x00, 0x11, 0x01,
        0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0xdc, 0x00, 0x00, 0x00, 0x15, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x16, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x17, 0x01,
        0x03, 0x00, 0x04, 0x00, 0x00, 0x00, 0xec, 0x00, 0x00, 0x00, 0x1a, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0x1b, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xfc, 0x00, 0x00, 0x00, 0x28, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x31, 0x01,
        0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x7b, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x20,
        0x5b, 0x31, 0x30, 0x2c, 0x20, 0x31, 0x35, 0x2c, 0x20, 0x31, 0x5d, 0x7d,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x3d, 0x01, 0x00, 0x00,
        0x6a, 0x01, 0x00, 0x00, 0x97, 0x01, 0x00, 0x00, 0x2d, 0x00, 0x2d, 0x00,
        0x2d, 0x00, 0x0f, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x74, 0x69, 0x66, 0x66,
        0x66, 0x69, 0x6c, 0x65, 0x2e, 0x70, 0x79, 0x00, 0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
        0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33,
        0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b,
        0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
        0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63,
        0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b,
        0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93,
        0x94, 0x95};

    auto p = JoinPath(path(), "strip.tiff");
    std::ofstream ofs(p);
    ofs.write(reinterpret_cast<const char*>(data), sizeof(data));
    return p;
  }

  std::string ZSTDUint16TileTiff() {
    static constexpr unsigned char data[] = {
        0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x02, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x03, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x50, 0xc3, 0x00, 0x00, 0x06, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x01,
        0x02, 0x00, 0x17, 0x00, 0x00, 0x00, 0xc2, 0x00, 0x00, 0x00, 0x15, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1a, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xe8, 0x00, 0x00, 0x00, 0x1b, 0x01,
        0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x28, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x31, 0x01,
        0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00, 0x42, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x44, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x45, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x61, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x7b, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x20,
        0x5b, 0x31, 0x36, 0x2c, 0x20, 0x31, 0x36, 0x2c, 0x20, 0x31, 0x5d, 0x7d,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x74, 0x69, 0x66, 0x66,
        0x66, 0x69, 0x6c, 0x65, 0x2e, 0x70, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0xb5, 0x2f, 0xfd,
        0x60, 0x00, 0x01, 0xbd, 0x0a, 0x00, 0x06, 0xe0, 0x54, 0x0a, 0x10, 0xf8,
        0x6c, 0x07, 0xff, 0xff, 0x3f, 0x5a, 0x32, 0x05, 0x4f, 0x00, 0x51, 0x00,
        0x51, 0x00, 0x7b, 0xe4, 0x71, 0x47, 0x1d, 0x73, 0xc4, 0xf1, 0x46, 0x1b,
        0x6b, 0xa4, 0x71, 0x46, 0x19, 0x63, 0x84, 0xf1, 0x45, 0x17, 0x5b, 0x64,
        0x71, 0x45, 0x15, 0x53, 0x44, 0xf1, 0x44, 0x13, 0x4b, 0x24, 0x71, 0x44,
        0x11, 0x43, 0x04, 0xf1, 0x43, 0x0f, 0x3b, 0xe4, 0x70, 0x43, 0x0d, 0x33,
        0xc4, 0xf0, 0x42, 0x0b, 0x2b, 0xa4, 0x70, 0x42, 0x09, 0x23, 0x84, 0xf0,
        0x41, 0x07, 0x1b, 0x64, 0x70, 0x41, 0x05, 0x13, 0x44, 0xf0, 0x40, 0x03,
        0x0b, 0x24, 0x70, 0x40, 0x01, 0x03, 0x04, 0xf0, 0xef, 0xfb, 0xe4, 0x73,
        0x4f, 0x3d, 0xf3, 0xc4, 0xf3, 0x4e, 0x3b, 0xeb, 0xa4, 0x73, 0x4e, 0x39,
        0xe3, 0x84, 0xf3, 0x4d, 0x37, 0xdb, 0x64, 0x73, 0x4d, 0x35, 0xd3, 0x44,
        0xf3, 0x4c, 0x33, 0xcb, 0x24, 0x73, 0x4c, 0x31, 0xc3, 0x04, 0xf3, 0x4b,
        0x2f, 0xbb, 0xe4, 0x72, 0x4b, 0x2d, 0xb3, 0xc4, 0xf2, 0x4a, 0x2b, 0xab,
        0xa4, 0x72, 0x4a, 0x29, 0xa3, 0x84, 0xf2, 0x49, 0x27, 0x9b, 0x64, 0x72,
        0x49, 0x25, 0x93, 0x44, 0xf2, 0x48, 0x23, 0x8b, 0x24, 0x72, 0x48, 0x21,
        0x83, 0x04, 0xf2, 0x47, 0x1f, 0x01, 0x7b, 0xe5, 0x75, 0x57, 0x5d, 0x73,
        0xc5, 0xf5, 0x56, 0x5b, 0x6b, 0xa5, 0x75, 0x56, 0x59, 0x63, 0x85, 0xf5,
        0x55, 0x57, 0x5b, 0x65, 0x75, 0x55, 0x55, 0x53, 0x45, 0xf5, 0x54, 0x53,
        0x4b, 0x25, 0x75, 0x54, 0x51, 0x43, 0x05, 0xf5, 0x53, 0x4f, 0x3b, 0xe5,
        0x74, 0x53, 0x4d, 0x33, 0xc5, 0xf4, 0x52, 0x4b, 0x2b, 0xa5, 0x74, 0x52,
        0x49, 0x23, 0x85, 0xf4, 0x51, 0x47, 0x1b, 0x65, 0x74, 0x51, 0x45, 0x13,
        0x45, 0xf4, 0x50, 0x43, 0x0b, 0x25, 0x74, 0x50, 0x41, 0x03, 0x05, 0xf4,
        0x4f, 0x3f, 0x01, 0xfb, 0xe5, 0x77, 0x5f, 0x7d, 0xf3, 0xc5, 0xf7, 0x5e,
        0x7b, 0xeb, 0xa5, 0x77, 0x5e, 0x79, 0xe3, 0x85, 0xf7, 0x5d, 0x77, 0xdb,
        0x65, 0x77, 0x5d, 0x75, 0xd3, 0x45, 0xf7, 0x5c, 0x73, 0xcb, 0x25, 0x77,
        0x5c, 0x71, 0xc3, 0x05, 0xf7, 0x5b, 0x6f, 0xbb, 0xe5, 0x76, 0x5b, 0x6d,
        0xb3, 0xc5, 0xf6, 0x5a, 0x6b, 0xab, 0xa5, 0x76, 0x5a, 0x69, 0xa3, 0x85,
        0xf6, 0x59, 0x67, 0x9b, 0x65, 0x76, 0x59, 0x65, 0x93, 0x45, 0xf6, 0x58,
        0x63, 0x8b, 0x25, 0x76, 0x58, 0x61, 0x83, 0x05, 0xf6, 0x57, 0x5f, 0x01,
        0x00};

    auto p = JoinPath(path(), "tile.tiff");
    std::ofstream ofs(p);
    ofs.write(reinterpret_cast<const char*>(data), sizeof(data));
    return p;
  }
};

::nlohmann::json GetFileSpec(std::string path) {
  return ::nlohmann::json{{"driver", "ometiff"},
                          {"kvstore", {{"driver", "file"}, {"path", path}}},
                          {"cache_pool", {{"total_bytes_limit", 100000000}}},
                          {"data_copy_concurrency", {{"limit", 1}}}};
}

template <typename Array>
void PrintCSVArray(Array&& data) {
  if (data.rank() == 0) {
    std::cout << data << std::endl;
    return;
  }

  // Iterate over the shape of the data array, which gives us one
  // reference for every element.
  //
  // The builtin streaming operator outputs data in C++ array initialization
  // syntax: {{0, 0}, {1, 0}}, but this routine prefers CSV-formatted output.
  //
  // The output of this function is equivalent to:
  //
  // for (int x = 0; x < data.shape()[0]; x++)
  //  for (int y = 0; y < data.shape()[1]; y++) {
  //     ...
  //       std::cout << data[x][y][...] << "\t";
  //  }
  //
  const auto max = data.shape()[data.rank() - 1] - 1;
  auto element_rep = data.dtype();

  // FIXME: We can't use operator() to get a value reference since that doesn't
  // work for tensorstore::ArrayView<const void, N>. However in the case of
  // printing, rank-0 arrays have been overloaded to print correctly, and so we
  // can do this:
  std::string s;
  tensorstore::IterateOverIndexRange(  //
      data.shape(), [&](tensorstore::span<const tensorstore::Index> idx) {
        element_rep->append_to_string(&s, data[idx].pointer());
        if (*idx.rbegin() == max) {
          std::cout << s << std::endl;
          s.clear();
        } else {
          s.append(" ");
        }
      });
  std::cout << s << std::endl;
}

TEST(OMETiffDriverTest, BasicTile) {
  TestData test_data;
  auto path = test_data.OffsetTileTiff();

  std::vector<uint8_t> expected_data(10 * 15);
  std::iota(expected_data.begin(), expected_data.end(), 0);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(GetFileSpec(path)).result());
  EXPECT_TRUE(!!store.base());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());
  EXPECT_THAT(array.shape(), ::testing::ElementsAre(10, 15));

  // Not sure how to reshape expected_data...there has to be an easier way.
  std::vector<uint8_t> data(array.num_elements());
  std::copy(static_cast<uint8_t*>(array.data()),
            static_cast<uint8_t*>(array.data()) + array.num_elements(),
            data.data());
  EXPECT_EQ(data, expected_data);
}

TEST(OMETiffDriverTest, BasicStrip) {
  TestData test_data;
  auto path = test_data.OffsetStripTiff();

  std::vector<uint8_t> expected_data(10 * 15);
  std::iota(expected_data.begin(), expected_data.end(), 0);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(GetFileSpec(path)).result());
  EXPECT_TRUE(!!store.base());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());
  EXPECT_THAT(array.shape(), ::testing::ElementsAre(10, 15));

  // Not sure how to reshape expected_data...there has to be an easier way.
  std::vector<uint8_t> data(array.num_elements());
  std::copy(static_cast<uint8_t*>(array.data()),
            static_cast<uint8_t*>(array.data()) + array.num_elements(),
            data.data());
  EXPECT_EQ(data, expected_data);
}

TEST(OMETiffDriverTest, ZSTD) {
  TestData test_data;
  auto path = test_data.ZSTDUint16TileTiff();

  std::vector<uint16_t> expected_data(16 * 16);
  std::iota(expected_data.begin(), expected_data.end(), 0);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(GetFileSpec(path)).result());
  EXPECT_TRUE(!!store.base());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());
  std::vector<uint16_t> data(array.num_elements());
  std::copy(static_cast<uint16_t*>(array.data()),
            static_cast<uint16_t*>(array.data()) + array.num_elements(),
            data.data());
  EXPECT_EQ(data, expected_data);
}

TEST(OMETiffDriverTest, ZSTDMultiTile32Bit) {
  std::vector<uint32_t> expected_data(48 * 32);
  std::iota(expected_data.begin(), expected_data.end(), 0);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          GetFileSpec(
              "/Users/hsidky/Code/tensorstore/"
              "tensorstore/driver/ometiff/testdata/multitile_32bit.tiff"))
          .result());
  EXPECT_TRUE(!!store.base());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());
  std::vector<uint32_t> data(array.num_elements());
  std::copy(static_cast<uint32_t*>(array.data()),
            static_cast<uint32_t*>(array.data()) + array.num_elements(),
            data.data());
  EXPECT_EQ(data, expected_data);
}

TEST(OMETiffDriverTest, ZSTDMultiStrip32Bit) {
  std::vector<uint32_t> expected_data(48 * 32);
  std::iota(expected_data.begin(), expected_data.end(), 0);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          GetFileSpec(
              "/Users/hsidky/Code/tensorstore/"
              "tensorstore/driver/ometiff/testdata/multistrip_32bit.tiff"))
          .result());
  EXPECT_TRUE(!!store.base());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());
  std::vector<uint32_t> data(array.num_elements());
  std::copy(static_cast<uint32_t*>(array.data()),
            static_cast<uint32_t*>(array.data()) + array.num_elements(),
            data.data());
  EXPECT_EQ(data, expected_data);
}

}  // namespace