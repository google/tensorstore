// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/tiff/tiff_test_util.h"

namespace tensorstore {
namespace internal_tiff_kvstore {
namespace testing {

TiffBuilder::TiffBuilder() {
  // Standard TIFF header
  data_ += "II";  // Little endian
  data_.push_back(42);
  data_.push_back(0);  // Magic number
  data_.push_back(8);
  data_.push_back(0);  // IFD offset (8)
  data_.push_back(0);
  data_.push_back(0);
}

TiffBuilder& TiffBuilder::StartIfd(uint16_t num_entries) {
  data_.push_back(num_entries & 0xFF);
  data_.push_back((num_entries >> 8) & 0xFF);
  return *this;
}

TiffBuilder& TiffBuilder::AddEntry(uint16_t tag, uint16_t type, uint32_t count,
                                   uint32_t value) {
  data_.push_back(tag & 0xFF);
  data_.push_back((tag >> 8) & 0xFF);
  data_.push_back(type & 0xFF);
  data_.push_back((type >> 8) & 0xFF);
  data_.push_back(count & 0xFF);
  data_.push_back((count >> 8) & 0xFF);
  data_.push_back((count >> 16) & 0xFF);
  data_.push_back((count >> 24) & 0xFF);
  data_.push_back(value & 0xFF);
  data_.push_back((value >> 8) & 0xFF);
  data_.push_back((value >> 16) & 0xFF);
  data_.push_back((value >> 24) & 0xFF);
  return *this;
}

TiffBuilder& TiffBuilder::EndIfd(uint32_t next_ifd_offset) {
  data_.push_back(next_ifd_offset & 0xFF);
  data_.push_back((next_ifd_offset >> 8) & 0xFF);
  data_.push_back((next_ifd_offset >> 16) & 0xFF);
  data_.push_back((next_ifd_offset >> 24) & 0xFF);
  return *this;
}

TiffBuilder& TiffBuilder::AddUint32Array(const std::vector<uint32_t>& values) {
  for (uint32_t val : values) {
    data_.push_back(val & 0xFF);
    data_.push_back((val >> 8) & 0xFF);
    data_.push_back((val >> 16) & 0xFF);
    data_.push_back((val >> 24) & 0xFF);
  }
  return *this;
}

TiffBuilder& TiffBuilder::AddUint16Array(const std::vector<uint16_t>& values) {
  for (uint16_t val : values) {
    data_.push_back(val & 0xFF);
    data_.push_back((val >> 8) & 0xFF);
  }
  return *this;
}

TiffBuilder& TiffBuilder::PadTo(size_t offset) {
  while (data_.size() < offset) {
    data_.push_back('X');
  }
  return *this;
}

std::string TiffBuilder::Build() const { return data_; }

void PutLE16(std::string& dst, uint16_t v) {
  dst.push_back(static_cast<char>(v & 0xff));
  dst.push_back(static_cast<char>(v >> 8));
}

void PutLE32(std::string& dst, uint32_t v) {
  dst.push_back(static_cast<char>(v & 0xff));
  dst.push_back(static_cast<char>(v >> 8));
  dst.push_back(static_cast<char>(v >> 16));
  dst.push_back(static_cast<char>(v >> 24));
}

std::string MakeTinyTiledTiff() {
  TiffBuilder builder;
  return builder
             .StartIfd(6)  // 6 entries
             .AddEntry(256, 3, 1, 256)
             .AddEntry(257, 3, 1, 256)  // width, length (256×256)
             .AddEntry(322, 3, 1, 256)
             .AddEntry(323, 3, 1, 256)  // tile width/length
             .AddEntry(324, 4, 1, 128)
             .AddEntry(325, 4, 1, 4)  // offset/bytecount
             .EndIfd()                // next IFD
             .PadTo(128)
             .Build() +
         "DATA";
}

std::string MakeTinyStripedTiff() {
  TiffBuilder builder;
  return builder
             .StartIfd(5)               // 5 entries
             .AddEntry(256, 3, 1, 4)    // ImageWidth = 4
             .AddEntry(257, 3, 1, 8)    // ImageLength = 8
             .AddEntry(278, 3, 1, 8)    // RowsPerStrip = 8
             .AddEntry(273, 4, 1, 128)  // StripOffsets = 128
             .AddEntry(279, 4, 1, 8)    // StripByteCounts = 8
             .EndIfd()                  // No more IFDs
             .PadTo(128)
             .Build() +
         "DATASTR!";
}

std::string MakeTwoStripedTiff() {
  TiffBuilder builder;
  return builder
             .StartIfd(6)               // 6 entries
             .AddEntry(256, 3, 1, 4)    // ImageWidth = 4
             .AddEntry(257, 3, 1, 8)    // ImageLength = 8
             .AddEntry(278, 3, 1, 4)    // RowsPerStrip = 4
             .AddEntry(273, 4, 2, 128)  // StripOffsets array at offset 128
             .AddEntry(279, 4, 2, 136)  // StripByteCounts array at offset 136
             .AddEntry(259, 3, 1, 1)    // Compression = none
             .EndIfd()                  // No more IFDs
             .PadTo(128)
             .AddUint32Array({200, 208})  // Strip offsets
             .PadTo(136)
             .AddUint32Array({4, 4})  // Strip byte counts
             .PadTo(200)
             .Build() +
         "AAAA" + std::string(4, '\0') + "BBBB";
}

std::string MakeReadOpTiff() {
  TiffBuilder builder;
  return builder
             .StartIfd(6)  // 6 entries
             .AddEntry(256, 3, 1, 16)
             .AddEntry(257, 3, 1, 16)  // width, length
             .AddEntry(322, 3, 1, 16)
             .AddEntry(323, 3, 1, 16)  // tile width/length
             .AddEntry(324, 4, 1, 128)
             .AddEntry(325, 4, 1, 16)  // offset/bytecount
             .EndIfd()                 // next IFD
             .PadTo(128)
             .Build() +
         "abcdefghijklmnop";
}

std::string MakeMalformedTiff() {
  std::string t;
  t += "MM";  // Bad endianness (motorola instead of intel)
  PutLE16(t, 42);
  PutLE32(t, 8);  // header
  PutLE16(t, 1);  // 1 IFD entry

  // Helper lambda for creating an entry
  auto E = [&](uint16_t tag, uint16_t type, uint32_t cnt, uint32_t val) {
    PutLE16(t, tag);
    PutLE16(t, type);
    PutLE32(t, cnt);
    PutLE32(t, val);
  };

  E(256, 3, 1, 16);  // Only width, missing other required tags
  PutLE32(t, 0);     // next IFD
  return t;
}

std::string MakeMultiIfdTiff() {
  TiffBuilder builder;
  return builder
             .StartIfd(6)  // 6 entries for first IFD
             .AddEntry(256, 3, 1, 256)
             .AddEntry(257, 3, 1, 256)  // width, length (256×256)
             .AddEntry(322, 3, 1, 256)
             .AddEntry(323, 3, 1, 256)  // tile width/length
             .AddEntry(324, 4, 1, 200)
             .AddEntry(325, 4, 1, 5)  // offset/bytecount for IFD 0
             .EndIfd(86)              // next IFD at offset 86
             .PadTo(86)               // pad to second IFD
             .StartIfd(6)             // 6 entries for second IFD
             .AddEntry(256, 3, 1, 128)
             .AddEntry(257, 3, 1, 128)  // width, length (128×128)
             .AddEntry(322, 3, 1, 128)
             .AddEntry(323, 3, 1, 128)  // tile width/length
             .AddEntry(324, 4, 1, 208)
             .AddEntry(325, 4, 1, 5)  // offset/bytecount for IFD 1
             .EndIfd()                // No more IFDs
             .PadTo(200)
             .Build() +
         "DATA1" + std::string(3, '\0') + "DATA2";
}

std::string MakeTiffMissingHeight() {
  std::string t;
  t += "II";  // Little endian
  PutLE16(t, 42);
  PutLE32(t, 8);  // header
  PutLE16(t, 1);  // 1 IFD entry

  // Helper lambda for creating an entry
  auto E = [&](uint16_t tag, uint16_t type, uint32_t cnt, uint32_t val) {
    PutLE16(t, tag);
    PutLE16(t, type);
    PutLE32(t, cnt);
    PutLE32(t, val);
  };

  E(256, 3, 1, 16);  // Width but no Height
  PutLE32(t, 0);     // next IFD
  return t;
}

}  // namespace testing
}  // namespace internal_tiff_kvstore
}  // namespace tensorstore
