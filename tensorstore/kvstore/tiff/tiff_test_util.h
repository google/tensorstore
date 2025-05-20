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

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_TEST_UTIL_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_TEST_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tensorstore {
namespace internal_tiff_kvstore {
namespace testing {

// Helper class for building test TIFF files
class TiffBuilder {
 public:
  TiffBuilder();

  // Start an IFD with specified number of entries
  TiffBuilder& StartIfd(uint16_t num_entries);

  // Add an IFD entry
  TiffBuilder& AddEntry(uint16_t tag, uint16_t type, uint32_t count,
                        uint32_t value);

  // End the current IFD and point to the next one at specified offset
  // Use 0 for no next IFD
  TiffBuilder& EndIfd(uint32_t next_ifd_offset = 0);

  // Add external uint32_t array data
  TiffBuilder& AddUint32Array(const std::vector<uint32_t>& values);

  // Add external uint16_t array data
  TiffBuilder& AddUint16Array(const std::vector<uint16_t>& values);

  // Pad to a specific offset
  TiffBuilder& PadTo(size_t offset);

  // Get the final TIFF data
  std::string Build() const;

  size_t CurrentOffset() const { return data_.size(); }

  std::string data_;
};

// Little‑endian byte helper functions
void PutLE16(std::string& dst, uint16_t v);
void PutLE32(std::string& dst, uint32_t v);

std::string MakeTinyTiledTiff();

std::string MakeTinyStripedTiff();

std::string MakeTwoStripedTiff();

std::string MakeReadOpTiff();

std::string MakeMalformedTiff();

std::string MakeMultiIfdTiff();

std::string MakeTiffMissingHeight();

}  // namespace testing
}  // namespace internal_tiff_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_TEST_UTIL_H_