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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_ZIP_EASY_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_ZIP_EASY_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zip {

/// Helper class for reading a ZIP file.
class EasyZipReader {
 public:
  explicit EasyZipReader(riegeli::Reader& reader ABSL_ATTRIBUTE_LIFETIME_BOUND,
                         bool validate_filename = true);

  /// Initializes the directory by reading the EOCD and central directory.
  absl::Status Initialize();

  /// Gets the directory entries, initializing if necessary.
  Result<tensorstore::span<const ZipEntry>> entries()
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

  /// Decompresses and returns the contents of a ZIP entry.
  Result<absl::Cord> ReadEntry(ZipEntry& entry);

  /// Finds and decompresses an entry by filename.
  Result<absl::Cord> ReadEntry(std::string_view filename);

 private:
  riegeli::Reader& reader_;
  ZipEOCD eocd_;
  std::vector<ZipEntry> entries_;
  bool initialized_ = false;
  bool validate_filename_ = true;
};

/// Helper class for writing a ZIP file.
class EasyZipWriter {
 public:
  explicit EasyZipWriter(riegeli::Writer& writer ABSL_ATTRIBUTE_LIFETIME_BOUND);

  /// Writes a configured ZIP entry. The entry values are updated with sizes,
  /// CRC, offsets, and metadata.
  absl::Status WriteEntry(ZipEntry& entry, const absl::Cord& data);

  absl::Status WriteEntry(
      const std::string& filename, const absl::Cord& data,
      ZipCompression compression_method = ZipCompression::kStore,
      absl::Time mtime = absl::Now(), std::string comment = "");

  /// Writes the Central Directory and EOCD record.
  /// If `eocd` is provided, its fields are updated and written.
  ///
  /// Note: Unlike EasyZipReader, this does NOT close the underlying writer.
  /// The caller must ensure that the writer is closed after Finalize.
  absl::Status Finalize(ZipEOCD* eocd = nullptr);

 private:
  riegeli::Writer& writer_;
  ZipEOCD eocd_;
  std::vector<ZipEntry> entries_;
};

}  // namespace internal_zip
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_ZIP_EASY_H_
