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

#include "tensorstore/kvstore/zip/cached_dir.h"

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/compression/zip_easy.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Result;
using ::tensorstore::internal_zip::EasyZipWriter;
using ::tensorstore::internal_zip::TryReadFullEOCD;
using ::tensorstore::internal_zip::ZipCompression;
using ::tensorstore::internal_zip::ZipEntry;
using ::tensorstore::internal_zip::ZipEOCD;
using ::tensorstore::internal_zip_kvstore::CachedDir;
using ::tensorstore::internal_zip_kvstore::DecodeDirectoryEntries;

// Decodes the ZIP central directory using the complete in-memory ZIP data.
tensorstore::Result<CachedDir> DecodeDirectory(const absl::Cord& zip_data) {
  riegeli::CordReader reader(&zip_data);
  ZipEOCD eocd;
  auto read_eocd_variant =
      TryReadFullEOCD(reader, eocd, /*offset_adjustment=*/0);
  if (auto* status = std::get_if<absl::Status>(&read_eocd_variant);
      status != nullptr && !status->ok()) {
    return std::move(*status);
  }
  if (std::holds_alternative<int64_t>(read_eocd_variant)) {
    return absl::InvalidArgumentError("Failed to read full EOCD");
  }

  if (!reader.Seek(eocd.cd_offset)) {
    return absl::InvalidArgumentError("Failed to seek to Central Directory");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      CachedDir dir,
      DecodeDirectoryEntries(reader, eocd.num_entries, eocd.cd_offset));

  // Full archive is available in zip_data, so seek-based reads are valid.
  dir.full_read = true;
  return dir;
}

TEST(CachedDirTest, DecodeDirectorySortingAndLocalHeaderAndDataSize) {
  absl::Cord zip_data;
  ZipEntry entry1;
  ZipEntry entry2;
  ZipEOCD eocd;
  {
    riegeli::CordWriter writer(&zip_data);
    EasyZipWriter zip_writer(writer);

    entry1.filename = "z_file";
    entry1.compression_method = ZipCompression::kStore;
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry1, absl::Cord("hello")));

    entry2.filename = "a_file";
    entry2.compression_method = ZipCompression::kStore;
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry2, absl::Cord("world")));

    TENSORSTORE_ASSERT_OK(zip_writer.Finalize(&eocd));
    ASSERT_TRUE(writer.Close());
  }

  // Decode the directory.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(CachedDir dir, DecodeDirectory(zip_data));

  // Entries must be sorted by filename.
  ASSERT_EQ(dir.entries.size(), 2);
  EXPECT_EQ(dir.entries[0].filename, "a_file");
  EXPECT_EQ(dir.entries[1].filename, "z_file");

  // EasyZipWriter populates entry.local_header_offset during WriteEntry.
  // "z_file" starts at offset 0, and ends at start of next file ("a_file").
  // So its local_header_and_data_size is key2 offset - key1 offset:
  EXPECT_EQ(dir.entries[1].local_header_and_data_size,
            entry2.local_header_offset);

  // "a_file" starts at entry2 offset, and ends at cd_offset
  // (EOCD/CD boundary).
  // So its local_header_and_data_size is cd_offset - key2 offset:
  EXPECT_EQ(dir.entries[0].local_header_and_data_size,
            eocd.cd_offset - entry2.local_header_offset);
}

TEST(CachedDirTest, DecodeInvalidZip) {
  absl::Cord bad_data("Definitely not a ZIP file.");
  auto result = DecodeDirectory(bad_data);
  EXPECT_FALSE(result.ok());
}

}  // namespace
