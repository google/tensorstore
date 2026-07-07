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

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zip_kvstore {

// Parses each central directory record sequentially, validates entries,
// and computes the local entry sizes using the offsets between headers.
Result<CachedDir> DecodeDirectoryEntries(riegeli::Reader& reader,
                                         size_t num_entries,
                                         int64_t cd_offset) {
  CachedDir dir{};
  dir.entries.reserve(num_entries);
  absl::flat_hash_set<std::string> seen_filenames;
  for (size_t i = 0; i < num_entries; ++i) {
    internal_zip::ZipEntry entry{};
    TENSORSTORE_RETURN_IF_ERROR(ReadCentralDirectoryEntry(reader, entry));
    auto [unused_it, inserted] = seen_filenames.insert(entry.filename);
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate filename in ZIP: %s", entry.filename));
    }
    // Only add validated entries to the zip directory.
    if (ValidateEntryIsSupported(entry).ok()) {
      dir.entries.push_back(CachedDir::Entry{entry});
    }
  }

  // Sort by local header offset first, then by name, to determine
  // the size of the encoded local header + compressed data for each entry.
  // Typically a ZIP file will already be ordered like this. Subsequently, the
  // gap between the headers is used to determine how many bytes to read.
  std::sort(dir.entries.begin(), dir.entries.end(),
            [](const auto& a, const auto& b) {
              return std::tie(a.local_header_offset, a.filename) <
                     std::tie(b.local_header_offset, b.filename);
            });
  auto last_header_offset = cd_offset;
  for (auto it = dir.entries.rbegin(); it != dir.entries.rend(); ++it) {
    it->local_header_and_data_size =
        last_header_offset - it->local_header_offset;
    last_header_offset = it->local_header_offset;
  }

  // Sort directory by filename.
  std::sort(dir.entries.begin(), dir.entries.end(),
            [](const auto& a, const auto& b) {
              return std::tie(a.filename, a.local_header_offset) <
                     std::tie(b.filename, b.local_header_offset);
            });

  return dir;
}

}  // namespace internal_zip_kvstore
}  // namespace tensorstore
