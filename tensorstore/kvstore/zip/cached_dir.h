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

#ifndef TENSORSTORE_KVSTORE_ZIP_CACHED_DIR_H_
#define TENSORSTORE_KVSTORE_ZIP_CACHED_DIR_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <vector>

#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zip_kvstore {

struct CachedDir {
  struct Entry : public internal_zip::ZipEntry {
    uint64_t local_header_and_data_size;
    // Lazily populated from local header reads; mutable because it is
    // updated through const ReadLock access.
    mutable std::atomic<uint64_t> local_header_size{0};

    Entry() = default;
    Entry(const internal_zip::ZipEntry& base,
          uint64_t local_header_and_data_size = 0)
        : internal_zip::ZipEntry(base),
          local_header_and_data_size(local_header_and_data_size),
          local_header_size(0) {}

    Entry(const Entry& other)
        : internal_zip::ZipEntry(
              static_cast<const internal_zip::ZipEntry&>(other)),
          local_header_and_data_size(other.local_header_and_data_size),
          local_header_size(
              other.local_header_size.load(std::memory_order_relaxed)) {}

    Entry& operator=(const Entry& other) {
      if (this != &other) {
        internal_zip::ZipEntry::operator=(
            static_cast<const internal_zip::ZipEntry&>(other));
        local_header_and_data_size = other.local_header_and_data_size;
        local_header_size.store(
            other.local_header_size.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
      }
      return *this;
    }

    Entry(Entry&& other) noexcept
        : internal_zip::ZipEntry(static_cast<internal_zip::ZipEntry&&>(other)),
          local_header_and_data_size(other.local_header_and_data_size),
          local_header_size(
              other.local_header_size.load(std::memory_order_relaxed)) {}

    Entry& operator=(Entry&& other) noexcept {
      if (this != &other) {
        internal_zip::ZipEntry::operator=(
            static_cast<internal_zip::ZipEntry&&>(other));
        local_header_and_data_size = other.local_header_and_data_size;
        local_header_size.store(
            other.local_header_size.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
      }
      return *this;
    }

    // Excludes local_header_size (runtime cache) and extra_field_length
    // (not identity).
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.filename, x.crc, x.compressed_size, x.uncompressed_size,
               x.local_header_offset, x.local_header_and_data_size,
               x.version_madeby, x.flags, x.compression_method, x.internal_fa,
               x.external_fa, x.mtime, x.comment, x.is_zip64);
    };

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Entry& entry) {
      absl::Format(&sink,
                   "Entry{filename=%v, crc=%v, compressed_size=%v, "
                   "uncompressed_size=%v, local_header_offset=%v, "
                   "local_header_and_data_size=%v, local_header_size=%v}",
                   entry.filename, entry.crc, entry.compressed_size,
                   entry.uncompressed_size, entry.local_header_offset,
                   entry.local_header_and_data_size,
                   entry.local_header_size.load(std::memory_order_relaxed));
    }
  };

  // Entries sorted by filename.
  std::vector<Entry> entries;

  // True when the entire archive was read (e.g. EOCD suffix read
  // failed). When set, individual entry reads use seek_pos instead
  // of byte-range requests since the data is already local.
  bool full_read = false;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.entries, x.full_read);
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CachedDir& dir) {
    absl::Format(&sink, "Directory{\n");
    for (const auto& entry : dir.entries) {
      absl::Format(&sink, "%v\n", entry);
    }
    absl::Format(&sink, "}");
  }
};

/// Decodes the central directory entries from the specified reader.
///
/// \param reader Reader positioned at the start of the central directory.
/// \param num_entries Expected number of directory entries.
/// \param cd_offset Offset of the central directory from the file start.
/// \returns The decoded directory representation on success, or an error.
Result<CachedDir> DecodeDirectoryEntries(riegeli::Reader& reader,
                                         size_t num_entries, int64_t cd_offset);

}  // namespace internal_zip_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZIP_CACHED_DIR_H_
