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

#include "tensorstore/internal/compression/zip_easy.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/digests/crc32_digester.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zip {

EasyZipReader::EasyZipReader(riegeli::Reader& reader) : reader_(reader) {}

absl::Status EasyZipReader::Initialize() {
  if (initialized_) return absl::OkStatus();
  int64_t initial_pos = reader_.pos();
  auto response = TryReadFullEOCD(reader_, eocd_, -1);
  if (std::holds_alternative<int64_t>(response)) {
    reader_.Seek(initial_pos);
    response = TryReadFullEOCD(reader_, eocd_, 0);
  }

  if (auto* status = std::get_if<absl::Status>(&response);
      status != nullptr && !status->ok()) {
    return std::move(*status);
  }
  if (std::holds_alternative<int64_t>(response)) {
    return absl::InternalError("ZIP incomplete");
  }

  // Attempt to read all the entries.
  reader_.Seek(eocd_.cd_offset);
  entries_.reserve(eocd_.num_entries);
  absl::flat_hash_set<std::string> filenames;
  for (size_t i = 0; i < eocd_.num_entries; ++i) {
    ZipEntry entry{};
    if (auto entry_status = ReadCentralDirectoryEntry(reader_, entry);
        !entry_status.ok()) {
      return entry_status;
    }
    auto [iter, inserted] = filenames.insert(entry.filename);
    if (!inserted) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Duplicate filename in ZIP archive: '%s'", entry.filename));
    }
    entries_.push_back(std::move(entry));
  }

  initialized_ = true;
  return absl::OkStatus();
}

Result<span<const ZipEntry>> EasyZipReader::entries() {
  TENSORSTORE_RETURN_IF_ERROR(Initialize());
  return entries_;
}

Result<absl::Cord> EasyZipReader::ReadEntry(ZipEntry& entry) {
  TENSORSTORE_RETURN_IF_ERROR(Initialize());
  if (!reader_.Seek(entry.local_header_offset)) {
    return reader_.status();
  }
  ZipEntry local_header = entry;
  uint64_t original_offset = entry.local_header_offset;
  TENSORSTORE_RETURN_IF_ERROR(ReadLocalEntry(reader_, local_header));
  local_header.local_header_offset = original_offset;

  if (local_header.filename != entry.filename) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ZIP entry local filename '%s' does not match "
                        "central directory filename '%s'",
                        local_header.filename, entry.filename));
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateEntryIsSupported(local_header));

  TENSORSTORE_ASSIGN_OR_RETURN(auto entry_reader,
                               GetReader(&reader_, local_header));
  absl::Cord content;
  if (!entry_reader->Read(local_header.uncompressed_size, content)) {
    if (entry_reader->status().ok()) {
      return absl::DataLossError("Failed to read all expected data");
    }
    return entry_reader->status();
  }
  entry = std::move(local_header);
  return content;
}

Result<absl::Cord> EasyZipReader::ReadEntry(std::string_view filename) {
  TENSORSTORE_RETURN_IF_ERROR(Initialize());
  for (auto& entry : entries_) {
    if (entry.filename == filename) {
      return ReadEntry(entry);
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("ZIP entry not found: %s", filename));
}

EasyZipWriter::EasyZipWriter(riegeli::Writer& writer) : writer_(writer) {}

absl::Status EasyZipWriter::WriteEntry(ZipEntry& entry,
                                       const absl::Cord& data) {
  // Compress the data if needed.
  std::array<ZipCompression, 1> methods = {
      entry.compression_method,
  };
  TENSORSTORE_ASSIGN_OR_RETURN(auto compression_result,
                               Compress(data, methods));

  entry.compression_method = compression_result.method;
  entry.uncompressed_size = data.size();
  const absl::Cord& compressed_data = compression_result.data;
  entry.compressed_size = compressed_data.size();

  // Calculate CRC32.
  riegeli::Crc32Digester digester;
  for (auto chunk : data.Chunks()) {
    digester.Write(chunk);
  }
  entry.crc = digester.Digest();

  entry.local_header_offset = writer_.pos();
  TENSORSTORE_RETURN_IF_ERROR(WriteLocalEntry(writer_, entry));
  writer_.Write(compressed_data);
  if (!writer_.ok()) {
    return writer_.status();
  }

  entries_.push_back(entry);
  return absl::OkStatus();
}

absl::Status EasyZipWriter::WriteEntry(const std::string& filename,
                                       const absl::Cord& data,
                                       ZipCompression compression_method,
                                       absl::Time mtime, std::string comment) {
  ZipEntry entry;
  entry.filename = filename;
  entry.compression_method = compression_method;
  entry.mtime = mtime;
  entry.comment = std::move(comment);
  return WriteEntry(entry, data);
}

absl::Status EasyZipWriter::Finalize(ZipEOCD* eocd) {
  if (eocd != nullptr) {
    eocd_ = *eocd;
  }
  eocd_.cd_offset = writer_.pos();
  for (const auto& entry : entries_) {
    TENSORSTORE_RETURN_IF_ERROR(WriteCentralDirectoryEntry(writer_, entry));
  }
  eocd_.cd_size = writer_.pos() - eocd_.cd_offset;
  eocd_.num_entries = entries_.size();

  TENSORSTORE_RETURN_IF_ERROR(WriteEOCD(writer_, eocd_));

  if (eocd != nullptr) {
    *eocd = eocd_;
  }

  if (!writer_.ok()) {
    return writer_.status();
  }
  return absl::OkStatus();
}

}  // namespace internal_zip
}  // namespace tensorstore
