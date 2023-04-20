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

#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"

#include <algorithm>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/varint/varint_reading.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/internal/ref_counted_string.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {
using PathLengthCodec = VarintCodec<PathLength>;
}  // namespace

void DataFileTableBuilder::Add(const DataFileId& data_file_id) {
  data_files_.emplace(data_file_id, 0);
}

bool DataFileTableBuilder::Finalize(riegeli::Writer& writer) {
  if (!riegeli::WriteVarint64(data_files_.size(), writer)) return false;
  if (data_files_.empty()) return true;
  std::vector<DataFileId> sorted_data_files;
  sorted_data_files.reserve(data_files_.size());
  for (const auto& p : data_files_) {
    sorted_data_files.push_back(p.first);
  }
  std::sort(sorted_data_files.begin(), sorted_data_files.end(),
            [&](const DataFileId& a, const DataFileId& b) {
              if (int c = std::string_view(a.base_path)
                              .compare(std::string_view(b.base_path));
                  c != 0) {
                return c < 0;
              }
              return std::string_view(a.relative_path) <
                     std::string_view(b.relative_path);
            });

  // Compute and encode prefix length
  std::vector<size_t> prefix_lengths(sorted_data_files.size());
  prefix_lengths[0] = 0;
  for (size_t i = 1; i < sorted_data_files.size(); ++i) {
    auto& cur = sorted_data_files[i];
    auto& prev = sorted_data_files[i - 1];
    std::string_view prev_base_path = prev.base_path;
    std::string_view cur_base_path = cur.base_path;
    size_t prefix_length =
        FindCommonPrefixLength(prev_base_path, cur_base_path);
    if (prev_base_path.size() == cur_base_path.size() &&
        cur_base_path.size() == prefix_length) {
      prefix_length +=
          FindCommonPrefixLength(prev.relative_path, cur.relative_path);
    }
    prefix_lengths[i] = prefix_length;
    if (!PathLengthCodec{}(writer, prefix_length)) return false;
  }

  // Encode suffix length.
  for (size_t i = 0; i < sorted_data_files.size(); ++i) {
    const auto& data_file = sorted_data_files[i];
    assert(data_file.base_path.size() + data_file.relative_path.size() <=
           kMaxPathLength);
    if (!PathLengthCodec{}(writer, data_file.base_path.size() +
                                       data_file.relative_path.size() -
                                       prefix_lengths[i])) {
      return false;
    }
  }

  // Encode base_path length.
  for (size_t i = 0; i < sorted_data_files.size(); ++i) {
    const auto& data_file = sorted_data_files[i];
    if (!PathLengthCodec{}(writer, data_file.base_path.size())) {
      return false;
    }
  }

  // Encode path_suffix
  for (size_t i = 0; i < sorted_data_files.size(); ++i) {
    const auto& data_file = sorted_data_files[i];
    size_t prefix_length = prefix_lengths[i];
    std::string_view base_path = data_file.base_path;
    size_t base_path_prefix_length = std::min(prefix_length, base_path.size());
    if (!writer.Write(base_path.substr(base_path_prefix_length))) return false;
    std::string_view relative_path = data_file.relative_path;
    if (!writer.Write(
            relative_path.substr(prefix_length - base_path_prefix_length))) {
      return false;
    }

    auto it = data_files_.find(data_file);
    assert(it != data_files_.end());
    it->second = i;
  }
  return true;
}

size_t DataFileTableBuilder::GetIndex(const DataFileId& data_file_id) const {
  return data_files_.at(data_file_id);
}

void DataFileTableBuilder::Clear() { data_files_.clear(); }

[[nodiscard]] bool ReadDataFileTable(riegeli::Reader& reader,
                                     const BasePath& transitive_path,
                                     DataFileTable& value) {
  ABSL_CHECK_LE(transitive_path.size(), kMaxPathLength);
  std::string_view transitive_path_sv = transitive_path;
  const size_t max_path_length = kMaxPathLength - transitive_path_sv.size();

  uint64_t num_files;
  if (!riegeli::ReadVarint64(reader, num_files)) return false;

  std::vector<PathLength> path_length_buffer;
  constexpr uint64_t kMaxReserve = 1024;
  path_length_buffer.reserve(std::min(kMaxReserve, num_files) * 3);

  // Initial prefix is 0
  path_length_buffer.push_back(0);

  for (uint64_t i = 1; i < num_files; ++i) {
    PathLength prefix_length;
    if (!PathLengthCodec{}(reader, prefix_length)) return false;
    path_length_buffer.push_back(prefix_length);
  }

  for (uint64_t i = 0; i < num_files; ++i) {
    PathLength suffix_length;
    if (!PathLengthCodec{}(reader, suffix_length)) return false;
    path_length_buffer.push_back(suffix_length);
  }

  PathLength prev_base_path_length = 0;
  for (uint64_t i = 0; i < num_files; ++i) {
    PathLength base_path_length;
    if (!PathLengthCodec{}(reader, base_path_length)) return false;
    size_t prefix_length = path_length_buffer[i];
    size_t suffix_length = path_length_buffer[num_files + i];
    size_t path_length = prefix_length + suffix_length;
    if (path_length > max_path_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "path_length[%d] = prefix_length(%d) + "
          "suffix_length(%d) = %d > %d - transitive_length(%d) = %d",
          i, prefix_length, suffix_length, path_length, kMaxPathLength,
          transitive_path.size(), max_path_length)));
      return false;
    }
    if (base_path_length > path_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "base_path_length[%d] = %d > path_length[%d] = %d = "
          "prefix_length(%d) + suffix_length(%d)",
          i, base_path_length, i, path_length, prefix_length, suffix_length)));
      return false;
    }

    if (prefix_length > std::min(prev_base_path_length, base_path_length) &&
        base_path_length != prev_base_path_length) {
      reader.Fail(absl::DataLossError(absl::StrFormat(
          "path_prefix_length[%d] = %d > "
          "min(base_path_length[%d] = %d, base_path_length[%d] = %d) is not "
          "valid if "
          "base_path_length[%d] != base_path_length[%d]",
          i - 1, prefix_length,          //
          i - 1, prev_base_path_length,  //
          i, base_path_length,           //
          i - 1, i)));
      return false;
    }
    path_length_buffer.push_back(base_path_length);
    prev_base_path_length = base_path_length;
  }

  auto& files = value.files;
  files.resize(num_files);

  size_t prev_relative_path_length = 0;
  for (uint64_t i = 0; i < num_files; ++i) {
    size_t prefix_length = path_length_buffer[i];
    size_t suffix_length = path_length_buffer[num_files + i];
    size_t base_path_length = path_length_buffer[2 * num_files + i];
    size_t relative_path_length =
        prefix_length + suffix_length - base_path_length;

    if (!reader.Pull(suffix_length)) return false;

    auto& file = files[i];

    if (base_path_length == 0) {
      file.base_path = transitive_path;
    } else if (prefix_length >= base_path_length) {
      // Base path is identical to previous base path, just create another
      // reference to the previous `RefCountedString`.
      assert(files[i - 1].base_path.size() ==
             base_path_length + transitive_path.size());
      file.base_path = files[i - 1].base_path;
      prefix_length -= base_path_length;
    } else {
      // Base path is not identical to the previous one.  Create a new
      // `RefCountedString`.
      internal::RefCountedStringWriter writer(base_path_length +
                                              transitive_path_sv.size());
      std::memcpy(writer.data(), transitive_path_sv.data(),
                  transitive_path_sv.size());
      size_t offset = transitive_path_sv.size();
      size_t base_suffix_length = base_path_length > prefix_length
                                      ? base_path_length - prefix_length
                                      : 0;
      if (prefix_length > 0) {
        std::string_view prev_base_path = files[i - 1].base_path;
        prev_base_path.remove_prefix(transitive_path_sv.size());
        size_t base_prefix_length = std::min(prefix_length, base_path_length);
        assert(base_prefix_length <= prev_base_path.size());
        std::memcpy(writer.data() + offset, prev_base_path.data(),
                    base_prefix_length);
        offset += base_prefix_length;
        prefix_length -= base_prefix_length;
      }
      if (base_suffix_length) {
        std::memcpy(writer.data() + offset, reader.cursor(),
                    base_suffix_length);
        reader.move_cursor(base_suffix_length);
        suffix_length -= base_suffix_length;
      }
      file.base_path = std::move(writer);
    }

    if (relative_path_length == 0) {
      assert(suffix_length == 0);
      prev_relative_path_length = 0;
      continue;
    }

    if (suffix_length == 0 &&
        relative_path_length == prev_relative_path_length) {
      // Since `relative_path_length != 0` but `suffix_length == 0`, the common
      // prefix extends into the `relative_path`, which implies that the
      // `base_path` is equal to the previous `base_path`.
      assert(file.base_path == files[i - 1].base_path);
      file.relative_path = files[i - 1].relative_path;
      continue;
    }

    internal::RefCountedStringWriter writer(relative_path_length);
    size_t offset = 0;
    if (prefix_length) {
      assert(file.base_path == files[i - 1].base_path);
      assert(prefix_length <= relative_path_length);
      std::memcpy(writer.data(), files[i - 1].relative_path.data(),
                  prefix_length);
      offset += prefix_length;
    }

    if (suffix_length > 0) {
      assert(offset + suffix_length == relative_path_length);
      std::memcpy(writer.data() + offset, reader.cursor(), suffix_length);
      reader.move_cursor(suffix_length);
    }
    file.relative_path = std::move(writer);
    prev_relative_path_length = relative_path_length;
  }
  return true;
}

[[nodiscard]] bool DataFileIdCodec<riegeli::Reader>::operator()(
    riegeli::Reader& reader, DataFileId& value) const {
  uint64_t index;
  if (!DataFileIndexCodec{}(reader, index)) return false;
  if (index >= data_file_table.files.size()) {
    reader.Fail(absl::DataLossError(
        absl::StrFormat("Data file id %d is outside range [0, %d)", index,
                        data_file_table.files.size())));
    return false;
  }
  value = data_file_table.files[index];
  return true;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
