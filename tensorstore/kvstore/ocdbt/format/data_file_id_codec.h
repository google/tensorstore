// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_CODEC_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_CODEC_H_

#include <vector>

#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"

namespace tensorstore {
namespace internal_ocdbt {

using DataFileIndexCodec = VarintCodec<uint64_t>;

// Builds and encodes a data file table, used only when writing.
class DataFileTableBuilder {
 public:
  // Adds a data file id that may later be encoded using
  // `DataFileIdCodec<riegeli::Writer>`.
  void Add(const DataFileId& data_file_id);

  // Finalizes the table and writes the encoded representation to `writer`.
  [[nodiscard]] bool Finalize(riegeli::Writer& writer);

  // Returns the index to the specified data file.
  //
  // It is required that `Add(data_file_id)` and `Finalize` were called
  // previously.
  size_t GetIndex(const DataFileId& data_file_id) const;

  void Clear();

 private:
  absl::flat_hash_map<DataFileId, size_t> data_files_;
};

// Decoded representation of a data file table, used only when reading.
struct DataFileTable {
  std::vector<DataFileId> files;
};

// Reads a data file table.
//
// The specified `transitive_path` is prepended to every decoded `base_path`
// value.
[[nodiscard]] bool ReadDataFileTable(riegeli::Reader& reader,
                                     const BasePath& transitive_path,
                                     DataFileTable& value);

// Encodes a `DataFileId` as an index into the data file table.
//
// When `IO = riegeli::Reader`, must be constructed with a reference to a
// previously decoded `DataFileTable`.
//
// When `IO = riegeli::Writer`, must be constructed with a
// `DataFileTableBuilder` for which `DataFileTableBuilder::Finalize` has already
// been called.
template <typename IO>
struct DataFileIdCodec;

template <>
struct DataFileIdCodec<riegeli::Reader> {
  const DataFileTable& data_file_table;
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                DataFileId& value) const;
};

template <>
struct DataFileIdCodec<riegeli::Writer> {
  const DataFileTableBuilder& table;
  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                const DataFileId& value) const {
    return DataFileIndexCodec{}(writer, table.GetIndex(value));
  }
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_CODEC_H_
