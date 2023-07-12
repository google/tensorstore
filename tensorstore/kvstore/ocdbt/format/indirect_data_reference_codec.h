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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_CODEC_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_CODEC_H_

/// \file
///
/// Internal codecs for `IndirectDataReference` and related types.

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"

namespace tensorstore {
namespace internal_ocdbt {

using DataFileOffsetCodec = VarintCodec<uint64_t>;
using DataFileLengthCodec = VarintCodec<uint64_t>;

template <typename DataFileTable, typename Getter>
struct IndirectDataReferenceArrayCodec {
  const DataFileTable& data_file_table;
  Getter getter;
  bool allow_missing = false;
  template <typename IO, typename Vec>
  [[nodiscard]] bool operator()(IO& io, Vec&& vec) const {
    static_assert(std::is_same_v<IO, riegeli::Reader> ||
                  std::is_same_v<IO, riegeli::Writer>);
    for (auto& entry : vec) {
      if (!DataFileIdCodec<IO>{data_file_table}(io, getter(entry).file_id)) {
        return false;
      }
    }

    for (auto& entry : vec) {
      if (!DataFileOffsetCodec{}(io, getter(entry).offset)) return false;
    }

    for (auto& entry : vec) {
      if (!DataFileLengthCodec{}(io, getter(entry).length)) return false;
    }

    if constexpr (std::is_same_v<IO, riegeli::Reader>) {
      // Validate length
      for (auto& v : vec) {
        auto& r = getter(v);
        TENSORSTORE_RETURN_IF_ERROR(r.Validate(allow_missing),
                                    (io.Fail(_), false));
      }
    }

    return true;
  }
};

template <typename DataFileTable, typename Getter>
IndirectDataReferenceArrayCodec(const DataFileTable&, Getter,
                                bool allow_missing = false)
    -> IndirectDataReferenceArrayCodec<DataFileTable, Getter>;

inline void AddDataFiles(DataFileTableBuilder& data_file_table,
                         const IndirectDataReference& ref) {
  data_file_table.Add(ref.file_id);
}

template <typename T>
void AddDataFiles(DataFileTableBuilder& data_file_table,
                  const std::vector<T>& entries) {
  for (const auto& entry : entries) {
    AddDataFiles(data_file_table, entry);
  }
}

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_CODEC_H_
