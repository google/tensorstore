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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"

namespace tensorstore {
namespace internal_ocdbt {

using DataFileIdCodec = RawBytesCodec<DataFileId>;

using DataFileOffsetCodec = VarintCodec<uint64_t>;
using DataFileLengthCodec = VarintCodec<uint64_t>;

template <typename Getter>
struct IndirectDataReferenceArrayCodec {
  Getter getter;
  bool allow_missing = false;
  template <typename IO, typename Vec>
  [[nodiscard]] bool operator()(IO& io, Vec&& vec) const {
    static_assert(std::is_same_v<IO, riegeli::Reader> ||
                  std::is_same_v<IO, riegeli::Writer>);
    for (auto& entry : vec) {
      if (!DataFileIdCodec{}(io, getter(entry).file_id)) return false;
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
        if (allow_missing && r.IsMissing()) continue;
        uint64_t end_offset;
        if (internal::AddOverflow(r.offset, r.length, &end_offset)) {
          io.Fail(absl::DataLossError(absl::StrFormat(
              "Invalid offset/length pair (%d, %d)", r.offset, r.length)));
          return false;
        }
      }
    }

    return true;
  }
};

template <typename Getter>
IndirectDataReferenceArrayCodec(Getter, bool allow_missing = false)
    -> IndirectDataReferenceArrayCodec<Getter>;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_CODEC_H_
