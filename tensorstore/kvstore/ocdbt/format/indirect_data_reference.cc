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

#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"

#include <string.h>

#include <ostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

void EncodeCacheKeyAdl(std::string* out, const IndirectDataReference& self) {
  const size_t total_size = sizeof(uint64_t) * 4 + self.file_id.size();
  out->resize(out->size() + total_size);
  char* buf_ptr = out->data() + out->size() - total_size;
  memcpy(buf_ptr, &self.offset, sizeof(uint64_t));
  buf_ptr += sizeof(uint64_t);
  memcpy(buf_ptr, &self.length, sizeof(uint64_t));
  buf_ptr += sizeof(uint64_t);
  const uint64_t base_path_length = self.file_id.base_path.size();
  memcpy(buf_ptr, &base_path_length, sizeof(uint64_t));
  buf_ptr += sizeof(uint64_t);
  const uint64_t relative_path_length = self.file_id.relative_path.size();
  memcpy(buf_ptr, &relative_path_length, sizeof(uint64_t));
  buf_ptr += sizeof(uint64_t);
  memcpy(buf_ptr, self.file_id.base_path.data(), base_path_length);
  buf_ptr += base_path_length;
  memcpy(buf_ptr, self.file_id.relative_path.data(), relative_path_length);
}

bool IndirectDataReference::DecodeCacheKey(std::string_view encoded) {
  if (encoded.size() < sizeof(uint64_t) * 4) return false;
  memcpy(&offset, encoded.data(), sizeof(uint64_t));
  encoded.remove_prefix(sizeof(uint64_t));
  memcpy(&length, encoded.data(), sizeof(uint64_t));
  encoded.remove_prefix(sizeof(uint64_t));
  uint64_t base_path_length;
  memcpy(&base_path_length, encoded.data(), sizeof(uint64_t));
  encoded.remove_prefix(sizeof(uint64_t));
  uint64_t relative_path_length;
  memcpy(&relative_path_length, encoded.data(), sizeof(uint64_t));
  encoded.remove_prefix(sizeof(uint64_t));
  if (base_path_length > encoded.size() ||
      encoded.size() - base_path_length != relative_path_length) {
    return false;
  }
  file_id.base_path = encoded.substr(0, base_path_length);
  file_id.relative_path = encoded.substr(base_path_length);
  return true;
}

bool operator==(const IndirectDataReference& a,
                const IndirectDataReference& b) {
  return a.file_id == b.file_id && a.offset == b.offset && a.length == b.length;
}
std::ostream& operator<<(std::ostream& os, const IndirectDataReference& x) {
  return os << "{file_id=" << x.file_id << ", offset=" << x.offset
            << ", length=" << x.length << "}";
}

absl::Status IndirectDataReference::Validate(bool allow_missing) const {
  if (!allow_missing || !IsMissing()) {
    uint64_t end_offset;
    if (internal::AddOverflow(offset, length, &end_offset) ||
        end_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return absl::DataLossError(
          tensorstore::StrCat("Invalid offset/length pair in ", *this));
    }
  }
  return absl::OkStatus();
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
