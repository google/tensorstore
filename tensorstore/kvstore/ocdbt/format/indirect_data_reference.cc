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

#include <ostream>
#include <string>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/strings/escaping.h"
#include <openssl/rand.h>
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

DataFileId GenerateDataFileId() {
  DataFileId file_id;
  ABSL_CHECK(RAND_bytes(reinterpret_cast<unsigned char*>(file_id.value.data()),
                        file_id.value.size()));
  return file_id;
}

std::string GetDataFilePath(DataFileId file_id) {
  return absl::BytesToHexString(
      std::string_view(reinterpret_cast<const char*>(file_id.value.data()),
                       file_id.value.size()));
}

std::string GetDataDirectoryPath(std::string_view base_path) {
  return tensorstore::StrCat(base_path, "d/");
}

bool operator==(const IndirectDataReference& a,
                const IndirectDataReference& b) {
  return a.file_id == b.file_id && a.offset == b.offset && a.length == b.length;
}
std::ostream& operator<<(std::ostream& os, const IndirectDataReference& x) {
  return os << "{file_id=" << x.file_id << ", offset=" << x.offset
            << ", length=" << x.length << "}";
}

std::ostream& operator<<(std::ostream& os, const DataFileId& x) {
  return os << GetDataFilePath(x);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
