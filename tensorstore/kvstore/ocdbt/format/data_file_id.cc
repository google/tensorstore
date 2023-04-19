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

#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"

#include <ostream>
#include <string>
#include <string_view>

#include "absl/log/absl_check.h"
#include <openssl/rand.h>
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

std::string DataFileId::FullPath() const {
  return tensorstore::StrCat(std::string_view(base_path),
                             std::string_view(relative_path));
}

DataFileId GenerateDataFileId() {
  constexpr size_t kIdBytes = 16;
  std::array<unsigned char, kIdBytes> id;
  ABSL_CHECK(
      RAND_bytes(reinterpret_cast<unsigned char*>(id.data()), id.size()));
  char buffer[kIdBytes * 2 + 2];
  buffer[0] = 'd';
  buffer[1] = '/';
  constexpr char kHexDigits[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                 '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  for (size_t i = 0; i < kIdBytes; ++i) {
    buffer[2 * i + 2] = kHexDigits[id[i] / 16];
    buffer[2 * i + 2 + 1] = kHexDigits[id[i] % 16];
  }
  DataFileId data_file_id;
  data_file_id.relative_path = std::string_view(buffer, sizeof(buffer));
  return data_file_id;
}

std::ostream& operator<<(std::ostream& os, const DataFileId& x) {
  return os << tensorstore::QuoteString(x.base_path) << "+"
            << tensorstore::QuoteString(x.relative_path);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
