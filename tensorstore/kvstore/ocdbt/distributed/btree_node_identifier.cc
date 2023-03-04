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

#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"

#include <ostream>
#include <string>
#include <string_view>

#include "absl/base/internal/endian.h"
#include <blake3.h>
#include "tensorstore/kvstore/key_range.h"

namespace tensorstore {
namespace internal_ocdbt {

std::ostream& operator<<(std::ostream& os, const BtreeNodeIdentifier& x) {
  return os << "{range=" << x.range << ", height=" << static_cast<int>(x.height)
            << "}";
}

std::string BtreeNodeIdentifier::GetKey(
    std::string_view database_identifier) const {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, database_identifier.data(),
                       database_identifier.size());
  char header[3];
  header[0] = static_cast<char>(range.full() ? 0 : height);
  absl::little_endian::Store16(&header[1], range.inclusive_min.size());
  blake3_hasher_update(&hasher, header, 3);
  blake3_hasher_update(&hasher, range.inclusive_min.data(),
                       range.inclusive_min.size());
  blake3_hasher_update(&hasher, range.exclusive_max.data(),
                       range.exclusive_max.size());
  std::string key;
  key.resize(20);
  blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t*>(key.data()),
                         key.size());
  return key;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
