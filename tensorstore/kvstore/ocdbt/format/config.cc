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

#include "tensorstore/kvstore/ocdbt/format/config.h"

#include <array>
#include <ostream>
#include <string_view>
#include <variant>

#include "absl/log/absl_check.h"
#include "absl/strings/escaping.h"
#include <openssl/rand.h>

namespace tensorstore {
namespace internal_ocdbt {

Uuid Uuid::Generate() {
  Uuid uuid;
  ABSL_CHECK(RAND_bytes(reinterpret_cast<unsigned char*>(uuid.value.data()),
                        uuid.value.size()));
  return uuid;
}

std::ostream& operator<<(std::ostream& os, const Uuid& value) {
  return os << absl::BytesToHexString(
             std::string_view(reinterpret_cast<const char*>(value.value.data()),
                              value.value.size()));
}

std::ostream& operator<<(std::ostream& os, ManifestKind x) {
  switch (x) {
    case ManifestKind::kSingle:
      os << "single";
      break;
    case ManifestKind::kNumbered:
      os << "numbered";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, Config::NoCompression) {
  return os << "raw";
}

bool operator==(Config::ZstdCompression a, Config::ZstdCompression b) {
  return a.level == b.level;
}

std::ostream& operator<<(std::ostream& os, Config::ZstdCompression x) {
  return os << "zstd{level=" << x.level << "}";
}

std::ostream& operator<<(std::ostream& os, const Config::Compression& x) {
  std::visit([&](const auto& v) { os << v; }, x);
  return os;
}

bool operator==(const Config& a, const Config& b) {
  return a.uuid == b.uuid && a.manifest_kind == b.manifest_kind &&
         a.max_inline_value_bytes == b.max_inline_value_bytes &&
         a.max_decoded_node_bytes == b.max_decoded_node_bytes &&
         a.version_tree_arity_log2 == b.version_tree_arity_log2 &&
         a.compression == b.compression;
}

std::ostream& operator<<(std::ostream& os, const Config& x) {
  return os << "{uuid=" << x.uuid << ", manifest_kind=" << x.manifest_kind
            << ", max_inline_value_bytes=" << x.max_inline_value_bytes
            << ", max_decoded_node_bytes=" << x.max_decoded_node_bytes
            << ", version_tree_arity_log2="
            << static_cast<int>(x.version_tree_arity_log2)
            << ", compression=" << x.compression << "}";
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
