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
#include "absl/strings/str_format.h"
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
  return os << absl::StreamFormat("%v", value);
}

std::ostream& operator<<(std::ostream& os, ManifestKind x) {
  return os << absl::StreamFormat("%v", x);
}

std::ostream& operator<<(std::ostream& os, Config::NoCompression x) {
  return os << absl::StreamFormat("%v", x);
}

bool operator==(Config::ZstdCompression a, Config::ZstdCompression b) {
  return a.level == b.level;
}

std::ostream& operator<<(std::ostream& os, Config::ZstdCompression x) {
  return os << absl::StreamFormat("%v", x);
}

std::ostream& operator<<(std::ostream& os, const Config::Compression& x) {
  return os << absl::StreamFormat("%v", x);
}

bool operator==(const Config& a, const Config& b) {
  return a.uuid == b.uuid && a.manifest_kind == b.manifest_kind &&
         a.max_inline_value_bytes == b.max_inline_value_bytes &&
         a.max_decoded_node_bytes == b.max_decoded_node_bytes &&
         a.version_tree_arity_log2 == b.version_tree_arity_log2 &&
         a.compression == b.compression;
}

std::ostream& operator<<(std::ostream& os, const Config& x) {
  return os << absl::StreamFormat("%v", x);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
