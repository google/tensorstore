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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <variant>

#include "tensorstore/util/apply_members/std_array.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Unique identifier of a database.
struct Uuid {
  std::array<uint8_t, 16> value = {};

  // Generates a new random id.
  static Uuid Generate();

  friend std::ostream& operator<<(std::ostream& os, const Uuid& value);
  friend bool operator==(const Uuid& a, const Uuid& b) {
    return a.value == b.value;
  }
  friend bool operator!=(const Uuid& a, const Uuid& b) { return !(a == b); }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.value);
  };
};

/// Database configuration stored in the manifest.
///
/// These options impact how data is written, and therefore must be known before
/// any data can be written.
struct Config {
  enum class ManifestKind {
    kSingle = 0,
    kNumbered = 1,
  };

  constexpr static ManifestKind kMaxManifestKind = ManifestKind::kNumbered;

  friend std::ostream& operator<<(std::ostream& os, ManifestKind);

  /// Unique identifier of the database.
  Uuid uuid;

  /// Specifies how the version tree is stored.
  ManifestKind manifest_kind = ManifestKind::kSingle;

  /// Maximum value size in bytes that will be stored inline within a leaf node.
  uint32_t max_inline_value_bytes = 100;

  /// Maximum size in bytes of a decoded b-tree node.
  uint32_t max_decoded_node_bytes = 8 * 1024 * 1024;

  /// Base-2 logarithm of the arity of the version tree, must be >= 1.
  uint8_t version_tree_arity_log2 = 4;

  struct NoCompression {
    friend bool operator==(NoCompression, NoCompression) { return true; }
    friend bool operator!=(NoCompression, NoCompression) { return false; }
    friend std::ostream& operator<<(std::ostream& os, NoCompression);
  };

  struct ZstdCompression {
    int32_t level;
    friend bool operator==(ZstdCompression a, ZstdCompression b);
    friend bool operator!=(ZstdCompression a, ZstdCompression b) {
      return !(a == b);
    }
    friend std::ostream& operator<<(std::ostream& os, ZstdCompression x);

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.level);
    };
  };

  /// Encoded as:
  ///   0 -> no compression
  ///   1 -> zstd
  using Compression = std::variant<NoCompression, ZstdCompression>;
  Compression compression = ZstdCompression{0};

  friend std::ostream& operator<<(std::ostream& os, const Compression& x);
  friend bool operator==(const Config& a, const Config& b);
  friend bool operator!=(const Config& a, const Config& b) { return !(a == b); }
  friend std::ostream& operator<<(std::ostream& os, const Config& x);
};

using ManifestKind = Config::ManifestKind;

constexpr size_t kMaxInlineValueLength = 1024 * 1024;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_H_
