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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_CODEC_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_CODEC_H_

/// \file
///
/// Internal codecs for `Config`, used by the manifest codec.

#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree_codec.h"

namespace tensorstore {
namespace internal_ocdbt {

using UuidCodec = RawBytesCodec<Uuid>;
using MaxInlineValueBytesCodec = VarintCodec<uint32_t>;
using MaxDecodedNodeBytesCodec = VarintCodec<uint32_t>;

struct CompressionConfigCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                Config::Compression& value) const;

  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                const Config::Compression& value) const;
};

struct ManifestKindCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader,
                                ManifestKind& value) const;

  [[nodiscard]] bool operator()(riegeli::Writer& writer,
                                ManifestKind value) const {
    return writer.WriteByte(static_cast<uint8_t>(value));
  }
};

struct ConfigCodec {
  template <typename IO, typename T>
  [[nodiscard]] bool operator()(IO& io, T&& value) const {
    return UuidCodec{}(io, value.uuid) &&
           ManifestKindCodec{}(io, value.manifest_kind) &&
           MaxInlineValueBytesCodec{}(io, value.max_inline_value_bytes) &&
           MaxDecodedNodeBytesCodec{}(io, value.max_decoded_node_bytes) &&
           VersionTreeArityLog2Codec{}(io, value.version_tree_arity_log2) &&
           CompressionConfigCodec{}(io, value.compression);
  }
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_CONFIG_CODEC_H_
