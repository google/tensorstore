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

#include "tensorstore/driver/zarr3/codec/crc32c.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/digests/crc32c_digester.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/riegeli/digest_suffixed_reader.h"
#include "tensorstore/internal/riegeli/digest_suffixed_writer.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_zarr3 {
namespace {

template <typename Digester, typename DigestWriterTraits,
          typename DigestVerifierTraits>
class DigestCodec : public ZarrBytesToBytesCodec {
 public:
  using DigestReader =
      internal::DigestSuffixedReader<Digester, DigestVerifierTraits>;
  using DigestWriter =
      internal::DigestSuffixedWriter<Digester, DigestWriterTraits>;
  using DigestType = typename DigestWriter::DigestType;

  static constexpr int64_t kChecksumSize =
      static_cast<int64_t>(DigestVerifierTraits::template Size<DigestType>());

  class State : public ZarrBytesToBytesCodec::PreparedState {
   public:
    explicit State(int64_t decoded_size) {
      if (decoded_size == -1 ||
          internal::AddOverflow(decoded_size, kChecksumSize, &encoded_size_)) {
        encoded_size_ = -1;
      }
    }
    Result<std::unique_ptr<riegeli::Writer>> GetEncodeWriter(
        riegeli::Writer& encoded_writer) const final {
      return std::make_unique<DigestWriter>(&encoded_writer);
    }

    Result<std::unique_ptr<riegeli::Reader>> GetDecodeReader(
        riegeli::Reader& encoded_reader) const final {
      std::optional<size_t> limit;
      if (encoded_size_ != -1) limit = encoded_size_ - kChecksumSize;
      return std::make_unique<DigestReader>(&encoded_reader, limit);
    }

    int64_t encoded_size() const override { return encoded_size_; }

    int64_t encoded_size_;
  };

  Result<PreparedState::Ptr> Prepare(int64_t decoded_size) const final {
    return internal::MakeIntrusivePtr<State>(decoded_size);
  }
};

using Crc32cCodec =
    DigestCodec<riegeli::Crc32cDigester, internal::LittleEndianDigestWriter,
                internal::LittleEndianDigestVerifier>;

}  // namespace

absl::Status Crc32cCodecSpec::MergeFrom(const ZarrCodecSpec& other,
                                        bool strict) {
  return absl::OkStatus();
}

ZarrCodecSpec::Ptr Crc32cCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<Crc32cCodecSpec>(*this);
}

Result<ZarrBytesToBytesCodec::Ptr> Crc32cCodecSpec::Resolve(
    BytesCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const {
  if (resolved_spec) resolved_spec->reset(this);
  return internal::MakeIntrusivePtr<Crc32cCodec>();
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = Crc32cCodecSpec;
  namespace jb = ::tensorstore::internal_json_binding;
  RegisterCodec<Self>("crc32c", jb::Sequence());
}

}  // namespace internal_zarr3
}  // namespace tensorstore
