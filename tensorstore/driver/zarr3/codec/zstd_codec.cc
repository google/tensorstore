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

#include "tensorstore/driver/zarr3/codec/zstd_codec.h"

#include <stdint.h>

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zarr3 {
namespace {

using ::riegeli::ZstdWriterBase;

class ZstdCodec : public ZarrBytesToBytesCodec {
 public:
  explicit ZstdCodec(int level, bool checksum)
      : level_(level), checksum_(checksum) {}

  class State : public ZarrBytesToBytesCodec::PreparedState {
   public:
    Result<std::unique_ptr<riegeli::Writer>> GetEncodeWriter(
        riegeli::Writer& encoded_writer) const final {
      using Writer = riegeli::ZstdWriter<riegeli::Writer*>;
      Writer::Options options;
      options.set_compression_level(level_);
      options.set_store_checksum(checksum_);
      if (decoded_size_ != -1) {
        options.set_pledged_size(decoded_size_);
      }
      return std::make_unique<Writer>(&encoded_writer, options);
    }

    Result<std::unique_ptr<riegeli::Reader>> GetDecodeReader(
        riegeli::Reader& encoded_reader) const final {
      using Reader = riegeli::ZstdReader<riegeli::Reader*>;
      Reader::Options options;
      return std::make_unique<Reader>(&encoded_reader, options);
    }

    int level_;
    bool checksum_;
    int64_t decoded_size_;
  };

  Result<PreparedState::Ptr> Prepare(int64_t decoded_size) const final {
    auto state = internal::MakeIntrusivePtr<State>();
    state->level_ = level_;
    state->checksum_ = checksum_;
    state->decoded_size_ = decoded_size;
    return state;
  }

 private:
  int level_;
  bool checksum_;
};

}  // namespace

absl::Status ZstdCodecSpec::MergeFrom(const ZarrCodecSpec& other, bool strict) {
  using Self = ZstdCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::level>("level", options, other_options));
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::checksum>("checksum", options, other_options));
  return absl::OkStatus();
}

ZarrCodecSpec::Ptr ZstdCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<ZstdCodecSpec>(*this);
}

Result<ZarrBytesToBytesCodec::Ptr> ZstdCodecSpec::Resolve(
    BytesCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const {
  auto resolved_level =
      options.level.value_or(ZstdWriterBase::Options::kDefaultCompressionLevel);
  auto resolved_checksum = options.checksum.value_or(false);
  if (resolved_spec) {
    if (options.level && options.checksum) {
      resolved_spec->reset(this);
    } else {
      resolved_spec->reset(
          new ZstdCodecSpec(Options{resolved_level, resolved_checksum}));
    }
  }
  return internal::MakeIntrusivePtr<ZstdCodec>(resolved_level,
                                               resolved_checksum);
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = ZstdCodecSpec;
  using Options = Self::Options;
  namespace jb = ::tensorstore::internal_json_binding;
  RegisterCodec<Self>(
      "zstd",
      jb::Projection<&Self::options>(jb::Sequence(
          jb::Member("level",
                     jb::Projection<&Options::level>(
                         OptionalIfConstraintsBinder(jb::Integer<int>(
                             ZstdWriterBase::Options::kMinCompressionLevel,
                             ZstdWriterBase::Options::kMaxCompressionLevel)))),
          jb::Member("checksum",
                     jb::Projection<&Options::checksum>(
                         OptionalIfConstraintsBinder())))  //
                                     ));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
