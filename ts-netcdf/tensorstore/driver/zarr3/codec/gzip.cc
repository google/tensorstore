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

#include "tensorstore/driver/zarr3/codec/gzip.h"

#include <stdint.h>

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"
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

constexpr int kDefaultLevel = 6;

class GzipCodec : public ZarrBytesToBytesCodec {
 public:
  explicit GzipCodec(int level) : level_(level) {}

  class State : public ZarrBytesToBytesCodec::PreparedState {
   public:
    Result<std::unique_ptr<riegeli::Writer>> GetEncodeWriter(
        riegeli::Writer& encoded_writer) const final {
      using Writer = riegeli::ZlibWriter<riegeli::Writer*>;
      Writer::Options options;
      options.set_compression_level(level_);
      options.set_header(Writer::Header::kGzip);
      return std::make_unique<Writer>(&encoded_writer, options);
    }

    Result<std::unique_ptr<riegeli::Reader>> GetDecodeReader(
        riegeli::Reader& encoded_reader) const final {
      using Reader = riegeli::ZlibReader<riegeli::Reader*>;
      Reader::Options options;
      options.set_header(Reader::Header::kGzip);
      return std::make_unique<Reader>(&encoded_reader, options);
    }

    int level_;
  };

  Result<PreparedState::Ptr> Prepare(int64_t decoded_size) const final {
    auto state = internal::MakeIntrusivePtr<State>();
    state->level_ = level_;
    return state;
  }

 private:
  int level_;
};

}  // namespace

absl::Status GzipCodecSpec::MergeFrom(const ZarrCodecSpec& other, bool strict) {
  using Self = GzipCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::level>("level", options, other_options));
  return absl::OkStatus();
}

ZarrCodecSpec::Ptr GzipCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<GzipCodecSpec>(*this);
}

Result<ZarrBytesToBytesCodec::Ptr> GzipCodecSpec::Resolve(
    BytesCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const {
  auto resolved_level = options.level.value_or(kDefaultLevel);
  if (resolved_spec) {
    resolved_spec->reset(
        options.level ? this : new GzipCodecSpec(Options{resolved_level}));
  }
  return internal::MakeIntrusivePtr<GzipCodec>(resolved_level);
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = GzipCodecSpec;
  using Options = Self::Options;
  namespace jb = ::tensorstore::internal_json_binding;
  RegisterCodec<Self>(
      "gzip",
      jb::Projection<&Self::options>(jb::Sequence(  //
          jb::Member("level", jb::Projection<&Options::level>(
                                  OptionalIfConstraintsBinder(
                                      jb::Integer<int>(0, 9))))  //
          )));
}

}  // namespace internal_zarr3
}  // namespace tensorstore
