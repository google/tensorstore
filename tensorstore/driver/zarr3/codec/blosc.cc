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

#include "tensorstore/driver/zarr3/codec/blosc.h"

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include <blosc.h>
#include "riegeli/base/chain.h"
#include "riegeli/bytes/chain_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/write.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/internal/compression/blosc.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {
namespace {

// Buffers writes to a `std::string`, and then in `Done`, calls `blosc::Encode`
// and forwards the result to another `Writer`.
class BloscDeferredWriter : public riegeli::CordWriter<absl::Cord> {
 public:
  explicit BloscDeferredWriter(blosc::Options options,
                               riegeli::Writer& base_writer)
      : CordWriter(riegeli::CordWriterBase::Options().set_max_block_size(
            std::numeric_limits<size_t>::max())),
        options_(std::move(options)),
        base_writer_(base_writer) {}

  void Done() override {
    CordWriter::Done();
    auto output = blosc::Encode(dest().Flatten(), options_);
    if (!output.ok()) {
      Fail(std::move(output).status());
      return;
    }
    auto status = riegeli::Write(*std::move(output), base_writer_);
    if (!status.ok()) {
      Fail(std::move(status));
      return;
    }
  }

 private:
  blosc::Options options_;
  riegeli::Writer& base_writer_;
};

class BloscCodec : public ZarrBytesToBytesCodec {
 public:
  class State : public ZarrBytesToBytesCodec::PreparedState {
   public:
    Result<std::unique_ptr<riegeli::Writer>> GetEncodeWriter(
        riegeli::Writer& encoded_writer) const final {
      return std::make_unique<BloscDeferredWriter>(
          blosc::Options{codec_->cname.c_str(), codec_->clevel, codec_->shuffle,
                         codec_->blocksize, codec_->typesize},
          encoded_writer);
    }

    Result<std::unique_ptr<riegeli::Reader>> GetDecodeReader(
        riegeli::Reader& encoded_reader) const final {
      auto output = riegeli::ReadAll(
          encoded_reader,
          [](absl::string_view input) -> absl::StatusOr<std::string> {
            auto output = blosc::Decode(input);
            if (!output.ok()) return std::move(output).status();
            return *std::move(output);
          });
      auto reader = std::make_unique<riegeli::ChainReader<riegeli::Chain>>(
          output.ok() ? riegeli::Chain(std::move(*output)) : riegeli::Chain());
      if (!output.ok()) {
        reader->Fail(std::move(output).status());
      }
      return reader;
    }

    const BloscCodec* codec_;
  };

  Result<PreparedState::Ptr> Prepare(int64_t decoded_size) const final {
    auto state = internal::MakeIntrusivePtr<State>();
    state->codec_ = this;
    return state;
  }

  std::string cname;
  int clevel;
  int shuffle;
  size_t typesize;
  size_t blocksize;
};

constexpr auto ShuffleBinder() {
  namespace jb = ::tensorstore::internal_json_binding;
  return jb::Enum<int, std::string_view>({{BLOSC_NOSHUFFLE, "noshuffle"},
                                          {BLOSC_SHUFFLE, "shuffle"},
                                          {BLOSC_BITSHUFFLE, "bitshuffle"}});
}

}  // namespace

absl::Status BloscCodecSpec::MergeFrom(const ZarrCodecSpec& other,
                                       bool strict) {
  using Self = BloscCodecSpec;
  const auto& other_options = static_cast<const Self&>(other).options;
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::cname>("cname", options, other_options));
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::clevel>("clevel", options, other_options));
  TENSORSTORE_RETURN_IF_ERROR(MergeConstraint<&Options::shuffle>(
      "shuffle", options, other_options, ShuffleBinder()));
  TENSORSTORE_RETURN_IF_ERROR(
      MergeConstraint<&Options::typesize>("typesize", options, other_options));
  TENSORSTORE_RETURN_IF_ERROR(MergeConstraint<&Options::blocksize>(
      "blocksize", options, other_options));
  return absl::OkStatus();
}

ZarrCodecSpec::Ptr BloscCodecSpec::Clone() const {
  return internal::MakeIntrusivePtr<BloscCodecSpec>(*this);
}

Result<ZarrBytesToBytesCodec::Ptr> BloscCodecSpec::Resolve(
    BytesCodecResolveParameters&& decoded, BytesCodecResolveParameters& encoded,
    ZarrBytesToBytesCodecSpec::Ptr* resolved_spec) const {
  auto codec = internal::MakeIntrusivePtr<BloscCodec>();
  codec->cname = options.cname.value_or(BLOSC_LZ4_COMPNAME);
  codec->clevel = options.clevel.value_or(5);
  std::optional<int> shuffle = options.shuffle;
  std::optional<size_t> typesize = options.typesize;
  if (shuffle != BLOSC_NOSHUFFLE && !typesize.has_value()) {
    if (decoded.item_bits == -1 || (decoded.item_bits % 8) != 0 ||
        decoded.item_bits / 8 > BLOSC_MAX_TYPESIZE) {
      if (!shuffle.has_value()) {
        // Disable shuffling due to invalid typesize.
        shuffle = BLOSC_NOSHUFFLE;
        typesize = 1;
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("typesize must be specified explicitly because "
                            "inferred itemsize %d/8 is not supported by Blosc",
                            decoded.item_bits));
      }
    } else {
      typesize = decoded.item_bits / 8;
    }
  }
  codec->typesize = typesize.value_or(1);
  codec->shuffle =
      shuffle.value_or(codec->typesize == 1 ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE);
  codec->blocksize = options.blocksize.value_or(0);
  if (resolved_spec) {
    auto spec = internal::MakeIntrusivePtr<BloscCodecSpec>();
    auto& resolved_options = spec->options;
    resolved_options.cname = codec->cname;
    resolved_options.clevel = codec->clevel;
    resolved_options.shuffle = codec->shuffle;
    if (resolved_options.shuffle != BLOSC_NOSHUFFLE) {
      resolved_options.typesize = codec->typesize;
    }
    resolved_options.blocksize = codec->blocksize;
    *resolved_spec = std::move(spec);
  }
  return codec;
}

namespace {
constexpr auto CodecBinder() {
  namespace jb = ::tensorstore::internal_json_binding;
  return jb::Validate([](const auto& options, std::string* cname) {
    if (cname->find('\0') != std::string::npos ||
        blosc_compname_to_compcode(cname->c_str()) == -1) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Expected one of ", blosc_list_compressors(),
                              " but received: ", QuoteString(*cname)));
    }
    return absl::OkStatus();
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  using Self = BloscCodecSpec;
  using Options = Self::Options;
  namespace jb = ::tensorstore::internal_json_binding;

  RegisterCodec<Self>(
      "blosc",
      jb::Projection<&Self::options>(jb::Sequence(
          jb::Member("cname", jb::Projection<&Options::cname>(
                                  OptionalIfConstraintsBinder(CodecBinder()))),
          jb::Member("clevel",
                     jb::Projection<&Options::clevel>(
                         OptionalIfConstraintsBinder(jb::Integer<int>(0, 9)))),
          jb::Member("shuffle",
                     jb::Projection<&Options::shuffle>(
                         OptionalIfConstraintsBinder(ShuffleBinder()))),
          jb::Member(
              "typesize",
              [](auto is_loading, const auto& options, auto* obj, auto* j) {
                if constexpr (is_loading) {
                  if (obj->shuffle == BLOSC_NOSHUFFLE && j->is_discarded()) {
                    return absl::OkStatus();
                  }
                } else {
                  if (obj->shuffle == BLOSC_NOSHUFFLE) {
                    return absl::OkStatus();
                  }
                }
                return jb::Projection<&Options::typesize>(
                    OptionalIfConstraintsBinder(jb::Integer<size_t>(
                        1, BLOSC_MAX_TYPESIZE)))(is_loading, options, obj, j);
              }),
          jb::Member(
              "blocksize",
              jb::Projection<&Options::blocksize>(OptionalIfConstraintsBinder(
                  jb::Integer<size_t>(0, BLOSC_MAX_BLOCKSIZE))))
          //
          )));
}
}  // namespace

}  // namespace internal_zarr3
}  // namespace tensorstore
