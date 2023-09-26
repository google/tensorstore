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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_READER_H_
#define TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_READER_H_

// Riegeli reader that validates a digest at the end of the data.

#include <stddef.h>

#include <optional>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "riegeli/base/arithmetic.h"
#include "riegeli/base/object.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/digests/digesting_reader.h"
#include "riegeli/endian/endian_reading.h"

namespace tensorstore {
namespace internal {

// Digest verifier for use with `DigestSuffixReader` where the `DigestType` is
// an integer type.  Reads the digest as a little endian number.
struct LittleEndianDigestVerifier {
  // Returns the size in bytes of the digest, for a given digest type `T`.
  template <typename T>
  static constexpr size_t Size() {
    return sizeof(T);
  }

  // Reads and verifies the digest.
  template <typename T>
  static absl::Status VerifyDigest(const T& digest, riegeli::Reader& reader) {
    T expected_digest;
    if (!riegeli::ReadLittleEndian<T>(reader, expected_digest)) {
      return reader.AnnotateStatus(
          absl::DataLossError("Unexpected end of input"));
    }
    if (expected_digest != digest) {
      return absl::DataLossError(absl::StrFormat(
          "Digest mismatch, stored digest is 0x%0*x but computed digest is "
          "0x%0*x",
          sizeof(T) * 2, expected_digest, sizeof(T) * 2, digest));
    }
    return absl::OkStatus();
  }
};

// Reader adapter that strips an appended digest from the end of the input, and
// validates it.
//
// \tparam Digester Riegeli-compatible digester, e.g. `riegeli::Crc32cDigester`.
// \tparam DigestVerifier Traits type like `LittleEndianDigestVerifier` that
//     defines static `Size` and `VerifyDigest` methods.  See
//     `LittleEndianDigestVerifier` for details.
//
// WARNING: Currently, the implementation is efficient only if the base reader
// supports `Size` or `payload_size` is specified explicitly.  Otherwise, this
// has to first read the entire input into an `absl::Cord`.
template <typename Digester, typename DigestVerifier>
class DigestSuffixedReader
    : public riegeli::DigestingReader<
          Digester, riegeli::LimitingReader<riegeli::Reader*>> {
  using Base =
      riegeli::DigestingReader<Digester,
                               riegeli::LimitingReader<riegeli::Reader*>>;

 public:
  using typename Base::DigestType;
  template <typename... DigesterArg>
  explicit DigestSuffixedReader(riegeli::Reader* src,
                                std::optional<size_t> payload_size = {},
                                DigesterArg&&... digester_arg)
      : Base(riegeli::Closed{}) {
    size_t inner_limit;
    if (payload_size) {
      inner_limit = *payload_size;
    } else {
      size_t limit;
      if (std::optional<size_t> size_opt;
          src->SupportsSize() && (size_opt = src->Size()).has_value()) {
        limit = *size_opt;
        auto pos = src->pos();
        limit -= riegeli::UnsignedMin(limit, pos);
      } else {
        absl::Cord cord;
        if (auto status = riegeli::ReadAll(src, cord); !status.ok()) {
          this->FailWithoutAnnotation(std::move(status));
          return;
        }
        limit = cord.size();
        cord_reader_.Reset(std::move(cord));
        src = &cord_reader_;
      }
      constexpr size_t digest_size =
          DigestVerifier::template Size<DigestType>();
      if (digest_size > limit) {
        this->FailWithoutAnnotation(absl::DataLossError(
            absl::StrFormat("Input size of %d is less than digest size of %d",
                            limit, digest_size)));
        return;
      }
      inner_limit = limit - digest_size;
    }
    Base::Reset(
        std::tuple(src, riegeli::LimitingReaderBase::Options().set_exact_length(
                            inner_limit)),
        std::forward<DigesterArg>(digester_arg)...);
  }

  bool SupportsSize() override { return Base::is_open(); }

  absl::optional<riegeli::Position> SizeImpl() override {
    if (!Base::is_open()) return absl::nullopt;
    return Base::src().max_pos();
  }

  void SetReadAllHintImpl(bool read_all_hint) override {
    if (!Base::is_open()) return;
    Base::src().SrcReader()->SetReadAllHint(read_all_hint);
  }

 private:
  void Done() override {
    riegeli::Reader* base_reader = Base::src().SrcReader();
    Base::Done();
    if (!this->ok()) return;
    auto status = DigestVerifier::VerifyDigest(Base::Digest(), *base_reader);
    if (!status.ok()) {
      this->FailWithoutAnnotation(std::move(status));
    }
  }

  riegeli::CordReader<absl::Cord> cord_reader_{riegeli::Closed{}};
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_READER_H_
