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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_WRITER_H_
#define TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_WRITER_H_

// Riegeli writer that appends a digest.

#include "riegeli/digests/digesting_writer.h"
#include "riegeli/endian/endian_writing.h"

namespace tensorstore {
namespace internal {

// Digest writer that writes the digest as little endian.
struct LittleEndianDigestWriter {
  template <typename T>
  static bool WriteDigest(const T& value, riegeli::Writer& dest) {
    return riegeli::WriteLittleEndian<T>(value, dest);
  }
};

// Writer adapter that appends a digest to the end of the output.
//
// \tparam Digester Riegeli-compatible digester, e.g. `riegeli::Crc32cDigester`.
// \tparam DigestWriter Traits type like `LittleEndianDigestWriter` that defines
//     a static `WriteDigest` method.
template <typename Digester, typename DigestWriter>
class DigestSuffixedWriter
    : public riegeli::DigestingWriter<Digester, riegeli::Writer*> {
  using Base = riegeli::DigestingWriter<Digester, riegeli::Writer*>;

 public:
  using Base::Base;

  void Done() override {
    if (!this->ok()) return;
    auto* base_writer = this->DestWriter();
    Base::Done();
    DigestWriter::WriteDigest(this->Digest(), *base_writer);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_DIGEST_SUFFIXED_WRITER_H_
