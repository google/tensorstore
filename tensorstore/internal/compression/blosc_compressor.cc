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

#include "tensorstore/internal/compression/blosc_compressor.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/chain.h"
#include "riegeli/bytes/chain_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/write.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/compression/blosc.h"

namespace tensorstore {
namespace internal {
namespace {

// Buffers writes to a `std::string`, and then in `Done`, calls `blosc::Encode`
// and forwards the result to another `Writer`.
class BloscDeferredWriter : public riegeli::CordWriter<absl::Cord> {
 public:
  explicit BloscDeferredWriter(blosc::Options options,
                               std::unique_ptr<riegeli::Writer> base_writer)
      : CordWriter(riegeli::CordWriterBase::Options().set_max_block_size(
            std::numeric_limits<size_t>::max())),
        options_(std::move(options)),
        base_writer_(std::move(base_writer)) {}

  void Done() override {
    CordWriter::Done();
    auto output = blosc::Encode(dest().Flatten(), options_);
    if (!output.ok()) {
      Fail(std::move(output).status());
      return;
    }
    auto status = riegeli::Write(*std::move(output), std::move(base_writer_));
    if (!status.ok()) {
      Fail(std::move(status));
      return;
    }
  }

 private:
  blosc::Options options_;
  std::unique_ptr<riegeli::Writer> base_writer_;
};

}  // namespace

std::unique_ptr<riegeli::Writer> BloscCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  return std::make_unique<BloscDeferredWriter>(
      blosc::Options{codec.c_str(), level, shuffle, blocksize, element_bytes},
      std::move(base_writer));
}

std::unique_ptr<riegeli::Reader> BloscCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  auto output = riegeli::ReadAll(
      std::move(base_reader),
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

}  // namespace internal
}  // namespace tensorstore
