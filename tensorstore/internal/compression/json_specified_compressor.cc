// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/compression/json_specified_compressor.h"

#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/write.h"
#include "riegeli/bytes/writer.h"

namespace tensorstore {
namespace internal {

JsonSpecifiedCompressor::~JsonSpecifiedCompressor() = default;

namespace {
// Buffers writes to a `absl::Cord`, and then in `Done`, calls `Encode` and
// forwards the result to another `Writer`.
class DeferredWriter : public riegeli::CordWriter<absl::Cord> {
  using Base = riegeli::CordWriter<absl::Cord>;

 public:
  explicit DeferredWriter(JsonSpecifiedCompressor::Ptr compressor,
                          std::unique_ptr<riegeli::Writer> base_writer,
                          size_t element_bytes)
      : compressor_(std::move(compressor)),
        base_writer_(std::move(base_writer)),
        element_bytes_(element_bytes) {}

  void Done() override {
    Base::Done();
    absl::Cord output;
    auto status = compressor_->Encode(dest(), &output, element_bytes_);
    if (!status.ok()) {
      Fail(std::move(status));
      return;
    }
    status = riegeli::Write(std::move(output), std::move(base_writer_));
    if (!status.ok()) {
      Fail(std::move(status));
      return;
    }
  }

 private:
  JsonSpecifiedCompressor::Ptr compressor_;
  std::unique_ptr<riegeli::Writer> base_writer_;
  size_t element_bytes_;
};
}  // namespace

std::unique_ptr<riegeli::Writer> JsonSpecifiedCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  // Note that the implementation strategy here necessarily differs from that in
  // `GetReader` because we must return a `Writer` that receives output as it is
  // provided.  In contrast, `GetReader` can just immediately consume all of the
  // input, decode, and return a `CordReader`.
  return std::make_unique<DeferredWriter>(JsonSpecifiedCompressor::Ptr(this),
                                          std::move(base_writer),
                                          element_bytes);
}

std::unique_ptr<riegeli::Reader> JsonSpecifiedCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  absl::Cord input;
  auto status = riegeli::ReadAll(std::move(base_reader), input);
  absl::Cord output;
  if (status.ok()) {
    status = Decode(input, &output, element_bytes);
  }
  auto reader =
      std::make_unique<riegeli::CordReader<absl::Cord>>(std::move(output));
  if (!status.ok()) {
    reader->Fail(std::move(status));
  }
  return reader;
}

absl::Status JsonSpecifiedCompressor::Encode(const absl::Cord& input,
                                             absl::Cord* output,
                                             std::size_t element_bytes) const {
  auto base_writer = std::make_unique<riegeli::CordWriter<absl::Cord*>>(output);
  auto writer = GetWriter(std::move(base_writer), element_bytes);
  return riegeli::Write(input, std::move(writer));
}

absl::Status JsonSpecifiedCompressor::Decode(const absl::Cord& input,
                                             absl::Cord* output,
                                             std::size_t element_bytes) const {
  auto base_reader =
      std::make_unique<riegeli::CordReader<const absl::Cord*>>(&input);
  auto reader = GetReader(std::move(base_reader), element_bytes);
  return riegeli::ReadAll(std::move(reader), *output);
}

}  // namespace internal
}  // namespace tensorstore
