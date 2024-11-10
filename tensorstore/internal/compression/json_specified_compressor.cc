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

#include <stddef.h>

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/write.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

JsonSpecifiedCompressor::~JsonSpecifiedCompressor() = default;

absl::Status JsonSpecifiedCompressor::Encode(const absl::Cord& input,
                                             absl::Cord* output,
                                             size_t element_bytes) const {
  riegeli::CordWriter<> base_writer(
      output, riegeli::CordWriterBase::Options().set_append(true));
  auto writer = GetWriter(base_writer, element_bytes);

  TENSORSTORE_RETURN_IF_ERROR(
      riegeli::Write(input, std::move(writer)),
      MaybeConvertStatusTo(_, absl::StatusCode::kInvalidArgument));
  if (!base_writer.Close()) return base_writer.status();
  return absl::OkStatus();
}

absl::Status JsonSpecifiedCompressor::Decode(const absl::Cord& input,
                                             absl::Cord* output,
                                             size_t element_bytes) const {
  riegeli::CordReader<> base_reader(&input);
  auto reader = GetReader(base_reader, element_bytes);

  TENSORSTORE_RETURN_IF_ERROR(
      riegeli::ReadAndAppendAll(std::move(reader), *output),
      MaybeConvertStatusTo(_, absl::StatusCode::kInvalidArgument));
  if (!base_reader.VerifyEndAndClose()) return base_reader.status();
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
