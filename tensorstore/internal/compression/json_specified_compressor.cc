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

#include <utility>

#include "absl/status/status.h"
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
                                             std::size_t element_bytes) const {
  auto base_writer = std::make_unique<riegeli::CordWriter<>>(
      output, riegeli::CordWriterBase::Options().set_append(true));
  auto writer = GetWriter(std::move(base_writer), element_bytes);

  TENSORSTORE_RETURN_IF_ERROR(
      riegeli::Write(input, std::move(writer)),
      MaybeConvertStatusTo(_, absl::StatusCode::kInvalidArgument));
  return absl::OkStatus();
}

absl::Status JsonSpecifiedCompressor::Decode(const absl::Cord& input,
                                             absl::Cord* output,
                                             std::size_t element_bytes) const {
  auto base_reader = std::make_unique<riegeli::CordReader<>>(&input);
  auto reader = GetReader(std::move(base_reader), element_bytes);

  TENSORSTORE_RETURN_IF_ERROR(
      riegeli::ReadAndAppendAll(std::move(reader), *output),
      MaybeConvertStatusTo(_, absl::StatusCode::kInvalidArgument));
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
