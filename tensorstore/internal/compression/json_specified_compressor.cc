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

namespace tensorstore {
namespace internal {

JsonSpecifiedCompressor::~JsonSpecifiedCompressor() = default;

bool JsonSpecifiedCompressor::valid() const { return true; }

absl::Status JsonSpecifiedCompressor::Encode(const absl::Cord& input,
                                             absl::Cord* output,
                                             std::size_t element_bytes) const {
  return absl::UnimplementedError("");
}

absl::Status JsonSpecifiedCompressor::Decode(const absl::Cord& input,
                                             absl::Cord* output,
                                             std::size_t element_bytes) const {
  return absl::UnimplementedError("");
}

bool JsonSpecifiedCompressor::Unregistered::valid() const { return false; }

}  // namespace internal
}  // namespace tensorstore
