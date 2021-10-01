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

#include "tensorstore/serialization/serialization.h"

#include <cstring>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace serialization {

namespace internal_serialization {
void FailNonNull(DecodeSource& source) {
  source.Fail(serialization::DecodeError("Expected non-null value"));
}
void FailEof(DecodeSource& source) {
  source.Fail(serialization::DecodeError("Unexpected end of input"));
}
}  // namespace internal_serialization

void EncodeSink::Fail(absl::Status status) {
  assert(!status.ok());
  writer().Fail(std::move(status));
}

void DecodeSource::Fail(absl::Status status) {
  assert(!status.ok());
  reader().Fail(std::move(status));
}

absl::Status DecodeError() {
  return absl::DataLossError("Failed to decode value");
}

absl::Status DecodeError(std::string_view message) {
  return absl::DataLossError(tensorstore::StrCat("Error decoding: ", message));
}

namespace internal_serialization {

absl::Status NonSerializableError() {
  return absl::InvalidArgumentError("Serialization not supported");
}

}  // namespace internal_serialization

}  // namespace serialization
}  // namespace tensorstore
