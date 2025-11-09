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

#include "tensorstore/internal/riegeli/delimited.h"

#include <string>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace serialization {

bool ReadDelimitedUtf8(riegeli::Reader& reader, std::string& dest) {
  if (!serialization::ReadDelimited(reader, dest)) return false;
  if (!internal::IsValidUtf8(dest)) {
    reader.Fail(absl::DataLossError(tensorstore::StrCat(
        "String is not valid utf-8: ", tensorstore::QuoteString(dest))));
    return false;
  }
  return true;
}

namespace internal_serialization {
void FailInvalidSize(riegeli::Reader& reader) {
  reader.Fail(absl::DataLossError("Failed to read size value as varint"));
}
}  // namespace internal_serialization

}  // namespace serialization
}  // namespace tensorstore
