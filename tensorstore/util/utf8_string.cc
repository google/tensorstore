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

#include "tensorstore/util/utf8_string.h"

#include "tensorstore/internal/utf8.h"
#include "tensorstore/serialization/riegeli_delimited.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace serialization {

bool Serializer<Utf8String>::Encode(EncodeSink& sink, const Utf8String& value) {
  return serialization::WriteDelimited(sink.writer(), value.utf8);
}

bool Serializer<Utf8String>::Decode(DecodeSource& source, Utf8String& value) {
  return serialization::ReadDelimitedUtf8(source.reader(), value.utf8);
}

}  // namespace serialization
}  // namespace tensorstore
