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

#include "tensorstore/serialization/json.h"

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/riegeli_json_input.h"
#include "tensorstore/internal/riegeli_json_output.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace serialization {

bool Serializer<::nlohmann::json>::Encode(EncodeSink& sink,
                                          const ::nlohmann::json& value) {
  return internal::WriteCbor(sink.writer(), value);
}

bool Serializer<::nlohmann::json>::Decode(DecodeSource& source,
                                          ::nlohmann::json& value) {
  return internal::ReadCbor(source.reader(), value, /*strict=*/false);
}

}  // namespace serialization
}  // namespace tensorstore
