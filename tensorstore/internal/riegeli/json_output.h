// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_JSON_OUTPUT_H_
#define TENSORSTORE_INTERNAL_RIEGELI_JSON_OUTPUT_H_

/// \file
///
/// Writes ::nlohmann::json as text or CBOR to `riegeli::Writer`.

#include <nlohmann/json.hpp>
#include "riegeli/bytes/writer.h"

namespace tensorstore {
namespace internal {

/// Writes the normal text-format JSON representation to a `riegeli::Writer`.
[[nodiscard]] bool WriteJson(riegeli::Writer& writer,
                             const ::nlohmann::json& value, int indent = -1,
                             char indent_char = ' ', bool ensure_ascii = false,
                             ::nlohmann::json::error_handler_t error_handler =
                                 ::nlohmann::json::error_handler_t::strict);

/// Writes the CBOR representation to a `riegeli::Reader`.
[[nodiscard]] bool WriteCbor(riegeli::Writer& writer,
                             const ::nlohmann::json& value);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_JSON_OUTPUT_H_
