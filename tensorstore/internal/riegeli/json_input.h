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

#ifndef TENSORSTORE_INTERNAL_RIEGELI_JSON_INPUT_H_
#define TENSORSTORE_INTERNAL_RIEGELI_JSON_INPUT_H_

/// \file
///
/// Reads ::nlohmann::json as text or CBOR from a `riegeli::Reader`.

#include <nlohmann/json.hpp>
#include "riegeli/bytes/reader.h"

namespace tensorstore {
namespace internal {

/// Parses normal text-format JSON from a `riegeli::Reader`.
///
/// If an error occurs, returns `false`.
[[nodiscard]] bool ReadJson(riegeli::Reader& reader, ::nlohmann::json& value,
                            bool ignore_comments = false);

/// Parses CBOR from a `riegeli::Reader`.
///
/// If an error occurs, returns `false`.
[[nodiscard]] bool ReadCbor(riegeli::Reader& reader, ::nlohmann::json& value,
                            bool strict = true,
                            ::nlohmann::json::cbor_tag_handler_t tag_handler =
                                ::nlohmann::json::cbor_tag_handler_t::error);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RIEGELI_JSON_INPUT_H_
