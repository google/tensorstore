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

#include "tensorstore/internal/riegeli_json_input.h"

#include <nlohmann/json.hpp>
#include "riegeli/bytes/reader.h"

namespace tensorstore {
namespace internal {

namespace {

struct RiegeliJsonInputAdapter {
  using char_type = char;
  std::char_traits<char>::int_type get_character() {
    char c;
    if (!reader.Read(c)) return std::char_traits<char>::eof();
    return std::char_traits<char>::to_int_type(c);
  }

  riegeli::Reader& reader;
};

}  // namespace

bool ReadJson(riegeli::Reader& reader, ::nlohmann::json& value,
              bool ignore_comments) {
  ::nlohmann::detail::parser<::nlohmann::json, RiegeliJsonInputAdapter>(
      RiegeliJsonInputAdapter{reader}, nullptr, /*allow_exceptions=*/false,
      ignore_comments)
      .parse(/*strict=*/true, value);
  if (value.is_discarded() || !reader.ok()) {
    reader.Fail(absl::DataLossError("Failed to parse JSON"));
    return false;
  }
  return true;
}

bool ReadCbor(riegeli::Reader& reader, ::nlohmann::json& value, bool strict,
              ::nlohmann::json::cbor_tag_handler_t tag_handler) {
  ::nlohmann::detail::json_sax_dom_parser<::nlohmann::json> sdp(
      value, /*allow_exceptions=*/false);
  if (!::nlohmann::detail::binary_reader<::nlohmann::json,
                                         RiegeliJsonInputAdapter>(
           RiegeliJsonInputAdapter{reader})
           .sax_parse(::nlohmann::detail::input_format_t::cbor, &sdp, strict,
                      tag_handler) ||
      !reader.ok()) {
    reader.Fail(absl::DataLossError("Failed to parse CBOR"));
    return false;
  }
  return true;
}

}  // namespace internal
}  // namespace tensorstore
