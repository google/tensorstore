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

#include "tensorstore/internal/riegeli_json_output.h"

#include <string_view>

#include <nlohmann/json.hpp>
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/unowned_to_shared.h"

namespace tensorstore {
namespace internal {

namespace {

struct RiegeliJsonOutputAdapter
    : public ::nlohmann::detail::output_adapter_protocol<char> {
  RiegeliJsonOutputAdapter(riegeli::Writer& writer) : writer_(writer) {}
  void write_character(char c) final { writer_.Write(c); }
  void write_characters(const char* s, size_t length) final {
    writer_.Write(std::string_view(s, length));
  }

  riegeli::Writer& writer_;
};

}  // namespace

bool WriteJson(riegeli::Writer& writer, const ::nlohmann::json& value,
               int indent, char indent_char, bool ensure_ascii,
               ::nlohmann::json::error_handler_t error_handler) {
  RiegeliJsonOutputAdapter output_adapter(writer);
  ::nlohmann::detail::serializer<::nlohmann::json> s(
      internal::UnownedToShared(&output_adapter), indent_char, error_handler);
  s.dump(value, /*pretty_print=*/indent >= 0, ensure_ascii,
         static_cast<unsigned int>(std::max(0, indent)));
  return writer.ok();
}

bool WriteCbor(riegeli::Writer& writer, const ::nlohmann::json& value) {
  if (value.is_discarded()) {
    writer.Fail(
        absl::InvalidArgumentError("Cannot encode discarded json value"));
    return false;
  }
  RiegeliJsonOutputAdapter output_adapter(writer);
  ::nlohmann::detail::binary_writer<::nlohmann::json, char>(
      internal::UnownedToShared(&output_adapter))
      .write_cbor(value);
  return writer.ok();
}

}  // namespace internal
}  // namespace tensorstore
