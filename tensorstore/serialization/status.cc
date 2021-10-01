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

#include "absl/status/status.h"

#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/status.h"

namespace tensorstore {
namespace serialization {

bool ErrorStatusSerializer::Encode(EncodeSink& sink,
                                   const absl::Status& status) {
  assert(!status.ok());
  return serialization::Encode(sink, status);
}

bool ErrorStatusSerializer::Decode(DecodeSource& source, absl::Status& status) {
  if (!serialization::Decode(source, status)) return false;
  if (status.ok()) {
    source.Fail(absl::DataLossError("Expected error status"));
    return false;
  }
  return true;
}

bool Serializer<absl::Status>::Encode(EncodeSink& sink,
                                      const absl::Status& value) {
  if (!serialization::Encode(sink, value.code())) return false;
  if (value.ok()) return true;
  if (!serialization::Encode(sink, value.message())) return false;

  bool ok = true;
  // Encode each payload by the bool value `true` followed by the payload and
  // url.  (absl::Status does not provide access to the number of payloads
  // except by iterating over them.)  Note that we encode the `payload` before
  // the `url`, as that way when decoding we avoid having to make an extra copy
  // of the `url`.
  value.ForEachPayload([&](std::string_view url, const absl::Cord& payload) {
    if (!ok) return;
    ok = serialization::EncodeTuple(sink, true, payload, url);
  });
  if (!ok) return false;
  // The final payload is indicated by `false`.
  return serialization::Encode(sink, false);
}

bool Serializer<absl::Status>::Decode(DecodeSource& source,
                                      absl::Status& value) {
  absl::StatusCode code;
  if (!serialization::Decode(source, code)) return false;
  if (code == absl::StatusCode::kOk) {
    value = absl::OkStatus();
    return true;
  }
  std::string_view message;
  if (!serialization::Decode(source, message)) return false;
  value = absl::Status(code, message);
  while (true) {
    bool has_payload;
    if (!serialization::Decode(source, has_payload)) return false;
    if (!has_payload) break;
    absl::Cord payload;
    std::string_view url;
    if (!serialization::DecodeTuple(source, payload, url)) return false;
    // Note: `url` must be decoded last, and is only valid until the next use of
    // `source`.
    value.SetPayload(url, payload);
  }
  return true;
}

}  // namespace serialization
}  // namespace tensorstore
