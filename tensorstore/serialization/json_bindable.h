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

#ifndef TENSORSTORE_SERIALIZATION_JSON_BINDABLE_H_
#define TENSORSTORE_SERIALIZATION_JSON_BINDABLE_H_

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/serialization/json.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace serialization {

template <typename T>
struct JsonBindableSerializer {
  using JsonValue = UnwrapResultType<decltype(std::declval<T>().ToJson())>;
  [[nodiscard]] static bool Encode(EncodeSink& sink, const T& value) {
    Result<JsonValue> json_result = value.ToJson();
    if (!json_result.ok()) {
      sink.Fail(std::move(json_result).status());
      return false;
    }
    return serialization::Encode(sink, *json_result);
  }

  [[nodiscard]] static bool Decode(DecodeSource& source, T& value) {
    JsonValue json;
    if (!serialization::Decode(source, json)) return false;
    TENSORSTORE_ASSIGN_OR_RETURN(value, T::FromJson(std::move(json)),
                                 (source.Fail(_), false));
    return true;
  }
};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_JSON_BINDABLE_H_
