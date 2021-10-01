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

#ifndef TENSORSTORE_SERIALIZATION_RESULT_H_
#define TENSORSTORE_SERIALIZATION_RESULT_H_

#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace serialization {

template <typename T>
struct Serializer<Result<T>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const Result<T>& value) {
    return serialization::Encode(sink, value.ok()) &&
           (value.ok() ? serialization::Encode(sink, *value)
                       : serialization::Encode(sink, value.status()));
  }

  [[nodiscard]] static bool Decode(DecodeSource& source, Result<T>& value) {
    bool has_value;
    if (!serialization::Decode(source, has_value)) return false;
    if (has_value) {
      return serialization::Decode(source, value.emplace());
    } else {
      absl::Status status;
      if (!ErrorStatusSerializer::Decode(source, status)) return false;
      value = std::move(status);
      return true;
    }
  }
};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_RESULT_H_
