// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_SERIALIZATION_STD_VARIANT_H_
#define TENSORSTORE_SERIALIZATION_STD_VARIANT_H_

#include <variant>

#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace serialization {

template <typename... T>
struct Serializer<std::variant<T...>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink,
                                   const std::variant<T...>& value) {
    return serialization::WriteSize(sink.writer(), value.index()) &&
           std::visit(
               [&sink](auto& x) { return serialization::Encode(sink, x); },
               value);
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   std::variant<T...>& value) {
    size_t index;
    if (!serialization::ReadSize(source.reader(), index)) return false;
    if (index >= sizeof...(T)) {
      source.Fail(absl::DataLossError("Invalid variant index"));
      return false;
    }
    return DecodeImpl(source, value, index, std::index_sequence_for<T...>{});
  }

  template <size_t... Is>
  [[nodiscard]] static bool DecodeImpl(DecodeSource& source,
                                       std::variant<T...>& value, size_t index,
                                       std::index_sequence<Is...>) {
    return ((index == Is
                 ? serialization::Decode(source, value.template emplace<Is>())
                 : true) &&
            ...);
  }

  constexpr static bool non_serializable() {
    return (IsNonSerializableLike<T> || ...);
  }
};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_STD_VARIANT_H_
