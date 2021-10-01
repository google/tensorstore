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

#ifndef TENSORSTORE_SERIALIZATION_SPAN_H_
#define TENSORSTORE_SERIALIZATION_SPAN_H_

#include <cstddef>
#include <type_traits>

#include "tensorstore/internal/attributes.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace serialization {

/// Serializer for contiguous ranges where the size is known.
///
/// Both static and dynamic extents `N` are supported.  Unlike
/// `ContainerSerializer`, the size is not encoded, even if
/// `N == dynamic_extent`.  Each element is simply encoded sequentially.
///
/// When decoding, the caller must supply a `span` of the correct length (if
/// `N == dynamic_extent`) and that references valid memory; the `span` object
/// itself is not modified by `Decode`, only the referenced elements.
///
/// Note: For static extent `N`, the encoded format is the same as
/// `std::array<T, N>`.
template <typename T, ptrdiff_t N,
          typename ElementSerializer = Serializer<std::remove_cv_t<T>>>
struct SpanSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, span<const T, N> value) const {
    for (const auto& element : value) {
      if (!element_serializer.Encode(sink, element)) return false;
    }
    return true;
  }
  [[nodiscard]] bool Decode(DecodeSource& source, span<T, N> value) const {
    for (auto& element : value) {
      if (!element_serializer.Decode(source, element)) return false;
    }
    return true;
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementSerializer
      element_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<ElementSerializer>;
  }
};

template <typename T, ptrdiff_t N>
struct Serializer<span<T, N>> : public SpanSerializer<T, N> {};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_SPAN_H_
