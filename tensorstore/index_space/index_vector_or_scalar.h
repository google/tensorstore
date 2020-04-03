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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_VECTOR_OR_SCALAR_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_VECTOR_OR_SCALAR_H_

#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// A type `T` satisfies the IsIndexVectorOrScalar concept if it is either:
///
///   1. convertible without narrowing to Index (scalar), in which case the
///      nested `extent` constant equals `dynamic_extent` and the nested
///      `normalized_type` alias equals `Index`; or
///
///   2. compatible with `span` with a `value_type` of `Index` (vector), in
///      which case the nested `normalized_type` alias is equal to the result
///      type of `span`, and the nested `extent` constant is equal to the
///      `extent` member of `normalized_type`.
///
/// If `T` satisfies the IsIndexVectorOrScalar concept, this metafunction has
/// defines a nested `value` member equal to `true`, as well as nested `extent`
/// and `normalized_type` members.  Otherwise, this metafunction defines a
/// nested `value` member equal to `false`, and no nested `normalized_type` or
/// `extent` members.
///
/// This concept is used to constrain the parameter types of some methods of
/// DimExpression.
template <typename T, typename = std::true_type>
struct IsIndexVectorOrScalar : public std::false_type {};

/// Specialization of IsIndexVectorOrScalar for the scalar case.
template <typename T>
struct IsIndexVectorOrScalar<T, typename internal::IsIndexPack<T>::type>
    : public std::true_type {
  using normalized_type = Index;
  constexpr static std::ptrdiff_t extent = dynamic_extent;
};

/// Specialization of IsIndexVectorOrScalar for the vector case.
template <typename T>
struct IsIndexVectorOrScalar<
    T, typename std::is_same<typename internal::ConstSpanType<T>::value_type,
                             Index>::type> : public std::true_type {
  using normalized_type = internal::ConstSpanType<T>;
  constexpr static std::ptrdiff_t extent = normalized_type::extent;
};

namespace internal_index_space {

/// Type-erased storage of an Index scalar or a `span<const Index>`.
class IndexVectorOrScalar {
 public:
  IndexVectorOrScalar(span<const Index> s)
      : pointer(s.data()), size_or_scalar(s.size()) {}
  IndexVectorOrScalar(const Index scalar)
      : pointer(nullptr), size_or_scalar(scalar) {}
  const Index* pointer;
  Index size_or_scalar;
  Index operator[](DimensionIndex i) const {
    return pointer ? pointer[i] : size_or_scalar;
  }
};

Status CheckIndexVectorSize(IndexVectorOrScalar indices, DimensionIndex size);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_VECTOR_OR_SCALAR_H_
