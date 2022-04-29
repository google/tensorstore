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
#include <variant>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// A type `T` satisfies the IsIndexVectorOrScalar concept if it is either:
///
///   1. convertible without narrowing to Index (scalar), in which case the
///      nested `extent` constant equals `dynamic_rank` and the nested
///      `normalized_type` alias equals `Index`; or
///
///   2. compatible with `span` with a `span::value_type` of `Index` (vector),
///      in which case the nested `normalized_type` alias is equal to the result
///      type of `span`, and the nested `extent` constant is equal to the
///      `span::extent` of `normalized_type`.
///
/// If `T` satisfies the `IsIndexVectorOrScalar` concept, this metafunction has
/// defines a nested `value` member equal to `true`, as well as nested `extent`
/// and `normalized_type` members.  Otherwise, this metafunction defines a
/// nested `value` member equal to `false`, and no nested `normalized_type` or
/// `extent` members.
///
/// This concept is used to constrain the parameter types of some methods of
/// `DimExpression`.
///
/// \relates DimExpression
template <typename T, typename = std::true_type>
struct IsIndexVectorOrScalar {
  /// Indicates whether `T` satisfies the concept.
  static constexpr bool value = false;

  /// Compile-time length of the vector, or `dynamic_rank` if `T` represents a
  /// scalar or the length is specified at run time.
  ///
  /// Only valid if `value == true`.
  static constexpr DimensionIndex extent = -1;

  /// Normalized scalar/vector type, equal to `Index` or `span<Index, extent>`.
  ///
  /// Only valid if `value == true`.
  using normalized_type = void;
};

// Specialization of IsIndexVectorOrScalar for the scalar case.
template <typename T>
struct IsIndexVectorOrScalar<
    T,
    std::integral_constant<bool, static_cast<bool>(internal::IsIndexPack<T>)>>
    : public std::true_type {
  using normalized_type = Index;
  constexpr static std::ptrdiff_t extent = dynamic_extent;
};

// Specialization of IsIndexVectorOrScalar for the vector case.
template <typename T>
struct IsIndexVectorOrScalar<
    T,
    std::integral_constant<
        bool, static_cast<bool>(
                  std::is_same_v<
                      typename internal::ConstSpanType<T>::value_type, Index>)>>
    : public std::true_type {
  using normalized_type = internal::ConstSpanType<T>;
  constexpr static std::ptrdiff_t extent = normalized_type::extent;
};

namespace internal_index_space {

/// Represents either an index vector or scalar, for use when
/// `IndexVectorOrScalarView` is not suitable.
using IndexVectorOrScalarContainer = std::variant<std::vector<Index>, Index>;

/// Type-erased storage of an Index scalar or a `span<const Index>`.
class IndexVectorOrScalarView {
 public:
  IndexVectorOrScalarView(const IndexVectorOrScalarContainer& c) {
    if (auto* v = std::get_if<std::vector<Index>>(&c)) {
      pointer = v->data();
      size_or_scalar = v->size();
    } else {
      pointer = nullptr;
      size_or_scalar = *std::get_if<Index>(&c);
    }
  }
  IndexVectorOrScalarView(span<const Index> s)
      : pointer(s.data()), size_or_scalar(s.size()) {}
  IndexVectorOrScalarView(const Index scalar)
      : pointer(nullptr), size_or_scalar(scalar) {}
  Index operator[](DimensionIndex i) const {
    return pointer ? pointer[i] : size_or_scalar;
  }
  const Index* pointer;
  Index size_or_scalar;
};

absl::Status CheckIndexVectorSize(IndexVectorOrScalarView indices,
                                  DimensionIndex size);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_VECTOR_OR_SCALAR_H_
