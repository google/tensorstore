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

#ifndef TENSORSTORE_UTIL_APPLY_MEMBERS_APPLY_MEMBERS_H_
#define TENSORSTORE_UTIL_APPLY_MEMBERS_APPLY_MEMBERS_H_

/// \file
///
/// Defines simple reflection interface for "aggregate" types.
///
/// Aggregate types are types that contain a fixed number of members, such as a
/// simple structs, tuples, pairs, fixed-size arrays.  This interface is not
/// suitable for dynamically-sized containers, like `std::vector`.
///
/// This reflection interface serves as a building block for defining
/// type-specific serialization, context binding, and cache key computation
/// operations based on a single `ApplyMembers` definition.
///
/// ApplyMembers is used to support the following mechanisms:
/// * ContextBindingTraits: internal/context_binding.h
/// * GarbageCollection: util/garbage_collection/garbage_collection.h
/// * Serializer: serialization/serialization.h
/// * EncodeCacheKey: internal/cache_key/cache_key.h
/// * EstimateHeapUsage: internal/estimate_heap_usage/estimate_heap_usage.h
///
/// Separate headers provide support for certain standard types:
///
///   - std::tuple: std_tuple.h
///   - std::pair: std_pair.h
///   - std::array: std_array.h
///   - std::complex: std_complex.h
///
/// Example usage:
///
///     struct Foo {
///       int x;
///       std::string y;
///       constexpr static auto ApplyMembers = [](auto &&self, auto f) {
///         return f(self.x, self.y);
///       };
///     };
///
///     struct Bar {
///       std::pair<int, Foo> x;
///       std::array<int, 3> y;
///       constexpr static auto ApplyMembers = [](auto &&self, auto f) {
///         return f(self.x, self.y);
///       };
///     };
///
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"

namespace half_float {
class half;
}

namespace tensorstore {

class bfloat16_t;

/// Provides access to the members of aggregate types.
///
/// If `T` is a supported type, `ApplyMembers<T>::Apply(x, f)`, where `x` is an
/// optionally const/ref-qualified `T`, returns the result of invoking `f` with
/// an argument pack consisting of references to all of the members of `x`.
///
/// Types are supported by `ApplyMembers` either by an explicit specialization
/// of `tensorstore::ApplyMembers`, or by defining a public static
/// `ApplyMembers` method.
///
/// Additionally, empty types are automatically supported.
///
/// The variable template `SupportsApplyMembers` may be used to check whether
/// `ApplyMembers` is supported for a given type.
template <typename T, typename SFINAE = void>
struct ApplyMembers {
  using NotSpecialized = void;
};

namespace internal_apply_members {

struct IgnoreMembers {
  template <typename... T>
  constexpr void operator()(const T&...) const {}
};

template <typename T, typename SFINAE = void>
struct SupportsApplyMembersImpl : public std::true_type {};

template <typename T>
struct SupportsApplyMembersImpl<T, typename ApplyMembers<T>::NotSpecialized>
    : public std::false_type {};

template <typename T>
using MemberApplyMembersCallExpr = decltype(T::ApplyMembers(
    std::declval<const T&>(), internal_apply_members::IgnoreMembers{}));

}  // namespace internal_apply_members

template <typename T>
struct ApplyMembers<
    T,
    std::enable_if_t<
        !std::is_empty_v<T>,
        std::void_t<internal_apply_members::MemberApplyMembersCallExpr<T>>>> {
  template <typename X, typename F>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static constexpr auto Apply(X&& x, F f) {
    return T::ApplyMembers(x, std::move(f));
  }
};

template <typename T>
struct ApplyMembers<T, std::enable_if_t<std::is_empty_v<T>>> {
  template <typename X, typename F>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static constexpr auto Apply(X&& x, F f) {
    return f();
  }
};

/// Evaluates to `true` if `T` supports `tensorstore::ApplyMembers`.
template <typename T>
constexpr inline bool SupportsApplyMembers =
    internal_apply_members::SupportsApplyMembersImpl<T>::value;

/// Evaluates to `true` if `T` can be safely serialized via memcpy.
template <typename T, typename SFINAE = void>
constexpr inline bool SerializeUsingMemcpy =
    std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_enum_v<T>;

template <>
constexpr inline bool SerializeUsingMemcpy<bfloat16_t> = true;

template <>
constexpr inline bool SerializeUsingMemcpy<half_float::half> = true;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_APPLY_MEMBERS_APPLY_MEMBERS_H_
