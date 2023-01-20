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

#ifndef TENSORSTORE_UTIL_SPAN_H_
#define TENSORSTORE_UTIL_SPAN_H_

/// \file
/// C++17-compatible implementation of C++20 span type.

#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/gdb_scripting.h"

TENSORSTORE_GDB_AUTO_SCRIPT("span_gdb.py")

namespace tensorstore {

/// Indicates that the extent is specified at run time.
///
/// \relates span
constexpr std::ptrdiff_t dynamic_extent = -1;

template <typename T, std::ptrdiff_t Extent = dynamic_extent>
class span;

namespace internal_span {

template <typename SourceElement, std::ptrdiff_t SourceExtent,
          typename DestElement, std::ptrdiff_t DestExtent>
constexpr inline bool IsSpanImplicitlyConvertible =
    std::is_convertible_v<SourceElement (*)[], DestElement (*)[]> &&
    (SourceExtent == DestExtent || DestExtent == dynamic_extent);

template <typename Container>
constexpr inline bool IsArrayOrSpan = false;

template <typename T, std::size_t N>
constexpr inline bool IsArrayOrSpan<std::array<T, N>> = true;

template <typename T, std::ptrdiff_t Extent>
constexpr inline bool IsArrayOrSpan<span<T, Extent>> = true;

template <typename T, typename Container,
          typename Pointer = decltype(std::declval<Container>().data()),
          typename Size = decltype(std::declval<Container>().size())>
using EnableIfCompatibleContainer = std::enable_if_t<
    !IsArrayOrSpan<std::remove_cv_t<Container>> &&
    std::is_convertible_v<std::remove_pointer_t<Pointer> (*)[], T (*)[]>>;

template <typename Container,
          typename Pointer = decltype(std::declval<Container>().data()),
          typename Size = decltype(std::declval<Container>().size())>
using ContainerElementType =
    std::enable_if_t<!IsArrayOrSpan<std::remove_cv_t<Container>>,
                     std::remove_pointer_t<Pointer>>;

template <typename T, typename SFINAE = void>
struct SpanTypeHelper {};

template <typename T, std::ptrdiff_t Extent>
struct SpanTypeHelper<span<T, Extent>> {
  using element_type = T;
  constexpr static std::ptrdiff_t extent = Extent;
};

template <typename T, std::ptrdiff_t Extent>
struct SpanTypeHelper<const span<T, Extent>> {
  using element_type = T;
  constexpr static std::ptrdiff_t extent = Extent;
};

template <typename T, std::size_t Extent>
struct SpanTypeHelper<T[Extent]> {
  using element_type = T;
  constexpr static std::ptrdiff_t extent = Extent;
};

template <typename T, std::size_t Extent>
struct SpanTypeHelper<std::array<T, Extent>> {
  using element_type = T;
  constexpr static std::ptrdiff_t extent = Extent;
};

template <typename T, std::size_t Extent>
struct SpanTypeHelper<const std::array<T, Extent>> {
  using element_type = const T;
  constexpr static std::ptrdiff_t extent = Extent;
};

template <typename T>
struct SpanTypeHelper<T, std::void_t<internal_span::ContainerElementType<T>>> {
  using element_type = internal_span::ContainerElementType<T>;
  constexpr static std::ptrdiff_t extent = dynamic_extent;
};

constexpr inline ptrdiff_t SubspanExtent(ptrdiff_t extent, ptrdiff_t offset,
                                         ptrdiff_t count) {
  return count == dynamic_extent && extent == dynamic_extent ? dynamic_extent
         : count == dynamic_extent                           ? extent - offset
                                                             : count;
}

}  // namespace internal_span

/// Unowned view of a contiguous 1-d array of elements, similar to `std::span`.
///
/// .. note::
///
///    This class is nearly identical to `std::span`; the main difference is
///    that in this implementation, `size` has a signed return type of
///    `ptrdiff_t` rather than `size_t`.
///
/// \tparam T Element type, must be a valid array element type.  May be
///     ``const`` or ``volatile`` qualified, but must not be a reference
///     type.
/// \tparam Extent Static extent of the array, or `dynamic_extent` to indicate
///     that the extent is specified at run time.
/// \ingroup utilities
template <typename T, std::ptrdiff_t Extent>
class span {
  static_assert(Extent == dynamic_extent || Extent >= 0,
                "Extent must be non-negative or -1.");

 public:
  /// Element type of the array.
  using element_type = T;

  /// Unqualified element type.
  using value_type = std::remove_cv_t<T>;

  /// Type used for indexing.
  using index_type = std::ptrdiff_t;

  /// Type used for indexing.
  using difference_type = std::ptrdiff_t;

  /// Pointer to an element.
  using pointer = T*;

  /// Pointer to a const-qualified element.
  using const_pointer = const T*;

  /// Reference to an element.
  using reference = T&;

  /// Reference to a const-qualified element.
  using const_reference = const T&;

  /// Iterator type.
  using iterator = pointer;

  /// Constant iterator type.
  using const_iterator = const_pointer;

  /// Reverse iterator type.
  using reverse_iterator = std::reverse_iterator<iterator>;

  /// Constant reverse iterator type.
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  /// Static extent.
  static constexpr std::ptrdiff_t extent = Extent;

  /// Constructs an empty/invalid array.
  ///
  /// If `Extent == dynamic_extent` or `Extent == 0`, then this constructs a
  /// valid view of an empty array, with `data() == nullptr` and `size() == 0`.
  ///
  /// If `Extent > 0`, then this constructs an invalid view of a non-empty
  /// array, with `data() == nullptr` and `size() == Extent`.
  ///
  /// \id default
  constexpr span() noexcept : data_(nullptr), size_{} {}

  /// Constructs from the specified pointer and length.
  ///
  /// \dchecks `Extent == dynamic_extent || count == Extent`
  /// \id pointer, count
  constexpr span(pointer ptr TENSORSTORE_LIFETIME_BOUND,
                 index_type count) noexcept
      : data_(ptr), size_{} {
    if constexpr (Extent == dynamic_extent) {
      assert(count >= 0);
      size_ = count;
    } else {
      assert(count == Extent);
    }
  }

  /// Constructs from begin and end pointers.
  ///
  /// \dchecks `Extent == dynamic_extent || Extent == (end - begin)`
  /// \id begin, end
  template <
      // Add an extra dummy template parameter to ensure this overload ranks
      // lower than the (pointer, index_type) overload in the case of a call of
      // the form (pointer, 0).
      //
      // See https://github.com/Microsoft/GSL/issues/541.
      typename = void>
  constexpr span(pointer begin TENSORSTORE_LIFETIME_BOUND,
                 pointer end TENSORSTORE_LIFETIME_BOUND) noexcept
      : span(begin, end - begin) {}

  /// Constructs from an array or `std::array`.
  ///
  /// \id array
  template <std::size_t N, typename = std::enable_if_t<
                               (Extent == dynamic_extent || Extent == N)>>
  constexpr span(T (&arr TENSORSTORE_LIFETIME_BOUND)[N]) noexcept
      : span(arr, N) {}
  template <std::size_t N, typename = std::enable_if_t<
                               (Extent == dynamic_extent || Extent == N)>>
  constexpr span(
      std::array<value_type, N>& arr TENSORSTORE_LIFETIME_BOUND) noexcept
      : span(arr.data(), N) {}
  template <
      std::size_t N, typename U = T,
      typename = std::enable_if_t<std::is_const_v<U> &&
                                  (Extent == dynamic_extent || Extent == N)>>
  constexpr span(
      const std::array<value_type, N>& arr TENSORSTORE_LIFETIME_BOUND) noexcept
      : span(arr.data(), N) {}

  /// Constructs from a container with ``data`` and ``size`` methods.
  ///
  /// \id container
  template <typename Container,
            typename = internal_span::EnableIfCompatibleContainer<T, Container>>
  constexpr span(Container& cont TENSORSTORE_LIFETIME_BOUND)
      : span(cont.data(), cont.size()) {}
  template <
      typename Container,
      typename = internal_span::EnableIfCompatibleContainer<T, const Container>>
  constexpr span(const Container& cont TENSORSTORE_LIFETIME_BOUND)
      : span(cont.data(), cont.size()) {}

  /// Converts from a compatible span type.
  ///
  /// \id convert
  template <typename U, std::ptrdiff_t N,
            typename = std::enable_if_t<
                internal_span::IsSpanImplicitlyConvertible<U, N, T, Extent>>>
  constexpr span(const span<U, N>& other) noexcept
      : span(other.data(), other.size()) {}

  /// Returns a pointer to the first element.
  constexpr pointer data() const noexcept { return data_; }

  /// Returns the size.
  constexpr index_type size() const noexcept { return size_; }

  /// Returns true if the span is empty.
  constexpr bool empty() const noexcept { return size() == 0; }

  /// Returns the size in bytes.
  constexpr index_type size_bytes() const noexcept {
    return size() * sizeof(element_type);
  }

  /// Returns begin/end iterators.
  constexpr iterator begin() const noexcept { return data(); }
  constexpr iterator end() const noexcept { return data() + size(); }
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  constexpr const_iterator cend() const noexcept { return end(); }

  /// Returns begin/end reverse iterators.
  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }
  constexpr const_reverse_iterator crend() const noexcept { return rend(); }

  /// Returns a static extent subspan of the first `Count` elements.
  ///
  /// \dchecks `size() >= Count`
  /// \id static
  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> first() const {
    static_assert(Count >= 0 && (Extent == dynamic_extent || Extent >= Count),
                  "Invalid Count");
    return {data(), Count};
  }

  /// Returns a dynamic extent subspan of the first `count` elements.
  ///
  /// \dchecks `size() >= count`
  /// \id dynamic
  constexpr span<element_type, dynamic_extent> first(
      std::ptrdiff_t count) const {
    assert(count <= size());
    return {data(), count};
  }

  /// Returns a static extent subspan of the last `Count` elements.
  ///
  /// \dchecks `size() >= Count`
  /// \id static
  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> last() const {
    static_assert(Count >= 0 && (Extent == dynamic_extent || Extent >= Count),
                  "Invalid Count");
    return {end() - Count, Count};
  }

  /// Returns a dynamic extent subspan of the last `count` elements.
  ///
  /// \dchecks `size() >= count`
  /// \id dynamic
  constexpr span<element_type, dynamic_extent> last(
      std::ptrdiff_t count) const {
    assert(count <= size());
    return {end() - count, count};
  }

  /// Returns a subspan from the starting offset `Offset` with the specified
  /// `Count`.
  ///
  /// \id static
  template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent>
  constexpr span<element_type,
                 internal_span::SubspanExtent(Extent, Offset, Count)>
  subspan() const {
    static_assert(Offset >= 0, "Offset must be non-negative.");
    static_assert(Count == dynamic_extent || Count >= 0,
                  "Count must be non-negative or dynamic_extent.");
    static_assert(Count == dynamic_extent || Extent == dynamic_extent ||
                      Offset + Count <= Extent,
                  "Offset must be non-negative.");
    return {begin() + Offset,
            Count == dynamic_extent ? size() - Offset : Count};
  }

  /// Returns a dynamic extent subspan from the starting `offset` and specified
  /// `count`.
  ///
  /// \id dynamic
  constexpr span<element_type, dynamic_extent> subspan(
      std::ptrdiff_t offset, std::ptrdiff_t count = dynamic_extent) const {
    assert(offset >= 0 && (count == dynamic_extent ||
                           (count >= 0 && offset + count <= size())));
    return {begin() + offset,
            count == dynamic_extent ? size() - offset : count};
  }

  /// Returns a reference to the element at the specified index.
  ///
  /// \dchecks `i >= 0 && i <= size()`
  constexpr reference operator[](index_type i) const noexcept {
    // operator[] is typically unchecked:
    //   "The behavior is undefined if idx is out of range."
    assert(i < size() && i >= 0);
    return *(data() + i);
  }

  // Support for non-standard accessors from
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1024r0.pdf

  /// Returns a reference to the element at the specified index.
  ///
  /// \checks `i >= 0 && i < size()`
  constexpr reference at(index_type i) const {
    if (ABSL_PREDICT_TRUE(i < size() && i >= 0)) {
      return *(data() + i);
    }
    ABSL_LOG(FATAL) << "span.at(" << i << (i >= 0 ? ") >= size()" : ") i < 0");
  }

  /// Returns a reference to the first element.
  ///
  /// \dchecks `size() > 0`
  constexpr reference front() const {
    assert(!empty());
    return *data();
  }

  /// Returns a reference to the last element.
  ///
  /// \dchecks `size() > 0`
  constexpr reference back() const {
    assert(!empty());
    return *(data() + (size() - 1));
  }

  // Support for absl::Hash.
  template <typename H>
  friend H AbslHashValue(H h, span v) {
    return H::combine(H::combine_contiguous(std::move(h), v.data(), v.size()),
                      v.size());
  }

 private:
  T* data_ = nullptr;
  // Note: To avoid miscompilation by MSVC, we must not specify a default member
  // initializer for `size_`.
  //
  // https://developercommunity.visualstudio.com/t/miscompilation-with-msvcno-unique-address-and-defa/1699750
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  std::conditional_t<Extent == dynamic_extent, ptrdiff_t,
                     std::integral_constant<ptrdiff_t, Extent>>
      size_;
};

template <typename T>
span(T* ptr, std::ptrdiff_t count) -> span<T>;

template <typename T>
span(T* begin, T* end) -> span<T>;

template <typename T, std::size_t N>
span(T (&arr)[N]) -> span<T, N>;

template <typename T, std::size_t N>
span(const T (&arr)[N]) -> span<const T, N>;

template <typename T, std::size_t N>
span(std::array<T, N>& arr) -> span<T, N>;

template <typename T, std::size_t N>
span(const std::array<T, N>& arr) -> span<const T, N>;

template <typename Container>
span(Container& cont) -> span<internal_span::ContainerElementType<Container>>;

template <typename Container>
span(const Container& cont)
    -> span<internal_span::ContainerElementType<const Container>>;

/// Support for the standard tuple access protocol, as proposed in:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1024r0.pdf
///
/// This definition (along with the partial specializations of std::tuple_size
/// and std::tuple_element defined below) makes fixed-size spans compatible with
/// C++17 structured bindings.

template <std::size_t I, typename T, std::ptrdiff_t Extent>
T& get(const ::tensorstore::span<T, Extent>& s) {
  static_assert(
      I >= 0 && (Extent == ::tensorstore::dynamic_extent || I < Extent),
      "Index I must be >= 0 and < Extent.");
  return s[I];
}

namespace internal {

/// Alias that evaluates to the deduced type of `span(std::declval<X>())`.
template <typename X>
using SpanType =
    span<typename internal_span::SpanTypeHelper<
             std::remove_reference_t<X>>::element_type,
         internal_span::SpanTypeHelper<std::remove_reference_t<X>>::extent>;

/// Same as `SpanType`, except the element type is always `const`-qualified.
template <typename X>
using ConstSpanType =
    span<const typename internal_span::SpanTypeHelper<
             std::remove_reference_t<X>>::element_type,
         internal_span::SpanTypeHelper<std::remove_reference_t<X>>::extent>;

/// RangesEqual returns whether the size and each value is equal
template <typename T, std::ptrdiff_t X, typename U, std::ptrdiff_t Y>
constexpr bool RangesEqual(span<T, X> l, span<U, Y> r) {
  return l.size() == r.size() &&
         (l.data() == r.data() || std::equal(l.begin(), l.end(), r.begin()));
}

}  // namespace internal

/// Returns the size of `span` as an `std::integral_constant` or
/// `DimensionIndex`.
///
/// The result is a valid `StaticOrDynamicRank` value.
///
/// \relates span
template <typename X, std::ptrdiff_t N>
std::integral_constant<std::ptrdiff_t, N> GetStaticOrDynamicExtent(span<X, N>) {
  return {};
}
template <typename X>
std::ptrdiff_t GetStaticOrDynamicExtent(span<X> s) {
  return s.size();
}

}  // namespace tensorstore

namespace std {

template <typename T, std::ptrdiff_t X>
struct tuple_size<::tensorstore::span<T, X>>
    : public std::integral_constant<std::size_t, X> {};

template <typename T>
struct tuple_size<::tensorstore::span<T, ::tensorstore::dynamic_extent>>;

template <std::size_t I, typename T, std::ptrdiff_t X>
struct tuple_element<I, ::tensorstore::span<T, X>> {
 public:
  using type = T;
};

template <std::size_t I, typename T, std::ptrdiff_t X>
struct tuple_element<I, const ::tensorstore::span<T, X>> {
 public:
  using type = T;
};

template <std::size_t I, typename T, std::ptrdiff_t X>
struct tuple_element<I, volatile ::tensorstore::span<T, X>> {
 public:
  using type = T;
};

template <std::size_t I, typename T, std::ptrdiff_t X>
struct tuple_element<I, const volatile ::tensorstore::span<T, X>> {
 public:
  using type = T;
};
}  // namespace std

#endif  // TENSORSTORE_UTIL_SPAN_H_
