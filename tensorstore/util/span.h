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
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/log_message.h"
#include "tensorstore/internal/source_location.h"

TENSORSTORE_GDB_AUTO_SCRIPT("span_gdb.py")

namespace tensorstore {

constexpr std::ptrdiff_t dynamic_extent = -1;

template <typename T, std::ptrdiff_t Extent = dynamic_extent>
class span;

namespace internal_span {

template <typename SourceElement, std::ptrdiff_t SourceExtent,
          typename DestElement, std::ptrdiff_t DestExtent>
struct IsSpanConvertible
    : public std::integral_constant<
          bool,
          (std::is_convertible<SourceElement (*)[], DestElement (*)[]>::value &&
           (SourceExtent == DestExtent || SourceExtent == dynamic_extent ||
            DestExtent == dynamic_extent))> {};

template <typename SourceElement, std::ptrdiff_t SourceExtent,
          typename DestElement, std::ptrdiff_t DestExtent>
struct IsSpanImplicitlyConvertible
    : public std::integral_constant<
          bool,
          std::is_convertible<SourceElement (*)[], DestElement (*)[]>::value &&
              (SourceExtent == DestExtent || DestExtent == dynamic_extent)> {};

template <typename Container>
struct IsArrayOrSpan : public std::false_type {};

template <typename T, std::size_t N>
struct IsArrayOrSpan<std::array<T, N>> : public std::true_type {};

template <typename T, std::ptrdiff_t Extent>
struct IsArrayOrSpan<span<T, Extent>> : public std::true_type {};

template <typename T, typename Container,
          typename Pointer = decltype(std::declval<Container>().data()),
          typename Size = decltype(std::declval<Container>().size())>
using EnableIfCompatibleContainer = std::enable_if_t<
    !IsArrayOrSpan<std::remove_cv_t<Container>>::value &&
    std::is_convertible<std::remove_pointer_t<Pointer> (*)[], T (*)[]>::value>;

template <typename Container,
          typename Pointer = decltype(std::declval<Container>().data()),
          typename Size = decltype(std::declval<Container>().size())>
using ContainerElementType =
    std::enable_if_t<!IsArrayOrSpan<std::remove_cv_t<Container>>::value,
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

// SpanBase owns the data and size members of a tensorstore::span<T>.
// It uses partial-specialization to avoid storing the length in the
// explicit-extent case.
template <typename T, std::ptrdiff_t Extent>
class SpanBase {
 protected:
  static_assert(Extent >= 0, "Extent must be non-negative or -1.");

  using index_type = std::ptrdiff_t;
  using pointer = T*;

  constexpr SpanBase() noexcept = default;

  constexpr SpanBase(pointer ptr, index_type count) noexcept : data_(ptr) {
    assert(count == Extent);
  }

  constexpr static index_type size() noexcept { return Extent; }
  constexpr pointer data() const noexcept { return data_; }

  T* data_ = nullptr;
};

template <typename T>
class SpanBase<T, dynamic_extent> {
 protected:
  using index_type = std::ptrdiff_t;
  using pointer = T*;

  constexpr SpanBase() noexcept = default;

  constexpr SpanBase(pointer ptr, index_type count) noexcept
      : data_(ptr), size_(count) {
    assert(count >= 0);
  }

  constexpr index_type size() const noexcept { return size_; }
  constexpr pointer data() const noexcept { return data_; }

  T* data_ = nullptr;
  index_type size_ = 0;
};

}  // namespace internal_span

template <typename T, std::ptrdiff_t Extent>
class span : private internal_span::SpanBase<T, Extent> {
  using Base = internal_span::SpanBase<T, Extent>;

 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using index_type = std::ptrdiff_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  static constexpr std::ptrdiff_t extent = Extent;

  constexpr span() noexcept = default;
  constexpr span(pointer ptr, index_type count) noexcept : Base(ptr, count) {}

  // Add an extra dummy template parameter to ensure this overload ranks lower
  // than the (pointer, index_type) overload in the case of a call of the form
  // (pointer, 0).
  //
  // See https://github.com/Microsoft/GSL/issues/541.
  template <typename U = void>
  constexpr span(pointer begin, pointer end) noexcept
      : Base(begin, end - begin) {}

  template <std::size_t N, typename = std::enable_if_t<
                               (Extent == dynamic_extent || Extent == N)>>
  constexpr span(T (&arr)[N]) noexcept : Base(arr, N) {}

  template <std::size_t N, typename = std::enable_if_t<
                               Extent == dynamic_extent || Extent == N>>
  constexpr span(std::array<value_type, N>& arr) noexcept
      : Base(arr.data(), N) {}

  template <
      std::size_t N, typename U = T,
      typename = std::enable_if_t<std::is_const_v<U> &&
                                  (Extent == dynamic_extent || Extent == N)>>
  constexpr span(const std::array<value_type, N>& arr) noexcept
      : Base(arr.data(), N) {}

  template <typename Container,
            typename = internal_span::EnableIfCompatibleContainer<T, Container>>
  constexpr span(Container& cont) : Base(cont.data(), cont.size()) {}

  template <
      typename Container,
      typename = internal_span::EnableIfCompatibleContainer<T, const Container>>
  constexpr span(const Container& cont) : Base(cont.data(), cont.size()) {}

  template <
      typename U, std::ptrdiff_t N,
      typename = std::enable_if_t<
          internal_span::IsSpanImplicitlyConvertible<U, N, T, Extent>::value>>
  constexpr span(const span<U, N>& other) noexcept
      : Base(other.data(), other.size()) {}

  using Base::data;
  using Base::size;

  constexpr bool empty() const noexcept { return size() == 0; }

  constexpr index_type size_bytes() const noexcept {
    return size() * sizeof(element_type);
  }

  constexpr iterator begin() const noexcept { return data(); }
  constexpr iterator end() const noexcept { return data() + size(); }
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  constexpr const_iterator cend() const noexcept { return end(); }

  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }
  constexpr const_reverse_iterator crend() const noexcept { return rend(); }

  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> first() const {
    static_assert(Count >= 0 && (Extent == dynamic_extent || Extent >= Count),
                  "Invalid Count");
    return {data(), Count};
  }

  constexpr span<element_type, dynamic_extent> first(
      std::ptrdiff_t count) const {
    assert(count <= size());
    return {data(), count};
  }

  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> last() const {
    static_assert(Count >= 0 && (Extent == dynamic_extent || Extent >= Count),
                  "Invalid Count");
    return {end() - Count, Count};
  }

  constexpr span<element_type, dynamic_extent> last(
      std::ptrdiff_t count) const {
    assert(count <= size());
    return {end() - count, count};
  }

  template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent>
  constexpr span<element_type,
                 (Count == dynamic_extent && Extent == dynamic_extent
                      ? dynamic_extent
                      : Count == dynamic_extent ? Extent - Offset : Count)>
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

  constexpr span<element_type, dynamic_extent> subspan(
      std::ptrdiff_t offset, std::ptrdiff_t count = dynamic_extent) const {
    assert(offset >= 0 && (count == dynamic_extent ||
                           (count >= 0 && offset + count <= size())));
    return {begin() + offset,
            count == dynamic_extent ? size() - offset : count};
  }

  constexpr reference operator[](index_type i) const noexcept {
    // operator[] is typically unchecked:
    //   "The behavior is undefined if idx is out of range."
    assert(i < size() && i >= 0);
    return *(data() + i);
  }

  // Support for non-standard accessors from
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1024r0.pdf
  constexpr reference at(index_type i) const {
    if (ABSL_PREDICT_TRUE(i < size() && i >= 0)) {
      return *(data() + i);
    }
    ::tensorstore::internal::LogMessageFatal(
        i >= 0 ? "span.at() i >= size()" : "span.at() i < 0", TENSORSTORE_LOC);
  }

  constexpr reference front() const {
    assert(!empty());
    return *data();
  }

  constexpr reference back() const {
    assert(!empty());
    return *(data() + (size() - 1));
  }
};

template <typename T>
span(T* ptr, std::ptrdiff_t count)->span<T>;

template <typename T>
span(T* begin, T* end)->span<T>;

template <typename T, std::size_t N>
span(T (&arr)[N])->span<T, N>;

template <typename T, std::size_t N>
span(const T (&arr)[N])->span<const T, N>;

template <typename T, std::size_t N>
span(std::array<T, N>& arr)->span<T, N>;

template <typename T, std::size_t N>
span(const std::array<T, N>& arr)->span<const T, N>;

template <typename Container>
span(Container& cont)->span<internal_span::ContainerElementType<Container>>;

template <typename Container>
span(const Container& cont)
    ->span<internal_span::ContainerElementType<const Container>>;

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

/// Returns the size of a fixed-size span as an `std::integral_constant`.  The
/// result is a valid `StaticOrDynamicRank` value.
template <typename X, std::ptrdiff_t N>
std::integral_constant<std::ptrdiff_t, N> GetStaticOrDynamicExtent(span<X, N>) {
  return {};
}

/// Returns the size of a `dynamic_extent` span as a `DimensionIndex`.  The
/// result is a valid `StaticOrDynamicRank` value.
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
