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

#ifndef TENSORSTORE_INTERNAL_INTRUSIVE_PTR_H_
#define TENSORSTORE_INTERNAL_INTRUSIVE_PTR_H_

/// \file
/// Intrusive reference-counted pointer implementation.
///
/// This implementation is inspired by `boost::intrusive_ptr` and P0468r1:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0468r1.html
///
/// Basic intrusive reference counting can be used as follows:
///
///     class X : public AtomicReferenceCount<X> {
///      public:
///       // ...
///     };
///
///     class Y : public X {
///       // ...
///     };
///
///     IntrusivePtr<X> x1(new X);
///     EXPECT_EQ(1, x1->use_count());
///
///     auto x2 = x1;
///     EXPECT_EQ(2, x1->use_count());
///
///     IntrusivePtr<Y> y1(new Y);
///     IntrusivePtr<X> y2 = y1;
///     IntrusivePtr<Y> y3 = static_pointer_cast<Y>(y2);
///     IntrusivePtr<Y> y3 = dynamic_pointer_cast<Y>(y2);
///
/// For classes that do not inherit from `AtomicReferenceCount`, the reference
/// counting behavior can be specified by defining non-member
/// `intrusive_ptr_increment` and `intrusive_ptr_decrement` functions that can
/// be found via argument-dependent lookup (ADL):
///
///     /// Uses non-atomic reference count for single-threaded efficiency.
///     class X {
///      public:
///       // ...
///       virtual ~X() = default;
///       friend void intrusive_ptr_increment(X *p) { ++p->ref_count_; }
///       friend void intrusive_ptr_decrement(X *p) {
///         if (--p->ref_count_ == 0) {
///           delete p;
///         }
///       }
///       std::uint32_t ref_count_{0};
///     };
///
///     class Y : public X {
///       // ...
///     };
///
///     IntrusivePtr<X> x1(new X);
///     EXPECT_EQ(1, x1->ref_count_);
///
///     IntrusivePtr<X> x2 = x1;
///     EXPECT_EQ(2, x2->ref_count_);
///
///     IntrusivePtr<Y> y1(new Y);
///     IntrusivePtr<X> y2 = y1;
///     IntrusivePtr<Y> y3 = dynamic_pointer_cast<Y>(y2);
///
/// As an alternative to defining the non-member `intrusive_ptr_increment` and
/// `intrusive_ptr_decrement` functions, a separate `Traits` class may be used
/// in place of `DefaultIntrusivePtrTraits`, which also permits the use of
/// custom pointer types (e.g. TaggedPtr):
///
///     class X {
///      public:
///       // ...
///       virtual ~X() = default;
///       std::uint32_t ref_count_{0};
///     };
///
///     class Y : public X {
///       // ...
///     };
///
///     struct XTraits {
///       template <typename U>
///       using pointer = U*;
///       static void increment(X *p) noexcept { ++p->ref_count_; }
///       static void decrement(X *p) noexcept {
///         if (--p->ref_count_ == 0) delete p;
///       }
///     };
///
///     IntrusivePtr<X, XTraits> x1(new X);
///     EXPECT_EQ(1, x1->ref_count_);
///
///     IntrusivePtr<X, XTraits> x2 = x1;
///     EXPECT_EQ(2, x2->ref_count_);
///
///     IntrusivePtr<Y, XTraits> y1(new Y);
///     IntrusivePtr<X, XTraits> y2 = y1;
///     IntrusivePtr<Y, XTraits> y3 = dynamic_pointer_cast<Y>(y2);
///

#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "absl/base/macros.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal {

/// CRTP base class that can be used with `IntrusivePtr` and
/// `DefaultIntrusivePtrTraits` to enable reference counting for objects
/// allocated using `operator new`.
///
/// \tparam Derived The derived class that inherits from this class.  When the
///     last reference is released, `intrusive_ptr_decrement` calls
///     `operator delete` using a `Derived*` pointer.  If `Derived` has a
///     virtual destructor, it need not be most derived type.
template <typename Derived>
class AtomicReferenceCount {
 public:
  AtomicReferenceCount() noexcept = default;
  // Defining the copy constructor below disables generation of the default move
  // constructor and move assignment operator, which means that the copy
  // constructor and copy assignment operator will be used for rvalues as well.
  AtomicReferenceCount(const AtomicReferenceCount&) noexcept {}
  AtomicReferenceCount& operator=(const AtomicReferenceCount&) noexcept {
    return *this;
  }
  std::uint32_t use_count() const noexcept {
    return ref_count_.load(std::memory_order_acquire);
  }

  /// Increments the reference count.
  ///
  /// This function is called by `DefaultIntrusivePtrTraits`.
  friend void intrusive_ptr_increment(const AtomicReferenceCount* p) noexcept {
    p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
  }

  /// Decrements the reference count.  If the reference count reaches 0, casts
  /// `p` to `Derived*` and calls `operator delete`.
  ///
  /// This function is called by `DefaultIntrusivePtrTraits`.
  friend void intrusive_ptr_decrement(const AtomicReferenceCount* p) noexcept {
    if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete static_cast<const Derived*>(p);
    }
  }

 private:
  mutable std::atomic<std::uint32_t> ref_count_{0};
};

/// Specifies the default behavior of `IntrusivePtr<T>`.
struct DefaultIntrusivePtrTraits {
  template <typename U>
  using pointer = U*;

  /// Called to acquire an additional reference to `p`.
  ///
  /// This simply calls `intrusive_ptr_increment`, which is found via ADL.
  template <typename Pointer>
  static void increment(Pointer p) noexcept {
    intrusive_ptr_increment(p);
  }

  /// Called to destroy a reference to `p`.
  ///
  /// This simply calls `intrusive_ptr_decrement`, which is found via ADL.
  template <typename Pointer>
  static void decrement(Pointer p) noexcept {
    intrusive_ptr_decrement(p);
  }
};

template <typename T, typename R>
class IntrusivePtr;

/// Tag type to indicate that a new reference to a given object should be
/// acquired.
struct acquire_object_ref_t {
  explicit constexpr acquire_object_ref_t() = default;
};

/// Tag type to indicate that an existing reference to a given object should be
/// adopted.
struct adopt_object_ref_t {
  explicit constexpr adopt_object_ref_t() = default;
};

constexpr acquire_object_ref_t acquire_object_ref{};
constexpr adopt_object_ref_t adopt_object_ref{};

template <typename T>
struct IsIntrusivePtr : public std::false_type {};

template <typename T, typename R>
struct IsIntrusivePtr<IntrusivePtr<T, R>> : public std::true_type {};

/// Intrusive reference-counting smart pointer.
///
/// \tparam T The pointee/element type.
/// \tparam R Traits type specifying the `pointer` type and
///     `increment`/`decrement` operations.  By default,
///     `DefaultIntrusivePtrTraits` is used.
///
/// The default traits type `R = DefaultIntrusivePtrTraits` is used,
/// `IntrusivePtr` stores a raw `T*` pointer and requires that functions of the
/// form:
///
///     void intrusive_ptr_increment(T *p);
///     void intrusive_ptr_decrement(T *p);
///
/// be found via argument-dependent lookup.  The `intrusive_ptr_increment`
/// function is called to acquire a new reference, while
/// `intrusive_ptr_decrement` is called to remove a reference (and possibly
/// destroy the object).  These functions may be defined manually, or `T` may
/// inherit from `AtomicReferenceCount<U>`.
///
/// Alternatively, a non-default traits type `R` may be specified with the
/// following members:
///
///     template <typename U>
///     using pointer = ...;
///     static void increment(pointer) noexcept;
///     static void decrement(pointer) noexcept;
///
/// The `pointer<T>` type specifies the pointer type stored by
/// `IntrusivePtr<T, Traits>`.  The `increment` function is called to acquire a
/// new reference, while `decrement` is called to release a reference.
///
/// Commonly, `pointer<U>` alias will be defined to be `U*`, but it could also
/// be a different type, such as a tagged pointer type.
///
/// Specifying a non-default traits type is the only way to use a custom pointer
/// type.
///
/// The requirements on the `Traits` type differ from the requirements specified
/// in P0468r1 in that in this implementation:
///
/// - the `pointer` member is a template alias in order to support
///   `{static,dynamic,const}_pointer_cast`;
///
/// - there is no `use_count` member (since it isn't used by IntrusivePtr, the
///   meaning and use is specific to a particular use case, and it can be
///   exposed through another means, e.g. as a member of `T`, if needed);
///
/// - there is no `default_action` member, as it is confusing and unnecessary.
template <typename T, typename R = DefaultIntrusivePtrTraits>
class IntrusivePtr {
 public:
  using element_type = T;
  using traits_type = R;
  using pointer = typename R::template pointer<T>;

  /// Constructs a null pointer.
  constexpr IntrusivePtr() noexcept : ptr_(pointer{}) {}

  /// Constructs from a given pointer.  If `p` is not null, acquires a new
  /// reference to `p` by calling `R::increment(p)`.
  explicit IntrusivePtr(pointer p,
                        acquire_object_ref_t = acquire_object_ref) noexcept
      : ptr_(p) {
    if (ptr_) R::increment(ptr_);
  }

  /// Constructs from a given pointer without acquiring a new reference.  If `p`
  /// is not null, this implicitly adopts an existing reference to `p`.
  explicit IntrusivePtr(pointer p, adopt_object_ref_t) noexcept : ptr_(p) {}

  /// Copy constructs from `rhs`.  If `rhs` is not null, acquires a new
  /// reference to `rhs.get()` by calling `R::increment(rhs.get())`.
  IntrusivePtr(const IntrusivePtr& rhs) noexcept
      : IntrusivePtr(rhs.get(), acquire_object_ref) {}

  template <typename U,
            std::enable_if_t<std::is_convertible<
                typename R::template pointer<U>, pointer>::value>* = nullptr>
  IntrusivePtr(const IntrusivePtr<U, R>& rhs)
      : IntrusivePtr(rhs.get(), acquire_object_ref) {}

  /// Move constructs from `rhs`.  If `rhs` is not null, transfers ownership of
  /// a reference from `rhs` to `*this`.
  IntrusivePtr(IntrusivePtr&& rhs) noexcept
      : IntrusivePtr(rhs.release(), adopt_object_ref) {}

  template <typename U,
            std::enable_if_t<std::is_convertible<
                typename R::template pointer<U>, pointer>::value>* = nullptr>
  IntrusivePtr(IntrusivePtr<U, R>&& rhs) noexcept
      : IntrusivePtr(rhs.release(), adopt_object_ref) {}

  ~IntrusivePtr() {
    if (pointer p = get()) R::decrement(p);
  }

  IntrusivePtr& operator=(const IntrusivePtr& rhs) {
    IntrusivePtr(rhs).swap(*this);
    return *this;
  }

  template <typename U>
  std::enable_if_t<
      std::is_convertible<typename R::template pointer<U>, pointer>::value,
      IntrusivePtr&>
  operator=(const IntrusivePtr<U, R>& rhs) {
    IntrusivePtr(std::move(rhs)).swap(*this);
    return *this;
  }

  IntrusivePtr& operator=(IntrusivePtr&& rhs) noexcept {
    IntrusivePtr(std::move(rhs)).swap(*this);
    return *this;
  }

  template <typename U>
  std::enable_if_t<
      std::is_convertible<typename R::template pointer<U>, pointer>::value,
      IntrusivePtr&>
  operator=(IntrusivePtr<U, R>&& rhs) noexcept {
    IntrusivePtr(std::move(rhs)).swap(*this);
    return *this;
  }

  /// Assigns the stored pointer to null.  If the prior stored pointer was
  /// non-null, calls `R::decrement` on it.
  void reset() noexcept { IntrusivePtr().swap(*this); }

  /// Assigns the stored pointer to `rhs`, and calls `R::increment` on `rhs` if
  /// non-null.  If the prior stored pointer was non-null, calls `R::decrement`
  /// on it.
  void reset(pointer rhs, acquire_object_ref_t = acquire_object_ref) {
    IntrusivePtr(rhs, acquire_object_ref).swap(*this);
  }

  /// Assigns the stored pointer to `rhs`, and if `rhs` is non-null, implicitly
  /// adopts an existing reference.  If the prior stored pointer was non-null,
  /// calls `R::decrement` on it.
  void reset(pointer rhs, adopt_object_ref_t) {
    IntrusivePtr(rhs, adopt_object_ref).swap(*this);
  }

  /// Returns `true` if the stored pointer is non-null.
  explicit operator bool() const { return static_cast<bool>(get()); }

  pointer get() const noexcept { return ptr_; }

  pointer operator->() const {
    pointer ptr = get();
    ABSL_ASSERT(static_cast<bool>(ptr));
    return ptr;
  }

  element_type& operator*() const {
    pointer ptr = get();
    ABSL_ASSERT(static_cast<bool>(ptr));
    return *ptr;
  }

  /// Assigns the stored pointer to null, and returns the prior value.
  ///
  /// `R::decrement` is not called.
  pointer release() noexcept {
    pointer ptr = get();
    ptr_ = pointer{};
    return ptr;
  }

  void swap(IntrusivePtr& rhs) noexcept { std::swap(ptr_, rhs.ptr_); }

  /// Abseil hash support.
  template <typename H>
  friend H AbslHashValue(H h, const IntrusivePtr& x) {
    return H::combine(std::move(h), x.get());
  }

  friend bool operator==(const IntrusivePtr& p, std::nullptr_t) { return !p; }
  friend bool operator!=(const IntrusivePtr& p, std::nullptr_t) {
    return static_cast<bool>(p);
  }

  friend bool operator==(std::nullptr_t, const IntrusivePtr& p) { return !p; }
  friend bool operator!=(std::nullptr_t, const IntrusivePtr& p) {
    return static_cast<bool>(p);
  }

 private:
  pointer ptr_;
};

template <typename T, typename R>
inline T* to_address(const IntrusivePtr<T, R>& p) {
  return to_address(p.get());
}

template <typename T, typename U, typename R>
inline std::enable_if_t<
    IsEqualityComparable<typename R::template pointer<T>,
                         typename R::template pointer<U>>::value,
    bool>
operator==(const IntrusivePtr<T, R>& x, const IntrusivePtr<U, R>& y) {
  return x.get() == y.get();
}

template <typename T, typename U, typename R>
inline std::enable_if_t<
    IsEqualityComparable<typename R::template pointer<T>,
                         typename R::template pointer<U>>::value,
    bool>
operator!=(const IntrusivePtr<T, R>& x, const IntrusivePtr<U, R>& y) {
  return x.get() != y.get();
}

template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> static_pointer_cast(IntrusivePtr<U, R> p) {
  return IntrusivePtr<T, R>(static_pointer_cast<T>(p.release()),
                            adopt_object_ref);
}

template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> const_pointer_cast(IntrusivePtr<U, R> p) {
  return IntrusivePtr<T, R>(const_pointer_cast<T>(p.release()),
                            adopt_object_ref);
}

template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> dynamic_pointer_cast(IntrusivePtr<U, R> p) {
  if (auto new_pointer = dynamic_pointer_cast<T>(p.get())) {
    p.release();
    return IntrusivePtr<T, R>(std::move(new_pointer), adopt_object_ref);
  } else {
    return IntrusivePtr<T, R>(std::move(new_pointer), adopt_object_ref);
  }
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTRUSIVE_PTR_H_
