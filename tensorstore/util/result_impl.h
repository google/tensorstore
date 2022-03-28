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

#ifndef TENSORSTORE_RESULT_IMPL_H_
#define TENSORSTORE_RESULT_IMPL_H_

#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/config.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/utility/utility.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"

// IWYU pragma: private, include "third_party/tensorstore/util/result.h"

namespace tensorstore {

template <typename T>
class Result;

namespace internal_result {

/// Result type traits helper structs for UnwrapResultType / FlatResultType.
template <typename T>
struct UnwrapResultHelper {
  static_assert(std::is_same_v<T, internal::remove_cvref_t<T>>,
                "Type argument to UnwrapResultType must be unqualified.");
  using type = T;
  using result_type = Result<T>;
};

template <typename T>
struct UnwrapResultHelper<Result<T>> {
  using type = T;
  using result_type = Result<T>;
};

template <>
struct UnwrapResultHelper<absl::Status> {
  using type = void;
  using result_type = Result<void>;
};

// Tag types to select internal constructors.
struct status_t {};
struct value_t {};
struct noinit_t {};

// ----------------------------------------------------------------
// Storage base classes for Result<T>
// ----------------------------------------------------------------

// ResultStorageBase
// * Owns the underlying absl::Status and T data objects.
// * Specialized construction and assignment for T=void vs. non-void.
//
template <class T>
struct ResultStorageBase {
  using value_type = T;
  using reference_type = T&;
  using const_reference_type = const T&;

  constexpr ResultStorageBase(noinit_t) noexcept
      : dummy_(), has_value_(false) {}

  template <typename... Args>
  constexpr explicit ResultStorageBase(value_t, Args&&... args)
      : value_(std::forward<Args>(args)...), has_value_(true) {}

  template <typename... Args>
  constexpr explicit ResultStorageBase(status_t, Args&&... args) noexcept
      : status_(std::forward<Args>(args)...), has_value_(false) {}

  ~ResultStorageBase() { destruct(); }

  inline void destruct() {
    if (has_value_) {
      destruct_value();
    } else {
      status_.~Status();
    }
  }

  inline void destruct_value() { value_.~T(); }

  template <typename... Args>
  inline void construct_value(Args&&... args) {
    ::new (&value_) T(std::forward<Args>(args)...);
    this->has_value_ = true;
  }

  struct dummy {};
  union {
    dummy dummy_;  // for constexpr initialization.
    T value_;
    absl::Status status_;
  };
  bool has_value_;
};

template <>
struct ResultStorageBase<void> {
  using value_type = void;
  using reference_type = void;
  using const_reference_type = void;

  constexpr ResultStorageBase(noinit_t) noexcept
      : value_(), has_value_(false) {}

  template <typename... Args>
  explicit constexpr ResultStorageBase(value_t, Args&&... args) noexcept
      : value_(), has_value_(true) {}

  template <typename... Args>
  explicit ResultStorageBase(status_t, Args&&... args) noexcept
      : status_(std::forward<Args>(args)...), has_value_(false) {}

  ~ResultStorageBase() { destruct(); }

  inline void destruct() {
    if (!has_value_) {
      status_.~Status();
    }
  }

  inline void destruct_value() noexcept {}

  template <typename... Args>
  inline void construct_value(Args&&...) noexcept {
    this->has_value_ = true;
  }

  struct dummy {};
  union {
    dummy value_;  // for constexpr initialization.
    absl::Status status_;
  };
  bool has_value_;
};

// ResultStorage is the storage base class for Result<T>,
// and implements the consturctors, copy, and assignment operators.
template <typename T>
struct ResultStorage : public ResultStorageBase<T> {
  using base = ResultStorageBase<T>;

  constexpr ResultStorage(noinit_t t) noexcept : base(t) {}

  template <typename... Args>
  explicit ResultStorage(value_t t, Args&&... args)
      : base(t, std::forward<Args>(args)...) {}

  template <typename... Args>
  explicit ResultStorage(status_t t, Args&&... args)
      : base(t, std::forward<Args>(args)...) {}

  ResultStorage(const ResultStorage& rhs) : base(noinit_t{}) {
    if (rhs.has_value_) {
      this->construct_value(rhs.value_);
    } else {
      construct_status(rhs.status_);
    }
  }

  ResultStorage(ResultStorage&& rhs) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : base(noinit_t{}) {
    if (rhs.has_value_) {
      this->construct_value(std::move(rhs).value_);
    } else {
      construct_status(std::move(rhs).status_);
    }
  }

  ResultStorage& operator=(const ResultStorage& rhs) {
    if (&rhs == this) return *this;
    if (rhs.has_value_) {
      emplace_value(rhs.value_);
    } else {
      assign_status(rhs.status_);
    }
    return *this;
  }

  ResultStorage& operator=(ResultStorage&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value) {
    if (&rhs == this) return *this;
    if (rhs.has_value_) {
      emplace_value(std::move(rhs).value_);
    } else {
      assign_status(std::move(rhs).status_);
    }
    return *this;
  }

  template <typename... Args>
  void construct_status(Args&&... args) noexcept {
    ::new (static_cast<void*>(&this->status_))
        absl::Status(std::forward<Args>(args)...);
    this->has_value_ = false;
  }

  template <typename Arg>
  void assign_status(Arg&& arg) noexcept {
    if (!this->has_value_) {
      this->status_ = std::forward<Arg>(arg);
    } else {
      this->destruct_value();
      construct_status(std::forward<Arg>(arg));
    }
  }

  template <typename... Args>
  void emplace_value(Args&&... args) {
    // NOTE: I think that using emplace_value in place of assignment is
    // misleading. We should use this->value_ = arg when a value already exists,
    // which is the same as std::optional<> and other monadic structures.
    this->destruct();
    this->construct_value(std::forward<Args>(args)...);
  }
};

// ----------------------------------------------------------------
// Mixin classes which selectively delete constructor and assignment
// variants through inheritance and template specialization.
// ----------------------------------------------------------------

template <typename T,
          bool = std::is_void_v<T> || std::is_copy_constructible_v<T>>
struct CopyCtorBase {
  CopyCtorBase() = default;
  CopyCtorBase(const CopyCtorBase&) = default;
  CopyCtorBase(CopyCtorBase&&) = default;
  CopyCtorBase& operator=(const CopyCtorBase&) = default;
  CopyCtorBase& operator=(CopyCtorBase&&) = default;
};

template <typename T>
struct CopyCtorBase<T, false> {
  CopyCtorBase() = default;
  CopyCtorBase(const CopyCtorBase&) = delete;
  CopyCtorBase(CopyCtorBase&&) = default;
  CopyCtorBase& operator=(const CopyCtorBase&) = default;
  CopyCtorBase& operator=(CopyCtorBase&&) = default;
};

template <typename T,
          bool = std::is_void_v<T> || std::is_move_constructible_v<T>>
struct MoveCtorBase {
  MoveCtorBase() = default;
  MoveCtorBase(const MoveCtorBase&) = default;
  MoveCtorBase(MoveCtorBase&&) = default;
  MoveCtorBase& operator=(const MoveCtorBase&) = default;
  MoveCtorBase& operator=(MoveCtorBase&&) = default;
};

template <typename T>
struct MoveCtorBase<T, false> {
  MoveCtorBase() = default;
  MoveCtorBase(const MoveCtorBase&) = default;
  MoveCtorBase(MoveCtorBase&&) = delete;
  MoveCtorBase& operator=(const MoveCtorBase&) = default;
  MoveCtorBase& operator=(MoveCtorBase&&) = default;
};

template <typename T,
          bool = std::is_void_v<T> || (std::is_copy_constructible_v<T> &&
                                       std::is_copy_assignable_v<T>)>
struct CopyAssignBase {
  CopyAssignBase() = default;
  CopyAssignBase(const CopyAssignBase&) = default;
  CopyAssignBase(CopyAssignBase&&) = default;
  CopyAssignBase& operator=(const CopyAssignBase&) = default;
  CopyAssignBase& operator=(CopyAssignBase&&) = default;
};

template <typename T>
struct CopyAssignBase<T, false> {
  CopyAssignBase() = default;
  CopyAssignBase(const CopyAssignBase&) = default;
  CopyAssignBase(CopyAssignBase&&) = default;
  CopyAssignBase& operator=(const CopyAssignBase&) = delete;
  CopyAssignBase& operator=(CopyAssignBase&&) = default;
};

template <typename T,
          bool = std::is_void_v<T> || (std::is_move_constructible_v<T> &&
                                       std::is_move_assignable_v<T>)>
struct MoveAssignBase {
  MoveAssignBase() = default;
  MoveAssignBase(const MoveAssignBase&) = default;
  MoveAssignBase(MoveAssignBase&&) = default;
  MoveAssignBase& operator=(const MoveAssignBase&) = default;
  MoveAssignBase& operator=(MoveAssignBase&&) = default;
};

template <typename T>
struct MoveAssignBase<T, false> {
  MoveAssignBase() = default;
  MoveAssignBase(const MoveAssignBase&) = default;
  MoveAssignBase(MoveAssignBase&&) = default;
  MoveAssignBase& operator=(const MoveAssignBase&) = default;
  MoveAssignBase& operator=(MoveAssignBase&&) = delete;
};

// Whether T is constructible or convertible from Result<U>.
template <typename T, typename U>
constexpr inline bool is_constructible_convertible_from = std::disjunction<
    std::is_constructible<T, U&>, std::is_constructible<T, const U&>,
    std::is_constructible<T, U&&>, std::is_constructible<T, const U&&>,
    std::is_convertible<U&, T>, std::is_convertible<const U&&, T>,
    std::is_convertible<U&, T>, std::is_convertible<const U&&, T>>::value;

// Metafunction for Result value constructor SFINAE overloads exclusions.
// These should parallel the other value constructors.
template <typename T>
constexpr inline bool is_result_status_or_inplace = false;
template <typename T>
constexpr inline bool is_result_status_or_inplace<Result<T>> = true;
template <>
constexpr inline bool is_result_status_or_inplace<absl::Status> = true;
template <>
constexpr inline bool is_result_status_or_inplace<std::in_place_t> = true;

// Define metafunctions to avoid the liberal MSVC error C2938:
// 'std::enable_if_t<false,bool>' : Failed to specialize alias template
// This happens, for instance, when using SFINAE on std::is_void_v<T>
// where T is a class template parameter.

// Metafunction for enabling the Result<T>::Result(Result<U>) constructor
// overloads.
template <typename T, typename U>
constexpr inline bool result_conversion = !std::is_same_v<T, U>;
template <typename U>
constexpr inline bool result_conversion<void, U> = false;

// Metafunction for enabling the Result<void>::Result(Result<U>) constructor
// overloads.
template <typename T, typename U>
constexpr inline bool result_void_conversion = false;
template <typename U>
constexpr inline bool result_void_conversion<void, U> = true;
template <>
constexpr inline bool result_void_conversion<void, void> = false;

// Metafunction for enabling the Result<T>::Result(U) constructor overloads.
template <typename T, typename U>
constexpr inline bool value_conversion =
    !is_result_status_or_inplace<internal::remove_cvref_t<U>>;
template <typename U>
constexpr inline bool value_conversion<void, U> = false;

/// Checks if `T` is equality comparable with `U`, but in the case that `U` is
/// itself an instance of `Result`, evaluates to `false` and avoids overload
/// resolution in order to prevent infinite SFINAE recursion.
template <typename T, typename U>
constexpr inline bool IsEqualityComparableIfNotResult =
    internal::IsEqualityComparable<T, U>;
template <typename T, typename U>
constexpr inline bool IsEqualityComparableIfNotResult<T, Result<U>> = false;

}  // namespace internal_result
}  // namespace tensorstore

#endif  // TENSORSTORE_RESULT_IMPL_H_
