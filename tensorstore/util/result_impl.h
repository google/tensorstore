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

#include <type_traits>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "tensorstore/internal/meta/type_traits.h"

// IWYU pragma: private, include "tensorstore/util/result.h"

namespace tensorstore {

template <typename T>
class Result;

namespace internal_result {

/// Result type traits helper structs for UnwrapResultType / FlatResultType.
template <typename T>
struct UnwrapResultHelper {
  static_assert(std::is_same_v<T, absl::remove_cvref_t<T>>,
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

// ----------------------------------------------------------------
// Storage base classes for Result<T>
// ----------------------------------------------------------------

// Move type-agnostic error handling to the .cc.
[[noreturn]] void CrashOnResultNotOk(const absl::Status& status);
void HandleInvalidResultCtor(absl::Status* absl_nonnull status);

// `ResultStorage` is the storage base class for `Result<T>` (where
// `T != void`), and implements the constructors, copy, and assignment
// operators.  This separate base class is needed so that we can default those
// methods on `Result` and have the operations either defined or deleted
// depending on `{Copy,Move},{Ctor,Assign}Base`.
template <typename T>
struct ResultStorage {
  template <typename U>
  friend struct ResultStorage;

  ResultStorage() = delete;

  ResultStorage(const ResultStorage& other) {
    if (other.ok()) {
      construct_value(other.value_);
      construct_status();
    } else {
      construct_status(other.status_);
    }
  }

  ResultStorage(ResultStorage&& other) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    if (other.ok()) {
      construct_value(std::move(other.value_));
      construct_status();
    } else {
      construct_status(std::move(other.status_));
    }
  }

  template <typename U>
  ResultStorage(const ResultStorage<U>& other) {
    if (other.ok()) {
      construct_value(other.value_);
      construct_status();
    } else {
      construct_status(other.status_);
    }
  }

  template <typename U>
  ResultStorage(ResultStorage<U>&& other) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    if (other.ok()) {
      construct_value(std::move(other.value_));
      construct_status();
    } else {
      construct_status(std::move(other.status_));
    }
  }

  template <typename... Args>
  explicit ResultStorage(value_t, Args&&... args)
      : value_(std::forward<Args>(args)...) {
    construct_status();
  }

  template <typename... Args>
  explicit ResultStorage(status_t, Args&&... args) noexcept
      : status_(std::forward<Args>(args)...) {
    EnsureNotOk();
  }

  ResultStorage& operator=(const ResultStorage& rhs) {
    if (&rhs == this) return *this;
    if (rhs.ok()) {
      assign_value(rhs.value_);
    } else {
      assign_status(rhs.status_);
    }
    return *this;
  }

  ResultStorage& operator=(ResultStorage&& rhs) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if (&rhs == this) return *this;
    if (rhs.ok()) {
      assign_value(std::move(rhs.value_));
    } else {
      assign_status(std::move(rhs.status_));
    }
    return *this;
  }

  ~ResultStorage() {
    if (ok()) {
      status_.~Status();
      if constexpr (!std::is_trivially_destructible_v<T>) {
        value_.~T();
      }
    } else {
      status_.~Status();
    }
  }

  template <typename... Args>
  inline void construct_value(Args&&... args) {
    ::new (&value_) T(std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline void construct_status(Args&&... args) {
    ::new (&status_) absl::Status(std::forward<Args>(args)...);
  }

  template <typename U>
  void assign_status(U&& value) noexcept {
    Clear();
    status_ = static_cast<absl::Status>(std::forward<U>(value));
    EnsureNotOk();
  }

  template <typename U>
  void assign_value(U&& value) {
    if (ok()) {
      value_ = std::forward<U>(value);
    } else {
      this->construct_value(std::forward<U>(value));
      status_ = absl::OkStatus();
    }
  }

  bool ok() const { return status_.ok(); }

 protected:
  void Clear() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      if (ok()) value_.~T();
    }
  }

  void EnsureOk() const {
    if (ABSL_PREDICT_FALSE(!ok())) internal_result::CrashOnResultNotOk(status_);
  }
  void EnsureNotOk() {
    if (ABSL_PREDICT_FALSE(ok()))
      internal_result::HandleInvalidResultCtor(&status_);
  }

  template <typename U>
  T ValueOrImpl(U&& default_value) const& {
    if (ok()) {
      return value_;
    }
    return std::forward<U>(default_value);
  }

  template <typename U>
  T ValueOrImpl(U&& default_value) && {
    if (ok()) {
      return std::move(value_);
    }
    return std::forward<U>(default_value);
  }

  // status_ will always be active, however by putting it in a union
  // we can ensure that the initialization is done exactly as needed.
  union {
    absl::Status status_;
  };
  union {
    T value_;
  };
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

// Metafunction for enabling the Result<T>::Result(U) constructor overloads.
template <typename T, typename U>
constexpr inline bool value_conversion =
    !is_result_status_or_inplace<absl::remove_cvref_t<U>>;
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
