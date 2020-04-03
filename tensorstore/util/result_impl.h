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
#include <type_traits>
#include <utility>

#include "absl/base/config.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/utility/utility.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"

// IWYU pragma: private, include "third_party/tensorstore/util/result.h"

namespace tensorstore {

template <typename T>
class Result;

namespace internal_result {

// Tag types to select internal constructors.
struct status_t {};
struct value_t {};
struct noinit_t {};

// ----------------------------------------------------------------
// Storage base classes for Result<T>
// ----------------------------------------------------------------

// ResultStorageBase
// * Owns the underlying Status and T data objects.
// * Specialized construction and assignment for T=void vs. non-void.
//
// FIXME: tensorstore::Status use of std::error_code means that the
// Status() constructor, and consequently all the constructors of
// Result<T> which take status cannot be constexpr.
//
template <class T>
struct ResultStorageBase {
  using value_type = T;
  using reference_type = T&;
  using const_reference_type = const T&;

  constexpr ResultStorageBase(noinit_t) noexcept
      : dummy_(), has_value_(false) {}

  template <typename... Args>
  explicit ResultStorageBase(value_t, Args&&... args)
      : value_(std::forward<Args>(args)...), has_value_(true) {}

  template <typename... Args>
  explicit ResultStorageBase(status_t, Args&&... args) noexcept
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
    Status status_;
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
    Status status_;
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
        Status(std::forward<Args>(args)...);
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

// Ordered by level of restriction, from low to high.
// Copyable implies movable.
enum class MixinTraits { kCopyable = 0, kMovable = 1, kNonMovable = 2 };

// Base class for enabling/disabling copy/move constructor.
template <MixinTraits>
class ResultConstructorMixinBase;

template <>
class ResultConstructorMixinBase<MixinTraits::kCopyable> {
 public:
  constexpr ResultConstructorMixinBase() = default;
  ResultConstructorMixinBase(const ResultConstructorMixinBase&) = default;
  ResultConstructorMixinBase(ResultConstructorMixinBase&&) = default;
  ResultConstructorMixinBase& operator=(const ResultConstructorMixinBase&) =
      default;
  ResultConstructorMixinBase& operator=(ResultConstructorMixinBase&&) = default;
};

template <>
class ResultConstructorMixinBase<MixinTraits::kMovable> {
 public:
  constexpr ResultConstructorMixinBase() = default;
  ResultConstructorMixinBase(const ResultConstructorMixinBase&) = delete;
  ResultConstructorMixinBase(ResultConstructorMixinBase&&) = default;
  ResultConstructorMixinBase& operator=(const ResultConstructorMixinBase&) =
      default;
  ResultConstructorMixinBase& operator=(ResultConstructorMixinBase&&) = default;
};

template <>
class ResultConstructorMixinBase<MixinTraits::kNonMovable> {
 public:
  constexpr ResultConstructorMixinBase() = default;
  ResultConstructorMixinBase(const ResultConstructorMixinBase&) = delete;
  ResultConstructorMixinBase(ResultConstructorMixinBase&&) = delete;
  ResultConstructorMixinBase& operator=(const ResultConstructorMixinBase&) =
      default;
  ResultConstructorMixinBase& operator=(ResultConstructorMixinBase&&) = default;
};

// Base class for enabling/disabling copy/move assignment.
template <MixinTraits>
class ResultAssignMixinBase;

template <>
class ResultAssignMixinBase<MixinTraits::kCopyable> {
 public:
  constexpr ResultAssignMixinBase() = default;
  ResultAssignMixinBase(const ResultAssignMixinBase&) = default;
  ResultAssignMixinBase(ResultAssignMixinBase&&) = default;
  ResultAssignMixinBase& operator=(const ResultAssignMixinBase&) = default;
  ResultAssignMixinBase& operator=(ResultAssignMixinBase&&) = default;
};

template <>
class ResultAssignMixinBase<MixinTraits::kMovable> {
 public:
  constexpr ResultAssignMixinBase() = default;
  ResultAssignMixinBase(const ResultAssignMixinBase&) = default;
  ResultAssignMixinBase(ResultAssignMixinBase&&) = default;
  ResultAssignMixinBase& operator=(const ResultAssignMixinBase&) = delete;
  ResultAssignMixinBase& operator=(ResultAssignMixinBase&&) = default;
};

template <>
class ResultAssignMixinBase<MixinTraits::kNonMovable> {
 public:
  constexpr ResultAssignMixinBase() = default;
  ResultAssignMixinBase(const ResultAssignMixinBase&) = default;
  ResultAssignMixinBase(ResultAssignMixinBase&&) = default;
  ResultAssignMixinBase& operator=(const ResultAssignMixinBase&) = delete;
  ResultAssignMixinBase& operator=(ResultAssignMixinBase&&) = delete;
};

// NOTE: This is used for both default-construction and default-assignment.
template <typename T>
constexpr MixinTraits GetConstructorMixinTraits() {
  return std::is_void<T>::value  // void<T> has all constructors.
             ? MixinTraits::kCopyable
             : std::is_copy_constructible<T>::value
                   ? MixinTraits::kCopyable
                   : std::is_move_constructible<T>::value
                         ? MixinTraits::kMovable
                         : MixinTraits::kNonMovable;
}

// Whether T is constructible from Result<U>.
template <typename T, typename U>
struct is_constructible_from_result
    : std::integral_constant<
          bool, std::is_constructible<T, Result<U>&>::value ||
                    std::is_constructible<T, Result<U>&&>::value ||
                    std::is_constructible<T, const Result<U>&>::value ||
                    std::is_constructible<T, const Result<U>&&>::value> {};

// Whether T is constructible or convertible from Result<U>.
template <typename T, typename U>
struct is_constructible_convertible_from_result
    : std::integral_constant<
          bool, is_constructible_from_result<T, U>::value ||
                    std::is_convertible<Result<U>&, T>::value ||
                    std::is_convertible<Result<U>&&, T>::value ||
                    std::is_convertible<const Result<U>&, T>::value ||
                    std::is_convertible<const Result<U>&&, T>::value> {};

}  // namespace internal_result
}  // namespace tensorstore

#endif  // TENSORSTORE_RESULT_IMPL_H_
