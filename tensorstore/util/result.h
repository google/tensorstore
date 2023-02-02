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

#ifndef TENSORSTORE_RESULT_H_
#define TENSORSTORE_RESULT_H_

#include <initializer_list>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/preprocessor/cat.h"
#include "tensorstore/internal/preprocessor/expand.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/result_impl.h"  // IWYU pragma: export
#include "tensorstore/util/status.h"

namespace tensorstore {

template <typename T>
class [[nodiscard]] Result;

/// IsResult evaluates to `true` if `T` is an instance of `Result`.
///
/// \relates Result
template <typename T>
constexpr inline bool IsResult = false;

template <typename T>
constexpr inline bool IsResult<Result<T>> = true;

/// UnwrapResultType<T> maps
///
///   Result<T> -> T
///   absl::Status -> void
///   T -> T
///
/// \relates Result
template <typename T>
using UnwrapResultType = typename internal_result::UnwrapResultHelper<T>::type;

/// As above, preserving const / volatile / reference qualifiers.
///
/// \relates Result
template <typename T>
using UnwrapQualifiedResultType =
    internal::CopyQualifiers<T, UnwrapResultType<internal::remove_cvref_t<T>>>;

/// FlatResult<T> maps
///
///     T -> Result<T>
///     Result<T> -> Result<T>
///     absl::Status -> Result<void>
///
/// \relates Result
template <typename T>
using FlatResult = typename internal_result::UnwrapResultHelper<T>::result_type;

/// Type alias that maps `Result<T>` to ``Result<U>``, where ``U = MapType<U>``.
///
/// \relates Result
template <template <typename...> class MapType, typename... T>
using FlatMapResultType = Result<MapType<UnwrapResultType<T>...>>;

/// Type alias used by initial overloads of the "Pipeline" operator|.
///
/// \relates Result
template <typename T, typename Func>
using PipelineResultType =
    std::enable_if_t<IsResult<std::invoke_result_t<Func&&, T>>,
                     std::invoke_result_t<Func&&, T>>;

// in_place and in_place_type are disambiguation tags that can be passed to the
// constructor of Result to indicate that the contained object should be
// constructed in-place.
using std::in_place;
using std::in_place_t;

/// `Result<T>` implements a value-or-error concept providing a discriminated
/// union of a usable value, `T`, or an `absl::Status` error describing why the
/// value is not present. `Result<void>` is permitted, which becomes a wrapper
/// for `absl::Status`, and which enables use of `Result<T>` in a wide variety
/// of template contexts.
///
/// The primary use case for `Result<T>` is as the return value of a
/// function which may fail.
///
/// Initialization with a non-error `absl::Status` is only allowed for
/// `Result<void>`, otherwise non-error `absl::Status` initilization is
/// nonsensical because it does not provide a value.
///
/// Conversion from `Result<T>` to `Result<void>` is allowed; the status
/// is retained but any value is discarded by such a conversion.
///
/// Assignment operators always destroy the existing value and reconstruct the
/// value. This may be surprising, since it is unlike `std::optional` and many
/// other monadic C++ types.
///
/// There are quite a few similar classes extant:
///
/// - The `absl::StatusOr<T>` concept used by Google.
///
/// - An `std::variant<T, absl::Status>`.
///
/// - The proposed ``std::expected<>``, except that the error type is not
///   template-selectable.
///   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0323r6.html
///
/// Example::
///
///   Result<Foo> result = GetTheStuff();
///   if (!result) {
///      return result.status();
///   }
///   result->DoTheStuff();
///
/// \tparam T Contained value type, or `void` to indicate no contained value.
///     Must not be a reference type or cv-qualified.
/// \ingroup Error handling
template <typename T>
class Result : private internal_result::ResultStorage<T>,
               private internal_result::CopyCtorBase<T>,
               private internal_result::MoveCtorBase<T>,
               private internal_result::CopyAssignBase<T>,
               private internal_result::MoveAssignBase<T> {
  static_assert(!std::is_reference_v<T>, "T must not be a reference");
  static_assert(!std::is_array_v<T>, "T must not be a C array type");
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>,
                "T must not be cv-qualified");

  using Base = internal_result::ResultStorage<T>;
  using Base::status_;

 public:
  /// The type of the contained success value.
  using value_type = T;

  /// Equal to `T&`, or `void` if `T` is void.
  using reference_type = T&;

  /// Equal to `const T&`, or `void` if `T` is `void`.
  using const_reference_type = const T&;

  /// Always equal to `absl::Status`.
  using error_type = absl::Status;

  template <typename U>
  using rebind = Result<U>;

  ~Result() = default;

  /// Constructs an error Result with a code of `absl::StatusCode::kUnknown`.
  ///
  /// \id default
  explicit Result()
      : Base(internal_result::status_t{}, absl::StatusCode::kUnknown, "") {}

  /// Constructs from an existing result.
  ///
  /// \id copy
  Result(const Result& src) = default;
  Result(Result&& src) = default;

  /// Assigns from an existing result.
  ///
  /// \id copy
  Result& operator=(const Result& src) = default;
  Result& operator=(Result&& src) = default;

  /// absl::Status construtors.

  /// Constructs from a status object.
  ///
  /// \pre `!status` unless `T` is `void`.
  /// \id status
  Result(const absl::Status& status)
      : Base(internal_result::status_t{}, status) {
    ABSL_CHECK(!status_.ok());
  }
  Result(absl::Status&& status)
      : Base(internal_result::status_t{}, std::move(status)) {
    ABSL_CHECK(!status_.ok());
  }

  /// Assigns from a status object.
  ///
  /// \pre `!status` unless `T` is `void`.
  /// \id status
  Result& operator=(const absl::Status& status) {
    ABSL_CHECK(!status.ok());
    this->assign_status(status);
    return *this;
  }
  Result& operator=(absl::Status&& status) {
    ABSL_CHECK(!status.ok());
    this->assign_status(std::move(status));
    return *this;
  }

  /// Converting constructors

  /// Constructs a Result<T> from a Result<U> when T is constructible
  /// from U with implicit and explicit specializations.
  ///
  /// Disallowed when T is constructible from Result<U>.

  /// Constructs from an existing result with a convertible value type.
  ///
  /// .. note::
  ///
  ///    This constructor is *conditionally* implicit if `T` is implicitly
  ///    constructible from `U`.
  ///
  /// \requires `T` is constructible from `U` and is not constructible from
  ///     `Result<U>`.
  /// \id convert
  template <
      typename U,
      std::enable_if_t<
          (internal_result::result_conversion<T, U> &&  //
           std::is_constructible_v<
               T, std::add_const_t<std::add_lvalue_reference_t<U>>> &&  //
           std::is_convertible_v<
               std::add_const_t<std::add_lvalue_reference_t<U>>, T> &&        //
           !internal_result::is_constructible_convertible_from<T, Result<U>>  //
           )>* = nullptr>
  Result(const Result<U>& rhs) {
    construct_from(rhs);
  }
  template <
      typename U,
      std::enable_if_t<
          (internal_result::result_conversion<T, U> &&                        //
           std::is_constructible_v<T, std::add_rvalue_reference_t<U>> &&      //
           std::is_convertible_v<std::add_rvalue_reference_t<U>, T> &&        //
           !internal_result::is_constructible_convertible_from<T, Result<U>>  //
           )>* = nullptr>
  Result(Result<U>&& rhs) {
    construct_from(std::move(rhs));
  }

  // Explicit overload of above.
  template <
      typename U,
      std::enable_if_t<
          (internal_result::result_conversion<T, U> &&  //
           std::is_constructible_v<
               T, std::add_const_t<std::add_lvalue_reference_t<U>>> &&  //
           !std::is_convertible_v<
               std::add_const_t<std::add_lvalue_reference_t<U>>, T> &&        //
           !internal_result::is_constructible_convertible_from<T, Result<U>>  //
           )>* = nullptr>
  explicit Result(const Result<U>& rhs) {
    construct_from(rhs);
  }

  // Explicit overload of above.
  template <
      typename U,
      std::enable_if_t<
          (internal_result::result_conversion<T, U> &&                        //
           std::is_constructible_v<T, std::add_rvalue_reference_t<U>> &&      //
           !std::is_convertible_v<std::add_rvalue_reference_t<U>, T> &&       //
           !internal_result::is_constructible_convertible_from<T, Result<U>>  //
           )>* = nullptr>
  explicit Result(Result<U>&& rhs) {
    construct_from(std::move(rhs));
  }

  /// Assigns from an existing result with a convertible value type.
  ///
  /// \requires `T` is `void`, or `T` is constructible from `U` and is not
  ///     constructible from `Result<U>`.
  /// \id convert
  template <
      typename U,
      std::enable_if_t<
          (internal_result::result_conversion<T, U> &&  //
           std::is_convertible_v<
               std::add_const_t<std::add_lvalue_reference_t<U>>, T> &&        //
           !internal_result::is_constructible_convertible_from<T, Result<U>>  //
           )>* = nullptr>
  Result& operator=(const Result<U>& rhs) {
    this->assign_from(rhs);
    return *this;
  }
  template <
      typename U,
      std::enable_if_t<
          (!internal_result::result_conversion<T, U> &&                      //
           !std::is_same_v<T, U> &&                                          //
           std::is_convertible_v<std::add_rvalue_reference_t<U>, T> &&       //
           !internal_result::is_constructible_convertible_from<T, Result<U>  //
                                                               >)>* = nullptr>
  Result& operator=(Result<U>&& rhs) {
    this->assign_from(std::move(rhs));
    return *this;
  }

  /// Forwarding constructors.

  /// Directly constructs the contained value from the specified arguments.
  /// \id in_place
  template <typename... Args>
  Result(std::in_place_t, Args&&... args)
      : Base(internal_result::value_t{}, std::forward<Args>(args)...) {}
  template <typename U, typename... Args>
  Result(std::in_place_t, std::initializer_list<U> il, Args&&... args)
      : Base(internal_result::value_t{}, il, std::forward<Args>(args)...) {}

  /// Single value forwarding constructors

  /// Disallowed when T is constructible from Result<U> except when T is U.
  /// Disallowed when absl::Status is constructible from U except when T is U.

  /// Constructs the contained value from a convertible value.
  ///
  /// .. note::
  ///
  ///    This constructor is *conditionally* implicit if `T` is implicitly
  ///    constructible from `U`.
  ///
  /// \requires `T` is constructible from `U` and either `T` is `U` or `T` is
  ///     not constructible from `Result<U>`.
  /// \id value
  template <typename U,
            std::enable_if_t<(internal_result::value_conversion<T, U> &&
                              std::is_constructible_v<T, U&&> &&
                              std::is_convertible_v<U&&, T>  //
                              )>* = nullptr>
  Result(U&& v) : Base(internal_result::value_t{}, std::forward<U>(v)) {}

  // Explicit overload.
  template <typename U,
            std::enable_if_t<(internal_result::value_conversion<T, U> &&
                              std::is_constructible_v<T, U&&> &&
                              !std::is_convertible_v<U&&, T>  //
                              )>* = nullptr>
  explicit Result(U&& v)
      : Base(internal_result::value_t{}, std::forward<U>(v)) {}

  /// Assigns the contained value from a convertible value.
  ///
  /// \requires `T` is constructible from `U` and either `T` is `U` or `T` is
  ///     not constructible from `Result<U>`.
  /// \id value
  template <typename U,
            std::enable_if_t<(internal_result::value_conversion<T, U> &&
                              std::is_constructible_v<T, U&&> &&
                              std::is_convertible_v<U&&, T>  //
                              )>* = nullptr>
  Result& operator=(U&& v) {
    this->emplace_value(std::forward<U>(v));
    return *this;
  }

  /// Reconstructs the contained value in-place with the given forwarded
  /// arguments.
  ///
  /// Example::
  ///
  ///   Result<Foo> opt = absl::UnknownError("");
  ///   opt.emplace(arg1,arg2,arg3);  // Constructs Foo(arg1,arg2,arg3)
  ///
  template <typename... Args>
  reference_type emplace(Args&&... args) {
    static_assert(sizeof...(Args) == 0 || !std::is_void_v<T>);
    this->emplace_value(std::forward<Args>(args)...);
    return this->value_;
  }
  template <typename U, typename... Args>
  reference_type emplace(std::initializer_list<U> il, Args&&... args) {
    this->emplace_value(il, std::forward<Args>(args)...);
    return this->value_;
  }

  /// Ignores the result. This method signals intent to ignore the result to
  /// suppress compiler warnings from ``[[nodiscard]]``.
  void IgnoreResult() const {}

  /// Result observers.

  /// Returns `true` if this represents a success state, `false` for a failure
  /// state.
  ///
  /// `value()` is valid only iff `has_value()` is `true`. `status()` is valid
  /// iff `has_value()` is false.
  bool ok() const { return status_.ok(); }
  bool has_value() const { return status_.ok(); }
  explicit operator bool() const noexcept { return status_.ok(); }

  /// Checked value accessor.
  ///
  /// Terminates the process if `*this` represents a failure state.
  ///
  /// \pre `has_value() == true`
  const_reference_type value() const& noexcept TENSORSTORE_LIFETIME_BOUND {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    return this->value_;
  }
  reference_type value() & noexcept TENSORSTORE_LIFETIME_BOUND {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    return this->value_;
  }
  value_type value() && noexcept {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    return std::move(this->value_);
  }

  /// Returns the error status.
  const absl::Status& status() const& noexcept TENSORSTORE_LIFETIME_BOUND {
    return status_;
  }
  absl::Status status() && {
    // Note: This relies on the fact that the moved-from `absl::Status` does not
    // have a status code of `absl::StatusCode::kOk`.
    return status_.ok() ? absl::OkStatus() : std::move(status_);
  }

  /// Returns a pointer to the contained value.
  ///
  /// \pre has_value() == true
  template <typename U = T>
  constexpr const U* operator->() const noexcept TENSORSTORE_LIFETIME_BOUND {
    assert_has_value();
    return &this->value_;
  }
  template <typename U = T>
  constexpr U* operator->() noexcept TENSORSTORE_LIFETIME_BOUND {
    assert_has_value();
    return &this->value_;
  }

  /// Returns a reference to the contained value.
  template <typename U = T>
  constexpr const U& operator*() const& noexcept TENSORSTORE_LIFETIME_BOUND {
    assert_has_value();
    return this->value_;
  }
  template <typename U = T>
  constexpr U& operator*() & noexcept TENSORSTORE_LIFETIME_BOUND {
    assert_has_value();
    return this->value_;
  }
  template <typename U = T>
  constexpr U&& operator*() && noexcept TENSORSTORE_LIFETIME_BOUND {
    assert_has_value();
    return std::move(this->value_);
  }

  /// "Pipeline" operator for `Result`.
  ///
  /// In the expression  ``x | y``, if
  ///
  /// - ``x`` is of type `Result<T>`
  /// - ``y`` is a function having signature ``U (T)`` or ``Result<U>(T)``
  ///
  /// Then `operator|` applies ``y`` to the value contained in ``x``, returning
  /// a ``Result<U>``. In other words, this function is roughly expressed as::
  ///
  ///    return !x.ok() ? x.status() : Result<U>(y(x.value()))
  ///
  template <typename Func>
  inline FlatResult<std::invoke_result_t<Func&&, reference_type>>  //
  operator|(Func&& func) const& {
    if (!ok()) return status();
    return static_cast<Func&&>(func)(value());
  }
  template <typename Func>
  inline FlatResult<std::invoke_result_t<Func&&, T>>  //
  operator|(Func&& func) && {
    if (!ok()) return status();
    return static_cast<Func&&>(func)(value());
  }

  /// Returns the contained value, or `default_value` if `*this` is in an error
  /// state.
  template <typename U>
  constexpr T value_or(U&& default_value) const& {
    return has_value() ? this->value_ : std::forward<U>(default_value);
  }
  template <typename U>
  constexpr T value_or(U&& default_value) && {
    return has_value() ? std::move(this->value_)
                       : std::forward<U>(default_value);
  }

  /// FIXME: Result::and_then()
  /// FIXME: Result::map()

  /// Compares two results for equality.
  ///
  /// Two results are considered equal if:
  ///
  /// - they both contain error values that are equal; or
  /// - they both contain values that are equal; or
  /// - they both contain values of type `void`.
  ///
  /// \requires `T` and `U` are equality comparable or both `void`.
  template <typename U>
  friend std::enable_if_t<(internal::IsEqualityComparable<T, U> ||
                           // Use `std::is_same_v<T, U>` rather than
                           // `std::is_void_v<T>` to ensure condition is
                           // dependent.
                           (std::is_same_v<T, U> && std::is_void_v<U>)),
                          bool>
  operator==(const Result<T>& a, const Result<U>& b) {
    if (a.has_value() != b.has_value()) {
      return false;
    }
    if (!a.has_value()) {
      return a.status() == b.status();
    }
    if constexpr (std::is_void_v<T>) {
      return true;
    } else {
      return *a == *b;
    }
  }
  template <typename U>
  friend std::enable_if_t<
      internal_result::IsEqualityComparableIfNotResult<T, U>, bool>
  operator==(const Result<T>& a, const U& b) {
    return a.has_value() ? (a.value() == b) : false;
  }
  template <typename U>
  friend std::enable_if_t<
      internal_result::IsEqualityComparableIfNotResult<T, U>, bool>
  operator==(const U& a, const Result<T>& b) {
    return b.has_value() ? (b.value() == a) : false;
  }

  /// Checks if two Result values are not equal.
  ///
  /// \requires `T` and `U` are equality comparable or both `void`.
  template <typename U>
  friend std::enable_if_t<(internal::IsEqualityComparable<T, U> ||
                           // Use `std::is_same_v<T, U>` rather than
                           // `std::is_void_v<T>` to ensure condition is
                           // dependent.
                           (std::is_same_v<T, U> && std::is_void_v<U>)),
                          bool>
  operator!=(const Result<T>& a, const Result<U>& b) {
    return !(a == b);
  }
  template <typename U>
  friend std::enable_if_t<
      internal_result::IsEqualityComparableIfNotResult<T, U>, bool>
  operator!=(const Result<T>& a, const U& b) {
    return !(a == b);
  }
  template <typename U>
  friend std::enable_if_t<
      internal_result::IsEqualityComparableIfNotResult<T, U>, bool>
  operator!=(const U& a, const Result<T>& b) {
    return !(a == b);
  }

 private:
  // Construct this from a Result<U>.
  template <typename Other>
  inline void construct_from(Other&& other) {
    if (other.has_value()) {
      this->construct_value(std::forward<Other>(other).value());
    } else {
      status_ = std::forward<Other>(other).status();
    }
  }

  // Assign to this from a Result<U>.
  template <typename Other>
  inline void assign_from(Other&& other) {
    if (other.has_value()) {
      this->emplace_value(std::forward<Other>(other).value());
    } else {
      this->assign_status(std::forward<Other>(other).status());
    }
  }

  // Functionally equivalent to assert(has_value()), except that it dumps the
  // internal status when the assert fails.
  inline void assert_has_value() const {
#if !defined(NDEBUG)
    if (!has_value()) {
      ABSL_LOG(FATAL) << "assert_has_value: " << this->status_;
    }
#endif
  }
};

// Specialization of `Result` for `void`.  This just wraps a plain
// `absl::Status` to provide the same API as `Result`.
template <>
class Result<void> {
 public:
  using value_type = void;
  using reference_type = void;
  using const_reference_type = void;
  using error_type = absl::Status;

  template <typename U>
  using rebind = Result<U>;

  explicit Result() : status_(absl::Status(absl::StatusCode::kUnknown, "")) {}

  Result(const absl::Status& status) : status_(status) {}
  Result(absl::Status&& status) : status_(std::move(status)) {}

  template <typename T>
  Result(const Result<T>& other) : status_(other.status()) {}

  template <typename T>
  Result(Result<T>&& other) : status_(std::move(other).status()) {}

  Result(std::in_place_t) {}

  Result& operator=(const absl::Status& status) {
    status_ = status;
    return *this;
  }

  Result& operator=(absl::Status&& status) {
    status_ = std::move(status);
    return *this;
  }

  template <typename T>
  Result& operator=(const Result<T>& other) {
    status_ = other.status();
    return *this;
  }

  template <typename T>
  Result& operator=(Result<T>&& other) {
    status_ = std::move(other).status();
    return *this;
  }

  void emplace() { status_ = absl::OkStatus(); }

  void IgnoreResult() const {}

  bool ok() const { return status_.ok(); }
  bool has_value() const { return status_.ok(); }
  explicit operator bool() const noexcept { return status_.ok(); }

  void value() const { TENSORSTORE_CHECK_OK(status()); }

  const absl::Status& status() const& noexcept TENSORSTORE_LIFETIME_BOUND {
    return status_;
  }

  absl::Status&& status() && { return std::move(status_); }

  friend bool operator==(const Result& a, const Result& b) {
    return a.status_ == b.status_;
  }

  friend bool operator!=(const Result& a, const Result& b) { return !(a == b); }

 private:
  absl::Status status_;
};

/// Returns a Result<T> with a (possibly-default) value.
///
/// Example:
///    Result<void> r = MakeResult();
///    Result<int>  x = MakeResult<int>();
///    auto result = MakeResult(7);
///
/// \relates Result
/// \id value
inline Result<void> MakeResult() { return {std::in_place}; }
template <int&... ExplicitArgumentBarrier, typename T>
inline Result<typename tensorstore::internal::remove_cvref_t<T>> MakeResult(
    T&& t) {
  return {std::in_place, std::forward<T>(t)};
}
template <typename U, typename... Args>
inline Result<U> MakeResult(Args&&... args) {
  return {std::in_place, std::forward<Args>(args)...};
}

/// Returns a Result corresponding to a success or error `status`.
///
/// \relates Result
/// \id status
inline Result<void> MakeResult(absl::Status status) {
  return Result<void>(std::move(status));
}
template <typename U>
inline Result<U> MakeResult(absl::Status status) {
  return status.ok() ? Result<U>(std::in_place) : Result<U>(std::move(status));
}

/// FIXME: It would be nice if the naming convention for UnwrapResult
/// and GetStatus were the same. I think that I'd prefer UnwrapStatus()
/// to return a absl::Status type() and UnwrapResult() to return a value type.

/// Returns the error status of `Result`, or `absl::OkStatus()` if `Result` has
/// a value.
///
/// \returns `result.status()`
/// \relates Result
/// \id result
template <typename T>
inline absl::Status GetStatus(const Result<T>& result) {
  return result.status();
}
template <typename T>
inline absl::Status GetStatus(Result<T>&& result) {
  return std::move(result).status();
}

/// UnwrapResult returns the value contained by the Result<T> instance,
/// `*t`, or the value, `t`.
///
/// \relates Result
template <typename T>
inline T&& UnwrapResult(T&& t) {
  return std::forward<T>(t);
}
template <typename T>
inline const T& UnwrapResult(const Result<T>& t) {
  return *t;
}
template <typename T>
inline T& UnwrapResult(Result<T>& t) {  // NOLINT
  return *t;
}
template <typename T>
inline T&& UnwrapResult(Result<T>&& t) {
  return *std::move(t);
}

/// Tries to call `func` with `Result`-wrapped arguments.
///
/// The return value of `func` is wrapped in a `Result` if it not already a
/// `Result` instance.
///
/// \returns `std::forward<Func>(func)(UnwrapResult(std::forward<T>(arg))...)`
///     if no `Result`-wrapped `arg` is an in error state.  Otherwise, returns
///     the error `absl::Status` of the first `Result`-wrapped `arg` in an error
///     state.
///
/// \relates Result
template <typename Func, typename... T>
FlatResult<std::invoke_result_t<Func&&, UnwrapQualifiedResultType<T>...>>
MapResult(Func&& func, T&&... arg) {
  absl::Status status;
  if (!([&] {
        if constexpr (IsResult<internal::remove_cvref_t<T>>) {
          if (!arg.ok()) {
            status = arg.status();
            return false;
          }
        }
        return true;
      }() &&
        ...)) {
    return status;
  }
  return std::forward<Func>(func)(UnwrapResult(std::forward<T>(arg))...);
}

namespace internal_result {

// Helper struct for ChainResultType.
template <typename T, typename... Func>
struct ChainResultTypeHelper;

template <typename T>
struct ChainResultTypeHelper<T> {
  using type =
      typename UnwrapResultHelper<internal::remove_cvref_t<T>>::result_type;
};

template <typename T, typename Func0, typename... Func>
struct ChainResultTypeHelper<T, Func0, Func...>
    : ChainResultTypeHelper<
          std::invoke_result_t<Func0&&, UnwrapQualifiedResultType<T>>,
          Func...> {};

/// Template alias that evaluates to the result of calling `ChainResult`.
///
/// For example:
///
///     ChainResultType<int>
///         -> Result<int>
///     ChainResultType<Result<int>>
///         -> Result<int>
///     ChainResultType<int, float(*)(int)>
///         -> Result<float>
///     ChainResultType<Result<int>, float(*)(int)>
///         -> Result<float>
///     ChainResultType<int, Result<float>(*)(int)>
///         -> Result<float>
///     ChainResultType<int, float(*)(int), std::string(*)(float)>
///         -> Result<std::string>
template <typename T, typename... Func>
using ChainResultType = typename ChainResultTypeHelper<T, Func...>::type;

}  // namespace internal_result

/// Applies a sequence of functions, which may optionally return
/// `Result`-wrapped values, to an optionally `Result`-wrapped value.
///
/// For example::
///
///     float func1(int x);
///     Result<std::string> func2(float x);
///     bool func3(std::string_view x);
///
///     Result<bool> y1 = ChainResult(Result<int>(3), func1, func2, func3);
///     Result<bool> y2 = ChainResult(3, func1, func2, func3);
///
/// \relates Result
template <typename T>
internal_result::ChainResultType<T> ChainResult(T&& arg) {
  return std::forward<T>(arg);
}
template <typename T, typename Func0, typename... Func>
internal_result::ChainResultType<T, Func0, Func...> ChainResult(
    T&& arg, Func0&& func0, Func&&... func) {
  return ChainResult(
      MapResult(std::forward<Func0>(func0), std::forward<T>(arg)),
      std::forward<Func>(func)...);
}

#define TENSORSTORE_INTERNAL_ASSIGN_OR_RETURN_IMPL(temp, decl, expr,      \
                                                   error_expr, ...)       \
  auto temp = (expr);                                                     \
  static_assert(tensorstore::IsResult<decltype(temp)>,                    \
                "TENSORSTORE_ASSIGN_OR_RETURN requires a Result value."); \
  if (ABSL_PREDICT_FALSE(!temp)) {                                        \
    auto _ = std::move(temp).status();                                    \
    static_cast<void>(_);                                                 \
    return (error_expr);                                                  \
  }                                                                       \
  decl = std::move(*temp);                                                \
  /**/

/// Convenience macro for propagating errors when calling a function that
/// returns a `tensorstore::Result`.
///
/// This macro generates multiple statements and should be invoked as follows::
///
///     Result<int> GetSomeResult();
///
///     TENSORSTORE_ASSIGN_OR_RETURN(int x, GetSomeResult());
///
/// An optional third argument specifies the return expression in the case of an
/// error.  A variable ``_`` bound to the error `absl::Status` value is in
/// scope within this expression.  For example::
///
///     TENSORSTORE_ASSIGN_OR_RETURN(int x, GetSomeResult(),
///                                  _.Annotate("Context message"));
///
/// \relates tensorstore::Result
#define TENSORSTORE_ASSIGN_OR_RETURN(decl, ...)                          \
  TENSORSTORE_PP_EXPAND(TENSORSTORE_INTERNAL_ASSIGN_OR_RETURN_IMPL(      \
      TENSORSTORE_PP_CAT(tensorstore_assign_or_return_, __LINE__), decl, \
      __VA_ARGS__, _))                                                   \
  /**/
// Note: the use of `TENSORSTORE_PP_EXPAND` above is a workaround for MSVC 2019
// preprocessor limitations.

#define TENSORSTORE_INTERNAL_CHECK_OK_AND_ASSIGN_IMPL(temp, decl, expr)      \
  auto temp = (expr);                                                        \
  static_assert(tensorstore::IsResult<decltype(temp)>,                       \
                "TENSORSTORE_CHECK_OK_AND_ASSIGN requires a Result value."); \
  if (ABSL_PREDICT_FALSE(!temp)) {                                           \
    TENSORSTORE_CHECK_OK(temp.status());                                     \
  }                                                                          \
  decl = std::move(*temp);

/// Convenience macro for checking errors when calling a function that returns a
/// `tensorstore::Result`.
///
/// This macro generates multiple statements and should be invoked as follows::
///
///     Result<int> GetSomeResult();
///
///     TENSORSTORE_CHECK_OK_AND_ASSIGN(int x, GetSomeResult());
///
/// If the expression returns a `tensorstore::Result` with a value, the value is
/// assigned to `decl`.  Otherwise, the error is logged and the program is
/// terminated.
///
/// \relates tensorstore::Result
#define TENSORSTORE_CHECK_OK_AND_ASSIGN(decl, ...)                         \
  TENSORSTORE_PP_EXPAND(TENSORSTORE_INTERNAL_CHECK_OK_AND_ASSIGN_IMPL(     \
      TENSORSTORE_PP_CAT(tensorstore_check_ok_or_return_, __LINE__), decl, \
      __VA_ARGS__))

}  // namespace tensorstore

#endif  // TENSORSTORE_RESULT_H_
