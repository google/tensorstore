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
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/result_impl.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

template <typename T>
class [[nodiscard]] Result;

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `Result`.
template <typename T>
struct IsResult : public std::false_type {};

template <typename T>
struct IsResult<Result<T>> : public std::true_type {};

namespace internal_result {

/// Result type traits helper structs for UnwrapResultType / FlatResultType.
template <typename T>
struct UnwrapResultHelper {
  static_assert(std::is_same<T, internal::remove_cvref_t<T>>::value,
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
struct UnwrapResultHelper<Status> {
  using type = void;
  using result_type = Result<void>;
};

}  // namespace internal_result

/// UnwrapResultType<T> maps
///
///   Result<T> -> T
///   Status -> void
///   T -> T
template <typename T>
using UnwrapResultType = typename internal_result::UnwrapResultHelper<T>::type;

/// As above, preserving const / volatile / reference qualifiers.
template <typename T>
using UnwrapQualifiedResultType =
    internal::CopyQualifiers<T, UnwrapResultType<internal::remove_cvref_t<T>>>;

/// FlatResult<T> maps
///
///     T -> Result<T>
///     Result<T> -> Result<T>
///     Status -> Result<void>
///
template <typename T>
using FlatResult = typename internal_result::UnwrapResultHelper<T>::result_type;

/// Type alias that maps `Result<T>` to `Result<U>`, where `U = MapType<U>`.
template <template <typename...> class MapType, typename... T>
using FlatMapResultType = Result<MapType<UnwrapResultType<T>...>>;

/// Type alias used by initial overloads of the "Pipeline" operator|.
template <typename T, typename Func>
using PipelineResultType =
    std::enable_if_t<IsResult<std::invoke_result_t<Func&&, T>>::value,
                     std::invoke_result_t<Func&&, T>>;

// in_place and in_place_type are disambiguation tags that can be passed to the
// constructor of Result to indicate that the contained object should be
// constructed in-place.
using std::in_place;
using std::in_place_t;

/// \brief Result<R> implements a value-or-error concept using the
/// existing tensorstore::Status mechanism. It provides a discriminated
/// union of a usable value, T, or an error Status explaining why the value
/// is not present.
///
/// The primary use case for Result<T> is as the return value of a
/// function which might fail.
///
/// `Result` requires explicit initialization using a non-default constructor.
/// Use `Result<T>{in_place, ...}` or `Result<T>{Status(...)}`
/// to construct a Result with the desired state.
///
/// Initialization with a non-error Status is not allowed.
///
/// There are quite a few similar classes extant:
///
/// * The StatusOr concept used by Google protocol buffers.
/// * A std::variant<T, Status>, except that it allows T=void.
/// * The proposed std::expected<>, except that the error type is not
///   template-selectable.
///   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0323r6.html
///
/// \example
///
///   Result<Foo> result = GetTheStuff();
///   if (!result) {
///      return result.status();
///   }
///   result->DoTheStuff();
///
template <typename T>
class Result : private internal_result::ResultStorage<T>,
               private internal_result::ResultConstructorMixinBase<
                   internal_result::GetConstructorMixinTraits<T>()>,
               private internal_result::ResultAssignMixinBase<
                   internal_result::GetConstructorMixinTraits<T>()> {
  static_assert(!std::is_reference<T>::value, "T must not be a reference");
  static_assert(!std::is_same<T, Status>::value, "T must not be a Status");
  static_assert(!std::is_convertible<T, Status>::value,
                "T must not be convertible to a Status");
  static_assert(!std::is_array<T>::value, "T must not be a std::array type");
  static_assert(!IsResult<T>::value, "T must not be a Result");

  static_assert(!std::is_const<T>::value && !std::is_volatile<T>::value,
                "T must not be cv-qualified");

  using storage = internal_result::ResultStorage<T>;

 public:
  /// The type of the contained success value.
  using value_type = typename storage::value_type;
  using reference_type = typename storage::reference_type;
  using const_reference_type = typename storage::const_reference_type;

  using error_type = Status;

  template <typename U>
  using rebind = Result<U>;

  explicit constexpr Result() = delete;

  /// Copy and move constructors, standard semantics
  Result(const Result& src) = default;
  Result(Result&& src) = default;

  /// \brief Construct a Result<T> with an error status.
  /// \requires `!status`
  Result(const Status& status) : storage(internal_result::status_t(), status) {
    TENSORSTORE_CHECK(!status.ok());
  }
  Result(Status&& status)
      : storage(internal_result::status_t(), std::move(status)) {
    TENSORSTORE_CHECK(!status.ok());
  }

  /// \brief Constructs a non-empty `Result` direct-initialized value of type
  /// `T` from the arguments `std::forward<Args>(args)...`  within the `Result`.
  template <typename... Args>
  Result(in_place_t, Args&&... args)
      : storage(internal_result::value_t(), std::forward<Args>(args)...) {}

  /// Constructs a non-empty `Result` direct-initialized value of type `T` from
  /// the arguments of an initializer_list and `std::forward<Args>(args)...`.
  template <typename U, typename... Args>
  Result(in_place_t, std::initializer_list<U> il, Args&&... args)
      : storage(internal_result::value_t(), il, std::forward<Args>(args)...) {}

  /// \brief Value constructor (implicit)
  template <typename U,
            std::enable_if_t<
                (!IsResult<internal::remove_cvref_t<U>>::value &&  //
                 !std::is_same<Status, internal::remove_cvref_t<U>>::value &&
                 std::is_convertible<U&&, T>::value  //
                 )>* = nullptr>
  Result(U&& v) : storage(internal_result::value_t(), std::forward<U>(v)) {}

  /// \brief Value constructor (explicit)
  template <typename U,
            std::enable_if_t<
                (!IsResult<internal::remove_cvref_t<U>>::value &&  //
                 !std::is_same<Status, internal::remove_cvref_t<U>>::value &&
                 !std::is_convertible<U&&, T>::value &&  //
                 std::is_constructible<T, U&&>::value    //
                 )>* = nullptr>
  explicit Result(U&& v)
      : storage(internal_result::value_t(), std::forward<U>(v)) {}

  /// FIXME: Allow explicit cast Result<T> -> Result<void>.

  /// \brief Converting copy constructor (implicit)
  template <typename U,
            std::enable_if_t<
                (!internal_result::is_constructible_convertible_from_result<
                     value_type, U>::value &&
                 std::is_convertible<const U&, value_type>::value  //
                 )>* = nullptr>
  Result(const Result<U>& rhs) : storage(internal_result::noinit_t{}) {
    construct_from(rhs);
  }

  /// \brief Converting copy constructor (explicit)
  template <typename U,
            std::enable_if_t<
                (!internal_result::is_constructible_convertible_from_result<
                     value_type, U>::value &&
                 !std::is_convertible<const U&, value_type>::value &&  //
                 std::is_constructible<T, const U&>::value             //
                 )>* = nullptr>
  explicit Result(const Result<U>& rhs) : storage(internal_result::noinit_t{}) {
    construct_from(rhs);
  }

  /// \brief Converting move constructor (implicit)
  template <typename U,
            std::enable_if_t<
                (!internal_result::is_constructible_convertible_from_result<
                     value_type, U>::value &&
                 std::is_convertible<U&&, value_type>::value  //
                 )>* = nullptr>
  Result(Result<U>&& rhs) : storage(internal_result::noinit_t{}) {
    construct_from(std::move(rhs));
  }

  /// \brief Converting move constructor (explicit)
  template <typename U,
            std::enable_if_t<
                (!internal_result::is_constructible_convertible_from_result<
                     value_type, U>::value &&
                 !std::is_convertible<U&&, value_type>::value &&  //
                 std::is_constructible<value_type, U&&>::value    //
                 )>* = nullptr>
  explicit Result(Result<U>&& rhs) : storage(internal_result::noinit_t{}) {
    construct_from(std::move(rhs));
  }

  ~Result() = default;

  // Assignment Operators

  /// Copy and move assignment operators, standard semantics
  Result& operator=(const Result& src) = default;
  Result& operator=(Result&& src) = default;

  /// \brief Copy and move assignment for Status.
  Result& operator=(const Status& status) {
    TENSORSTORE_CHECK(!status.ok());
    this->assign_status(status);
    return *this;
  }
  Result& operator=(Status&& status) {
    TENSORSTORE_CHECK(!status.ok());
    this->assign_status(std::move(status));
    return *this;
  }

  /// \brief Value assignment
  /// These assignment operators always destroy the existing value and
  /// reconstruct the value. This may be surprising, since it is unlike
  /// std::optional and many other monadic C++ types.
  template <typename U>
  typename std::enable_if_t<
      (!IsResult<internal::remove_cvref_t<U>>::value &&
       !std::is_same<Status, internal::remove_cvref_t<U>>::value &&
       std::is_constructible<value_type, U&&>::value),
      Result&>
  operator=(U&& v) {
    this->emplace_value(std::forward<U>(v));
    return *this;
  }

  template <typename U>
  typename std::enable_if_t<std::is_constructible<value_type, const U&>::value,
                            Result&>
  operator=(const Result<U>& rhs) {
    this->assign_from(rhs);
    return *this;
  }

  template <typename U>
  typename std::enable_if_t<std::is_constructible<value_type, U&&>::value,
                            Result&>
  operator=(Result<U>&& rhs) {
    this->assign_from(std::move(rhs));
    return *this;
  }

  /// \brief Result::emplace() reconstructs the underlying `T` in-place with
  /// the given forwarded arguments.
  ///
  /// Example:
  ///
  ///   Result<Foo> opt = absl::UnknownError("");
  ///   opt.emplace(arg1,arg2,arg3);  // Constructs Foo(arg1,arg2,arg3)
  ///
  template <typename... Args>
  typename std::enable_if_t<((std::is_void_v<value_type> &&
                              sizeof...(Args) == 0) ||
                             std::is_constructible_v<value_type, Args&&...>),
                            reference_type>
  emplace(Args&&... args) {
    this->emplace_value(std::forward<Args>(args)...);
    if constexpr (!std::is_void_v<value_type>) {
      return this->value_;
    }
  }

  template <typename U, typename... Args>
  typename std::enable_if_t<
      std::is_constructible<value_type, std::initializer_list<U>&,
                            Args&&...>::value,
      reference_type>
  emplace(std::initializer_list<U> il, Args&&... args) {
    this->emplace_value(il, std::forward<Args>(args)...);
    return this->value_;
  }

  void Construct(const Status& status) {
    TENSORSTORE_CHECK(!status.ok());
    this->assign_status(status);
  }
  void Construct(Status&& status) {
    TENSORSTORE_CHECK(!status.ok());
    this->assign_status(std::move(status));
  }

  template <typename U>
  std::enable_if_t<
      (std::is_void<U>::value && std::is_void<T>::value) ||
      std::is_constructible<T, std::add_lvalue_reference_t<const U>>::value>
  Construct(const Result<U>& u) {
    *this = u;
  }

  template <typename U>
  std::enable_if_t<
      (std::is_void<U>::value && std::is_void<T>::value) ||
      std::is_constructible<T, std::add_rvalue_reference_t<U>>::value>
  Construct(Result<U>&& u) {
    *this = std::move(u);
  }

  template <typename U>
  std::enable_if_t<!IsResult<internal::remove_cvref_t<U>>::value &&
                   std::is_constructible<T, U&&>::value>
  Construct(U&& u) {
    this->emplace_value(std::forward<U>(u));
  }

  template <typename... Args>
  std::enable_if_t<(std::is_void<T>::value && sizeof...(Args) == 0) ||
                   std::is_constructible<T, Args&&...>::value>
  Construct(in_place_t, Args&&... args) {
    this->emplace_value(std::forward<Args>(args)...);
  }

  template <typename U, typename... Args>
  std::enable_if_t<
      std::is_constructible<T, std::initializer_list<U>, Args&&...>::value>
  Construct(in_place_t, std::initializer_list<U> il, Args&&... args) {
    this->emplace_value(il, std::forward<Args>(args)...);
  }

  /// Ignores the result. This method signals intent to ignore the result
  /// to suppress compiler warnings from [[nodiscard]].
  void IgnoreResult() const {}

  /// Result observers.

  /// Returns true if this represents a success state, false for a failure
  /// state. value() is only iff has_value() is true. status() is valid iff
  /// has_value is false.
  constexpr bool has_value() const { return this->has_value_; }

  /// Returns true if *this represents a success state.
  explicit constexpr operator bool() const noexcept { return has_value(); }
  constexpr bool ok() const { return has_value(); }

  /// \brief Checked value accessor.
  /// Terminates the process if *this represents a failure state.
  /// \pre has_value() == true
  const_reference_type value() const& noexcept {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    if constexpr (!std::is_void_v<value_type>) {
      return this->value_;
    }
  }

  reference_type value() & noexcept {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    if constexpr (!std::is_void_v<value_type>) {
      return this->value_;
    }
  }

  value_type value() && noexcept {
    if (!has_value()) TENSORSTORE_CHECK_OK(status());
    if constexpr (!std::is_void_v<value_type>) {
      return std::move(this->value_);
    }
  }

  const Status& status() const& noexcept {
    TENSORSTORE_CHECK(!has_value());
    return this->status_;
  }

  Status status() && {
    TENSORSTORE_CHECK(!has_value());
    return std::move(this->status_);
  }

  /// operators
  /// \pre has_value() == true
  template <typename U = T>
  constexpr const U* operator->() const noexcept {
    assert(has_value());
    return &this->value_;
  }
  template <typename U = T>
  constexpr U* operator->() noexcept {
    assert(has_value());
    return &this->value_;
  }

  template <typename U = T>
  constexpr const U& operator*() const& noexcept {
    assert(has_value());
    return this->value_;
  }
  template <typename U = T>
  constexpr U& operator*() & noexcept {
    assert(has_value());
    return this->value_;
  }
  template <typename U = T>
  constexpr U&& operator*() && noexcept {
    assert(has_value());
    return std::move(this->value_);
  }

  /// "Pipeline" operator for `Result`.
  ///
  /// In the expression  `x | y`, if
  ///   * x is of type `Result<T>`
  ///   * y is a function having signature `U (T)` or `Result<U>(T)`
  ///
  /// Then operator| applies y to the value contained in x, returning a
  /// Result<U>. In other words, this function is roughly expressed as:
  ///
  ///    `return !x.ok() ? x.status() | StatusOr<U>(y(x.value()))`
  ///
  template <typename Func>
  inline FlatResult<std::invoke_result_t<Func&&, reference_type>>  //
  operator|(Func&& func) const& {
    if (!ok()) return status();
    return static_cast<Func&&>(func)(value());
  }
  template <typename Func>
  inline FlatResult<std::invoke_result_t<Func&&, value_type>>  //
  operator|(Func&& func) && {
    if (!ok()) return status();
    return static_cast<Func&&>(func)(value());
  }

  template <typename U>
  constexpr value_type value_or(U&& default_value) const& {
    return has_value() ? this->value_ : std::forward<U>(default_value);
  }

  template <typename U>
  constexpr value_type value_or(U&& default_value) && {
    return has_value() ? std::move(this->value_)
                       : std::forward<U>(default_value);
  }

  /// FIXME: Result::and_then()
  /// FIXME: Result::map()

  /// The `set_value`, `set_cancel`, `set_error`, and `submit` functions defined
  /// below make `Result<T>` model the `Receiver<Status, T>` and
  /// `Sender<Status, T>` concepts.
  ///
  /// These are defined as friend functions rather than member functions to
  /// allow `std::reference_wrapper<Result<T>>` and other types implicitly
  /// convertible to `Result<T>` to also model `Receiver<Status, T>` and
  /// `Sender<Status, T>`.

  /// Implements the Receiver `set_value` operation.
  ///
  /// This destroys any existing value/error and constructs the contained value
  /// from `v...`.
  template <typename... V>
  friend std::enable_if_t<((std::is_void<value_type>::value &&
                            sizeof...(V) == 0) ||
                           std::is_constructible<value_type, V&&...>::value)>
  set_value(Result& result, V&&... v) {
    result.emplace(std::forward<V>(v)...);
  }

  /// Implements the Receiver `set_error` operation.
  ///
  /// Overrides the existing value/error with `status`.
  friend void set_error(Result& result, Status status) {
    result = std::move(status);
  }

  /// Implements the Receiver `set_cancel` operation.
  ///
  /// This overrides the existing value/error with `absl::CancelledError("")`.
  friend void set_cancel(Result& result) { result = absl::CancelledError(""); }

  /// Implements the Sender `submit` operation.
  ///
  /// If `has_value() == true`, calls `set_value` with an lvalue reference to
  /// the contained value.
  ///
  /// If in an error state with an error code of `absl::StatusCode::kCancelled`,
  /// calls `set_cancel`.
  ///
  /// Otherwise, calls `set_error` with `status()`.
  template <typename Receiver>
  friend std::void_t<decltype(execution::set_value, std::declval<Receiver&>(),
                              std::declval<T>()),
                     decltype(execution::set_error, std::declval<Receiver&>(),
                              std::declval<Status>()),
                     decltype(execution::set_cancel, std::declval<Receiver&>())>
  submit(Result& result, Receiver&& receiver) {
    if (result.has_value()) {
      execution::set_value(receiver, *result);
    } else {
      if (result.status().code() == absl::StatusCode::kCancelled) {
        execution::set_cancel(receiver);
      } else {
        execution::set_error(receiver, result.status());
      }
    }
  }

 private:
  // Construct this from a Result<U>.
  template <typename Other>
  inline void construct_from(Other&& other) {
    if (other.has_value()) {
      this->construct_value(std::forward<Other>(other).value());
    } else {
      this->construct_status(std::forward<Other>(other).status());
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
};

/// \brief Checks if two Result values are equal.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator==(const Result<T>& a, const Result<U>& b) {
  if (a.has_value() != b.has_value()) {
    return false;
  }
  return a.has_value() ? a.value() == b.value() : a.status() == b.status();
}

inline bool operator==(const Result<void>& a, const Result<void>& b) {
  return (a.has_value() == b.has_value()) &&
         (a.has_value() || (a.status() == b.status()));
}

/// \brief Checks if two Result values are not equal.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator!=(const Result<T>& a, const Result<U>& b) {
  return !(a == b);
}

inline bool operator!=(const Result<void>& a, const Result<void>& b) {
  return !(a == b);
}

/// \brief Checks if a Result has a success value equal to a given value.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator==(const Result<T>& a, const U& b) {
  return a && a.value() == b;
}

/// \brief Checks if a Result does not have a success value equal to a given
/// value.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator!=(const Result<T>& a, const U& b) {
  return !(a == b);
}

/// \brief Checks if a Result has a success value equal to a given value.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator==(const U& b, const Result<T>& a) {
  return a && a.value() == b;
}

/// \brief Checks if a Result does not have a success value equal to a given
/// value.
template <typename T, typename U>
std::enable_if_t<
    std::is_same<bool, decltype(std::declval<T>() == std::declval<U>())>::value,
    bool>
operator!=(const U& b, const Result<T>& a) {
  return !(a == b);
}

/// Returns a Result<T> with a (possibly-default) value.
/// Example:
///    Result<void> r = MakeResult();
///    Result<int>  x = MakeResult<int>();
///    auto result = MakeResult(7);
inline Result<void> MakeResult() { return {std::in_place}; }

template <int&... ExplicitArgumentBarrier, typename T>
inline Result<typename tensorstore::internal::remove_cvref_t<T>> MakeResult(
    T&& t) {
  return {in_place, std::forward<T>(t)};
}

template <typename U, typename... Args>
inline Result<U> MakeResult(Args&&... args) {
  return {in_place, std::forward<Args>(args)...};
}

/// Returns a Result corresponding to a success or error `status`.
inline Result<void> MakeResult(Status status) {
  return status.ok() ? Result<void>(in_place) : Result<void>(std::move(status));
}

template <typename U>
inline Result<U> MakeResult(Status status) {
  return status.ok() ? Result<U>(in_place) : Result<U>(std::move(status));
}

/// FIXME: It would be nice if the naming convention for UnwrapResult
/// and GetStatus were the same. I think that I'd prefer UnwrapStatus()
/// to return a Status type() and UnwrapResult() to return a value type.

/// Returns the associated Status value, or the success value `Status()` if
/// there is no associated Status value.
///
/// FIXME: GetStatus should return a const Status&
///
/// This overload handles the case of a non-Status, non-Result type.
/// \returns `Status()`
template <typename T>
inline Status GetStatus(const T& other) {
  return absl::OkStatus();
}

/// Overload for the case of an argument that is an instance of `Result`.
/// \returns `result.status()`
template <typename T>
inline Status GetStatus(const Result<T>& result) {
  return result.has_value() ? Status() : result.status();
}

template <typename T>
inline Status GetStatus(Result<T>&& result) {
  return result.has_value() ? Status() : std::move(result).status();
}

/// UnwrapResult returns the value contained by the Result<T> instance,
/// `*t`, or the value, `t`.
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
///     the error Status of the first `Result`-wrapped `arg` in an error state.
template <typename Func, typename... T>
FlatResult<std::invoke_result_t<Func&&, UnwrapQualifiedResultType<T>...>>
MapResult(Func&& func, T&&... arg) {
  TENSORSTORE_RETURN_IF_ERROR(
      GetFirstErrorStatus(GetStatus(std::forward<T>(arg))...));
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

/// ChainResult() applies a sequence of functions, which may optionally
/// return `Result`-wrapped values, to an optionally `Result`-wrapped value.
///
/// For example:
///
///     float func1(int x);
///     Result<std::string> func2(float x);
///     bool func3(absl::string_view x);
///
///     Result<bool> y1 = ChainResult(Result<int>(3), func1, func2, func3);
///     Result<bool> y2 = ChainResult(3, func1, func2, func3);

/// This overload handles the base case of zero functions.
template <typename T>
internal_result::ChainResultType<T> ChainResult(T&& arg) {
  return std::forward<T>(arg);
}

/// Overload that handles the case of at least one function.
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
  static_assert(tensorstore::IsResult<decltype(temp)>::value,             \
                "TENSORSTORE_ASSIGN_OR_RETURN requires a Result value."); \
  if (ABSL_PREDICT_FALSE(!temp)) {                                        \
    auto _ = std::move(temp).status();                                    \
    static_cast<void>(_);                                                 \
    return (error_expr);                                                  \
  }                                                                       \
  decl = std::move(*temp);                                                \
  /**/

/// Convenience macro for propagating errors when calling a function that
/// returns a `Result`.
///
/// This macro generates multiple statements and should be invoked as follows:
///
///     Result<int> GetSomeResult();
///
///     TENSORSTORE_ASSIGN_OR_RETURN(int x, GetSomeResult());
///
/// An optional third argument specifies the return expression in the case of an
/// error.  A variable `_` bound to the error Status value is in scope within
/// this expression.  For example:
///
///     TENSORSTORE_ASSIGN_OR_RETURN(int x, GetSomeResult(),
///                                  _.Annotate("Context message"));
#define TENSORSTORE_ASSIGN_OR_RETURN(decl, ...)                          \
  TENSORSTORE_PP_EXPAND(TENSORSTORE_INTERNAL_ASSIGN_OR_RETURN_IMPL(      \
      TENSORSTORE_PP_CAT(tensorstore_assign_or_return_, __LINE__), decl, \
      __VA_ARGS__, _))                                                   \
  /**/
// Note: the use of `TENSORSTORE_PP_EXPAND` above is a workaround for MSVC 2019
// preprocessor limitations.

}  // namespace tensorstore

#endif  // TENSORSTORE_RESULT_H_
