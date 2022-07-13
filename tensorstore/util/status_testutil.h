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

#ifndef TENSORSTORE_UTIL_STATUS_TESTUTIL_H_
#define TENSORSTORE_UTIL_STATUS_TESTUTIL_H_

/// \file
/// Implements GMock matchers for absl::Status, Result, and Future.
/// For example, to test an Ok result, perhaps with a value, use:
///
///   EXPECT_THAT(DoSomething(), ::tensorstore::IsOk());
///   EXPECT_THAT(DoSomething(), ::tensorstore::IsOkAndHolds(7));
///
/// A shorthand for IsOk() is provided:
///
///   TENSORSTORE_EXPECT_OK(DoSomething());
///
/// To test an error expectation, use:
///
///   EXPECT_THAT(DoSomething(),
///               ::tensorstore::StatusIs(absl::StatusCode::kInternal));
///   EXPECT_THAT(DoSomething(),
///               ::tensorstore::StatusIs(absl::StatusCode::kInternal,
///                                       ::testing::HasSubstr("foo")));
///
///   EXPECT_THAT(DoSomething(),
///               ::tensorstore::MatchesStatus(absl::StatusCode::kInternal,
///               "foo"));
///

#include <ostream>
#include <string>
#include <system_error>  // NOLINT

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

// Prints a debugging representation of a result.
template <typename T>
void PrintTo(const Result<T>& result, std::ostream* os) {
  if (result) {
    if constexpr (std::is_void_v<T>) {
      *os << "Result{}";
    } else {
      *os << "Result{" << ::testing::PrintToString(*result) << "}";
    }
  } else {
    *os << result.status();
  }
}

// Prints a debugging representation of a future.
template <typename T>
void PrintTo(const Future<T>& future, std::ostream* os) {
  if (!future.ready()) {
    *os << "Future{<not ready>}";
    return;
  }
  if (future.status().ok()) {
    if constexpr (std::is_void_v<T>) {
      *os << "Future{}";
    } else {
      *os << "Future{" << ::testing::PrintToString(future.value()) << "}";
    }
  } else {
    *os << future.status();
  }
}

namespace internal_status {

// Monomorphic implementation of matcher IsOkAndHolds(m).
// StatusType is a const reference to Status, Result<T>, or Future<T>.
template <typename StatusType>
class IsOkAndHoldsMatcherImpl : public ::testing::MatcherInterface<StatusType> {
 public:
  typedef
      typename std::remove_reference<StatusType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }
  bool MatchAndExplain(
      StatusType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    auto status = ::tensorstore::GetStatus(actual_value);  // avoid ADL.
    if (!status.ok()) {
      *result_listener << "whose status code is "
                       << absl::StatusCodeToString(status.code());
      return false;
    }
    ::testing::StringMatchResultListener inner_listener;
    if (!inner_matcher_.MatchAndExplain(actual_value.value(),
                                        &inner_listener)) {
      *result_listener << "whose value "
                       << ::testing::PrintToString(actual_value.value())
                       << " doesn't match";
      if (!inner_listener.str().empty()) {
        *result_listener << ", " << inner_listener.str();
      }
      return false;
    }
    return true;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.
  template <typename StatusType>
  operator ::testing::Matcher<StatusType>() const {  // NOLINT
    return ::testing::Matcher<StatusType>(
        new IsOkAndHoldsMatcherImpl<const StatusType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// StatusType is a const reference to Status, Result<T>, or Future<T>.
template <typename StatusType>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<StatusType> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(
      StatusType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    return ::tensorstore::GetStatus(actual_value).ok();  // avoid ADL.
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename StatusType>
  operator ::testing::Matcher<StatusType>() const {  // NOLINT
    return ::testing::Matcher<StatusType>(
        new MonoIsOkMatcherImpl<const StatusType&>());
  }
};

// Monomorphic implementation of matcher StatusIs().
// StatusType is a reference to Status, Result<T>, or Future<T>.
template <typename StatusType>
class StatusIsMatcherImpl : public ::testing::MatcherInterface<StatusType> {
 public:
  explicit StatusIsMatcherImpl(
      testing::Matcher<absl::StatusCode> code_matcher,
      testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "has a status code that ";
    code_matcher_.DescribeTo(os);
    *os << ", and has an error message that ";
    message_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "has a status code that ";
    code_matcher_.DescribeNegationTo(os);
    *os << ", or has an error message that ";
    message_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    auto status = ::tensorstore::GetStatus(actual_value);  // avoid ADL.

    testing::StringMatchResultListener inner_listener;
    if (!code_matcher_.MatchAndExplain(status.code(), &inner_listener)) {
      *result_listener << "whose status code "
                       << absl::StatusCodeToString(status.code())
                       << " doesn't match";
      const std::string inner_explanation = inner_listener.str();
      if (!inner_explanation.empty()) {
        *result_listener << ", " << inner_explanation;
      }
      return false;
    }

    if (!message_matcher_.Matches(std::string(status.message()))) {
      *result_listener << "whose error message is wrong";
      return false;
    }

    return true;
  }

 private:
  const testing::Matcher<absl::StatusCode> code_matcher_;
  const testing::Matcher<const std::string&> message_matcher_;
};

// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  StatusIsMatcher(testing::Matcher<absl::StatusCode> code_matcher,
                  testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.
  template <typename StatusType>
  operator ::testing::Matcher<StatusType>() const {  // NOLINT
    return ::testing::Matcher<StatusType>(
        new StatusIsMatcherImpl<const StatusType&>(code_matcher_,
                                                   message_matcher_));
  }

 private:
  const testing::Matcher<absl::StatusCode> code_matcher_;
  const testing::Matcher<const std::string&> message_matcher_;
};

}  // namespace internal_status

// Returns a gMock matcher that matches an OK Status/Result/Future.
inline internal_status::IsOkMatcher IsOk() {
  return internal_status::IsOkMatcher();
}

// Returns a gMock matcher that matches an OK Status/Result/Future
// and whose value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a matcher that matches a Status/Result/Future whose code
// matches code_matcher, and whose error message matches message_matcher.
template <typename CodeMatcher, typename MessageMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher,
                                          MessageMatcher message_matcher) {
  return internal_status::StatusIsMatcher(std::move(code_matcher),
                                          std::move(message_matcher));
}

// Returns a matcher that matches a Status/Result/Future whose code
// matches code_matcher.
template <typename CodeMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher) {
  return internal_status::StatusIsMatcher(std::move(code_matcher),
                                          ::testing::_);
}

// Returns a matcher that matches a Status/Result/Future whose code
// is code_matcher.
inline internal_status::StatusIsMatcher MatchesStatus(
    absl::StatusCode status_code) {
  return internal_status::StatusIsMatcher(status_code, ::testing::_);
}

// Returns a matcher that matches a Status/Result/Future whose code
// is code_matcher, and whose message matches the provided regex pattern.
internal_status::StatusIsMatcher MatchesStatus(
    absl::StatusCode status_code, const std::string& message_pattern);

}  // namespace tensorstore

/// EXPECT assertion that the argument, when converted to an `absl::Status` via
/// `tensorstore::GetStatus`, has a code of `absl::StatusCode::kOk`.
///
/// Example::
///
///     absl::Status DoSomething();
///     tensorstore::Result<int> GetResult();
///
///     TENSORSTORE_EXPECT_OK(DoSomething());
///     TENSORSTORE_EXPECT_OK(GetResult());
///
/// \relates tensorstore::Result
/// \membergroup Test support
#define TENSORSTORE_EXPECT_OK(expr) EXPECT_THAT(expr, ::tensorstore::IsOk())

/// Same as `TENSORSTORE_EXPECT_OK`, but returns in the case of an error.
///
/// \relates tensorstore::Result
/// \membergroup Test support
#define TENSORSTORE_ASSERT_OK(expr) ASSERT_THAT(expr, ::tensorstore::IsOk())

/// ASSERTs that `expr` is a `tensorstore::Result` with a value, and assigns the
/// value to `decl`.
///
/// Example:
///
///     tensorstore::Result<int> GetResult();
///
///     TENSORSTORE_ASSERT_OK_AND_ASSIGN(int x, GetResult());
///
/// \relates tensorstore::Result
/// \membergroup Test support
#define TENSORSTORE_ASSERT_OK_AND_ASSIGN(decl, expr)                      \
  TENSORSTORE_ASSIGN_OR_RETURN(decl, expr,                                \
                               ([&] { FAIL() << #expr << ": " << _; })()) \
  /**/

#endif  // TENSORSTORE_UTIL_STATUS_TESTUTIL_H_
