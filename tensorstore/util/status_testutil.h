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
/// Implements GMock matchers for absl::Status and Result.

#include <ostream>
#include <string>
#include <system_error>  // NOLINT

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_status {

class ErrorCodeMatchesStatusMatcher {
 public:
  ErrorCodeMatchesStatusMatcher(absl::StatusCode status_code,
                                const std::string& message_pattern);

  bool MatchAndExplain(const absl::Status& status,
                       ::testing::MatchResultListener* listener) const;

  template <typename T>
  bool MatchAndExplain(const Result<T>& result,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplain(GetStatus(result), listener);
  }

  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(::std::ostream* os) const;

 private:
  absl::StatusCode status_code_;
  std::string message_pattern_;
};
}  // namespace internal_status

/// Returns a
/// `GMock <https://google.github.io/googletest/reference/matchers.html>`__
/// matcher for a given status code.
///
/// The matcher can be applied to either `Result` or `absl::Status`.
///
/// \relates Result
/// \membergroup Test support
::testing::PolymorphicMatcher<internal_status::ErrorCodeMatchesStatusMatcher>
MatchesStatus(absl::StatusCode status_code,
              const std::string& message_pattern = "[^]*");

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
#define TENSORSTORE_EXPECT_OK(...)                               \
  EXPECT_EQ(absl::Status(), tensorstore::GetStatus(__VA_ARGS__)) \
  /**/

/// Same as `TENSORSTORE_EXPECT_OK`, but returns in the case of an error.
///
/// \relates tensorstore::Result
/// \membergroup Test support
#define TENSORSTORE_ASSERT_OK(...)                               \
  ASSERT_EQ(absl::Status(), tensorstore::GetStatus(__VA_ARGS__)) \
  /**/

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

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STATUS_TESTUTIL_H_
