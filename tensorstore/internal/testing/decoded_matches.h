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

#ifndef TENSORSTORE_INTERNAL_TESTING_DECODED_MATCHES_H_
#define TENSORSTORE_INTERNAL_TESTING_DECODED_MATCHES_H_

#include <functional>
#include <string>
#include <string_view>

#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Returns a GoogleMock `absl::Cord` matcher that matches if `decoder`
/// successfully decodes the input argument and `value_matcher` matches the
/// result.
///
/// \param value_matcher Matcher for the decoded value.
/// \param decoder Decodes the input argument.
///
/// Example usage:
///
///     EXPECT_THAT(some_blosc_compressed_string,
///                 DecodedMatches("expected value",
///                                &blosc::Decode));
::testing::Matcher<absl::Cord> DecodedMatches(
    ::testing::Matcher<std::string_view> value_matcher,
    std::function<Result<std::string>(std::string_view)> decoder);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TESTING_DECODED_MATCHES_H_
