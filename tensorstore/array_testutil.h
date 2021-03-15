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

#ifndef TENSORSTORE_ARRAY_TESTUTIL_H_
#define TENSORSTORE_ARRAY_TESTUTIL_H_

/// \file
/// Define a GMock matcher for `tensorstore::Array`.

#include <string>

#include <gmock/gmock.h>
#include "tensorstore/array.h"
#include "tensorstore/util/iterate_over_index_range.h"

namespace tensorstore {

namespace internal_array {
template <typename Element>
class ArrayMatcherImpl
    : public ::testing::MatcherInterface<OffsetArrayView<const void>> {
 public:
  ArrayMatcherImpl(
      SharedOffsetArray<const ::testing::Matcher<Element>> element_matchers)
      : element_matchers_(std::move(element_matchers)) {}

  bool MatchAndExplain(
      OffsetArrayView<const void> value_untyped,
      ::testing::MatchResultListener* listener) const override {
    const bool listener_interested = listener->IsInterested();
    if (value_untyped.dtype() != dtype_v<Element>) {
      if (listener_interested) {
        *listener << "which has a data type of " << value_untyped.dtype();
      }
      return false;
    }
    if (element_matchers_.domain() != value_untyped.domain()) {
      return false;
    }
    auto value = StaticDataTypeCast<const Element, unchecked>(value_untyped);
    SharedOffsetArray<std::string> explanations;
    if (listener_interested) {
      explanations = AllocateArray<std::string>(value.domain());
    }
    std::vector<Index> mismatch_indices;
    bool matches =
        IterateOverIndexRange(value.domain(), [&](span<const Index> indices) {
          const Element& element = value(indices);
          const auto& matcher = element_matchers_(indices);
          bool element_matches;
          if (listener_interested) {
            ::testing::StringMatchResultListener s;
            element_matches = matcher.MatchAndExplain(element, &s);
            explanations(indices) = s.str();
          } else {
            element_matches = matcher.Matches(element);
          }
          if (!element_matches) {
            mismatch_indices.assign(indices.begin(), indices.end());
          }
          return element_matches;
        });
    if (!matches) {
      if (listener_interested) {
        *listener << "whose element at " << span(mismatch_indices)
                  << " doesn't match";
        const auto& explanation = explanations(mismatch_indices);
        if (!explanation.empty()) {
          *listener << ", " << explanation;
        }
      }
      return false;
    }

    // Every element matches its expectation.
    if (listener_interested) {
      bool reason_printed = false;
      IterateOverIndexRange(value.domain(), [&](span<const Index> indices) {
        const std::string& explanation = explanations(indices);
        if (explanation.empty()) return;
        if (reason_printed) *listener << ",\nand ";
        *listener << "whose element at " << span(indices) << " matches, "
                  << explanation;
        reason_printed = true;
      });
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "has a data type of " << dtype_v<Element> << " and a domain of "
        << element_matchers_.domain();
    if (!element_matchers_.domain().is_empty()) {
      *os << " where\n";
      bool is_first = true;
      IterateOverIndexRange(element_matchers_.domain(),
                            [&](span<const Index> indices) {
                              if (!is_first) {
                                *os << ",\n";
                              }
                              is_first = false;
                              *os << "element at " << indices << " ";
                              element_matchers_(indices).DescribeTo(os);
                            });
    }
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "doesn't have a data type of "
        << dtype_v<Element> << ", or\ndoesn't have a domain of "
        << element_matchers_.domain();
    IterateOverIndexRange(element_matchers_.domain(),
                          [&](span<const Index> indices) {
                            *os << ", or\nelement at " << indices << " ";
                            element_matchers_(indices).DescribeNegationTo(os);
                          });
  }

 private:
  SharedOffsetArray<const ::testing::Matcher<Element>> element_matchers_;
};
}  // namespace internal_array

using ArrayMatcher = ::testing::Matcher<OffsetArrayView<const void>>;

/// Returns a GMock matcher that matches arrays with a domain of
/// `matcher_array.domain()` and where each element matches the corresponding
/// matcher in `matcher_array`.
template <typename Element>
ArrayMatcher MatchesArray(
    SharedOffsetArray<const ::testing::Matcher<Element>> matcher_array) {
  return ::testing::MakeMatcher(
      new internal_array::ArrayMatcherImpl<Element>(std::move(matcher_array)));
}

/// Returns a GMock matcher that matches a rank-0 array whose single element
/// matches `matcher`.
template <typename Element>
ArrayMatcher MatchesScalarArray(const ::testing::Matcher<Element>& matcher) {
  return MatchesArray<Element>(
      MakeScalarArray<::testing::Matcher<Element>>(matcher));
}

// [BEGIN GENERATED: generate_matches_array_overloads.py]

/// Returns a GMock matcher that matches a rank-1 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0>
ArrayMatcher MatchesArray(
    const ::testing::Matcher<Element> (&element_matchers)[N0]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-1 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0>
ArrayMatcher MatchesArray(
    span<const Index, 1> origin,
    const ::testing::Matcher<Element> (&element_matchers)[N0]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-1 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(
    const Index (&origin)[OriginRank],
    const ::testing::Matcher<Element> (&element_matchers)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

// Defines MatchesArray overloads for multi-dimensional arrays of rank 2 to 6.
#include "tensorstore/array_testutil_matches_array.inc"
// [END GENERATED: generate_matches_array_overloads.py]

}  // namespace tensorstore

#endif  // TENSORSTORE_ARRAY_TESTUTIL_H_
