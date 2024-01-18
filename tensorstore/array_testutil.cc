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

#include "tensorstore/array_testutil.h"

#include <ostream>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

namespace internal_array {

class ArrayMatcherImpl
    : public ::testing::MatcherInterface<OffsetArrayView<const void>> {
 public:
  ArrayMatcherImpl(SharedOffsetArray<const void> expected,
                   EqualityComparisonKind comparison_kind)
      : expected_(std::move(expected)), comparison_kind_(comparison_kind) {}

  bool MatchAndExplain(
      OffsetArrayView<const void> value,
      ::testing::MatchResultListener* listener) const override {
    const bool listener_interested = listener->IsInterested();
    if (value.dtype() != expected_.dtype()) {
      if (listener_interested) {
        *listener << "which has a data type of " << value.dtype();
      }
      return false;
    }
    if (expected_.domain() != value.domain()) {
      if (listener_interested) {
        *listener << "which has a domain of " << value.domain();
      }
      return false;
    }
    if (AreArraysEqual(expected_, value, comparison_kind_)) {
      return true;
    }
    if (!listener_interested) return false;
    bool reason_printed = false;
    IterateOverIndexRange(value.domain(), [&](span<const Index> indices) {
      if (!AreArraysEqual(value[indices], expected_[indices],
                          comparison_kind_)) {
        if (reason_printed) {
          *listener << ", ";
        }
        *listener << "whose element at " << indices
                  << " doesn't match, expected=" << expected_[indices]
                  << ", actual=" << value[indices];
        reason_printed = true;
      }
    });
    return false;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "has a data type of " << expected_.dtype() << " and a domain of "
        << expected_.domain() << " and is "
        << (comparison_kind_ == EqualityComparisonKind::equal ? "equal"
                                                              : "identical")
        << " to " << expected_;
  }

 private:
  SharedOffsetArray<const void> expected_;
  EqualityComparisonKind comparison_kind_;
};

}  // namespace internal_array

ArrayMatcher MatchesArray(SharedOffsetArray<const void> expected,
                          EqualityComparisonKind comparison_kind) {
  return ::testing::MakeMatcher(new internal_array::ArrayMatcherImpl(
      std::move(expected), comparison_kind));
}

}  // namespace tensorstore
