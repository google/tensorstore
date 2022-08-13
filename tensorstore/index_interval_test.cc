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

#include "tensorstore/index_interval.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::AreCompatibleOrUnbounded;
using ::tensorstore::ComputeStridedSliceMap;
using ::tensorstore::container;
using ::tensorstore::DividePositiveRoundOut;
using ::tensorstore::ExplicitIndexOr;
using ::tensorstore::ExtractClosedStridedSlice;
using ::tensorstore::ExtractHalfOpenStridedSlice;
using ::tensorstore::ExtractSizedStridedSlice;
using ::tensorstore::GetAffineTransformInverseDomain;
using ::tensorstore::ImplicitOrEqual;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainDimension;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexIntervalRef;
using ::tensorstore::Intersect;
using ::tensorstore::IntervalForm;
using ::tensorstore::kImplicit;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::kMinFiniteIndex;
using ::tensorstore::MatchesStatus;
using ::tensorstore::MergeDimensionLabels;
using ::tensorstore::MergeOptionallyImplicitIndexIntervals;
using ::tensorstore::OptionallyImplicitIndexInterval;
using ::tensorstore::ShiftInterval;
using ::tensorstore::ShiftIntervalBackward;
using ::tensorstore::ShiftIntervalTo;
using ::tensorstore::StrCat;
using ::tensorstore::view;
using ::tensorstore::serialization::TestSerializationRoundTrip;
using ::testing::Optional;
using ::testing::Pair;

TEST(IndexIntervalTest, DefaultConstruct) {
  IndexInterval x;
  EXPECT_EQ(-kInfIndex, x.inclusive_min());
  EXPECT_EQ(-kInfIndex - 1, x.exclusive_min());
  EXPECT_EQ(kInfIndex, x.inclusive_max());
  EXPECT_EQ(kInfIndex + 1, x.exclusive_max());
  EXPECT_EQ(kInfSize, x.size());
  EXPECT_FALSE(x.empty());
}

TEST(IndexIntervalTest, Empty) {
  EXPECT_TRUE(IndexInterval::UncheckedSized(1, 0).empty());
}

TEST(IndexIntervalTest, ValidSized) {
  EXPECT_TRUE(IndexInterval::ValidSized(0, 0));
  EXPECT_TRUE(IndexInterval::ValidSized(-kInfIndex, kInfSize));
  EXPECT_TRUE(IndexInterval::ValidSized(-kInfIndex, 100));
  EXPECT_TRUE(IndexInterval::ValidSized(kInfIndex - 5, 6));
  EXPECT_TRUE(IndexInterval::ValidSized(-kInfIndex, 2));
  EXPECT_FALSE(IndexInterval::ValidSized(-kInfIndex - 1, 0));
  EXPECT_FALSE(IndexInterval::ValidSized(5, -1));
  EXPECT_FALSE(IndexInterval::ValidSized(kInfIndex - 5, 7));
  EXPECT_FALSE(IndexInterval::ValidSized(-kInfIndex, 0));
  EXPECT_FALSE(IndexInterval::ValidSized(-kInfIndex, 1));
  EXPECT_FALSE(IndexInterval::ValidSized(kInfIndex, 1));
  EXPECT_FALSE(IndexInterval::ValidSized(kInfIndex, 0));
}

TEST(IndexIntervalTest, ValidClosed) {
  EXPECT_TRUE(IndexInterval::ValidClosed(0, 0));
  EXPECT_TRUE(IndexInterval::ValidClosed(0, -1));
  EXPECT_TRUE(IndexInterval::ValidClosed(-kInfIndex, kInfIndex));
  EXPECT_TRUE(IndexInterval::ValidClosed(-5, kInfIndex));
  EXPECT_TRUE(IndexInterval::ValidClosed(-kInfIndex, -kInfIndex + 1));
  EXPECT_FALSE(IndexInterval::ValidClosed(0, -2));
  EXPECT_FALSE(IndexInterval::ValidClosed(-kInfIndex - 1, 0));
  EXPECT_FALSE(IndexInterval::ValidClosed(0, kInfIndex + 1));
  EXPECT_FALSE(IndexInterval::ValidClosed(-kInfIndex, -kInfIndex));
  EXPECT_FALSE(IndexInterval::ValidClosed(+kInfIndex, +kInfIndex));
}

TEST(IndexIntervalTest, ValidHalfOpen) {
  EXPECT_TRUE(IndexInterval::ValidHalfOpen(0, 0));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(0, -1));
  EXPECT_TRUE(IndexInterval::ValidHalfOpen(-kInfIndex, kInfIndex + 1));
  EXPECT_TRUE(IndexInterval::ValidHalfOpen(-5, kInfIndex + 1));
  EXPECT_TRUE(IndexInterval::ValidHalfOpen(-kInfIndex, -kInfIndex + 2));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(-kInfIndex - 1, 0));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(0, kInfIndex + 2));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(-kInfIndex, -kInfIndex));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(-kInfIndex, -kInfIndex + 1));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(kInfIndex, kInfIndex));
  EXPECT_FALSE(IndexInterval::ValidHalfOpen(kInfIndex, kInfIndex + 1));
}

TEST(IndexIntervalTest, Sized) {
  EXPECT_EQ(IndexInterval::UncheckedSized(0, 5), IndexInterval::Sized(0, 5));
  EXPECT_THAT(IndexInterval::Sized(0, -1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexIntervalTest, UncheckedSized) {
  auto x = IndexInterval::UncheckedSized(1, 5);
  EXPECT_EQ(1, x.inclusive_min());
  EXPECT_EQ(0, x.exclusive_min());
  EXPECT_EQ(5, x.size());
  EXPECT_EQ(5, x.inclusive_max());
  EXPECT_EQ(6, x.exclusive_max());
}

TEST(IndexIntervalTest, Equality) {
  EXPECT_TRUE(IndexInterval::UncheckedSized(1, 2) ==
              IndexInterval::UncheckedSized(1, 2));
  EXPECT_FALSE(IndexInterval::UncheckedSized(1, 2) !=
               IndexInterval::UncheckedSized(1, 2));

  EXPECT_FALSE(IndexInterval::UncheckedSized(1, 3) ==
               IndexInterval::UncheckedSized(1, 2));
  EXPECT_FALSE(IndexInterval::UncheckedSized(2, 2) ==
               IndexInterval::UncheckedSized(1, 2));
  EXPECT_TRUE(IndexInterval::UncheckedSized(2, 3) ==
              IndexInterval::UncheckedClosed(2, 4));
  EXPECT_TRUE(IndexInterval::UncheckedSized(2, 3) ==
              IndexInterval::UncheckedHalfOpen(2, 5));
}

TEST(IndexIntervalTest, UncheckedClosed) {
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 3),
            IndexInterval::UncheckedClosed(2, 4));
}

TEST(IndexIntervalTest, Closed) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(2, 4), IndexInterval::Closed(2, 4));
  EXPECT_THAT(IndexInterval::Closed(2, 0),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexIntervalTest, UncheckedHalfOpen) {
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 2),
            IndexInterval::UncheckedHalfOpen(2, 4));
}

TEST(IndexIntervalTest, HalfOpen) {
  EXPECT_EQ(IndexInterval::UncheckedHalfOpen(2, 4),
            IndexInterval::HalfOpen(2, 4));
  EXPECT_THAT(IndexInterval::HalfOpen(2, 0),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexIntervalTest, ContainsIndex) {
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15), 5));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15), 3));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15), 15));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(3, 15), 2));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(3, 15), 16));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(-kInfIndex, 15),
                       kMinFiniteIndex));
  EXPECT_FALSE(
      Contains(IndexInterval::UncheckedClosed(-kInfIndex, 15), -kInfIndex));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(-kInfIndex, 15), 16));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, kInfIndex), 16));
  EXPECT_TRUE(
      Contains(IndexInterval::UncheckedClosed(3, kInfIndex), kMaxFiniteIndex));
  EXPECT_FALSE(
      Contains(IndexInterval::UncheckedClosed(3, kInfIndex), kInfIndex));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex),
                        -kInfIndex));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex),
                        kInfIndex));
  EXPECT_TRUE(
      Contains(IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex), 3));
}

TEST(IndexIntervalTest, ContainsInterval) {
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15),
                       IndexInterval::UncheckedClosed(3, 15)));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15),
                       IndexInterval::UncheckedClosed(4, 15)));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15),
                       IndexInterval::UncheckedClosed(3, 14)));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15),
                       IndexInterval::UncheckedClosed(6, 8)));
  EXPECT_TRUE(Contains(IndexInterval::UncheckedClosed(3, 15),
                       IndexInterval::UncheckedSized(20, 0)));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(3, 15),
                        IndexInterval::UncheckedClosed(2, 10)));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(3, 15),
                        IndexInterval::UncheckedClosed(3, 16)));
  EXPECT_FALSE(Contains(IndexInterval::UncheckedClosed(3, 15),
                        IndexInterval::UncheckedClosed(5, 16)));
}

TEST(IndexIntervalTest, IsFinite) {
  EXPECT_TRUE(IsFinite(IndexInterval::UncheckedClosed(3, 15)));
  EXPECT_FALSE(IsFinite(IndexInterval::UncheckedClosed(-kInfIndex, 15)));
  EXPECT_FALSE(IsFinite(IndexInterval::UncheckedClosed(3, kInfIndex)));
  EXPECT_FALSE(IsFinite(IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex)));
}

TEST(IndexIntervalTest, Intersect) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(3, 5),
            Intersect(IndexInterval::UncheckedClosed(-3, 5),
                      IndexInterval::UncheckedClosed(3, 10)));

  EXPECT_EQ(IndexInterval::UncheckedClosed(3, 5),
            Intersect(IndexInterval::UncheckedClosed(3, 10),
                      IndexInterval::UncheckedClosed(-3, 5)));

  EXPECT_EQ(IndexInterval::UncheckedClosed(3, 10),
            Intersect(IndexInterval::UncheckedClosed(3, 10),
                      IndexInterval::UncheckedClosed(-3, 11)));

  EXPECT_EQ(IndexInterval::UncheckedSized(3, 0),
            Intersect(IndexInterval::UncheckedClosed(-3, 0),
                      IndexInterval::UncheckedClosed(3, 5)));
}

TEST(IndexIntervalTest, IntersectOptionallyImplicit) {
  using OIII = OptionallyImplicitIndexInterval;

  // Values
  EXPECT_THAT(
      Intersect(OIII{IndexInterval::UncheckedClosed(1, 5), false, false},
                OIII{IndexInterval::UncheckedClosed(2, 6), false, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(2, 5), false, false}));

  EXPECT_THAT(
      Intersect(OIII{IndexInterval::UncheckedClosed(2, 5), false, true},
                OIII{IndexInterval::UncheckedClosed(1, 6), true, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(2, 5), false, true}));

  // implicit/explicit does not change the result.
  for (int x = 0; x < 16; x++) {
    const bool a = ((x & 1) != 0);
    const bool b = ((x & 2) != 0);
    const bool c = ((x & 4) != 0);
    const bool d = ((x & 8) != 0);

    EXPECT_THAT(Intersect(OIII{IndexInterval::UncheckedClosed(1, 5), a, b},
                          OIII{IndexInterval::UncheckedClosed(1, 5), c, d}),
                ::testing::Eq(
                    OIII{IndexInterval::UncheckedClosed(1, 5), a && c, b && d}))
        << x;

    EXPECT_THAT(Intersect(OIII{IndexInterval::UncheckedClosed(-3, 5), a, b},
                          OIII{IndexInterval::UncheckedClosed(3, 10), c, d}),
                ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), c, b}))
        << x;
  }

  // Edge cases
  EXPECT_THAT(
      Intersect(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               true, true},
          OIII{IndexInterval::UncheckedClosed(0, 10), false, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(0, 10), false, false}));

  EXPECT_THAT(
      Intersect(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               true, true},
          OIII{IndexInterval::UncheckedClosed(kMinFiniteIndex, kInfIndex),
               false, false}),
      ::testing::Eq(
          OIII{IndexInterval::UncheckedClosed(kMinFiniteIndex, kMaxFiniteIndex),
               false, true}));

  EXPECT_THAT(
      Intersect(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               false, false},
          OIII{IndexInterval::UncheckedClosed(0, 10), true, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(0, 10), true, true}));
}

TEST(IndexIntervalTest, IntersectPreferringExplicit) {
  using OIII = OptionallyImplicitIndexInterval;

  // Equal but for implicit flags.
  for (int x = 0; x < 16; x++) {
    const bool a = ((x & 1) != 0);
    const bool b = ((x & 2) != 0);
    const bool c = ((x & 4) != 0);
    const bool d = ((x & 8) != 0);

    EXPECT_THAT(Intersect(OIII{IndexInterval::UncheckedClosed(1, 5), a, b},
                          OIII{IndexInterval::UncheckedClosed(1, 5), c, d}),
                ::testing::Eq(
                    OIII{IndexInterval::UncheckedClosed(1, 5), a && c, b && d}))
        << x;

    EXPECT_THAT(Intersect(OIII{IndexInterval::UncheckedClosed(-3, 5), a, b},
                          OIII{IndexInterval::UncheckedClosed(3, 10), a, b}),
                ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), a, b}))
        << x;
  }

  // Values
  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(1, 5), false, false},
          OIII{IndexInterval::UncheckedClosed(2, 6), false, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(2, 5), false, false}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(2, 5), false, true},
          OIII{IndexInterval::UncheckedClosed(1, 6), true, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(2, 5), false, true}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-3, 5), true, false},
          OIII{IndexInterval::UncheckedClosed(3, 10), true, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), true, false}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-3, 5), false, false},
          OIII{IndexInterval::UncheckedClosed(3, 10), false, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), false, false}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-3, 5), false, true},
          OIII{IndexInterval::UncheckedClosed(3, 10), false, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), false, true}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-3, 5), true, true},
          OIII{IndexInterval::UncheckedClosed(3, 10), true, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(3, 5), true, true}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-3, 5), true, false},
          OIII{IndexInterval::UncheckedClosed(-5, 10), false, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(-5, 5), false, false}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-5, 10), false, false},
          OIII{IndexInterval::UncheckedClosed(-3, 5), true, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(-5, 5), false, false}));

  EXPECT_THAT(IntersectPreferringExplicit(
                  OIII{IndexInterval::UncheckedClosed(-3, 12), true, false},
                  OIII{IndexInterval::UncheckedClosed(-5, 10), false, true}),
              ::testing::Eq(
                  OIII{IndexInterval::UncheckedClosed(-5, 12), false, false}));

  EXPECT_THAT(IntersectPreferringExplicit(
                  OIII{IndexInterval::UncheckedClosed(-5, 10), false, true},
                  OIII{IndexInterval::UncheckedClosed(-3, 12), true, false}),
              ::testing::Eq(
                  OIII{IndexInterval::UncheckedClosed(-5, 12), false, false}));

  // Edge cases
  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               true, true},
          OIII{IndexInterval::UncheckedClosed(0, 10), false, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(0, 10), false, false}));

  // These may surprise you!  explicit takes prededence over implicit!
  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               true, true},
          OIII{IndexInterval::UncheckedClosed(kMinFiniteIndex, kInfIndex),
               false, false}),
      ::testing::Eq(
          OIII{IndexInterval::UncheckedClosed(kMinFiniteIndex, kInfIndex),
               false, false}));

  EXPECT_THAT(
      IntersectPreferringExplicit(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               false, false},
          OIII{IndexInterval::UncheckedClosed(0, 10), true, true}),
      ::testing::Eq(
          OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
               false, false}));
}

TEST(IndexIntervalTest, Hull) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(3, 15),
            Hull(IndexInterval::UncheckedClosed(3, 5),
                 IndexInterval::UncheckedClosed(10, 15)));
  EXPECT_EQ(IndexInterval::UncheckedClosed(5, 15),
            Hull(IndexInterval::UncheckedClosed(0, -1),
                 IndexInterval::UncheckedClosed(5, 15)));
  EXPECT_EQ(IndexInterval::UncheckedClosed(5, 15),
            Hull(IndexInterval::UncheckedClosed(5, 15),
                 IndexInterval::UncheckedClosed(0, -1)));
  EXPECT_EQ(IndexInterval::UncheckedClosed(0, -1),
            Hull(IndexInterval::UncheckedClosed(5, 4),
                 IndexInterval::UncheckedClosed(0, -1)));
}

TEST(IndexIntervalTest, HullOptionallyImplicit) {
  using OIII = OptionallyImplicitIndexInterval;

  EXPECT_THAT(
      Hull(OIII{IndexInterval::UncheckedClosed(1, 5), false, true},
           OIII{IndexInterval::UncheckedClosed(2, 6), false, true}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(1, 6), false, true}));

  // implicit/explicit does not change the result.
  for (int x = 0; x < 16; x++) {
    const bool a = ((x & 1) != 0);
    const bool b = ((x & 2) != 0);
    const bool c = ((x & 4) != 0);
    const bool d = ((x & 8) != 0);

    EXPECT_THAT(Hull(OIII{IndexInterval::UncheckedClosed(1, 5), a, b},
                     OIII{IndexInterval::UncheckedClosed(1, 5), c, d}),
                ::testing::Eq(
                    OIII{IndexInterval::UncheckedClosed(1, 5), a && c, b && d}))
        << x;

    EXPECT_THAT(
        Hull(OIII{IndexInterval::UncheckedClosed(-3, 5), a, b},
             OIII{IndexInterval::UncheckedClosed(3, 10), c, d}),
        ::testing::Eq(OIII{IndexInterval::UncheckedClosed(-3, 10), a, d}))
        << x;
  }

  // Edge cases
  EXPECT_THAT(
      Hull(OIII{IndexInterval::UncheckedClosed(-kInfIndex, kMaxFiniteIndex),
                true, true},
           OIII{IndexInterval::UncheckedClosed(kMinFiniteIndex, kInfIndex),
                false, false}),
      ::testing::Eq(OIII{IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex),
                         true, false}));
}

TEST(IndexIntervalTest, ContainsOrUnbounded) {
  EXPECT_TRUE(
      ContainsOrUnbounded(IndexInterval::UncheckedClosed(5, 10),
                          IndexInterval::UncheckedClosed(-kInfIndex, 10)));
  EXPECT_TRUE(ContainsOrUnbounded(IndexInterval::UncheckedClosed(5, 10),
                                  IndexInterval::UncheckedClosed(6, 9)));
  EXPECT_FALSE(ContainsOrUnbounded(IndexInterval::UncheckedClosed(5, 10),
                                   IndexInterval::UncheckedClosed(4, 10)));
  EXPECT_TRUE(
      ContainsOrUnbounded(IndexInterval::UncheckedClosed(5, 10),
                          IndexInterval::UncheckedClosed(5, kInfIndex)));
  EXPECT_FALSE(ContainsOrUnbounded(IndexInterval::UncheckedClosed(5, 10),
                                   IndexInterval::UncheckedClosed(5, 11)));
  EXPECT_TRUE(ContainsOrUnbounded(
      IndexInterval::UncheckedClosed(5, 10),
      IndexInterval::UncheckedClosed(-kInfIndex, +kInfIndex)));
}

TEST(IndexIntervalTest, AreCompatibleOrUnbounded) {
  EXPECT_TRUE(AreCompatibleOrUnbounded(IndexInterval(), IndexInterval()));
  EXPECT_TRUE(AreCompatibleOrUnbounded(IndexInterval(),
                                       IndexInterval::UncheckedSized(1, 4)));
  EXPECT_TRUE(AreCompatibleOrUnbounded(IndexInterval::UncheckedSized(1, 4),
                                       IndexInterval()));
  EXPECT_FALSE(AreCompatibleOrUnbounded(IndexInterval::UncheckedSized(1, 4),
                                        IndexInterval::UncheckedSized(1, 5)));
  EXPECT_FALSE(AreCompatibleOrUnbounded(IndexInterval::UncheckedSized(1, 4),
                                        IndexInterval::UncheckedSized(2, 3)));
  EXPECT_TRUE(
      AreCompatibleOrUnbounded(IndexInterval::UncheckedClosed(1, 4),
                               IndexInterval::UncheckedClosed(-kInfIndex, 4)));
  EXPECT_TRUE(
      AreCompatibleOrUnbounded(IndexInterval::UncheckedClosed(1, 4),
                               IndexInterval::UncheckedClosed(1, kInfIndex)));
}

TEST(IndexIntervalTest, Ostream) {
  EXPECT_EQ("[1, 3)", StrCat(IndexInterval::UncheckedClosed(1, 2)));
  EXPECT_EQ("(-inf, 3)", StrCat(IndexInterval::UncheckedClosed(-kInfIndex, 2)));
  EXPECT_EQ("[7, +inf)", StrCat(IndexInterval::UncheckedClosed(7, kInfIndex)));
}

TEST(IndexIntervalTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      IndexInterval(),
      IndexInterval::UncheckedSized(0, 1),
      IndexInterval::UncheckedSized(0, 0),
      IndexInterval::UncheckedSized(0, 2),
      IndexInterval::UncheckedSized(1, 2),
  }));
}

TEST(IndexIntervalTest, ShiftInterval) {
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(1, 8), 2),
              Optional(IndexInterval::UncheckedClosed(3, 10)));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(-kInfIndex, 8), 2),
              Optional(IndexInterval::UncheckedClosed(-kInfIndex, 10)));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(1, kInfIndex), 2),
              Optional(IndexInterval::UncheckedClosed(3, kInfIndex)));
  EXPECT_THAT(ShiftInterval(
                  IndexInterval::UncheckedClosed(kMinFiniteIndex + 1, 101), -1),
              Optional(IndexInterval::Closed(kMinFiniteIndex, 100)));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(5, 10), -kInfIndex),
              Optional(IndexInterval::UncheckedClosed(-kInfIndex + 5,
                                                      -kInfIndex + 10)));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(5, 10), kInfIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "5 \\+ [0-9]+ is outside valid range .*"));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(5, 10), kMaxFiniteIndex),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "5 \\+ [0-9]+ is outside valid range .*"));

  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(-1, 10), kMinFiniteIndex),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "-1 \\+ -[0-9]+ is outside valid range .*"));

  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(-kInfIndex, -5),
                            kMinFiniteIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "-5 \\+ -[0-9]+ is outside valid range .*"));
}

TEST(IndexIntervalTest, ShiftIntervalBackward) {
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval(), std::numeric_limits<Index>::min()),
      Optional(IndexInterval()));
  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(1, 8), -2),
              Optional(IndexInterval::UncheckedClosed(3, 10)));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(-kInfIndex, 8), -2),
      Optional(IndexInterval::UncheckedClosed(-kInfIndex, 10)));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(1, kInfIndex), -2),
      Optional(IndexInterval::UncheckedClosed(3, kInfIndex)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(kMinFiniteIndex + 1, 101), 1),
              Optional(IndexInterval::Closed(kMinFiniteIndex, 100)));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(5, 10), kInfIndex),
      Optional(
          IndexInterval::UncheckedClosed(-kInfIndex + 5, -kInfIndex + 10)));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(5, 10), -kInfIndex),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "5 \\+ [0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(5, 10),
                                    kMinFiniteIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "5 \\+ [0-9]+ is outside valid range .*"));

  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(-1, 10),
                                    kMaxFiniteIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "-1 \\+ -[0-9]+ is outside valid range .*"));

  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(-kInfIndex, -5),
                            kMaxFiniteIndex),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "-5 \\+ -[0-9]+ is outside valid range .*"));
}

TEST(IndexIntervalTest, ShiftIntervalSeparateOffsets) {
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(1, 8), 2, 5),
              Optional(IndexInterval::UncheckedClosed(3, 13)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8), 0, 5),
      Optional(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 13)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8), 1, 5),
      Optional(IndexInterval::UncheckedClosed(-kMaxFiniteIndex + 1, 13)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8), -1, 5),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "-[0-9]+ \\+ -1 is outside valid range .*"));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(-1, 8),
                            std::numeric_limits<Index>::min(), 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "-1 \\+ -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(2, kMaxFiniteIndex), -1, 0),
      Optional(IndexInterval::UncheckedClosed(1, kMaxFiniteIndex)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(2, kMaxFiniteIndex), -1, 1),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "[0-9]+ \\+ 1 is outside valid range .*"));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(2, 1), -1,
                            std::numeric_limits<Index>::max()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "1 \\+ [0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(0, 8),
                            std::numeric_limits<Index>::min(), 5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "0 \\+ -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftInterval(IndexInterval::UncheckedClosed(1, 8), 2, 5),
              Optional(IndexInterval::UncheckedClosed(3, 13)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(-kInfIndex, 8), 2, 5),
      Optional(IndexInterval::UncheckedClosed(-kInfIndex, 13)));
  EXPECT_THAT(
      ShiftInterval(IndexInterval::UncheckedClosed(1, +kInfIndex), 2, 5),
      Optional(IndexInterval::UncheckedClosed(3, +kInfIndex)));
  EXPECT_THAT(ShiftInterval(
                  IndexInterval::UncheckedClosed(-kInfIndex, +kInfIndex), 2, 5),
              Optional(IndexInterval::UncheckedClosed(-kInfIndex, +kInfIndex)));
}

TEST(IndexIntervalTest, ShiftIntervalBackwardSeparateOffsets) {
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(1, 8), -2, -5),
      Optional(IndexInterval::UncheckedClosed(3, 13)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8), 0, -5),
              Optional(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 13)));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8),
                            -1, -5),
      Optional(IndexInterval::UncheckedClosed(-kMaxFiniteIndex + 1, 13)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(-kMaxFiniteIndex, 8), 1, -5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "-[0-9]+ \\+ -1 is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(-1, 8),
                                    std::numeric_limits<Index>::max(), -5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "-1 \\+ -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(2, kMaxFiniteIndex), 1, 0),
              Optional(IndexInterval::UncheckedClosed(1, kMaxFiniteIndex)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(2, kMaxFiniteIndex), 1, -1),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "[0-9]+ \\+ 1 is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(2, 1), 1,
                                    std::numeric_limits<Index>::min()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "1 \\+ -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalBackward(IndexInterval::UncheckedClosed(0, 8),
                                    std::numeric_limits<Index>::max(), -5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "0 \\+ -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(
      ShiftIntervalBackward(IndexInterval::UncheckedClosed(1, 8), -2, -5),
      Optional(IndexInterval::UncheckedClosed(3, 13)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(-kInfIndex, 8), -2, -5),
              Optional(IndexInterval::UncheckedClosed(-kInfIndex, 13)));
  EXPECT_THAT(ShiftIntervalBackward(
                  IndexInterval::UncheckedClosed(1, +kInfIndex), -2, -5),
              Optional(IndexInterval::UncheckedClosed(3, +kInfIndex)));
  EXPECT_THAT(
      ShiftIntervalBackward(
          IndexInterval::UncheckedClosed(-kInfIndex, +kInfIndex), -2, -5),
      Optional(IndexInterval::UncheckedClosed(-kInfIndex, +kInfIndex)));
}

TEST(IndexIntervalTest, ShiftIntervalTo) {
  EXPECT_THAT(ShiftIntervalTo(IndexInterval::UncheckedClosed(1, 8), 3),
              Optional(IndexInterval::UncheckedClosed(3, 10)));

  EXPECT_THAT(ShiftIntervalTo(IndexInterval::UncheckedClosed(-kInfIndex, 8), 2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Interval .* is not bounded below"));

  EXPECT_THAT(ShiftIntervalTo(IndexInterval::UncheckedClosed(1, kInfIndex), 3),
              Optional(IndexInterval::UncheckedClosed(3, kInfIndex)));

  EXPECT_THAT(
      ShiftIntervalTo(IndexInterval::UncheckedClosed(kMinFiniteIndex + 1, 101),
                      kMinFiniteIndex),
      Optional(IndexInterval::Closed(kMinFiniteIndex, 100)));

  EXPECT_THAT(
      ShiftIntervalTo(IndexInterval::UncheckedClosed(5, 10), -kInfIndex),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Origin -[0-9]+ is outside valid range .*"));
  EXPECT_THAT(ShiftIntervalTo(IndexInterval::UncheckedClosed(5, 10), kInfIndex),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Origin [0-9]+ is outside valid range .*"));
  EXPECT_THAT(
      ShiftIntervalTo(IndexInterval::UncheckedClosed(5, 10), kMaxFiniteIndex),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "10 \\+ [0-9]+ is outside valid range .*"));
}

TEST(ExtractStridedSliceTest, Closed) {
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 6, 9, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(6, 4), false, false}, 6));

  // Tests that implicit bounds are not constraints.
  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), true, true}, 3, 15, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(3, 15), false, false}, 3));

  // Tests that implicit_lower and implicit_upper are propagated and swapped
  // when `stride < 0`.
  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), true, false}, kImplicit,
          kImplicit, -1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-10, -5), false, true}, 10));

  // Same as above, but with opposite values of implicit_lower and
  // implicit_upper.
  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, true}, kImplicit,
          kImplicit, -1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-10, -5), true, false}, 10));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, 6, -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(-4, 2), false, false}, 9));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false},
                  kImplicit, 9, 1)
                  .value(),
              Pair(OIII{IndexInterval::UncheckedSized(5, 5), false, false}, 5));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false},
                  -kInfIndex, 9, 1),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Slice interval \\(-inf, 10\\) is not contained "
                            "within domain \\[5, 11\\)"));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, kImplicit, 6,
          -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(-5, 3), false, false}, 10));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 9,
                  -kInfIndex, -2),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Slice interval \\(-inf, 10\\) is not contained "
                            "within domain \\[5, 11\\)"));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, kImplicit,
          -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(-4, 3), false, false}, 9));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 7,
                  kImplicit, 2)
                  .value(),
              Pair(OIII{IndexInterval::UncheckedSized(3, 2), false, false}, 7));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex), false, false},
          kImplicit, 10, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-kInfIndex, 10), false, false},
           -kInfIndex));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex), false, false},
          5, kImplicit, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(5, kInfIndex), false, false},
           5));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex), false, false},
          kImplicit, 5, -1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-kInfIndex, -5), false, false},
           kInfIndex));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, kImplicit, 6,
          0),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid stride 0"));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false},
                  kImplicit, 6, std::numeric_limits<Index>::min()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid stride -[0-9]+"));

  EXPECT_THAT(
      ExtractClosedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 4, 6, 1),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Slice interval \\[4, 7\\) is not contained within domain "
                    "\\[5, 11\\)"));

  EXPECT_THAT(ExtractClosedStridedSlice(
                  {IndexInterval::UncheckedClosed(3, 10), false, false},
                  -kInfIndex - 1, 10, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid start index -[0-9]+"));
}

TEST(ExtractStridedSliceTest, Sized) {
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, 3, -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-4, -2), false, false}, 9));

  // Tests that implicit_upper is propagated when `kImplicit` is specified as
  // the size.
  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, true}, 7, kImplicit, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(7, 10), false, true}, 7));

  // Tests that implicit_{lower,upper} are propagated when `kImplicit` is
  // specified as the start and size.
  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), true, true},
                  kImplicit, kImplicit, 1)
                  .value(),
              Pair(OIII{IndexInterval::UncheckedClosed(5, 10), true, true}, 5));

  // Tests that implicit_{lower,upper} are swapped when `stride < 0`.
  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), true, false}, kImplicit,
          kImplicit, -1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-10, -5), false, true}, 10));

  // Same as above, but with opposite values of implicit_{lower,upper}.
  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, true}, kImplicit,
          kImplicit, -1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-10, -5), true, false}, 10));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, kImplicit,
          -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-4, -2), false, false}, 9));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 7, kImplicit,
          2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(3, 4), false, false}, 7));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 7, 0, 2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(3, 0), false, false}, 7));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 7, 0, -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedSized(-3, 0), false, false}, 7));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, -1, -2),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Negative size -1 specified for sized interval"));

  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false},
                  std::numeric_limits<Index>::min() + 1, 0, 2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid start index -[0-9]+"));

  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false},
                  std::numeric_limits<Index>::max(), 0, -2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid start index [0-9]+"));

  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 5, 100,
                  kInfIndex),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Integer overflow computing slice result"));

  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 5,
                  kInfIndex, 2),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Integer overflow computing slice result"));

  EXPECT_THAT(
      ExtractSizedStridedSlice(
          {IndexInterval::UncheckedClosed(-kInfIndex, 10), false, false},
          kImplicit, kImplicit, 2),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Slicing with non-unit stride of 2 requires a "
                    "finite start index"));

  EXPECT_THAT(ExtractSizedStridedSlice(
                  {IndexInterval::UncheckedClosed(3, kInfIndex), false, false},
                  kImplicit, kImplicit, -2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Slicing with non-unit stride of -2 requires a "
                            "finite start index"));
}

TEST(ExtractStridedSliceTest, HalfOpen) {
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(
      ExtractHalfOpenStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, false}, 9, 7, -2)
          .value(),
      Pair(OIII{IndexInterval::UncheckedClosed(-4, -4), false, false}, 9));

  // Tests that implicit_lower remains true when kImplicit is specified.
  EXPECT_THAT(
      ExtractHalfOpenStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), true, false}, kImplicit, 8, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedHalfOpen(5, 8), true, false}, 5));

  // Tests that implicit_upper remains true when kImplicit is specified.
  EXPECT_THAT(
      ExtractHalfOpenStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, true}, 6, kImplicit, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedHalfOpen(6, 11), false, true}, 6));

  // Tests that an implicit lower bound is not a constraint.
  EXPECT_THAT(
      ExtractHalfOpenStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), true, false}, 3, 8, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedHalfOpen(3, 8), false, false}, 3));

  // Tests that an implicit upper bound is not a constraint.
  EXPECT_THAT(
      ExtractHalfOpenStridedSlice(
          {IndexInterval::UncheckedClosed(5, 10), false, true}, 6, 15, 1)
          .value(),
      Pair(OIII{IndexInterval::UncheckedHalfOpen(6, 15), false, false}, 6));

  EXPECT_THAT(ExtractHalfOpenStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 9,
                  std::numeric_limits<Index>::min() + 1, 2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".* do not specify a valid closed index interval"));

  EXPECT_THAT(ExtractHalfOpenStridedSlice(
                  {IndexInterval::UncheckedClosed(5, 10), false, false}, 9,
                  std::numeric_limits<Index>::max(), -2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".* do not specify a valid closed index interval"));
}

TEST(ComputeStridedSliceMapTest, NoTranslationStride1) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_EQ(absl::OkStatus(),
            ComputeStridedSliceMap(
                OptionallyImplicitIndexInterval{
                    IndexInterval::UncheckedHalfOpen(1, 10), false, false},
                IntervalForm::half_open,
                /*translate_origin_to=*/kImplicit,
                /*start=*/2,
                /*stop_or_size=*/8,
                /*stride=*/1, &new_domain, &output_offset));
  EXPECT_EQ((OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedHalfOpen(2, 8), false, false}),
            new_domain);
  EXPECT_EQ(0, output_offset);
}

TEST(ComputeStridedSliceMapTest, NoTranslationStride2) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_EQ(absl::OkStatus(),
            ComputeStridedSliceMap(
                OptionallyImplicitIndexInterval{
                    IndexInterval::UncheckedHalfOpen(1, 10), false, false},
                IntervalForm::half_open,
                /*translate_origin_to=*/kImplicit,
                /*start=*/2,
                /*stop_or_size=*/8,
                /*stride=*/2, &new_domain, &output_offset));
  EXPECT_EQ((OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedHalfOpen(1, 4), false, false}),
            new_domain);
  EXPECT_EQ(0, output_offset);
}

TEST(ComputeStridedSliceMapTest, NoTranslationStrideNegative2) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_EQ(absl::OkStatus(),
            ComputeStridedSliceMap(
                OptionallyImplicitIndexInterval{
                    IndexInterval::UncheckedHalfOpen(1, 10), false, false},
                IntervalForm::half_open,
                /*translate_origin_to=*/kImplicit,
                /*start=*/9,
                /*stop_or_size=*/2,
                /*stride=*/-2, &new_domain, &output_offset));
  EXPECT_EQ((OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedHalfOpen(-4, 0), false, false}),
            new_domain);
  EXPECT_EQ(1, output_offset);
}

TEST(ComputeStridedSliceMapTest, TranslationStride1) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_EQ(absl::OkStatus(),
            ComputeStridedSliceMap(
                OptionallyImplicitIndexInterval{
                    IndexInterval::UncheckedHalfOpen(1, 10), false, false},
                IntervalForm::half_open,
                /*translate_origin_to=*/7,
                /*start=*/2,
                /*stop_or_size=*/8,
                /*stride=*/1, &new_domain, &output_offset));
  EXPECT_EQ((OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedHalfOpen(7, 13), false, false}),
            new_domain);
  EXPECT_EQ(-5, output_offset);
}

TEST(ComputeStridedSliceMapTest, TranslationError) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_THAT(ComputeStridedSliceMap(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedHalfOpen(1, 10), false, false},
                  IntervalForm::half_open,
                  /*translate_origin_to=*/kMaxFiniteIndex,
                  /*start=*/2,
                  /*stop_or_size=*/8,
                  /*stride=*/1, &new_domain, &output_offset),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(ComputeStridedSliceMapTest, SliceError) {
  OptionallyImplicitIndexInterval new_domain;
  Index output_offset;
  EXPECT_THAT(ComputeStridedSliceMap(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedHalfOpen(3, 10), false, false},
                  IntervalForm::half_open,
                  /*translate_origin_to=*/kMaxFiniteIndex,
                  /*start=*/2,
                  /*stop_or_size=*/8,
                  /*stride=*/1, &new_domain, &output_offset),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(GetAffineTransformDomainTest, Divisor1) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-9, -1),
            GetAffineTransformDomain(IndexInterval::UncheckedClosed(1, 9),
                                     /*offset=*/10, /*divisor=*/1)
                .value());
}

TEST(GetAffineTransformDomainTest, Divisor2) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-2, 1),
            GetAffineTransformDomain(IndexInterval::UncheckedClosed(1, 9),
                                     /*offset=*/6, /*divisor=*/2)
                .value());
}

TEST(GetAffineTransformDomainTest, DivisorNegative1) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-3, 5),
            GetAffineTransformDomain(IndexInterval::UncheckedClosed(1, 9),
                                     /*offset=*/6, /*divisor=*/-1)
                .value());
}

TEST(GetAffineTransformDomainTest, DivisorNegative2) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-1, 2),
            GetAffineTransformDomain(IndexInterval::UncheckedClosed(1, 9),
                                     /*offset=*/6, /*divisor=*/-2)
                .value());
}

TEST(GetAffineTransformDomainTest, DivisorNegative2LargeMagnitude) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-(kInfIndex - 10) / 2, 5),
            GetAffineTransformDomain(
                IndexInterval::UncheckedClosed(-10, kInfIndex - 10),
                /*offset=*/0, /*divisor=*/-2)
                .value());
}

TEST(GetAffineTransformDomainTest, EmptyInterval) {
  EXPECT_EQ(IndexInterval::UncheckedSized(-2, 0),
            GetAffineTransformDomain(IndexInterval::UncheckedSized(10, 0),
                                     /*offset=*/5, /*divisor=*/-2)
                .value());
}

TEST(GetAffineTransformDomainTest, DivisorInvalid) {
  EXPECT_THAT(GetAffineTransformDomain(
                  IndexInterval::UncheckedClosed(1, 10),
                  /*offset=*/0, /*divisor=*/std::numeric_limits<Index>::min()),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GetAffineTransformDomainTest, OffsetInvalid) {
  EXPECT_THAT(GetAffineTransformDomain(
                  IndexInterval::UncheckedClosed(1, 10),
                  /*offset=*/std::numeric_limits<Index>::min(), /*divisor=*/-1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

void TestGetAffineTransformRangeRoundTrip(IndexInterval domain, Index offset,
                                          Index multiplier,
                                          IndexInterval range) {
  EXPECT_THAT(GetAffineTransformRange(domain, offset, multiplier),
              ::testing::Optional(range))
      << "domain=" << domain << ", offset=" << offset
      << ", multiplier=" << multiplier << ", range=" << range;
  EXPECT_THAT(GetAffineTransformDomain(range, offset, multiplier),
              ::testing::Optional(domain))
      << "domain=" << domain << ", offset=" << offset
      << ", multiplier=" << multiplier << ", range=" << range;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto inv_domain,
      GetAffineTransformInverseDomain(domain, offset, multiplier));
  EXPECT_THAT(GetAffineTransformDomain(inv_domain, offset, multiplier),
              ::testing::Optional(domain))
      << "domain=" << domain << ", offset=" << offset
      << ", multiplier=" << multiplier << ", range=" << range
      << ", inv_domain=" << inv_domain;
}

// Tests that GetAffineTransformDomain inverts GetAffineTransformRange when the
// multiplier is non-zero.
TEST(GetAffineTransformRangeTest, SerializationRoundTrip) {
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval::UncheckedClosed(1, 10), /*offset=*/3,
      /*multiplier=*/1,
      /*range=*/IndexInterval::UncheckedClosed(4, 13));
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval::UncheckedClosed(1, 10), /*offset=*/3,
      /*multiplier=*/2,
      /*range=*/IndexInterval::UncheckedClosed(2 + 3, 10 * 2 + 3));

  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval::UncheckedSized(4, 0), /*offset=*/3,
      /*multiplier=*/2, /*range=*/IndexInterval::UncheckedSized(2 * 4 + 3, 0));
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval::UncheckedSized(4, 0), /*offset=*/3,
      /*multiplier=*/-2,
      /*range=*/IndexInterval::UncheckedSized(-2 * 4 + 3, 0));

  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/std::numeric_limits<Index>::min(),
      /*multiplier=*/1,
      /*range=*/IndexInterval());
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/std::numeric_limits<Index>::max(),
      /*multiplier=*/1,
      /*range=*/IndexInterval());
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/0,
      /*multiplier=*/1,
      /*range=*/IndexInterval());
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/std::numeric_limits<Index>::min(),
      /*multiplier=*/-1,
      /*range=*/IndexInterval());
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/std::numeric_limits<Index>::max(),
      /*multiplier=*/-1,
      /*range=*/IndexInterval());
  TestGetAffineTransformRangeRoundTrip(
      /*domain=*/IndexInterval(), /*offset=*/0,
      /*multiplier=*/-1,
      /*range=*/IndexInterval());
}

TEST(GetAffineTransformRangeTest, ZeroMultiplier) {
  // Can't round trip with a zero multiplier.
  EXPECT_EQ(IndexInterval::UncheckedSized(3, 1),
            GetAffineTransformRange(IndexInterval::UncheckedClosed(4, 10), 3, 0)
                .value());
}

TEST(GetAffineTransformRangeTest, ErrorCases) {
  EXPECT_THAT(GetAffineTransformRange(IndexInterval::UncheckedClosed(3, 10),
                                      kInfIndex, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(GetAffineTransformRange(IndexInterval::UncheckedClosed(3, 10), 5,
                                      kInfIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetAffineTransformRange(
                  IndexInterval::UncheckedClosed(-1, 1),
                  std::numeric_limits<Index>::max() - kInfIndex + 1, kInfIndex),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GetAffineTransformInverseDomainTest, Examples) {
  EXPECT_THAT(
      GetAffineTransformRange(IndexInterval::UncheckedClosed(2, 4), 1, 3),
      ::testing::Optional(IndexInterval::UncheckedClosed(7, 13)));
  EXPECT_THAT(GetAffineTransformInverseDomain(
                  IndexInterval::UncheckedClosed(2, 4), 1, 3),
              ::testing::Optional(IndexInterval::UncheckedClosed(7, 15)));
  EXPECT_THAT(
      GetAffineTransformRange(IndexInterval::UncheckedClosed(2, 4), 1, -3),
      ::testing::Optional(IndexInterval::UncheckedClosed(-11, -5)));
  EXPECT_THAT(GetAffineTransformInverseDomain(
                  IndexInterval::UncheckedClosed(2, 4), 1, -3),
              ::testing::Optional(IndexInterval::UncheckedClosed(-13, -5)));
}

void TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
    OptionallyImplicitIndexInterval domain, Index offset, Index multiplier,
    OptionallyImplicitIndexInterval range) {
  EXPECT_EQ(GetAffineTransformRange(domain, offset, multiplier).value(), range)
      << "domain=" << domain << ", offset=" << offset
      << ", multiplier=" << multiplier << ", range=" << range;
  EXPECT_EQ(GetAffineTransformDomain(range, offset, multiplier).value(), domain)
      << "domain=" << domain << ", offset=" << offset
      << ", multiplier=" << multiplier << ", range=" << range;
}

TEST(GetAffineTransformRangeTest, OptionallyImplicitRoundTrip) {
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), true, false},
      /*offset=*/3,
      /*multiplier=*/1, {IndexInterval::UncheckedClosed(4, 13), true, false});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), true, true},
      /*offset=*/3,
      /*multiplier=*/1, {IndexInterval::UncheckedClosed(4, 13), true, true});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), false, false},
      /*offset=*/3,
      /*multiplier=*/1, {IndexInterval::UncheckedClosed(4, 13), false, false});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), false, true},
      /*offset=*/3,
      /*multiplier=*/1, {IndexInterval::UncheckedClosed(4, 13), false, true});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), false, true},
      /*offset=*/-3,
      /*multiplier=*/1, {IndexInterval::UncheckedClosed(-2, 7), false, true});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), false, true},
      /*offset=*/3,
      /*multiplier=*/-1, {IndexInterval::UncheckedClosed(-7, 2), true, false});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedClosed(1, 10), true, false},
      /*offset=*/3,
      /*multiplier=*/-1, {IndexInterval::UncheckedClosed(-7, 2), false, true});
  TestGetAffineTransformRangeOptionallyImplicitRoundTrip(
      {IndexInterval::UncheckedSized(4, 0), true, false},
      /*offset=*/3,
      /*multiplier=*/-2,
      {IndexInterval::UncheckedSized(-2 * 4 + 3, 0), false, true});
}

// Errors from regular GetAffineTransformRange simply pass through.
TEST(GetAffineTransformRangeTest, OptionallyImplicitErrorCases) {
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(GetAffineTransformRange(
                  OIII{IndexInterval::UncheckedClosed(3, 10), true, false},
                  kInfIndex, 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Errors from regular GetAffineTransformDomain simply pass through.
TEST(GetAffineTransformDomainTest, OptionallyImplicitErrorCases) {
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(GetAffineTransformDomain(
                  OIII{IndexInterval::UncheckedClosed(1, 10), true, false},
                  /*offset=*/std::numeric_limits<Index>::min(), /*divisor=*/-1),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexIntervalRefTest, Basic) {
  Index inclusive_min = 5, size = 10;
  IndexIntervalRef ref = IndexIntervalRef::UncheckedSized(inclusive_min, size);
  EXPECT_EQ(5, ref.inclusive_min());
  EXPECT_EQ(4, ref.exclusive_min());
  EXPECT_EQ(10, ref.size());
  EXPECT_EQ(15, ref.exclusive_max());
  EXPECT_EQ(14, ref.inclusive_max());
  EXPECT_EQ(IndexInterval::UncheckedSized(5, 10),
            static_cast<IndexInterval>(ref));
  ref = IndexInterval::UncheckedSized(6, 9);
  EXPECT_EQ(6, inclusive_min);
  EXPECT_EQ(9, size);
  EXPECT_FALSE(ref.empty());
  size = 0;
  EXPECT_TRUE(ref.empty());
}

TEST(IndexIntervalRefTest, ConstructFromIndexInterval) {
  IndexInterval interval = IndexInterval::UncheckedSized(5, 10);
  IndexIntervalRef ref(interval);
  ref = IndexInterval::UncheckedSized(3, 6);
  EXPECT_EQ(interval, IndexInterval::UncheckedSized(3, 6));
}

// Tests that the implicit conversion from IndexIntervalRef to IndexInterval
// works, and that functions like IsFinite, Contains, Intersect, and Hull can be
// called on IndexIntervalRef objects by way of that conversion.
TEST(IndexIntervalRefTest, ImplicitConversion) {
  IndexInterval interval = IndexInterval::UncheckedSized(5, 10);
  IndexIntervalRef ref(interval);
  IndexInterval interval2 = ref;
  EXPECT_EQ(interval, interval2);
  EXPECT_TRUE(IsFinite(ref));
  EXPECT_TRUE(Contains(ref, ref.inclusive_min()));
  EXPECT_TRUE(Contains(ref, ref));
  EXPECT_EQ(ref, Intersect(ref, ref));
  EXPECT_EQ(ref, Hull(ref, ref));
}

TEST(IndexIntervalRefTest, Assignment) {
  IndexInterval interval = IndexInterval::UncheckedSized(5, 10);
  IndexIntervalRef ref(interval);
  IndexInterval interval2 = ref;
  IndexIntervalRef ref2(interval2);
  ref2 = ref;
  EXPECT_EQ(IndexInterval::UncheckedSized(5, 10), interval2);
  EXPECT_EQ(IndexInterval::UncheckedSized(5, 10), interval);
}

TEST(OptionallyImplicitIndexIntervalTest, EffectiveInterval) {
  EXPECT_EQ(IndexInterval::UncheckedClosed(-kInfIndex, 2),
            OptionallyImplicitIndexInterval(
                IndexInterval::UncheckedClosed(1, 2), true, false)
                .effective_interval());
  EXPECT_EQ(IndexInterval::UncheckedClosed(1, +kInfIndex),
            OptionallyImplicitIndexInterval(
                IndexInterval::UncheckedClosed(1, 2), false, true)
                .effective_interval());
  EXPECT_EQ(IndexInterval(),
            OptionallyImplicitIndexInterval(
                IndexInterval::UncheckedClosed(1, 2), true, true)
                .effective_interval());
}

TEST(OptionallyImplicitIndexIntervalTest, Ostream) {
  EXPECT_EQ("[1*, 3)", StrCat(OptionallyImplicitIndexInterval{
                           IndexInterval::UncheckedClosed(1, 2), true, false}));
  EXPECT_EQ("(-inf, 3*)",
            StrCat(OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedClosed(-kInfIndex, 2), false, true}));
  EXPECT_EQ("[7*, +inf*)",
            StrCat(OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedClosed(7, kInfIndex), true, true}));
}

TEST(OptionallyImplicitIndexIntervalTest, Comparison) {
  OptionallyImplicitIndexInterval a{};
  OptionallyImplicitIndexInterval b{IndexInterval::UncheckedSized(0, 1), false,
                                    false};
  OptionallyImplicitIndexInterval c{IndexInterval::UncheckedSized(0, 1), false,
                                    true};
  OptionallyImplicitIndexInterval d{IndexInterval::UncheckedSized(0, 1), true,
                                    false};
  OptionallyImplicitIndexInterval e{IndexInterval::UncheckedSized(0, 1), true,
                                    true};
  OptionallyImplicitIndexInterval f{IndexInterval::UncheckedSized(0, 0), false,
                                    false};
  OptionallyImplicitIndexInterval g{IndexInterval::UncheckedSized(0, 2), false,
                                    false};
  OptionallyImplicitIndexInterval h{IndexInterval::UncheckedSized(1, 2), false,
                                    false};
  OptionallyImplicitIndexInterval i{IndexInterval::UncheckedSized(1, 2), false,
                                    true};
  OptionallyImplicitIndexInterval j{IndexInterval::UncheckedSized(1, 2), true,
                                    false};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_EQ(e, e);
  EXPECT_EQ(f, f);
  EXPECT_EQ(g, g);
  EXPECT_EQ(h, h);
  EXPECT_EQ(i, i);
  EXPECT_EQ(j, j);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(a, e);
  EXPECT_NE(a, f);
  EXPECT_NE(a, g);
  EXPECT_NE(a, h);
  EXPECT_NE(a, i);
  EXPECT_NE(a, j);
  EXPECT_NE(g, j);
  EXPECT_NE(g, h);
  EXPECT_NE(g, i);
  EXPECT_NE(g, j);
}

TEST(OptionallyImplicitIndexIntervalTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      OptionallyImplicitIndexInterval{},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 1),
                                      false, false},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 1),
                                      false, true},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 1), true,
                                      false},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 1), true,
                                      true},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 0),
                                      false, false},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(0, 2),
                                      false, false},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(1, 2),
                                      false, false},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(1, 2),
                                      false, true},
      OptionallyImplicitIndexInterval{IndexInterval::UncheckedSized(1, 2), true,
                                      false},
  }));
}

static_assert(std::is_convertible_v<IndexDomainDimension<container>,
                                    IndexDomainDimension<view>>);
static_assert(std::is_convertible_v<IndexDomainDimension<view>,
                                    IndexDomainDimension<container>>);
static_assert(std::is_assignable_v<IndexDomainDimension<container>,
                                   IndexDomainDimension<view>>);
static_assert(std::is_assignable_v<IndexDomainDimension<view>,
                                   IndexDomainDimension<container>>);

TEST(IndexDomainDimensionTest, DefaultConstruct) {
  IndexDomainDimension<> d;
  EXPECT_EQ(OptionallyImplicitIndexInterval(),
            d.optionally_implicit_interval());
  EXPECT_EQ("", d.label());
}

TEST(IndexDomainDimensionTest, ConstructFromOptionallyImplicitIndexInterval) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<> d = interval;
  EXPECT_EQ(interval, d.optionally_implicit_interval());
  EXPECT_EQ("", d.label());
}

TEST(IndexDomainDimensionTest, ConstructLabel) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<> d = {interval, "label"};
  EXPECT_EQ(interval, d.optionally_implicit_interval());
  EXPECT_EQ("label", d.label());
}

TEST(IndexDomainDimensionTest, ConstructContainerFromView) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<view> d_view = {interval, "label"};
  IndexDomainDimension<> d(d_view);
  EXPECT_EQ(interval, d.optionally_implicit_interval());
  EXPECT_EQ("label", d.label());
}

TEST(IndexDomainDimensionTest, ConstructViewFromContainer) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<> d = {interval, "label"};
  IndexDomainDimension<view> d_view = d;
  EXPECT_EQ(interval, d_view.optionally_implicit_interval());
  EXPECT_EQ("label", d_view.label());
}

TEST(IndexDomainDimensionTest, AssignContainerFromView) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<view> d_view = {interval, "label"};
  IndexDomainDimension<> d;
  d = d_view;
  EXPECT_EQ(interval, d.optionally_implicit_interval());
  EXPECT_EQ("label", d.label());
}

TEST(IndexDomainDimensionTest, AssignViewFromContainer) {
  OptionallyImplicitIndexInterval interval{IndexInterval::UncheckedSized(0, 10),
                                           false, true};
  IndexDomainDimension<> d = {interval, "label"};
  IndexDomainDimension<view> d_view;
  d_view = d;
  EXPECT_EQ(interval, d_view.optionally_implicit_interval());
  EXPECT_EQ("label", d_view.label());
}

TEST(IndexDomainDimensionTest, PrintToOstream) {
  EXPECT_EQ("[0, 10*)",
            StrCat(IndexDomainDimension<>{
                {IndexInterval::UncheckedSized(0, 10), false, true}, ""}));
  EXPECT_EQ("[0, 10*)",
            StrCat(IndexDomainDimension<view>{
                {IndexInterval::UncheckedSized(0, 10), false, true}, ""}));
  EXPECT_EQ("\"label\": [0, 10*)",
            StrCat(IndexDomainDimension<>{
                {IndexInterval::UncheckedSized(0, 10), false, true}, "label"}));
}

TEST(IndexDomainDimensionTest, Compare) {
  IndexDomainDimension<> d1 = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label"};
  IndexDomainDimension<view> d1_view = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label"};
  IndexDomainDimension<> d2 = {
      {IndexInterval::UncheckedSized(3, 10), false, true}, "label"};
  IndexDomainDimension<view> d2_view = {
      {IndexInterval::UncheckedSized(3, 10), false, true}, "label"};
  IndexDomainDimension<> d3 = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label2"};
  EXPECT_EQ(d1, d1);
  EXPECT_EQ(d1, d1_view);
  EXPECT_EQ(d1_view, d1);
  EXPECT_EQ(d1_view, d1_view);
  EXPECT_EQ(d2, d2);
  EXPECT_EQ(d3, d3);
  EXPECT_NE(d1, d2);
  EXPECT_NE(d1, d2_view);
  EXPECT_NE(d1_view, d2);
  EXPECT_NE(d1_view, d2_view);
  EXPECT_NE(d1, d3);
}

TEST(IndexDomainDimensionTest, Hash) {
  IndexDomainDimension<> d1 = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label"};
  IndexDomainDimension<view> d1_view = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label"};
  IndexDomainDimension<> d2 = {
      {IndexInterval::UncheckedSized(3, 10), false, true}, "label"};
  IndexDomainDimension<view> d2_view = {
      {IndexInterval::UncheckedSized(3, 10), false, true}, "label"};
  IndexDomainDimension<> d3 = {
      {IndexInterval::UncheckedSized(0, 10), false, true}, "label2"};
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({d1, d2, d3}));
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({d1_view, d2_view}));
}

static_assert(ExplicitIndexOr(10, 11) == 10);
static_assert(ExplicitIndexOr(kImplicit, 11) == 11);

static_assert(ImplicitOrEqual(10, 10));
static_assert(ImplicitOrEqual(kImplicit, 10));
static_assert(!ImplicitOrEqual(10, 11));

static_assert(DividePositiveRoundOut(IndexInterval::UncheckedHalfOpen(3, 10),
                                     2) ==
              IndexInterval::UncheckedHalfOpen(1, 5));

static_assert(DividePositiveRoundOut(IndexInterval::UncheckedHalfOpen(3, 11),
                                     2) ==
              IndexInterval::UncheckedHalfOpen(1, 6));

static_assert(DividePositiveRoundOut(IndexInterval::UncheckedHalfOpen(-3, 10),
                                     2) ==
              IndexInterval::UncheckedHalfOpen(-2, 5));

TEST(IndexIntervalTest, Negate) {
  EXPECT_EQ(IndexInterval::UncheckedSized(0, 0),
            -IndexInterval::UncheckedSized(0, 0));
  EXPECT_EQ(IndexInterval::UncheckedSized(5, 0),
            -IndexInterval::UncheckedSized(-5, 0));
  EXPECT_EQ(
      IndexInterval::UncheckedClosed(kMaxFiniteIndex, kMaxFiniteIndex),
      -IndexInterval::UncheckedClosed(-kMaxFiniteIndex, -kMaxFiniteIndex));
  EXPECT_EQ(IndexInterval(), -IndexInterval());
  EXPECT_EQ(IndexInterval::UncheckedClosed(-5, 6),
            -IndexInterval::UncheckedClosed(-6, 5));
  EXPECT_EQ(IndexInterval::UncheckedClosed(5, 30),
            -IndexInterval::UncheckedClosed(-30, -5));
}

TEST(MergeDimensionLabelsTest, Basic) {
  EXPECT_THAT(MergeDimensionLabels("a", ""),
              ::testing::Optional(std::string("a")));
  EXPECT_THAT(MergeDimensionLabels("a", "a"),
              ::testing::Optional(std::string("a")));
  EXPECT_THAT(MergeDimensionLabels("", "a"),
              ::testing::Optional(std::string("a")));
  EXPECT_THAT(MergeDimensionLabels("", ""),
              ::testing::Optional(std::string("")));
  EXPECT_THAT(MergeDimensionLabels("a", "b"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Dimension labels do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, EqualExplicit) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false}),
              ::testing::Optional(OptionallyImplicitIndexInterval{
                  IndexInterval::UncheckedClosed(1, 5), false, false}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, EqualImplicit) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), true, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), true, false}),
              ::testing::Optional(OptionallyImplicitIndexInterval{
                  IndexInterval::UncheckedClosed(1, 5), true, false}));
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, true},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, true}),
              ::testing::Optional(OptionallyImplicitIndexInterval{
                  IndexInterval::UncheckedClosed(1, 5), false, true}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, UpperUnspecified) {
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(1, kInfIndex), false, true},
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false}),
      ::testing::Optional(OptionallyImplicitIndexInterval{
          IndexInterval::UncheckedClosed(1, 5), false, false}));
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false},
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(1, kInfIndex), false, true}),
      ::testing::Optional(OptionallyImplicitIndexInterval{
          IndexInterval::UncheckedClosed(1, 5), false, false}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, LowerUnspecified) {
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(-kInfIndex, 5), true, false},
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false}),
      ::testing::Optional(OptionallyImplicitIndexInterval{
          IndexInterval::UncheckedClosed(1, 5), false, false}));
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false},
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(-kInfIndex, 5), true, false}),
      ::testing::Optional(OptionallyImplicitIndexInterval{
          IndexInterval::UncheckedClosed(1, 5), false, false}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, MismatchLower) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(2, 5), false, false}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Lower bounds do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, MismatchLowerInfinite) {
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false},
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(-kInfIndex, 5), false, false}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Lower bounds do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, LowerImplicitMerge) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), true, false}),
              ::testing::Optional(OptionallyImplicitIndexInterval{
                  IndexInterval::UncheckedClosed(1, 5), false, false}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, UpperImplicitMerge) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, true},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false}),
              ::testing::Optional(OptionallyImplicitIndexInterval{
                  IndexInterval::UncheckedClosed(1, 5), false, false}));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, MismatchUpper) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 6), false, false}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Upper bounds do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, MismatchUpperInfinite) {
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{IndexInterval::UncheckedClosed(1, 5),
                                          false, false},
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(1, kInfIndex), false, false}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Upper bounds do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, MismatchUpperImplicit) {
  EXPECT_THAT(MergeOptionallyImplicitIndexIntervals(
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 5), false, false},
                  OptionallyImplicitIndexInterval{
                      IndexInterval::UncheckedClosed(1, 6), false, true}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Upper bounds do not match"));
}

TEST(MergeOptionallyImplicitIndexIntervalsTest, InvalidInterval) {
  EXPECT_THAT(
      MergeOptionallyImplicitIndexIntervals(
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(-kInfIndex, -5), true, false},
          OptionallyImplicitIndexInterval{
              IndexInterval::UncheckedClosed(5, kInfIndex), false, true}),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "\\(5, -5\\) do not specify a valid closed index interval"));
}

TEST(IndexIntervalSerializationTest, Basic) {
  TestSerializationRoundTrip(IndexInterval::UncheckedSized(1, 2));
}

}  // namespace
