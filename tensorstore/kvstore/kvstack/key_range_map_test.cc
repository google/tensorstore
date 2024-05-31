// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/kvstack/key_range_map.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "tensorstore/kvstore/key_range.h"

using ::tensorstore::KeyRange;
using ::tensorstore::internal_kvstack::KeyRangeMap;

namespace tensorstore {
namespace internal_kvstack {

using IntRange = KeyRangeMap<int>::Value;

bool operator==(const IntRange& a, const IntRange& b) {
  return a.range == b.range && a.value == b.value;
}
bool operator!=(const IntRange& a, const IntRange& b) { return !(a == b); }

template <typename V, typename Sink>
void AbslStringify(Sink& sink, const IntRange& r) {
  absl::Format(&sink, "{%s, %s}", r.range.inclusive_min, r.range.exclusive_max);
}

}  // namespace internal_kvstack
}  // namespace tensorstore

namespace {

using IntRange = ::tensorstore::internal_kvstack::IntRange;

TEST(RangeMapTest, Dense) {
  KeyRangeMap<int> m;

  // Record passed ranges and values in "r"
  m.Set(KeyRange({}, {}), 1);
  m.Set(KeyRange("a", "z"), 20);
  m.Set(KeyRange::Prefix("a"), 30);
  m.Set(KeyRange::Singleton("a/b.c"), 40);  // hidden
  m.Set(KeyRange::Prefix("a/b"), 50);

  EXPECT_THAT(m, testing::ElementsAre(IntRange{KeyRange("", "a"), 1},        //
                                      IntRange{KeyRange("a", "a/b"), 30},    //
                                      IntRange{KeyRange("a/b", "a/c"), 50},  //
                                      IntRange{KeyRange("a/c", "b"), 30},    //
                                      IntRange{KeyRange("b", "z"), 20},      //
                                      IntRange{KeyRange("z", ""), 1}));

  ASSERT_NE(m.range_containing(""), m.end());
  EXPECT_THAT(*m.range_containing(""), (IntRange{KeyRange("", "a"), 1}));

  ASSERT_NE(m.range_containing("a"), m.end());
  EXPECT_THAT(*m.range_containing("a"), (IntRange{KeyRange("a", "a/b"), 30}));

  ASSERT_NE(m.range_containing("z"), m.end());
  EXPECT_THAT(*m.range_containing("z"), (IntRange{KeyRange("z", ""), 1}));

  ASSERT_NE(m.range_containing("b"), m.end());
  EXPECT_THAT(*m.range_containing("b"), (IntRange{KeyRange("b", "z"), 20}));

  ASSERT_NE(m.range_containing("d"), m.end());
  EXPECT_THAT(*m.range_containing("d"), (IntRange{KeyRange("b", "z"), 20}));

  ASSERT_NE(m.range_containing("a/d"), m.end());
  EXPECT_THAT(*m.range_containing("a/d"), (IntRange{KeyRange("a/c", "b"), 30}));

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange("", ""),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_THAT(ranges, testing::ElementsAre(
                            KeyRange("", "a"), KeyRange("a", "a/b"),
                            KeyRange("a/b", "a/c"), KeyRange("a/c", "b"),
                            KeyRange("b", "z"), KeyRange("z", "")));
  }

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange::EmptyRange(),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_TRUE(ranges.empty());
  }

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange("", "a/z"),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_THAT(ranges, testing::ElementsAre(
                            KeyRange("", "a"), KeyRange("a", "a/b"),
                            KeyRange("a/b", "a/c"), KeyRange("a/c", "a/z")));
  }
}

TEST(RangeMapTest, Sparse) {
  KeyRangeMap<int> m;

  // Record passed ranges and values in "r"
  m.Set(KeyRange("a", "z"), 20);
  m.Set(KeyRange::Prefix("a"), 30);
  m.Set(KeyRange::Singleton("a/b.c"), 40);  // hidden
  m.Set(KeyRange::Prefix("a/b"), 50);

  for (const auto& v : m) {
    ABSL_LOG(INFO) << v.range << ": " << v.value;
  }

  EXPECT_THAT(m, testing::ElementsAre(IntRange{KeyRange("a", "a/b"), 30},    //
                                      IntRange{KeyRange("a/b", "a/c"), 50},  //
                                      IntRange{KeyRange("a/c", "b"), 30},    //
                                      IntRange{KeyRange("b", "z"), 20}));

  ASSERT_EQ(m.range_containing(""), m.end());
  ASSERT_EQ(m.range_containing("z"), m.end());

  ASSERT_NE(m.range_containing("a"), m.end());
  EXPECT_THAT(*m.range_containing("a"), (IntRange{KeyRange("a", "a/b"), 30}));

  ASSERT_NE(m.range_containing("b"), m.end());
  EXPECT_THAT(*m.range_containing("b"), (IntRange{KeyRange("b", "z"), 20}));

  ASSERT_NE(m.range_containing("d"), m.end());
  EXPECT_THAT(*m.range_containing("d"), (IntRange{KeyRange("b", "z"), 20}));

  ASSERT_NE(m.range_containing("a/d"), m.end());
  EXPECT_THAT(*m.range_containing("a/d"), (IntRange{KeyRange("a/c", "b"), 30}));

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange("", ""),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_THAT(ranges, testing::ElementsAre(
                            KeyRange("a", "a/b"), KeyRange("a/b", "a/c"),
                            KeyRange("a/c", "b"), KeyRange("b", "z")));
  }

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange::EmptyRange(),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_TRUE(ranges.empty());
  }

  {
    std::vector<KeyRange> ranges;
    m.VisitRange(KeyRange("", "a/z"),
                 [&](KeyRange r, auto& value) { ranges.push_back(r); });

    EXPECT_THAT(ranges, testing::ElementsAre(KeyRange("a", "a/b"),
                                             KeyRange("a/b", "a/c"),
                                             KeyRange("a/c", "a/z")));
  }
}

}  // namespace
