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

#include "tensorstore/internal/container/hash_set_of_any.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::HashSetOfAny;

template <typename T>
struct Entry : public HashSetOfAny::Entry {
  using KeyParam = T;

  Entry(T key) : key_(key) {}

  static Entry& FindOrInsert(HashSetOfAny& set, T key) {
    return *set.FindOrInsert<Entry<T>>(
                   key, [&] { return std::make_unique<Entry<T>>(key); })
                .first;
  }

  T key_;

  T key() const { return key_; }
};

TEST(HashSetOfAnyTest, Basic) {
  HashSetOfAny set;

  auto& a = Entry<int>::FindOrInsert(set, 5);
  auto& b = Entry<int>::FindOrInsert(set, 5);
  auto& c = Entry<int>::FindOrInsert(set, 6);
  auto& e = Entry<float>::FindOrInsert(set, 1.5);
  auto& f = Entry<float>::FindOrInsert(set, 2.5);
  EXPECT_EQ(set.size(), 4);
  EXPECT_FALSE(set.empty());
  EXPECT_EQ(&a, &b);
  EXPECT_NE(&a, &c);
  EXPECT_NE(&e, &f);

  for (auto* entry : set) {
    delete entry;
  }
}

}  // namespace
