// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/poly/storage.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_poly_storage::ActualInlineSize;
using ::tensorstore::internal_poly_storage::GetVTableBase;
using ::tensorstore::internal_poly_storage::HeapStorageOps;
using ::tensorstore::internal_poly_storage::InlineStorageOps;
using ::tensorstore::internal_poly_storage::Storage;
using ::tensorstore::internal_poly_storage::VTableBase;

static constexpr size_t kStorageSize = ActualInlineSize(8);

static_assert(80 == ActualInlineSize(79));
static_assert(80 == ActualInlineSize(80));

TEST(ObjectOps, InlineTrivial) {
  using S = Storage<kStorageSize, true>;
  using Ops = typename S::Ops<int>;
  static_assert(std::is_same_v<Ops, InlineStorageOps<int>>);
  static_assert(Ops::UsesInlineStorage());

  S a, b;

  EXPECT_EQ(nullptr, a.template get_if<int>());

  Ops::Construct(a.storage(), 7);
  Ops::Relocate(b.storage(), a.storage());
  Ops::Copy(a.storage(), b.storage());

  EXPECT_EQ(7, Ops::Get(a.storage()));
  EXPECT_EQ(7, Ops::Get(b.storage()));

  Ops::Destroy(a.storage());
  Ops::Destroy(b.storage());
}

TEST(ObjectOps, NotInlineTrivial) {
  struct X {
    double x;
    double y;
    double z;
  };

  using S = Storage<kStorageSize, true>;
  using Ops = typename S::Ops<X>;
  static_assert(std::is_same_v<Ops, HeapStorageOps<X>>);

  static_assert(!Ops::UsesInlineStorage());

  S a, b;
  EXPECT_EQ(nullptr, a.get_if<int>());

  Ops::Construct(a.storage(), X{7, 8, 9});
  Ops::Relocate(b.storage(), a.storage());
  Ops::Copy(a.storage(), b.storage());

  EXPECT_EQ(7, Ops::Get(a.storage()).x);
  EXPECT_EQ(9, Ops::Get(b.storage()).z);

  Ops::Destroy(a.storage());
  Ops::Destroy(b.storage());
}

template <typename Ops, bool Copyable>
static const VTableBase* GetVTable() {
  static VTableBase vtable = GetVTableBase<Ops, Copyable>();
  return &vtable;
}

TEST(Storage, MoveOnly) {
  using S = Storage<16, false>;
  using Ops = typename S::Ops<int>;

  {
    S a;
    EXPECT_TRUE(a.null());
    EXPECT_EQ(nullptr, a.get_if<int>());
  }

  {
    S a;
    a.ConstructT<int>(GetVTable<Ops, false>(), 7);
    ASSERT_FALSE(a.null());

    ASSERT_NE(nullptr, a.get_if<int>());
    EXPECT_EQ(7, *a.get_if<int>());
  }

  {
    S a;
    a.ConstructT<int>(GetVTable<Ops, false>(), 8);

    S b = std::move(a);
    ASSERT_FALSE(b.null());

    ASSERT_NE(nullptr, b.get_if<int>());
    EXPECT_EQ(8, *b.get_if<int>());

    S c(std::move(b));
    ASSERT_FALSE(c.null());

    ASSERT_NE(nullptr, c.get_if<int>());
    EXPECT_EQ(8, *c.get_if<int>());
  }
}

TEST(Storage, Copy) {
  using S = Storage<16, true>;
  using Ops = typename S::Ops<int>;

  {
    S a;
    EXPECT_TRUE(a.null());
    EXPECT_EQ(nullptr, a.get_if<int>());
  }

  {
    S a;
    a.ConstructT<int>(GetVTable<Ops, true>(), 7);
    ASSERT_FALSE(a.null());

    ASSERT_NE(nullptr, a.get_if<int>());
    EXPECT_EQ(7, *a.get_if<int>());
  }

  {
    S a;
    a.ConstructT<int>(GetVTable<Ops, true>(), 8);

    S b = a;
    ASSERT_NE(nullptr, b.get_if<int>());
    EXPECT_EQ(8, *b.get_if<int>());

    S c(a);
    EXPECT_FALSE(a.null());
    ASSERT_FALSE(c.null());
    ASSERT_NE(nullptr, c.get_if<int>());
    EXPECT_EQ(8, *c.get_if<int>());

    a.Destroy();
    EXPECT_TRUE(a.null());
    EXPECT_EQ(nullptr, a.get_if<int>());
  }
}

}  // namespace
