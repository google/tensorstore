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

#include "tensorstore/internal/tagged_ptr.h"

#include <memory>
#include <type_traits>

#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"

namespace {
using ::tensorstore::internal::const_pointer_cast;
using ::tensorstore::internal::dynamic_pointer_cast;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::static_pointer_cast;
using ::tensorstore::internal::TaggedPtr;

struct alignas(8) X {
  virtual ~X() = default;
};
struct Y : public X {
  virtual ~Y() = default;
};

static_assert(!std::is_convertible_v<TaggedPtr<Y, 1>, TaggedPtr<Y, 2>>);
static_assert(std::is_convertible_v<TaggedPtr<Y, 2>, TaggedPtr<X, 2>>);
static_assert(!std::is_convertible_v<TaggedPtr<Y, 1>, TaggedPtr<X, 2>>);
static_assert(std::is_convertible_v<Y*, TaggedPtr<X, 2>>);
static_assert(!std::is_convertible_v<TaggedPtr<Y, 2>, TaggedPtr<Y, 1>>);
static_assert(!std::is_convertible_v<TaggedPtr<X, 2>, TaggedPtr<Y, 2>>);
static_assert(!std::is_convertible_v<TaggedPtr<X, 2>, TaggedPtr<Y, 1>>);
static_assert(!std::is_convertible_v<X*, TaggedPtr<Y, 2>>);

static_assert(std::is_assignable_v<TaggedPtr<X, 2>, TaggedPtr<Y, 2>>);
static_assert(!std::is_assignable_v<TaggedPtr<X, 2>, TaggedPtr<X, 1>>);
static_assert(!std::is_assignable_v<TaggedPtr<X, 2>, TaggedPtr<Y, 1>>);
static_assert(!std::is_assignable_v<TaggedPtr<Y, 2>, TaggedPtr<Y, 3>>);
static_assert(!std::is_assignable_v<TaggedPtr<Y, 2>, TaggedPtr<X, 2>>);
static_assert(!std::is_assignable_v<TaggedPtr<Y, 2>, TaggedPtr<X, 3>>);

TEST(TaggedPtr, DefaultConstruct) {
  TaggedPtr<X, 3> p;
  EXPECT_EQ(nullptr, p.get());
  EXPECT_EQ(0u, p.tag());
}

TEST(TaggedPtr, Construct) {
  X x;
  TaggedPtr<X, 3> p(&x, 5);
  EXPECT_EQ(&x, p.get());
  EXPECT_EQ(5u, p.tag());
}

TEST(TaggedPtr, ConstructNullptr) {
  TaggedPtr<X, 3> p(nullptr, 5);
  EXPECT_EQ(nullptr, p.get());
  EXPECT_EQ(5u, p.tag());
}

TEST(TaggedPtr, CopyConstruct) {
  X x;
  TaggedPtr<X, 3> p(&x, 5);
  TaggedPtr<X, 3> p2(p);
  EXPECT_EQ(&x, p2.get());
  EXPECT_EQ(&x, p.get());
  EXPECT_EQ(5u, p.tag());
  EXPECT_EQ(5u, p2.tag());
}

TEST(TaggedPtr, CopyAssignTaggedPtr) {
  X x;
  TaggedPtr<X, 3> p(&x, 5);
  TaggedPtr<X, 3> p2;
  p2 = p;
  EXPECT_EQ(&x, p2.get());
  EXPECT_EQ(&x, p.get());
  EXPECT_EQ(5u, p2.tag());
  EXPECT_EQ(5u, p.tag());
}

TEST(TaggedPtr, CopyAssignPointer) {
  X x;
  TaggedPtr<X, 3> p(nullptr, 5);
  p = &x;
  EXPECT_EQ(&x, p.get());
  EXPECT_EQ(0u, p.tag());
}

TEST(TaggedPtr, CopyAssignNullptr) {
  X x;
  TaggedPtr<X, 3> p(&x, 5);
  p = nullptr;
  EXPECT_EQ(nullptr, p.get());
  EXPECT_EQ(0u, p.tag());
}

TEST(TaggedPtr, GetAndSetTag) {
  X x;
  TaggedPtr<X, 3> p(&x, 3);
  EXPECT_EQ(3u, p.tag());
  p.set_tag(4);
  EXPECT_EQ(4u, p.tag());
  EXPECT_TRUE(p.tag<2>());
  EXPECT_FALSE(p.tag<0>());
  EXPECT_FALSE(p.tag<1>());
  p.set_tag<0>(true);
  EXPECT_EQ(5u, p.tag());
  p.set_tag<2>(false);
  EXPECT_EQ(1u, p.tag());
}

TEST(TaggedPtr, TagComparison) {
  X x;
  X x2;
  TaggedPtr<X, 2> p(&x, 3);
  TaggedPtr<X, 2> p2(&x, 1);
  TaggedPtr<X, 2> p3(&x2, 3);

  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
  EXPECT_NE(p, p3);
}

TEST(TaggedPtr, StaticPointerCast) {
  Y y;
  TaggedPtr<X, 3> p(&y, 5);
  TaggedPtr<Y, 3> p2 = static_pointer_cast<Y>(p);
  EXPECT_EQ(&y, p2.get());
  EXPECT_EQ(5u, p2.tag());
}

TEST(TaggedPtr, ConstPointerCast) {
  X x;
  TaggedPtr<const X, 3> p(&x, 5);
  TaggedPtr<X, 3> p2 = const_pointer_cast<X>(p);
  EXPECT_EQ(&x, p2.get());
  EXPECT_EQ(5u, p2.tag());
}

TEST(TaggedPtr, DynamicPointerCastSuccess) {
  Y y;
  TaggedPtr<X, 3> p(&y, 5);
  TaggedPtr<Y, 3> p2 = dynamic_pointer_cast<Y>(p);
  EXPECT_EQ(&y, p2.get());
  EXPECT_EQ(5u, p2.tag());
}

TEST(TaggedPtr, DynamicPointerCastFailure) {
  X x;
  TaggedPtr<X, 3> p(&x, 5);
  TaggedPtr<Y, 3> p2 = dynamic_pointer_cast<Y>(p);
  EXPECT_EQ(nullptr, p2.get());
  EXPECT_EQ(5u, p2.tag());
}

struct alignas(8) X2 : public tensorstore::internal::AtomicReferenceCount<X2> {
  int value;
  virtual ~X2() = default;
};
struct Y2 : public X2 {
  virtual ~Y2() = default;
};

template <int TagBits>
struct TaggedIntrusivePtrTraits
    : public tensorstore::internal::DefaultIntrusivePtrTraits {
  template <typename U>
  using pointer = TaggedPtr<U, TagBits>;
};

template <typename T, int TagBits>
using TaggedIntrusivePtr = IntrusivePtr<T, TaggedIntrusivePtrTraits<TagBits>>;

TEST(IntrusivePtrTest, Basic) {
  Y2* x = new Y2;
  TaggedIntrusivePtr<Y2, 3> p(x);
  EXPECT_EQ(1u, p->use_count());
  EXPECT_EQ(x, p.get().get());
  EXPECT_EQ(0u, p.get().tag());

  TaggedIntrusivePtr<Y2, 3> p2({x, 5});
  EXPECT_EQ(2u, p2->use_count());
  EXPECT_EQ(x, p2.get().get());
  EXPECT_EQ(5u, p2.get().tag());

  TaggedIntrusivePtr<const X2, 3> p3 = p2;
  EXPECT_EQ(3u, p3->use_count());
  EXPECT_EQ(x, p3.get().get());
  EXPECT_EQ(5u, p3.get().tag());

  auto p4 = static_pointer_cast<const Y2>(p3);
  static_assert(std::is_same_v<TaggedIntrusivePtr<const Y2, 3>, decltype(p4)>);
  EXPECT_EQ(4u, p4->use_count());
  EXPECT_EQ(x, p4.get().get());
  EXPECT_EQ(5u, p4.get().tag());
}

}  // namespace
