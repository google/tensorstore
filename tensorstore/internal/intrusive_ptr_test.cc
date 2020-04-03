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

/// Tests of intrusive_ptr.h.

#include "tensorstore/internal/intrusive_ptr.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/internal/memory.h"

namespace {
using tensorstore::internal::acquire_object_ref;
using tensorstore::internal::adopt_object_ref;
using tensorstore::internal::AtomicReferenceCount;
using tensorstore::internal::const_pointer_cast;
using tensorstore::internal::dynamic_pointer_cast;
using tensorstore::internal::IntrusivePtr;
using tensorstore::internal::static_pointer_cast;

namespace default_behavior {

struct X : public AtomicReferenceCount<X> {
  virtual ~X() = default;
};
struct Y : public X {
  virtual ~Y() = default;
};

TEST(IntrusivePtrTest, DefaultConstructor) {
  IntrusivePtr<X> p;
  EXPECT_EQ(p.get(), nullptr);
  EXPECT_EQ(p, p);
  EXPECT_EQ(p.get(), static_cast<X*>(nullptr));
  EXPECT_EQ(p, nullptr);
  EXPECT_EQ(nullptr, p);
}

TEST(IntrusivePtrTest, PointerConstructor) {
  X* x = new X;
  IntrusivePtr<X> p(x, acquire_object_ref);
  EXPECT_EQ(p.get(), x);
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(p, p);
  EXPECT_NE(p, nullptr);
  EXPECT_NE(nullptr, p);
  EXPECT_EQ(x, p.operator->());
  EXPECT_EQ(x, &*p);
}

TEST(IntrusivePtrTest, ConstructFromDerivedPointer) {
  IntrusivePtr<X> p(new Y);
}

TEST(IntrusivePtrTest, PointerConstructorNoAddRef) {
  X* x = new X;
  intrusive_ptr_increment(x);
  EXPECT_EQ(1, x->use_count());
  IntrusivePtr<X> p(x, adopt_object_ref);
  EXPECT_EQ(p.get(), x);
  EXPECT_EQ(1, x->use_count());
}

TEST(IntrusivePtrTest, CopyConstructorNonNull) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(p);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p.get());
  EXPECT_EQ(x, p2.get());
}

TEST(IntrusivePtrTest, CopyConstructorNull) {
  IntrusivePtr<X> p;
  IntrusivePtr<X> p2(p);
  EXPECT_EQ(nullptr, p.get());
  EXPECT_EQ(nullptr, p2.get());
}

TEST(IntrusivePtrTest, MoveConstructorNonNull) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(std::move(p));
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(x, p2.get());
  EXPECT_EQ(nullptr, p.get());  // NOLINT
}

TEST(IntrusivePtrTest, MoveConstructorNull) {
  IntrusivePtr<X> p;
  IntrusivePtr<X> p2(std::move(p));
  EXPECT_EQ(nullptr, p.get());  // NOLINT
  EXPECT_EQ(nullptr, p2.get());
}

TEST(IntrusivePtrTest, ConvertingCopyConstructorNonNull) {
  Y* y = new Y;
  IntrusivePtr<Y> p(y);
  IntrusivePtr<X> p2(p);
  EXPECT_EQ(2, y->use_count());
  EXPECT_EQ(y, p2.get());
  EXPECT_EQ(y, p.get());
}

TEST(IntrusivePtrTest, ConvertingMoveConstructorNonNull) {
  Y* y = new Y;
  IntrusivePtr<Y> p(y);
  IntrusivePtr<X> p2(std::move(p));
  EXPECT_EQ(1, y->use_count());
  EXPECT_EQ(y, p2.get());
  EXPECT_EQ(nullptr, p.get());  // NOLINT
}

TEST(IntrusivePtrTest, ConvertingCopyConstructorNull) {
  IntrusivePtr<Y> p;
  IntrusivePtr<X> p2(p);
  EXPECT_EQ(nullptr, p2.get());
  EXPECT_EQ(nullptr, p.get());
}

TEST(IntrusivePtrTest, ConvertingMoveConstructorNull) {
  IntrusivePtr<Y> p;
  IntrusivePtr<X> p2(std::move(p));
  EXPECT_EQ(nullptr, p2.get());
  EXPECT_EQ(nullptr, p.get());  // NOLINT
}

TEST(IntrusivePtrTest, CopyAssignment) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  X* x2 = new X;
  IntrusivePtr<X> p3(x2);
  IntrusivePtr<X> p2(x2);
  p2 = p;
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p.get());
  EXPECT_EQ(x, p2.get());
  EXPECT_EQ(1, x2->use_count());
}

TEST(IntrusivePtrTest, CopyAssignmentSelf) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  auto& p_ref = p;  // Avoid compiler warning for self-assignment.
  p = p_ref;
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(x, p.get());
}

TEST(IntrusivePtrTest, MoveAssignment) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  X* x2 = new X;
  IntrusivePtr<X> p3(x2);
  IntrusivePtr<X> p2(x2);
  p2 = std::move(p);
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(nullptr, p.get());  // NOLINT
  EXPECT_EQ(x, p2.get());
  EXPECT_EQ(1, x2->use_count());
}

TEST(IntrusivePtrTest, MoveAssignmentSelf) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  auto& p_ref = p;
  p = std::move(p_ref);
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(x, p.get());
}

TEST(IntrusivePtrTest, ConvertingCopyAssignment) {
  Y* y = new Y;
  IntrusivePtr<Y> p(y);
  X* x2 = new X;
  IntrusivePtr<X> p3(x2);
  IntrusivePtr<X> p2(x2);
  p2 = p;
  EXPECT_EQ(2, y->use_count());
  EXPECT_EQ(y, p.get());
  EXPECT_EQ(y, p2.get());
  EXPECT_EQ(1, x2->use_count());
}

TEST(IntrusivePtrTest, ConvertingMoveAssignment) {
  Y* y = new Y;
  IntrusivePtr<Y> p(y);
  X* x2 = new X;
  IntrusivePtr<X> p3(x2);
  IntrusivePtr<X> p2(x2);
  p2 = std::move(p);
  EXPECT_EQ(1, y->use_count());
  EXPECT_EQ(nullptr, p.get());  // NOLINT
  EXPECT_EQ(y, p2.get());
  EXPECT_EQ(1, x2->use_count());
}

TEST(IntrusivePtrTest, Swap) {
  X* x = new X;
  X* x2 = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x2);
  p.swap(p2);
  EXPECT_EQ(x, p2.get());
  EXPECT_EQ(x2, p.get());
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(1, x2->use_count());
}

TEST(IntrusivePtrTest, BoolConversion) {
  IntrusivePtr<X> p;
  EXPECT_FALSE(static_cast<bool>(p));

  IntrusivePtr<X> p2(new X);
  EXPECT_TRUE(static_cast<bool>(p2));
}

TEST(IntrusivePtrTest, Detach) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p.release());
  EXPECT_EQ(nullptr, p.get());
  EXPECT_EQ(2, x->use_count());
  p.reset(x, adopt_object_ref);
}

TEST(IntrusivePtrTest, ResetNoArg) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x);
  EXPECT_EQ(2, x->use_count());
  p.reset();
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(nullptr, p.get());
}

TEST(IntrusivePtrTest, ResetNullptr) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x);
  EXPECT_EQ(2, x->use_count());
  p.reset(static_cast<X*>(nullptr));
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(nullptr, p.get());
}

TEST(IntrusivePtrTest, ResetPointerAddRef) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x);
  IntrusivePtr<X> p3(new X);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(1, p3->use_count());
  p.reset(p3.get());
  EXPECT_EQ(2, p3->use_count());
  EXPECT_EQ(p3.get(), p.get());
  EXPECT_EQ(1, x->use_count());
}

TEST(IntrusivePtrTest, ResetPointerNoAddRef) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x);
  IntrusivePtr<X> p3(new X);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(1, p3->use_count());
  p.reset(p3.get(), adopt_object_ref);
  EXPECT_EQ(1, p3->use_count());
  EXPECT_EQ(p3.get(), p.get());
  EXPECT_EQ(1, x->use_count());
  p.release();
}

TEST(IntrusivePtrTest, Comparison) {
  X* x = new X;
  X* x2 = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<X> p2(x2);
  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
  EXPECT_NE(p, nullptr);
  EXPECT_NE(nullptr, p);
}

TEST(IntrusivePtrTest, StaticPointerCast) {
  X* x = new Y;
  IntrusivePtr<X> p(x);
  IntrusivePtr<Y> p2 = static_pointer_cast<Y>(p);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p2.get());
}

TEST(IntrusivePtrTest, ConstPointerCast) {
  X* x = new X;
  IntrusivePtr<const X> p(x);
  IntrusivePtr<X> p2 = const_pointer_cast<X>(p);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p2.get());
}

TEST(IntrusivePtrTest, DynamicPointerCastSuccess) {
  X* x = new Y;
  IntrusivePtr<X> p(x);
  IntrusivePtr<Y> p2 = dynamic_pointer_cast<Y>(p);
  EXPECT_EQ(2, x->use_count());
  EXPECT_EQ(x, p2.get());
}

TEST(IntrusivePtrTest, DynamicPointerCastFailure) {
  X* x = new X;
  IntrusivePtr<X> p(x);
  IntrusivePtr<Y> p2 = dynamic_pointer_cast<Y>(p);
  EXPECT_EQ(1, x->use_count());
  EXPECT_EQ(nullptr, p2.get());
}

}  // namespace default_behavior

namespace custom_increment_decrement_functions {
class X {
 public:
  virtual ~X() = default;
  friend void intrusive_ptr_increment(X* p) { ++p->ref_count_; }
  friend void intrusive_ptr_decrement(X* p) {
    if (--p->ref_count_ == 0) {
      delete p;
    }
  }
  std::uint32_t ref_count_{0};
};

class Y : public X {
  // ...
};

TEST(IntrusivePtrTest, CustomIncrementDecrementFunctions) {
  IntrusivePtr<X> x1(new X);
  EXPECT_EQ(1, x1->ref_count_);

  IntrusivePtr<X> x2 = x1;
  EXPECT_EQ(2, x2->ref_count_);

  IntrusivePtr<Y> y1(new Y);
  IntrusivePtr<X> y2 = y1;
  IntrusivePtr<Y> y3 = dynamic_pointer_cast<Y>(y2);
  EXPECT_EQ(y2, y1);
  EXPECT_EQ(y3, y1);
}

}  // namespace custom_increment_decrement_functions

namespace custom_traits {

class X {
 public:
  // ...
  virtual ~X() = default;
  std::uint32_t ref_count_{0};
};

class Y : public X {
  // ...
};

struct XTraits {
  template <typename U>
  using pointer = U*;
  static void increment(X* p) noexcept { ++p->ref_count_; }
  static void decrement(X* p) noexcept {
    if (--p->ref_count_ == 0) delete p;
  }
};

TEST(IntrusivePtrTest, CustomTraits) {
  IntrusivePtr<X, XTraits> x1(new X);
  EXPECT_EQ(1, x1->ref_count_);

  IntrusivePtr<X, XTraits> x2 = x1;
  EXPECT_EQ(2, x2->ref_count_);

  IntrusivePtr<Y, XTraits> y1(new Y);
  IntrusivePtr<X, XTraits> y2 = y1;
  IntrusivePtr<Y, XTraits> y3 = dynamic_pointer_cast<Y>(y2);
  EXPECT_EQ(y2, y1);
  EXPECT_EQ(y3, y1);
}

}  // namespace custom_traits

}  // namespace
