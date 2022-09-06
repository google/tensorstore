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

#include "tensorstore/internal/poly/poly.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "tensorstore/util/result.h"

namespace {

using ::tensorstore::internal_poly::CallPolyApplyResult;
using ::tensorstore::internal_poly::HasPolyApply;
using ::tensorstore::internal_poly::IsCallPolyApplyResultConvertible;
using ::tensorstore::poly::Poly;

struct GetWidth {};
struct GetHeight {};
struct Scale {};
using PolyRectangle = Poly<sizeof(double), true, double(GetWidth) const,
                           double(GetHeight) const, void(Scale, double scalar)>;

struct Rectangle {
  double width;
  double height;
  double operator()(GetWidth) const { return width; }
  double operator()(GetHeight) const { return height; }
  void operator()(Scale, double scalar) {
    width *= scalar;
    height *= scalar;
  }
};
struct Square {
  double size;
  double operator()(GetWidth) const { return size; }
  double operator()(GetHeight) const { return size; }
};
// Define Scale operation on Square non-intrusively via PolyApply.
void PolyApply(Square& self, Scale, double scalar) { self.size *= scalar; }

template <typename T, typename P>
bool IsStoredInline(P& p) {
  auto min = reinterpret_cast<uintptr_t>(&p);
  auto t = reinterpret_cast<uintptr_t>(p.template target<T>());
  return t >= min && t <= (min + sizeof(p));
}

TEST(PolyTest, Example) {
  // No heap allocation because `sizeof(Square) <= sizeof(double)`.
  PolyRectangle square = Square{5};
  EXPECT_EQ(5, square(GetWidth{}));
  EXPECT_EQ(5, square(GetHeight{}));
  square(Scale{}, 2);
  EXPECT_EQ(10, square(GetWidth{}));
  EXPECT_EQ(10, square(GetHeight{}));

  // Heap-allocated because `sizeof(Rectangle) > sizeof(double)`.
  PolyRectangle rect = Rectangle{6, 7};
  EXPECT_EQ(6, rect(GetWidth{}));
  EXPECT_EQ(7, rect(GetHeight{}));
  rect(Scale{}, 2);
  EXPECT_EQ(12, rect(GetWidth{}));
  EXPECT_EQ(14, rect(GetHeight{}));
}

TEST(PolyTest, Interface) {
  // Using poly to define a golang-style interface type.
  // The drawback to this is that the methods have to be defined as operator()
  // overload sets, and Poly cannot detect named functions.
  class RectangleInterface {
   public:
    RectangleInterface(PolyRectangle poly) : poly(std::move(poly)) {}
    operator PolyRectangle() { return poly; }

    double GetHeight() const { return poly(::GetHeight{}); }
    double GetWidth() const { return poly(::GetWidth{}); }
    double GetArea() const { return GetHeight() * GetWidth(); }

    void Scale(double scalar) { poly(::Scale{}, scalar); }

   private:
    PolyRectangle poly;
  };

  // No heap allocation because `sizeof(Square) <= sizeof(double)`.
  {
    RectangleInterface rect(Square{5});
    EXPECT_EQ(5, rect.GetWidth());
    EXPECT_EQ(5, rect.GetHeight());
    EXPECT_EQ(25, rect.GetArea());
    rect.Scale(2);
    EXPECT_EQ(10, rect.GetWidth());
    EXPECT_EQ(10, rect.GetHeight());
  }

  // Heap-allocated because `sizeof(Rectangle) > sizeof(double)`.
  {
    RectangleInterface rect(Rectangle{6, 7});
    EXPECT_EQ(6, rect.GetWidth());
    EXPECT_EQ(7, rect.GetHeight());
    EXPECT_EQ(42, rect.GetArea());
    rect.Scale(2);
    EXPECT_EQ(12, rect.GetWidth());
    EXPECT_EQ(14, rect.GetHeight());
  }
}

std::string Foo(Poly<0, true, std::string()> poly) { return "Text: " + poly(); }
int Foo(Poly<0, true, int()> poly) { return 3 + poly(); }

TEST(PolyTest, ConstructorOverloadResolution) {
  EXPECT_EQ(6, Foo([] { return 3; }));
  EXPECT_EQ("Text: Message", Foo([] { return "Message"; }));
}

struct Add {
  std::shared_ptr<int> value;
  Add(std::shared_ptr<int> value) : value(value) {}
  template <typename T>
  T operator()(T x) const {
    return x + *value;
  }
};

TEST(PolyTest, DefaultConstruct) {
  Poly<1, true, int(int)&, float(float)&> poly;
  EXPECT_FALSE(poly);
  EXPECT_EQ(nullptr, poly.target<Add>());
  const auto& const_poly = poly;
  EXPECT_EQ(nullptr, const_poly.target<Add>());
}

TEST(PolyTest, NullptrConstruct) {
  Poly<1, true, int(int)&, float(float)&> poly(nullptr);
  EXPECT_FALSE(poly);
}

TEST(PolyTest, NullCopy) {
  Poly<1, true, int(int)&, float(float)&> poly;
  EXPECT_FALSE(poly);

  auto poly2 = poly;
  EXPECT_FALSE(poly2);
}

TEST(PolyTest, InlineConstruct) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_EQ(2, amount.use_count());
    EXPECT_TRUE(poly);

    EXPECT_TRUE(IsStoredInline<Add>(poly));

    auto* contained_obj = poly.target<Add>();
    ASSERT_NE(nullptr, contained_obj);
    EXPECT_EQ(amount, contained_obj->value);
    EXPECT_EQ(3, poly(2));
    EXPECT_EQ(3.5, poly(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, ConstructInplace) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<sizeof(Add), true, int(int)&, double(double)&> poly(
        std::in_place_type_t<Add>{}, amount);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_TRUE(poly);
    EXPECT_EQ(3, poly(2));
    EXPECT_EQ(3.5, poly(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, Emplace) {
  auto amount = std::make_shared<int>(1);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly;
  poly.emplace(Add{amount});
  EXPECT_TRUE(poly);
  EXPECT_EQ(2, amount.use_count());
  EXPECT_EQ(3, poly(2));
  EXPECT_EQ(3.5, poly(2.5));
  auto amount2 = std::make_shared<int>(2);
  poly.emplace(Add{amount2});
  EXPECT_TRUE(poly);
  EXPECT_EQ(1, amount.use_count());
  EXPECT_EQ(2, amount2.use_count());
  EXPECT_EQ(4, poly(2));
  EXPECT_EQ(4.5, poly(2.5));
}

TEST(PolyTest, EmplaceInplace) {
  auto amount = std::make_shared<int>(1);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly;
  poly.emplace<Add>(amount);
  EXPECT_TRUE(poly);
  EXPECT_EQ(2, amount.use_count());
  EXPECT_EQ(3, poly(2));
  EXPECT_EQ(3.5, poly(2.5));
  auto amount2 = std::make_shared<int>(2);
  poly.emplace<Add>(amount2);
  EXPECT_TRUE(poly);
  EXPECT_EQ(1, amount.use_count());
  EXPECT_EQ(2, amount2.use_count());
  EXPECT_EQ(4, poly(2));
  EXPECT_EQ(4.5, poly(2.5));
}

TEST(PolyTest, AssignNullptr) {
  auto amount = std::make_shared<int>(1);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
  EXPECT_EQ(2, amount.use_count());
  EXPECT_TRUE(poly);
  poly = nullptr;
  EXPECT_EQ(1, amount.use_count());
  EXPECT_FALSE(poly);
}

TEST(PolyTest, AssignObject) {
  auto amount = std::make_shared<int>(1);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
  EXPECT_TRUE(poly);
  EXPECT_EQ(2, amount.use_count());
  EXPECT_EQ(3, poly(2));
  EXPECT_EQ(3.5, poly(2.5));
  auto amount2 = std::make_shared<int>(2);
  poly = Add{amount2};
  EXPECT_TRUE(poly);
  EXPECT_EQ(1, amount.use_count());
  EXPECT_EQ(2, amount2.use_count());
  EXPECT_EQ(4, poly(2));
  EXPECT_EQ(4.5, poly(2.5));
}

TEST(PolyTest, CopyAssign) {
  auto amount = std::make_shared<int>(1);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
  EXPECT_TRUE(poly);
  EXPECT_EQ(2, amount.use_count());
  auto amount2 = std::make_shared<int>(2);
  Poly<sizeof(Add), true, int(int)&, double(double)&> poly2(Add{amount2});
  EXPECT_TRUE(poly2);
  EXPECT_EQ(2, amount2.use_count());
  poly2 =
      static_cast<const Poly<sizeof(Add), true, int(int)&, double(double)&>&>(
          poly);
  EXPECT_EQ(1, amount2.use_count());
  EXPECT_EQ(3, amount.use_count());
  EXPECT_EQ(3, poly(2));
  EXPECT_EQ(3.5, poly(2.5));
}

TEST(PolyTest, InlineMove) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_TRUE(poly);
    //    EXPECT_TRUE(poly.is_inline());
    EXPECT_EQ(2, amount.use_count());

    auto poly2 = std::move(poly);
    EXPECT_TRUE(poly2);
    //    EXPECT_TRUE(poly2.is_inline());
    EXPECT_FALSE(poly);  // NOLINT
    EXPECT_EQ(2, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, InlineCopy) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<sizeof(Add), true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_TRUE(poly);
    EXPECT_EQ(2, amount.use_count());

    EXPECT_TRUE(IsStoredInline<Add>(poly));

    auto poly2 = poly;
    EXPECT_TRUE(poly2);
    EXPECT_TRUE(IsStoredInline<Add>(poly2));

    EXPECT_TRUE(poly);  // NOLINT
    EXPECT_EQ(3, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, HeapConstruct) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<0, true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_TRUE(poly);

    EXPECT_TRUE(poly.target<Add>());
    EXPECT_FALSE(IsStoredInline<Add>(poly));

    EXPECT_EQ(amount, poly.target<Add>()->value);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_EQ(3, poly(2));
    EXPECT_EQ(3.5, poly(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, HeapMove) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<0, true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_TRUE(poly);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_FALSE(IsStoredInline<Add>(poly));

    auto poly2 = std::move(poly);
    EXPECT_TRUE(poly2);

    EXPECT_FALSE(poly);  // NOLINT
    EXPECT_EQ(2, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, HeapCopy) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<0, true, int(int)&, double(double)&> poly(Add{amount});
    EXPECT_TRUE(poly);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_FALSE(IsStoredInline<Add>(poly));

    auto poly2 = poly;
    EXPECT_TRUE(poly2);
    EXPECT_TRUE(poly);  // NOLINT
    EXPECT_EQ(3, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

struct AddPolyApply {
  std::shared_ptr<int> value;
  template <typename T>
  friend T PolyApply(const AddPolyApply& self, T x) {
    return x + *self.value;
  }
};

static_assert(HasPolyApply<AddPolyApply, int>);
static_assert(!HasPolyApply<AddPolyApply, int, int>);
static_assert(!HasPolyApply<Add, int>);
static_assert(!HasPolyApply<Add, int, int>);
static_assert(std::is_same_v<CallPolyApplyResult<AddPolyApply, int>, int>);
static_assert(
    std::is_same_v<CallPolyApplyResult<AddPolyApply, double>, double>);
static_assert(std::is_same_v<CallPolyApplyResult<Add, int>, int>);
static_assert(std::is_same_v<CallPolyApplyResult<Add, double>, double>);
static_assert(IsCallPolyApplyResultConvertible<Add, int, double>::value);
static_assert(IsCallPolyApplyResultConvertible<Add, double, double>::value);
static_assert(!IsCallPolyApplyResultConvertible<Add, int*, double>::value);
static_assert(IsCallPolyApplyResultConvertible<Add, void, double>::value);
static_assert(!IsCallPolyApplyResultConvertible<Add, void, int, int>::value);
static_assert(!IsCallPolyApplyResultConvertible<Add, int, int, int>::value);
static_assert(
    IsCallPolyApplyResultConvertible<AddPolyApply, int, double>::value);
static_assert(
    IsCallPolyApplyResultConvertible<AddPolyApply, double, double>::value);
static_assert(IsCallPolyApplyResultConvertible<AddPolyApply, void, int>::value);
static_assert(
    !IsCallPolyApplyResultConvertible<AddPolyApply, int*, int>::value);
static_assert(
    !IsCallPolyApplyResultConvertible<AddPolyApply, void, int, int>::value);

TEST(PolyTest, PolyApply) {
  auto amount = std::make_shared<int>(1);
  {
    Poly<sizeof(AddPolyApply), true, int(int)&, double(double)&> poly(
        AddPolyApply{amount});
    EXPECT_EQ(2, amount.use_count());
    EXPECT_TRUE(poly);
    EXPECT_EQ(3, poly(2));
    EXPECT_EQ(3.5, poly(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, MoveOnly) {
  struct Callable {
    std::unique_ptr<int> value;
    int operator()() const { return *value; }
  };

  using PolyT = Poly<sizeof(Callable), false, int() const>;
  static_assert(!std::is_constructible_v<Poly<0, true, int() const>, Callable>);
  static_assert(std::is_constructible_v<Poly<0, false, int() const>, Callable>);
  PolyT poly(Callable{std::unique_ptr<int>(new int(5))});
  auto poly2 = std::move(poly);
  EXPECT_FALSE(poly);  // NOLINT
  EXPECT_EQ(5, poly2());
}

struct IntGetterSetter {
  int operator()() { return value; }
  void operator()(int v) { value = v; }
  int value;
};

/// Tests that copy constructing from an existing Poly object with an
/// incompatible vtable results in a double-wrapped Poly.
TEST(PolyTest, CopyConstructFromPolyWithIncompatibleVTable) {
  auto amount = std::make_shared<int>(1);
  {
    using Poly1 = Poly<sizeof(Add), true, int(int)&, double(double)&>;
    using Poly2 = Poly<sizeof(Add), true, double(double)&, int(int)&>;
    Poly1 poly(Add{amount});
    EXPECT_EQ(2, amount.use_count());
    EXPECT_TRUE(poly.target<Add>());

    Poly2 poly2 = poly;
    EXPECT_TRUE(poly2);
    EXPECT_FALSE(poly2.target<Add>());
    EXPECT_TRUE(poly2.target<Poly1>());
    EXPECT_EQ(3, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

/// Tests that move constructing from an existing Poly object with an
/// incompatible vtable results in a double-wrapped Poly.
TEST(PolyTest, MoveConstructFromPolyWithIncompatibleVTable) {
  auto amount = std::make_shared<int>(1);
  {
    using Poly1 = Poly<sizeof(Add), true, int(int)&, double(double)&>;
    using Poly2 = Poly<sizeof(Add), true, double(double)&, int(int)&>;
    Poly1 poly(Add{amount});
    EXPECT_EQ(2, amount.use_count());

    Poly2 poly2 = std::move(poly);
    EXPECT_FALSE(poly);  // NOLINT
    EXPECT_FALSE(poly2.target<Add>());
    EXPECT_TRUE(poly2.target<Poly1>());
    EXPECT_TRUE(poly2);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, EmplaceFromPolyWithIncompatibleVTable) {
  auto amount = std::make_shared<int>(1);
  {
    using Poly1 = Poly<sizeof(Add), true, int(int)&, double(double)&>;
    using Poly2 = Poly<sizeof(Add), true, double(double)&, int(int)&>;
    Poly1 poly(Add{amount});
    EXPECT_EQ(2, amount.use_count());

    Poly2 poly2;
    poly2.emplace(std::move(poly));
    EXPECT_FALSE(poly);  // NOLINT
    EXPECT_FALSE(poly2.target<Add>());
    EXPECT_TRUE(poly2.target<Poly1>());
    EXPECT_TRUE(poly2);
    EXPECT_EQ(2, amount.use_count());
    EXPECT_EQ(3, poly2(2));
    EXPECT_EQ(3.5, poly2(2.5));
  }
  EXPECT_EQ(1, amount.use_count());
}

TEST(PolyTest, CopyConstructFromPolyWithCompatibleVTable) {
  Poly<0, true, void(int), int()> poly1 = IntGetterSetter{5};
  EXPECT_EQ(5, poly1());
  poly1(6);
  EXPECT_EQ(6, poly1());

  Poly<0, true, int()> poly2{poly1};
  EXPECT_TRUE(poly2.target<IntGetterSetter>());
  EXPECT_EQ(6, poly2());
}

TEST(PolyTest, MoveConstructFromPolyWithCompatibleVTable) {
  Poly<0, true, void(int), int()> poly1 = IntGetterSetter{5};
  EXPECT_EQ(5, poly1());
  poly1(6);
  EXPECT_EQ(6, poly1());

  Poly<0, true, int()> poly2{std::move(poly1)};
  EXPECT_TRUE(poly2.target<IntGetterSetter>());
  EXPECT_EQ(6, poly2());
  EXPECT_FALSE(poly1);  // NOLINT
}

TEST(PolyTest, EmplacePolyWithCompatibleVTable) {
  Poly<0, true, void(int), int()> poly1 = IntGetterSetter{5};
  EXPECT_EQ(5, poly1());
  poly1(6);
  EXPECT_EQ(6, poly1());

  Poly<0, true, int()> poly2;
  poly2.emplace(std::move(poly1));
  EXPECT_TRUE(poly2.target<IntGetterSetter>());
  EXPECT_EQ(6, poly2());
  EXPECT_FALSE(poly1);  // NOLINT
}

template <typename T>
using SinglePoly = Poly<0, false, T>;

template <template <typename> class OptionalLike,
          template <typename> class FunctionLike>
void TestAvoidsSfinaeLoop() {
  using Poly1 = FunctionLike<void()>;
  using Poly2 = FunctionLike<OptionalLike<Poly1>()>;

  struct X {
    void operator()() const {}
  };

  struct Y {
    OptionalLike<Poly1> operator()() const { return X{}; }
  };

  auto use_poly2 = [](Poly2) { /*noop*/ };
  use_poly2(Poly2{Y{}});
}

// Tests that a nested Poly/Result type avoids a SFINAE loop.
TEST(PolyTest, AvoidsSfinaeLoop) {
  TestAvoidsSfinaeLoop<tensorstore::Result, absl::FunctionRef>();
  TestAvoidsSfinaeLoop<tensorstore::Result, std::function>();
  TestAvoidsSfinaeLoop<std::optional, SinglePoly>();
  TestAvoidsSfinaeLoop<tensorstore::Result, SinglePoly>();
}

}  // namespace
