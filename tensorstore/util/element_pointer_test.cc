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

#include "tensorstore/util/element_pointer.h"

#include <memory>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/data_type.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::DataType;
using tensorstore::DataTypeOf;
using tensorstore::ElementPointer;
using tensorstore::ElementTagTraits;
using tensorstore::IsElementTag;
using tensorstore::MatchesStatus;
using tensorstore::PointerElementTag;
using tensorstore::Result;
using tensorstore::Shared;
using tensorstore::SharedElementPointer;
using tensorstore::StaticDataTypeCast;
using tensorstore::Status;

static_assert(IsElementTag<int>::value, "");
static_assert(IsElementTag<void>::value, "");
static_assert(IsElementTag<const void>::value, "");
static_assert(IsElementTag<int*>::value, "");
static_assert(IsElementTag<const int>::value, "");
static_assert(!IsElementTag<volatile int>::value, "");
static_assert(!IsElementTag<int(int)>::value, "");
static_assert(IsElementTag<int (*)(int)>::value, "");
static_assert(IsElementTag<Shared<int>>::value, "");
static_assert(!IsElementTag<const Shared<int>>::value, "");
static_assert(!IsElementTag<Shared<Shared<int>>>::value, "");
static_assert(!IsElementTag<Shared<const Shared<int>>>::value, "");
static_assert(!IsElementTag<Shared<const Shared<Shared<int>>>>::value, "");

static_assert(std::is_same<ElementTagTraits<int>::Pointer, int*>::value, "");
static_assert(std::is_same<ElementTagTraits<int>::rebind<float>, float>::value,
              "");
static_assert(std::is_same<ElementTagTraits<Shared<int>>::Pointer,
                           std::shared_ptr<int>>::value,
              "");
static_assert(std::is_same<ElementTagTraits<Shared<int>>::rebind<float>,
                           Shared<float>>::value,
              "");
static_assert(std::is_same<PointerElementTag<int*>, int>::value, "");
static_assert(
    std::is_same<PointerElementTag<std::shared_ptr<int>>, Shared<int>>::value,
    "");

// ElementPointer conversion tests
static_assert(
    std::is_convertible<ElementPointer<int>, ElementPointer<const int>>::value,
    "");
static_assert(
    !std::is_convertible<ElementPointer<const int>, ElementPointer<int>>::value,
    "");
static_assert(
    std::is_convertible<ElementPointer<int>, ElementPointer<void>>::value, "");
static_assert(
    !std::is_convertible<ElementPointer<void>, ElementPointer<int>>::value, "");
static_assert(
    !std::is_convertible<ElementPointer<const int>, ElementPointer<int>>::value,
    "");
static_assert(!std::is_convertible<ElementPointer<const int>,
                                   ElementPointer<void>>::value,
              "");
static_assert(std::is_convertible<ElementPointer<const int>,
                                  ElementPointer<const void>>::value,
              "");
static_assert(std::is_convertible<int*, ElementPointer<int>>::value, "");
static_assert(!std::is_convertible<const int*, ElementPointer<int>>::value, "");
static_assert(std::is_convertible<const int*, ElementPointer<const int>>::value,
              "");
static_assert(std::is_convertible<int*, ElementPointer<void>>::value, "");
static_assert(!std::is_convertible<const int*, ElementPointer<void>>::value,
              "");
static_assert(std::is_convertible<int*, ElementPointer<const int>>::value, "");
static_assert(std::is_convertible<int*, ElementPointer<const void>>::value, "");
static_assert(
    std::is_constructible<ElementPointer<void>, void*, DataType>::value, "");
static_assert(
    std::is_constructible<ElementPointer<const void>, void*, DataType>::value,
    "");

static_assert(!std::is_constructible<ElementPointer<void>, void*>::value, "");

// SharedElementPointer conversion tests
static_assert(std::is_convertible<SharedElementPointer<int>,
                                  SharedElementPointer<const int>>::value,
              "");
static_assert(!std::is_convertible<SharedElementPointer<const int>,
                                   SharedElementPointer<int>>::value,
              "");
static_assert(std::is_convertible<SharedElementPointer<int>,
                                  SharedElementPointer<void>>::value,
              "");
static_assert(!std::is_convertible<SharedElementPointer<void>,
                                   SharedElementPointer<int>>::value,
              "");
static_assert(!std::is_convertible<SharedElementPointer<const int>,
                                   SharedElementPointer<int>>::value,
              "");
static_assert(!std::is_convertible<SharedElementPointer<const int>,
                                   SharedElementPointer<void>>::value,
              "");
static_assert(std::is_convertible<SharedElementPointer<const int>,
                                  SharedElementPointer<const void>>::value,
              "");
static_assert(
    std::is_convertible<std::shared_ptr<int>, SharedElementPointer<int>>::value,
    "");
static_assert(std::is_convertible<std::shared_ptr<const int>,
                                  SharedElementPointer<const int>>::value,
              "");
static_assert(std::is_convertible<std::shared_ptr<int>,
                                  SharedElementPointer<void>>::value,
              "");
static_assert(!std::is_convertible<std::shared_ptr<const int>,
                                   SharedElementPointer<void>>::value,
              "");
static_assert(std::is_convertible<std::shared_ptr<int>,
                                  SharedElementPointer<const int>>::value,
              "");
static_assert(std::is_convertible<std::shared_ptr<int>,
                                  SharedElementPointer<const void>>::value,
              "");
static_assert(std::is_constructible<SharedElementPointer<void>,
                                    std::shared_ptr<void>, DataType>::value,
              "");
static_assert(std::is_constructible<SharedElementPointer<const void>,
                                    std::shared_ptr<void>, DataType>::value,
              "");

// SharedElementPointer -> ElementPointer conversion tests
static_assert(
    std::is_convertible<SharedElementPointer<int>, ElementPointer<int>>::value,
    "");
static_assert(std::is_convertible<SharedElementPointer<int>,
                                  ElementPointer<const int>>::value,
              "");
static_assert(!std::is_convertible<SharedElementPointer<void>,
                                   ElementPointer<const int>>::value,
              "");
static_assert(!std::is_constructible<ElementPointer<const int>,
                                     SharedElementPointer<void>>::value,
              "");
static_assert(!std::is_constructible<ElementPointer<int>,
                                     SharedElementPointer<const void>>::value,
              "");

TEST(ElementPointerTest, StaticType) {
  {
    ElementPointer<float> p_null;
    EXPECT_EQ(nullptr, p_null.data());
  }

  {
    ElementPointer<float> p_null = nullptr;
    EXPECT_EQ(nullptr, p_null.data());
  }

  float value;
  ElementPointer<float> p = &value;
  EXPECT_EQ(&value, p.data());
  EXPECT_EQ(&value, p.pointer());
  EXPECT_EQ(DataTypeOf<float>(), p.data_type());

  // Test implicit construction of `ElementPointer<const float>` from
  // `ElementPointer<float>`.
  {
    ElementPointer<const float> p_const = p;
    EXPECT_EQ(&value, p_const.data());
    EXPECT_EQ(DataTypeOf<float>(), p_const.data_type());
    p_const.pointer() = nullptr;
    EXPECT_EQ(nullptr, p_const.data());
  }

  {
    // Test copy construction of `ElementPointer<float>` from
    // `ElementPointer<float>`.
    ElementPointer<float> p_copy = p;
    EXPECT_EQ(&value, p_copy.data());
  }

  // Test implicit conversion to const void.
  ElementPointer<const void> other = p;
  EXPECT_EQ(&value, other.data());
  EXPECT_EQ(other.data_type(), p.data_type());

  // Test explicit conversion from another ElementPointer.
  {
    auto p2 = tensorstore::StaticDataTypeCast<const float>(other);
    static_assert(
        std::is_same<decltype(p2), Result<ElementPointer<const float>>>::value,
        "");
    ASSERT_EQ(Status(), GetStatus(p2));
    EXPECT_EQ(&value, p2->data());
  }

  // Test assignment from an ElementPointer<float>.
  {
    ElementPointer<const float> p_const;
    p_const = p;
    EXPECT_EQ(&value, p_const.data());

    p_const = nullptr;
    EXPECT_EQ(nullptr, p_const.data());
  }

  static_assert(!std::is_assignable<ElementPointer<float>,
                                    ElementPointer<const float>>::value,
                "");

  static_assert(
      !std::is_assignable<ElementPointer<int>, ElementPointer<float>>::value,
      "");

  static_assert(!std::is_assignable<ElementPointer<int>, float*>::value, "");
  static_assert(!std::is_assignable<ElementPointer<void>, void*>::value, "");

  static_assert(!std::is_assignable<ElementPointer<const float>,
                                    ElementPointer<const void>>::value,
                "");
  static_assert(!std::is_assignable<ElementPointer<float>, void*>::value, "");
}

TEST(ElementPointerTest, DynamicType) {
  {
    ElementPointer<void> p_null;
    EXPECT_EQ(nullptr, p_null.data());
    EXPECT_EQ(DataType(), p_null.data_type());
  }

  {
    ElementPointer<void> p_null = nullptr;
    EXPECT_EQ(nullptr, p_null.data());
    EXPECT_EQ(DataType(), p_null.data_type());
  }

  float value;
  {
    ElementPointer<void> p = &value;
    EXPECT_EQ(&value, p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }
  {
    ElementPointer<void> p = {static_cast<void*>(&value), DataTypeOf<float>()};
    EXPECT_EQ(&value, p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }

  static_assert(!std::is_assignable<ElementPointer<void>,
                                    ElementPointer<const float>>::value,
                "");
  static_assert(!std::is_assignable<ElementPointer<void>,
                                    SharedElementPointer<const float>>::value,
                "");

  static_assert(!std::is_assignable<ElementPointer<void>,
                                    ElementPointer<const void>>::value,
                "");
  static_assert(!std::is_assignable<ElementPointer<void>,
                                    SharedElementPointer<const void>>::value,
                "");
  static_assert(!std::is_assignable<ElementPointer<void>, void*>::value, "");
  static_assert(!std::is_assignable<ElementPointer<void>, const float*>::value,
                "");

  // Copy construction
  {
    ElementPointer<void> p;
    p = ElementPointer<float>(&value);

    ElementPointer<void> p_copy = p;
    EXPECT_EQ(&value, p_copy.data());
    EXPECT_EQ(DataTypeOf<float>(), p_copy.data_type());
  }

  // Copy assignment from `ElementPointer<void>`.
  {
    ElementPointer<void> p;
    p = ElementPointer<float>(&value);

    ElementPointer<void> p_copy;
    p_copy = p;
    EXPECT_EQ(&value, p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }

  // Assignment from ElementPointer<float>.
  {
    ElementPointer<void> p;
    p = ElementPointer<float>(&value);
    EXPECT_EQ(&value, p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }

  // Assignment from float*.
  {
    ElementPointer<void> p;
    p = &value;
    EXPECT_EQ(&value, p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }

  // Assignment from SharedElementPointer<float>.
  {
    ElementPointer<void> p;
    std::shared_ptr<float> shared_value = std::make_shared<float>();
    p = SharedElementPointer<float>(shared_value);
    EXPECT_EQ(shared_value.get(), p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }

  // Assignment from SharedElementPointer<float>.
  {
    ElementPointer<void> p;
    std::shared_ptr<float> shared_value = std::make_shared<float>();
    p = SharedElementPointer<void>(shared_value);
    EXPECT_EQ(shared_value.get(), p.data());
    EXPECT_EQ(DataTypeOf<float>(), p.data_type());
  }
}

TEST(ElementPointerTest, StaticDataTypeCast) {
  float value;
  EXPECT_THAT(StaticDataTypeCast<std::int32_t>(ElementPointer<void>(&value)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot cast pointer with data type of float32 to "
                            "pointer with data type of int32"));

  EXPECT_THAT(StaticDataTypeCast<std::int32_t>(
                  SharedElementPointer<void>(std::make_shared<float>())),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot cast pointer with data type of float32 to "
                            "pointer with data type of int32"));
}

TEST(SharedElementPointerTest, StaticType) {
  std::shared_ptr<float> value = std::make_shared<float>();
  SharedElementPointer<float> p = value;
  EXPECT_EQ(value.get(), p.data());
  EXPECT_EQ(DataTypeOf<float>(), p.data_type());

  {
    // Test copy construction of `SharedElementPointer<float>` from
    // `SharedElementPointer<float>`.
    SharedElementPointer<float> p_copy = p;
    EXPECT_EQ(value.get(), p_copy.data());

    // Test move construction of `SharedElementPointer<float>` from
    // `SharedElementPointer<float>`.
    SharedElementPointer<float> p_move = std::move(p_copy);
    EXPECT_EQ(value.get(), p_move.data());

    /// Test move construction of `SharedElementPointer<const float>` from
    /// `SharedElementPointer<float>`.
    SharedElementPointer<const float> p_const_move = std::move(p_move);
    EXPECT_EQ(value.get(), p_const_move.data());
  }

  {
    SharedElementPointer<const void> other = p;
    EXPECT_EQ(value.get(), other.data());
    EXPECT_EQ(other.data_type(), p.data_type());
    EXPECT_EQ(3, value.use_count());
  }
  EXPECT_EQ(2, value.use_count());

  // Test implicit conversion to ElementPointer of same type.
  {
    ElementPointer<float> x = p;
    EXPECT_EQ(value.get(), x.data());
  }

  // Test implicit conversion to ElementPointer with change in type.
  {
    ElementPointer<const void> x = p;
    EXPECT_EQ(value.get(), x.data());
  }

  // Test explicit conversion to ElementPointer.
  {
    SharedElementPointer<void> shared_p_void = p;
    auto p_float = StaticDataTypeCast<float>(shared_p_void).value();
    static_assert(
        std::is_same<decltype(p_float), SharedElementPointer<float>>::value,
        "");
    EXPECT_EQ(value.get(), p_float.data());
  }

  // Test UnownedToShared conversion.
  {
    float fvalue;
    auto f_pointer = UnownedToShared(ElementPointer<float>(&fvalue));
    static_assert(
        std::is_same<decltype(f_pointer), SharedElementPointer<float>>::value,
        "");
    EXPECT_EQ(&fvalue, f_pointer.data());
  }

  // Assignment from SharedElementPointer<float>.
  {
    SharedElementPointer<float> p2;
    EXPECT_TRUE(p2 == nullptr);
    EXPECT_TRUE(nullptr == p2);
    EXPECT_FALSE(p2 != nullptr);
    EXPECT_FALSE(nullptr != p2);
    EXPECT_FALSE(p2 == p);
    EXPECT_TRUE(p2 != p);
    p2 = p;
    EXPECT_EQ(value.get(), p2.data());
    EXPECT_FALSE(p2 == nullptr);
    EXPECT_FALSE(nullptr == p2);
    EXPECT_TRUE(p2 != nullptr);
    EXPECT_TRUE(nullptr != p2);
    EXPECT_TRUE(p2 == p);
    EXPECT_FALSE(p2 != p);
  }

  // Move assignment of `SharedElementPointer<float>` from
  // `SharedElementPointer<float>`.
  {
    SharedElementPointer<float> p2 = p;
    SharedElementPointer<float> p2_move;
    p2_move = std::move(p2);
    EXPECT_TRUE(p2 == nullptr);  // NOLINT
    EXPECT_EQ(value.get(), p2_move.data());
  }

  // Move assignment of `SharedElementPointer<const float>` from
  // `SharedElementPointer<float>`.
  {
    SharedElementPointer<float> p2 = p;
    SharedElementPointer<const float> p2_move;
    p2_move = std::move(p2);
    EXPECT_TRUE(p2 == nullptr);  // NOLINT
    EXPECT_EQ(value.get(), p2_move.data());
  }

  // Move assignment of `SharedElementPointer<float>` from `pointer()`.
  {
    SharedElementPointer<float> p2 = p;
    SharedElementPointer<float> p2_move;
    p2_move = std::move(p2.pointer());
    EXPECT_TRUE(p2 == nullptr);
    EXPECT_EQ(value.get(), p2_move.data());
  }

  static_assert(!std::is_assignable<SharedElementPointer<float>,
                                    SharedElementPointer<void>>::value,
                "");
  static_assert(!std::is_assignable<SharedElementPointer<float>,
                                    std::shared_ptr<void>>::value,
                "");
}

TEST(SharedElementPointerTest, DynamicType) {
  std::shared_ptr<float> value = std::make_shared<float>();
  SharedElementPointer<void> p = value;
  EXPECT_EQ(value.get(), p.data());
  EXPECT_EQ(DataTypeOf<float>(), p.data_type());
}

TEST(ElementPointerTest, Deduction) {
  int* raw_int_ptr;
  std::shared_ptr<int> shared_int_ptr;
  ElementPointer<int> el_ptr;
  ElementPointer<void> el_void_ptr;
  SharedElementPointer<int> shared_el_ptr;
  SharedElementPointer<void> shared_void_el_ptr;

  {
    auto x = ElementPointer(raw_int_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<int>>);
  }
  {
    auto x = ElementPointer(shared_int_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<Shared<int>>>);
  }
  {
    auto x = ElementPointer(el_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<int>>);
  }
  {
    auto x = ElementPointer(el_void_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<void>>);
  }
  {
    auto x = ElementPointer(shared_el_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<Shared<int>>>);
  }
  {
    auto x = ElementPointer(shared_void_el_ptr);
    static_assert(std::is_same_v<decltype(x), ElementPointer<Shared<void>>>);
  }
}

}  // namespace
