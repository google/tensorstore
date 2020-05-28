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

#include "tensorstore/util/result.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::ChainResult;
using tensorstore::FlatMapResultType;
using tensorstore::FlatResult;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::UnwrapQualifiedResultType;
using tensorstore::UnwrapResultType;

static_assert(std::is_convertible<Result<int>, Result<float>>::value, "");

static_assert(!std::is_convertible<Result<int>, Result<std::string>>::value,
              "");

static_assert(std::is_same<int, Result<int>::value_type>::value, "");

TEST(ResultTest, ConstructDefault) {
  Result<int> result{absl::in_place};
  EXPECT_TRUE(result.has_value());
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result);
}

TEST(ResultTest, ConstructValue) {
  Result<int> result(3);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result.has_value()) << result.status().ToString();
  EXPECT_TRUE(result) << result.status().ToString();
  EXPECT_EQ(3, *result);
  // NO result.status()
}

TEST(ResultDeathTest, ConstructSuccess) {
  tensorstore::Status status;
  ASSERT_DEATH(Result<int> result(status), "status");
}

TEST(ResultDeathTest, AssignSuccess) {
  // Cannot get blood from a stone.
  tensorstore::Status status;
  Result<int> result{absl::in_place};
  ASSERT_DEATH(result = status, "status");
}

TEST(ResultDeathTest, ConstructDefaultHasNoStatus) {
  Result<int> result{absl::in_place};
  ASSERT_DEATH(result.status(), "has_value");
}

TEST(ResultDeathTest, ConstructStatusHasNoValue) {
  // Cannot get blood from a stone.
  tensorstore::Status status(absl::StatusCode::kUnknown,
                             "My custom error message");
  Result<int> result(status);

  ASSERT_DEATH(result.value(), "");
  ASSERT_DEATH(static_cast<const Result<int>&>(result).value(), "");
  ASSERT_DEATH(std::move(result).value(), "My custom error message");
}

TEST(ResultDeathTest, ValidResultChecksOnStatus) {
  // Cannot get blood from a stone.
  Result<int> result = absl::UnknownError("Message");
  EXPECT_FALSE(result.has_value());

  result = 3;
  EXPECT_TRUE(result.has_value());

  ASSERT_DEATH(result.status(), "");
}

TEST(ResultTest, ConstructStatus) {
  tensorstore::Status status(absl::StatusCode::kUnknown, "Message");
  Result<int> result(status);
  EXPECT_FALSE(result);
  EXPECT_FALSE(result.ok());

  EXPECT_EQ(status, result.status());
}

TEST(ResultTest, ConstructStatusMove) {
  tensorstore::Status status(absl::StatusCode::kUnknown, "Message");
  auto message = status.message();  // string_view
  Result<int> result(std::move(status));
  EXPECT_EQ(message.data(), result.status().message().data());
}

TEST(ResultTest, ConstructValueMove) {
  std::unique_ptr<int> value(new int(3));
  Result<std::unique_ptr<int>> result(std::move(value));
  EXPECT_FALSE(value);
  EXPECT_TRUE(result);
  EXPECT_TRUE(result.value());
  EXPECT_EQ(3, *result.value().get());

  Result<std::unique_ptr<int>> result2(std::move(result));
  EXPECT_TRUE(result);  // NOLINT
  EXPECT_TRUE(result2);
  EXPECT_FALSE(result.value());
  EXPECT_TRUE(result2.value());
  EXPECT_EQ(3, *result2.value().get());
}

TEST(ResultTest, ConstructConvertMove) {
  Result<std::unique_ptr<int>> result(std::unique_ptr<int>(new int(3)));
  Result<std::shared_ptr<int>> result2(std::move(result));
  EXPECT_FALSE(result.value());  // NOLINT
  EXPECT_TRUE(result2.value());
  EXPECT_EQ(3, *result2.value());
}

TEST(ResultTest, ConstructConvertCopy) {
  Result<int> result(2);
  Result<float> result2(result);
  EXPECT_EQ(2.0f, result2.value());
}

TEST(ResultTest, ConstructCopySuccess) {
  Result<int> a(3);
  EXPECT_TRUE(a.ok());
  Result<int> b(a);
  EXPECT_TRUE(b);
  EXPECT_EQ(3, b.value());
}

TEST(ResultTest, ConstructCopyFailure) {
  Result<int> a(absl::UnknownError(""));
  Result<int> b(a);
  EXPECT_FALSE(b);
  EXPECT_EQ(absl::StatusCode::kUnknown, b.status().code());
}

TEST(ResultTest, Comparison) {
  // Compare with value.
  const Result<int> r(1);
  EXPECT_EQ(true, r == Result<float>(1));
  EXPECT_EQ(false, r != Result<float>(1));
  EXPECT_EQ(false, Result<int>(2) == Result<float>(1));
  EXPECT_EQ(true, Result<int>(2) != Result<float>(1));
  EXPECT_EQ(true, Result<int>{absl::in_place} == Result<int>{absl::in_place});
  EXPECT_EQ(false, Result<int>{absl::in_place} == r);
  EXPECT_EQ(false, Result<int>{absl::in_place} == r);

  Result<int> err{absl::UnknownError("Message")};
  Result<int> err2 = err;

  // Compare with error.
  EXPECT_EQ(true, err == err2);
  EXPECT_EQ(false, err != err2);
  EXPECT_EQ(false, err == Result<int>(1));
  EXPECT_EQ(true, err != Result<int>(1));
  EXPECT_EQ(false, err == Result<int>(absl::in_place));
  EXPECT_EQ(true, err != Result<int>(absl::in_place));

  // Compare with values.
  EXPECT_EQ(true, r == 1);
  EXPECT_EQ(false, r == 2);
  EXPECT_EQ(false, r != 1);
  EXPECT_EQ(true, r != 2);

  // Compare Result<void>
  const Result<void> rv = tensorstore::MakeResult();
  Result<void> err3 = absl::UnknownError("Message");
  EXPECT_TRUE(rv == rv);
  EXPECT_FALSE(rv != rv);
  EXPECT_TRUE(err3 == err3);
  EXPECT_FALSE(err3 != err3);
  EXPECT_FALSE(rv == err3);
  EXPECT_FALSE(err3 == rv);
  EXPECT_TRUE(rv != err3);
  EXPECT_TRUE(err3 != rv);
}

TEST(ResultTest, AssignMoveFailure) {
  Result<int> result(absl::UnknownError("Hello"));
  auto message = result.status().message();
  Result<int> result2{absl::in_place};
  result2 = std::move(result);
  EXPECT_FALSE(result2);
  EXPECT_EQ(message.data(), result2.status().message().data());

  Result<int> other(absl::in_place);
  result2 = std::move(other);
  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(0, *result2);
}

TEST(ResultTest, AssignMoveSuccess) {
  Result<std::unique_ptr<int>> result(std::unique_ptr<int>(new int(3)));
  Result<std::unique_ptr<int>> result2{absl::UnknownError("")};
  result2 = std::move(result);
  EXPECT_TRUE(result2);
  EXPECT_TRUE(result2.value());
  EXPECT_EQ(3, *result2.value());
}

TEST(ResultTest, AssignCopyFailure) {
  Result<int> result(absl::UnknownError("Hello"));
  Result<int> result2{absl::in_place};
  result2 = result;
  EXPECT_EQ(result2, result);
  EXPECT_FALSE(result2);
  EXPECT_EQ(result2.status(), result.status());

  result2 = Result<int>(absl::in_place);
  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(0, *result2);
}

TEST(ResultTest, AssignCopySuccess) {
  Result<int> result(3);
  Result<int> result2{absl::in_place};
  result2 = result;
  EXPECT_EQ(result2, result);
  EXPECT_TRUE(result2);
  EXPECT_EQ(3, result2.value());
}

TEST(ResultTest, ConvertingValueConstructor) {
  float a = 3.0f;
  const tensorstore::Status b(absl::StatusCode::kUnknown, "Hello");

  Result<double> c(a);
  EXPECT_TRUE(c);
  c = b;
  EXPECT_FALSE(c);

  Result<double> d(b);
  EXPECT_FALSE(c);
  d = a;
  EXPECT_TRUE(d);

  d = b;
  d = std::move(a);
  EXPECT_TRUE(d);

  a = 4.0f;
  Result<double> e(std::move(a));
  EXPECT_TRUE(e);
}

TEST(ResultTest, ConvertingConstructor) {
  Result<float> a(3.0);
  const Result<float> b(absl::UnknownError("Hello"));

  Result<double> c(a);
  EXPECT_TRUE(c);
  c = b;
  EXPECT_FALSE(c);

  Result<double> d(b);
  EXPECT_FALSE(c);
  d = a;
  EXPECT_TRUE(d);

  d = b;
  d = std::move(a);
  EXPECT_TRUE(d);

  Result<float> aa(4.0);
  Result<double> e(std::move(aa));
  EXPECT_TRUE(e);
}

TEST(ResultTest, Swap) {
  using std::swap;

  tensorstore::Status status(absl::StatusCode::kUnknown, "Message");

  Result<int> result(status);
  Result<int> result2(3);

  swap(result, result2);

  EXPECT_TRUE(result.has_value());
  EXPECT_FALSE(result2.has_value());

  ASSERT_EQ(3, result.value());
  ASSERT_EQ(status, result2.status());
}

TEST(ResultTest, InitializerList) {
  Result<std::vector<int>> a(tensorstore::in_place, {1, 2, 3});
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(3, a.value()[2]);
}

struct Aggregate {
  int x, y, z;
};

TEST(ResultTest, Aggregate) {
  // NOTE: c++ does not treate Aggregate initialization as a constructor,
  // so we cannot use emplace or in_place_t constructors here.
  Result<Aggregate> a(Aggregate{1, 2, 3});
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(3, a.value().z);

  Result<Aggregate> b = std::move(a);
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(3, b.value().z);

  a = Aggregate{4, 5, 6};
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(6, a.value().z);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign"
#endif

TEST(ResultTest, AssignCopySelf) {
  Result<int> result(3);
  result = result;
  EXPECT_EQ(3, result.value());
}

TEST(ResultTest, AssignCopySelf2) {
  auto ptr = std::make_shared<int>(5);
  EXPECT_EQ(1, ptr.use_count());

  Result<std::shared_ptr<int>> result(ptr);
  EXPECT_EQ(2, ptr.use_count());
  result = result;
  EXPECT_EQ(2, ptr.use_count());
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

TEST(ResultTest, AssignMoveConvert) {
  Result<std::unique_ptr<int>> result(std::unique_ptr<int>(new int(3)));

  auto ptr = std::make_shared<int>(5);
  Result<std::shared_ptr<int>> result2(ptr);

  result2 = std::move(result);
  EXPECT_FALSE(result.value());  // NOLINT
  EXPECT_TRUE(result2.value());
  EXPECT_EQ(1, ptr.use_count());
  EXPECT_EQ(3, *result2.value());
}

TEST(ResultTest, AssignCopyConvert) {
  struct Base {
    int value;
    Base(int value) : value(value) {}
  };
  struct Derived : Base {
    using Base::Base;
  };
  Result<std::shared_ptr<Derived>> result(
      std::shared_ptr<Derived>(new Derived{3}));

  auto ptr = std::make_shared<Base>(5);
  Result<std::shared_ptr<Base>> result2(ptr);

  result2 = result;
  EXPECT_TRUE(result.value());
  EXPECT_TRUE(result2.value());
  EXPECT_EQ(1, ptr.use_count());
  EXPECT_EQ(3, result2.value()->value);
}

TEST(ResultTest, AssignValue) {
  auto ptr = std::make_shared<int>(5);
  Result<std::shared_ptr<int>> result(ptr);

  auto ptr2 = std::make_shared<int>(3);
  result = ptr2;

  EXPECT_TRUE(result.value());
  EXPECT_EQ(1, ptr.use_count());
  EXPECT_EQ(2, ptr2.use_count());
}

TEST(ResultTest, AssignStatus) {
  Result<std::shared_ptr<int>> result(tensorstore::in_place, nullptr);

  EXPECT_TRUE(result);

  result = absl::UnknownError("Hello");

  EXPECT_FALSE(result);
  EXPECT_EQ(absl::UnknownError("Hello"), result.status());
}

TEST(ResultTest, AssignErrorCallsDtor) {
  auto ptr = std::make_shared<int>(5);
  EXPECT_EQ(1, ptr.use_count());

  Result<std::shared_ptr<int>> result(ptr);
  EXPECT_EQ(2, ptr.use_count());
  result = absl::UnknownError("");

  EXPECT_FALSE(result);
  EXPECT_EQ(1, ptr.use_count());

  EXPECT_EQ(absl::UnknownError(""), result.status());
}

TEST(ResultTest, Error) {
  Result<int> result = absl::UnknownError("");
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnknown);
}

TEST(ResultTest, Message) {
  Result<int> result(absl::UnknownError("Hello"));
  EXPECT_EQ("Hello", result.status().message());
  EXPECT_EQ("Hello", std::move(result).status().message());
}

TEST(ResultTest, Value) {
  static_assert(
      std::is_same<decltype(std::declval<const Result<int>&>().value()),
                   const int&>::value,
      "");
  static_assert(
      std::is_same<decltype(std::declval<Result<int>&>().value()), int&>::value,
      "");
  static_assert(
      std::is_same<decltype(std::declval<Result<int>&&>().value()), int>::value,
      "");

  static_assert(std::is_same<decltype(*std::declval<const Result<int>&>()),
                             const int&>::value,
                "");
  static_assert(
      std::is_same<decltype(*std::declval<Result<int>&>()), int&>::value, "");

  static_assert(
      std::is_same<decltype(std::declval<const Result<int>&>().operator->()),
                   const int*>::value,
      "");
  static_assert(
      std::is_same<decltype(std::declval<Result<int>&>().operator->()),
                   int*>::value,
      "");

  Result<int> result = 3;
  EXPECT_EQ(3, result.value());
  EXPECT_EQ(3, *result);
  EXPECT_EQ(3, result.value_or(4));
  EXPECT_EQ(3, static_cast<const Result<int>&>(result).value());
  EXPECT_EQ(3, std::move(result).value());

  std::vector<int> vec{1, 2, 3};
  const int* data = vec.data();
  Result<std::vector<int>> result2(std::move(vec));
  EXPECT_EQ(3, result2->size());
  EXPECT_EQ(3, static_cast<const Result<std::vector<int>>&>(result2)->size());
  std::vector<int> vec2 = std::move(result2).value();
  EXPECT_EQ(data, vec2.data());
}

TEST(ResultTest, Emplace) {
  {
    Result<int> result = absl::UnknownError("");
    result.emplace(3);
    EXPECT_TRUE(result);
    EXPECT_EQ(3, result.value());
  }

  {
    Result<std::vector<int>> result = absl::UnknownError("");
    result.emplace({3, 4, 5});
    EXPECT_TRUE(result);
    std::vector<int> x = std::move(*result);
    EXPECT_EQ(3, x.size());
  }
}

TEST(ResultTest, ValueOrInt) {
  {
    Result<int> result = absl::UnknownError("");
    EXPECT_EQ(3, std::move(result).value_or(3));
  }

  {
    Result<int> result = absl::UnknownError("");
    EXPECT_EQ(3, result.value_or(3));
    result = 4;
    EXPECT_EQ(4, result.value_or(3));

    int x = std::move(result).value_or(5);
    EXPECT_EQ(4, x);
  }
}

TEST(ResultTest, ValueOrVectorInt) {
  Result<std::vector<int>> result = absl::UnknownError("");

  std::vector<int> a({4, 5, 6});
  EXPECT_EQ(a, result.value_or(a));

  std::vector<int> b({3, 2, 1});
  result = a;

  EXPECT_EQ(a, result.value_or(b));
}

TEST(ResultTest, ConvertAssignmentWithCopy) {
  Result<int> err{absl::UnknownError("")};
  Result<int> a(123);

  // Constructor
  {
    Result<std::int64_t> c(a);
    EXPECT_TRUE(c.has_value());
    EXPECT_EQ(123, c.value());
  }
  {
    Result<std::int64_t> c(err);
    EXPECT_FALSE(c.has_value());
  }

  // Assignment
  {
    Result<std::int64_t> b = a;
    EXPECT_TRUE(b.has_value());
    EXPECT_EQ(123, b.value());
  }
  {
    Result<std::int64_t> b = err;
    EXPECT_FALSE(b.has_value());
  }
}

TEST(ResultTest, ConvertAssignmentWithMove) {
  {
    Result<int> a{absl::UnknownError("")};
    Result<std::int64_t> b = std::move(a);
    EXPECT_FALSE(b.has_value());
  }
  {
    Result<int> a{absl::UnknownError("")};
    Result<std::int64_t> c(std::move(a));
    EXPECT_FALSE(c.has_value());
  }
}

TEST(UnwrapResult, Basic) {
  EXPECT_EQ(3, tensorstore::UnwrapResult(3));
  EXPECT_EQ(3, tensorstore::UnwrapResult(Result<int>(3)));
  EXPECT_EQ(3, tensorstore::UnwrapResult(Result<int>(3)));

  const Result<int> r(3);
  EXPECT_EQ(3, tensorstore::UnwrapResult(r));
}

TEST(ResultVoidTest, MoveConstructFromStatus) {
  Result<void> r = absl::UnknownError("C");
  EXPECT_EQ(absl::UnknownError("C"), r.status());
}

TEST(ResultVoidTest, CopyConstructFromStatus) {
  Status s = absl::UnknownError("C");
  Result<void> r = s;
  EXPECT_EQ(absl::UnknownError("C"), r.status());
}

TEST(ResultVoidTest, CopyAssignFromStatus) {
  Status s = absl::UnknownError("C");
  Result<void> t(absl::in_place);
  Result<void> r{absl::in_place};
  r = s;
  EXPECT_EQ(absl::UnknownError("C"), r.status());
  r = t;
  EXPECT_TRUE(r.has_value());
  r = t;
  EXPECT_TRUE(r.has_value());
}

TEST(ResultVoidTest, MoveAssignFromStatus) {
  Result<void> r{absl::in_place};
  r = absl::UnknownError("C");
  EXPECT_EQ(absl::UnknownError("C"), r.status());
  r = Result<void>(absl::in_place);
  EXPECT_TRUE(r.has_value());
  r = Result<void>(absl::in_place);
  EXPECT_TRUE(r.has_value());
}

TEST(ResultVoidTest, Emplace) {
  Result<void> r = absl::UnknownError("C");
  EXPECT_FALSE(r.has_value());
  r.emplace();
  EXPECT_TRUE(r.has_value());
  EXPECT_TRUE(r);
}

TEST(ResultVoidTest, MakeResultVoid) {
  // See result_nc_test.cc as well.
  Result<void> r = tensorstore::MakeResult();
  EXPECT_TRUE(r.has_value());
}

TEST(ResultVoidTest, ReturnVoid) {
  auto fn = []() -> Result<void> { return {absl::in_place}; };

  Result<void> r = fn();
  EXPECT_TRUE(r.has_value());
}

struct MoveOnly {
  MoveOnly(int value) : value(value) {}

  MoveOnly(MoveOnly const&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  int value;
};

Result<MoveOnly> MakeMoveOnly(int x) {
  if (x % 2 == 1) {
    return x;
  }
  if (x % 5 == 1) {
    MoveOnly y(x);
    return std::move(y);
  }
  return absl::UnknownError("");
}

TEST(ResultTest, MoveOnlyConstruct) {
  {
    Result<MoveOnly> x(3);
    EXPECT_TRUE(x);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<MoveOnly> x(absl::in_place, 3);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<MoveOnly> y{absl::UnknownError("C")};
    y.emplace(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    MoveOnly w(3);
    Result<MoveOnly> y(std::move(w));
    EXPECT_EQ(3, y.value().value);
  }
  {
    MoveOnly w(3);
    Result<MoveOnly> y(absl::in_place, 4);
    y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    MoveOnly w(3);
    Result<MoveOnly> y{absl::UnknownError("C")};
    y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    MoveOnly w(3);
    Result<MoveOnly> y{absl::UnknownError("C")};
    y = MoveOnly(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Result<MoveOnly> y = MoveOnly(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    MoveOnly w(3);
    Result<MoveOnly> y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Result<MoveOnly> y{absl::in_place, 4};
    y = absl::UnknownError("D");
    EXPECT_FALSE(y);
  }
}

TEST(ResultTest, MoveOnlyAssign) {
  MoveOnly m(3);
  Result<MoveOnly> x = std::move(m);
  EXPECT_EQ(3, x.value().value);
}

TEST(ResultTest, MoveOnlyTypeEmplace) {
  Result<MoveOnly> x(3);
  x.emplace(3);
  EXPECT_EQ(3, x.value().value);
}

TEST(ResultTest, MoveOnlyFunc) {
  Result<MoveOnly> x = MakeMoveOnly(3);
  MoveOnly y = std::move(x.value());
  EXPECT_EQ(3, y.value);
}

struct CopyOnly {
  CopyOnly(int value) : value(value) {}

  CopyOnly(CopyOnly const&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&) = delete;
  CopyOnly& operator=(CopyOnly&&) = delete;

  int value;
};

Result<CopyOnly> MakeCopyOnly(int x) {
  if (x % 2 == 1) {
    return x;
  }
  if (x % 5 == 1) {
    CopyOnly y(x);
    return y;
  }
  return absl::UnknownError("");
}

TEST(ResultTest, CopyOnlyConstruct) {
  {
    Result<CopyOnly> x(3);
    EXPECT_TRUE(x);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<CopyOnly> x(absl::in_place, 3);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<CopyOnly> y{absl::UnknownError("C")};
    y.emplace(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    CopyOnly w(3);
    Result<CopyOnly> y(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    CopyOnly w(3);
    Result<CopyOnly> y(absl::in_place, 4);
    y = w;
    EXPECT_EQ(3, y.value().value);
  }
  {
    CopyOnly w(3);
    Result<CopyOnly> y = w;
    EXPECT_EQ(3, y.value().value);
  }
  {
    CopyOnly w(3);
    Result<CopyOnly> y{absl::UnknownError("C")};
    y = w;
    EXPECT_EQ(3, y.value().value);
  }
  // No overload for y = CopyOnly(3)
  {
    Result<CopyOnly> y{absl::in_place, 4};
    y = absl::UnknownError("D");
    EXPECT_FALSE(y);
  }
}

TEST(ResultTest, CopyOnlyFunc) {
  Result<CopyOnly> x = MakeCopyOnly(3);
  CopyOnly y = x.value();
  EXPECT_EQ(3, y.value);
}

struct Explicit {
  explicit Explicit(int value) : value(value) {}
  explicit Explicit(MoveOnly v) : value(v.value) {}
  explicit Explicit(CopyOnly v) : value(v.value) {}

  Explicit(const Explicit& x) : value(x.value) {}
  Explicit& operator=(const Explicit& x) {
    value = x.value;
    return *this;
  }
  Explicit(Explicit&& x) : value(x.value) {}
  Explicit& operator=(Explicit&& x) {
    value = std::move(x).value;
    return *this;
  }

  int value;
};

TEST(ResultTest, ExplicitConstruct) {
  {
    Result<Explicit> x(3);
    EXPECT_TRUE(x);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<Explicit> x(absl::in_place, 3);
    EXPECT_EQ(3, x.value().value);
  }
  {
    Result<Explicit> y{absl::UnknownError("C")};
    y.emplace(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y(absl::in_place, 4);
    y = w;
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y = w;
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y{absl::UnknownError("C")};
    y = w;
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y(absl::in_place, 4);
    y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y{absl::UnknownError("C")};
    y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Result<Explicit> y{absl::UnknownError("C")};
    y = Explicit(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Result<Explicit> y = Explicit(3);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Explicit w(3);
    Result<Explicit> y = std::move(w);
    EXPECT_EQ(3, y.value().value);
  }
  {
    Result<Explicit> y{absl::in_place, 4};
    y = absl::UnknownError("D");
    EXPECT_FALSE(y);
  }
}

TEST(ResultTest, ExplicitConvertingConstructor) {
  {
    Result<CopyOnly> w{absl::in_place, 4};
    Result<Explicit> y{w};
    EXPECT_TRUE(y);
    EXPECT_EQ(4, y.value().value);
  }
  {
    Result<MoveOnly> w{absl::in_place, 4};
    Result<Explicit> y{std::move(w)};
    EXPECT_TRUE(y);
    EXPECT_EQ(4, y.value().value);
  }
}

struct X {
  X(Result<int>) {}
};

TEST(ResultTest, AdvancedConversion) {
  // See result_nc_test.cc as well.
  Result<int> int_result(3);
  Result<X> x_result(absl::in_place, int_result);
}

TEST(ResultTest, MakeResult) {
  // See result_nc_test.cc as well.
  {
    auto result = tensorstore::MakeResult();
    EXPECT_TRUE(result.has_value());
  }
  {
    auto result = tensorstore::MakeResult(7);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(7, *result);
  }
  {
    auto result = tensorstore::MakeResult<int>();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(0, *result);
  }

  const Status err(absl::StatusCode::kUnknown, "C");
  {
    auto result = tensorstore::MakeResult(err);
    EXPECT_FALSE(result.has_value());
  }
  {
    auto result = tensorstore::MakeResult<int>(err);
    EXPECT_FALSE(result.has_value());
  }
}

// Tests of Result::Construct.

TEST(ResultConstructTest, AssignStatusCopy) {
  Result<int> res = absl::UnknownError("");
  const Status status(absl::StatusCode::kUnknown, "new");
  res.Construct(status);
  EXPECT_FALSE(res);
  EXPECT_EQ(status, res.status());
}

TEST(ResultConstructTest, AssignStatusMove) {
  Result<int> res = absl::UnknownError("");
  Status status(absl::StatusCode::kUnknown, "new");
  res.Construct(absl::UnknownError("new"));
  EXPECT_FALSE(res);
  EXPECT_EQ(status, res.status());
}

TEST(ResultConstructTest, AssignResultCopy) {
  Result<int> res = absl::UnknownError("");
  Result<int> other(3);
  res.Construct(other);
  EXPECT_EQ(res, other);
}

TEST(ResultConstructTest, AssignResultMove) {
  Result<int> res = absl::UnknownError("");
  Result<int> other(3);
  res.Construct(Result<int>(3));
  EXPECT_EQ(res, other);
}

TEST(ResultConstructTest, AssignValue) {
  Result<int> res = absl::UnknownError("");
  Result<int> other(3);
  res.Construct(3);
  EXPECT_EQ(res, other);
}

TEST(ResultConstructTest, AssignInPlace) {
  Result<int> res = absl::UnknownError("");
  Result<int> other(3);
  res.Construct(tensorstore::in_place, 3);
  EXPECT_EQ(res, other);
}

TEST(ResultConstructTest, AssignInPlaceInitializerList) {
  Result<std::vector<int>> res = absl::UnknownError("");
  Result<std::vector<int>> other(tensorstore::in_place, {1, 2, 3});
  res.Construct(tensorstore::in_place, {1, 2, 3});
  EXPECT_EQ(res, other);
}

TEST(ResultTest, AssignOrReturn) {
  const auto Helper = [](Result<int> r) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto x, r);
    static_assert(std::is_same_v<decltype(x), int>);
    EXPECT_EQ(3, x);
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(3));
  EXPECT_EQ(absl::UnknownError("Got error"),
            Helper(absl::UnknownError("Got error")));
}

TEST(ResultTest, AssignOrReturnAnnotate) {
  const auto Helper = [](Result<int> r) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto x, r, tensorstore::MaybeAnnotateStatus(_, "Annotated"));
    static_assert(std::is_same_v<decltype(x), int>);
    EXPECT_EQ(3, x);
    return absl::UnknownError("No error");
  };
  EXPECT_EQ(absl::UnknownError("No error"), Helper(3));
  EXPECT_EQ(absl::UnknownError("Annotated: Got error"),
            Helper(absl::UnknownError("Got error")));
}

/// FIXME: Is FlatMapResultType pulling it's weight?

static_assert(std::is_same<UnwrapResultType<int>, int>::value);
static_assert(std::is_same<UnwrapResultType<Result<int>>, int>::value);
static_assert(std::is_same<UnwrapResultType<Status>, void>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<Status>, void>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<Result<int>>, int>::value);
static_assert(
    std::is_same<UnwrapQualifiedResultType<Result<int>&>, int&>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<const Result<int>&>,
                           const int&>::value);

static_assert(
    std::is_same<UnwrapQualifiedResultType<Result<int>&&>, int&&>::value);

/// FIXME: Typically a meta-function like FlatResult would be named MakeResult<>
/// or similar.

static_assert(std::is_same<FlatResult<Result<int>>, Result<int>>::value);

static_assert(std::is_same<FlatResult<int>, Result<int>>::value);

TEST(ChainResultTest, Example) {
  auto func1 = [](int x) -> float { return 1.0f + x; };
  auto func2 = [](float x) -> Result<std::string> {
    return absl::StrCat("fn.", x);
  };
  auto func3 = [](absl::string_view x) -> bool { return x.length() > 4; };

  Result<bool> y1 = ChainResult(Result<int>(3), func1, func2, func3);
  Result<bool> y2 = ChainResult(3, func1, func2, func3);

  EXPECT_TRUE(y1.has_value());
  EXPECT_TRUE(y2.has_value());

  EXPECT_EQ(y1.value(), y2.value());
}

TEST(ChainResultTest, Basic) {
  EXPECT_EQ(Result<int>(2), ChainResult(2));
  EXPECT_EQ(Result<int>(2), ChainResult(Result<int>(2)));

  EXPECT_EQ(Result<int>(3),
            ChainResult(Result<int>(2), [](int x) { return x + 1; }));

  EXPECT_EQ(Result<float>(1.5), ChainResult(
                                    Result<int>(2), [](int x) { return x + 1; },
                                    [](int x) { return x / 2.0f; }));

  EXPECT_EQ(Result<int>(absl::UnknownError("A")),
            ChainResult(Result<int>(absl::UnknownError("A")),
                        [](int x) { return x + 1; }));
}

TEST(MapResultTest, Basic) {
  tensorstore::Status status;

  EXPECT_EQ(Result<int>(absl::UnknownError("A")),
            tensorstore::MapResult(std::plus<int>(),
                                   Result<int>(absl::UnknownError("A")),
                                   Result<int>(absl::UnknownError("B"))));
  EXPECT_EQ(Result<int>(absl::UnknownError("B")),
            tensorstore::MapResult(std::plus<int>(), 1,
                                   Result<int>(absl::UnknownError("B"))));
  EXPECT_EQ(Result<int>(3), tensorstore::MapResult(std::plus<int>(), 1, 2));
  EXPECT_EQ(
      Result<int>(absl::UnknownError("C")),
      tensorstore::MapResult(
          [](int a, int b) { return Result<int>(absl::UnknownError("C")); }, 1,
          2));
}

TEST(PipelineOperator, Basic) {
  auto func1 = [](int x) -> float { return 1.0f + x; };
  auto func2 = [](float x) -> Result<std::string> {
    return absl::StrCat("fn.", x);
  };
  auto func3 = [](absl::string_view x) -> bool { return x.length() > 4; };

  auto y1 = Result<int>(3) | func1 | func2 | func3;
  static_assert(std::is_same_v<decltype(y1), Result<bool>>);
  EXPECT_THAT(y1, ::testing::Optional(false));

  auto y2 = Result<float>(2.5) | func2;
  static_assert(std::is_same_v<decltype(y2), Result<std::string>>);
  EXPECT_THAT(y2, ::testing::Optional(std::string("fn.2.5")));
}

}  // namespace
