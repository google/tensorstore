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

#include "tensorstore/internal/no_destructor.h"

#include <string>
#include <type_traits>

#include <gtest/gtest.h>

namespace {
using tensorstore::internal::NoDestructor;

static_assert(std::is_trivially_destructible<NoDestructor<std::string>>::value,
              "");
static_assert(
    std::is_constructible<NoDestructor<std::string>, std::size_t, char>::value,
    "");
static_assert(
    std::is_constructible<NoDestructor<std::string>, std::string>::value, "");
static_assert(
    std::is_constructible<NoDestructor<std::string>, const char*>::value, "");

NoDestructor<std::string> test_obj("test");
TEST(NoDestructorTest, Basic) {
  const NoDestructor<std::string>& const_obj = test_obj;
  auto& s_ref = *test_obj;
  auto& s_const_ref = *const_obj;
  auto s_ptr = test_obj.get();
  auto s_const_ptr = const_obj.get();
  static_assert(std::is_same<const std::string*, decltype(s_const_ptr)>::value,
                "");
  static_assert(std::is_same<std::string*, decltype(s_ptr)>::value, "");
  static_assert(std::is_same<const std::string&, decltype(s_const_ref)>::value,
                "");
  static_assert(std::is_same<std::string&, decltype(s_ref)>::value, "");
  EXPECT_EQ("test", s_ref);
  EXPECT_EQ(&s_ref, &s_const_ref);
  EXPECT_EQ(s_ptr, &s_ref);
  EXPECT_EQ(s_ptr, s_const_ptr);
}
}  // namespace
