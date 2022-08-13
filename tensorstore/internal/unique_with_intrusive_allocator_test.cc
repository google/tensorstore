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

#include "tensorstore/internal/unique_with_intrusive_allocator.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/internal/arena.h"

namespace {

using ::tensorstore::internal::Arena;
using ::tensorstore::internal::ArenaAllocator;
using ::tensorstore::internal::IntrusiveAllocatorBase;
using ::tensorstore::internal::MakeUniqueWithIntrusiveAllocator;
using ::tensorstore::internal::MakeUniqueWithVirtualIntrusiveAllocator;

class Base {
 public:
  virtual void Destroy() = 0;
  virtual ~Base() = default;
};

class Derived : public IntrusiveAllocatorBase<Derived, Base> {
 public:
  Derived(ArenaAllocator<> allocator)
      :  // Ensure detectable memory leak if destructor is not called.
        vec(100, allocator) {}
  ArenaAllocator<> get_allocator() const { return vec.get_allocator(); }
  std::vector<double, ArenaAllocator<double>> vec;
};

TEST(UniqueWithVirtualIntrusiveAllocatorTest, Basic) {
  Arena arena;
  std::unique_ptr<Base, tensorstore::internal::VirtualDestroyDeleter> ptr =
      MakeUniqueWithVirtualIntrusiveAllocator<Derived>(
          ArenaAllocator<>(&arena));
}

class Foo {
 public:
  using allocator_type = ArenaAllocator<int>;

  Foo(std::size_t n, ArenaAllocator<int> allocator) : vec_(n, allocator) {}

  allocator_type get_allocator() const { return vec_.get_allocator(); }

  int operator()(int x) const { return vec_[x]; }
  void operator()(int x, int y) { vec_[x] = y; }

 private:
  std::vector<int, allocator_type> vec_;
};

TEST(UniqueWithIntrusiveAllocatorTest, Basic) {
  unsigned char buffer[200];
  Arena arena(buffer);
  auto ptr =
      MakeUniqueWithIntrusiveAllocator<Foo>(ArenaAllocator<>(&arena), 10);
  (*ptr)(2, 3);
  EXPECT_EQ(3, (*ptr)(2));
  EXPECT_EQ(3, (static_cast<const Foo&>(*ptr)(2)));
}

}  // namespace
