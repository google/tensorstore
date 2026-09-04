// Copyright 2026 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

#include "tensorstore/internal/grpc/stub_pool.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "grpcpp/channel.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"

namespace {

using ::tensorstore::internal_grpc::StubPool;

struct DummyStub {
  int id;
};

struct DummyService {
  using StubInterface = DummyStub;
  static std::shared_ptr<DummyStub> NewStub(
      const std::shared_ptr<grpc::Channel>& channel) {
    return std::make_shared<DummyStub>(DummyStub{channel ? 1 : 0});
  }
};

TEST(StubPoolTest, RoundRobin) {
  std::vector<std::shared_ptr<grpc::Channel>> channels(3);
  std::vector<std::shared_ptr<DummyStub>> stubs = {
      std::make_shared<DummyStub>(DummyStub{0}),
      std::make_shared<DummyStub>(DummyStub{1}),
      std::make_shared<DummyStub>(DummyStub{2}),
  };

  StubPool<DummyStub> pool("localhost:8080", channels, stubs);
  EXPECT_EQ(pool.size(), 3);
  EXPECT_EQ(pool.address(), "localhost:8080");
  EXPECT_EQ(pool.channels().size(), 3);
  EXPECT_EQ(pool.stubs().size(), 3);

  EXPECT_EQ(pool.get_next_stub()->id, 0);
  EXPECT_EQ(pool.get_next_stub()->id, 1);
  EXPECT_EQ(pool.get_next_stub()->id, 2);
  EXPECT_EQ(pool.get_next_stub()->id, 0);
}

TEST(StubPoolTest, SingleStub) {
  std::vector<std::shared_ptr<grpc::Channel>> channels(1);
  std::vector<std::shared_ptr<DummyStub>> stubs = {
      std::make_shared<DummyStub>(DummyStub{42}),
  };

  StubPool<DummyStub> pool("localhost:8080", channels, stubs);
  EXPECT_EQ(pool.size(), 1);

  // Single-stub pool always returns the same stub without
  // incrementing the atomic counter (size <= 1 fast path).
  EXPECT_EQ(pool.get_next_stub()->id, 42);
  EXPECT_EQ(pool.get_next_stub()->id, 42);
  EXPECT_EQ(pool.get_next_stub()->id, 42);
}

TEST(StubPoolTest, EmptyPool) {
  StubPool<DummyStub> pool("localhost:8080", {}, {});
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.get_next_stub(), nullptr);
}

TEST(StubPoolTest, CreateStubPool) {
  auto auth =
      ::tensorstore::internal_grpc::CreateInsecureAuthenticationStrategy();
  auto pool = ::tensorstore::internal_grpc::CreateStubPool<DummyService>(
      "localhost:8080", 2, *auth);
  EXPECT_EQ(pool->size(), 2);
  EXPECT_EQ(pool->address(), "localhost:8080");
  EXPECT_NE(pool->get_next_stub(), nullptr);
  EXPECT_NE(pool->get_next_stub(), nullptr);
}

TEST(StubPoolTest, WaitForConnected) {
  std::vector<std::shared_ptr<grpc::Channel>> channels = {nullptr};
  std::vector<std::shared_ptr<DummyStub>> stubs = {
      std::make_shared<DummyStub>(DummyStub{0}),
  };
  StubPool<DummyStub> pool("localhost:8080", channels, stubs);
  pool.WaitForConnected(absl::Milliseconds(1));
}

}  // namespace
