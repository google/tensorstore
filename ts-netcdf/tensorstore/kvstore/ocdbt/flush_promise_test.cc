// Copyright 2023 The TensorStore Authors
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

#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::HaveSameSharedState;
using ::tensorstore::PromiseFuturePair;
using ::tensorstore::internal_future::FutureReferenceCount;
using ::tensorstore::internal_future::kFutureReferencesPerLink;
using ::tensorstore::internal_ocdbt::FlushPromise;

TEST(FlushPromiseTest, None) {
  Future<const void> flush_future;
  {
    FlushPromise flush_promise;
    flush_future = std::move(flush_promise).future();
  }
  EXPECT_TRUE(flush_future.null());
}

TEST(FlushPromiseTest, A) {
  auto [promise, future] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise;
    flush_promise.Link(future);
    flush_future = std::move(flush_promise).future();
  }
  EXPECT_TRUE(HaveSameSharedState(flush_future, future));
}

TEST(FlushPromiseTest, AA) {
  auto [promise, future] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise;
    flush_promise.Link(future);
    flush_promise.Link(future);
    flush_future = std::move(flush_promise).future();
  }
  EXPECT_TRUE(HaveSameSharedState(flush_future, future));
}

TEST(FlushPromiseTest, AABB) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  EXPECT_EQ(1, FutureReferenceCount(fa));
  EXPECT_EQ(1, FutureReferenceCount(fb));

  Future<const void> flush_future;
  {
    FlushPromise flush_promise;
    flush_promise.Link(fa);
    flush_promise.Link(fa);
    flush_promise.Link(fb);
    flush_promise.Link(fb);
    flush_future = std::move(flush_promise).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AABBAA) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  EXPECT_EQ(1, FutureReferenceCount(fa));
  EXPECT_EQ(1, FutureReferenceCount(fb));

  Future<const void> flush_future;
  {
    FlushPromise flush_promise;
    flush_promise.Link(fa);
    flush_promise.Link(fa);
    flush_promise.Link(fb);
    flush_promise.Link(fb);
    flush_promise.Link(fa);
    flush_promise.Link(fa);
    flush_future = std::move(flush_promise).future();
  }

  EXPECT_EQ(1 + 2 * kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AmergeABB) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  EXPECT_EQ(1, FutureReferenceCount(fa));
  EXPECT_EQ(1, FutureReferenceCount(fb));

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(std::move(flush_promise1));
    flush_promise2.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(fb);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AB_BCmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();
  auto [pd, fd] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(fd);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fc));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fd));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pd.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, A_Amerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(fa);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_TRUE(HaveSameSharedState(flush_future, fa));
}

TEST(FlushPromiseTest, AB_Bmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    flush_promise2.Link(fb);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, B_ABmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fb);
    flush_promise2.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AB_CBmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();

  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(fb);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + 2 * kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fc));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, MergeNull) {
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_TRUE(flush_future.null());
}

TEST(FlushPromiseTest, null_Amerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise2.Link(fa);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_TRUE(HaveSameSharedState(flush_future, fa));
}

TEST(FlushPromiseTest, A_merge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_TRUE(HaveSameSharedState(flush_future, fa));
}

TEST(FlushPromiseTest, A_BCmergeC) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(std::move(flush_promise1));
    flush_promise2.Link(fc);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + 2 * kFutureReferencesPerLink, FutureReferenceCount(fc));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, A_BCmergeA) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(std::move(flush_promise1));
    flush_promise2.Link(fa);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fc));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AB_Cmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fc));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, AB_CmergeB) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  auto [pc, fc] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    flush_promise2.Link(fc);
    flush_promise2.Link(std::move(flush_promise1));
    flush_promise2.Link(fb);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fc));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pc.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, A_Bmerge) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(std::move(flush_promise1));
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, A_BmergeA) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    FlushPromise flush_promise2;
    flush_promise1.Link(fa);
    flush_promise2.Link(fb);
    flush_promise2.Link(std::move(flush_promise1));
    flush_promise2.Link(fa);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, ABmove) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    FlushPromise flush_promise2{std::move(flush_promise1)};
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

TEST(FlushPromiseTest, ABmoveB) {
  auto [pa, fa] = PromiseFuturePair<void>::Make();
  auto [pb, fb] = PromiseFuturePair<void>::Make();
  Future<const void> flush_future;
  {
    FlushPromise flush_promise1;
    flush_promise1.Link(fa);
    flush_promise1.Link(fb);
    FlushPromise flush_promise2{std::move(flush_promise1)};
    flush_promise2.Link(fb);
    flush_future = std::move(flush_promise2).future();
  }

  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fa));
  EXPECT_EQ(1 + kFutureReferencesPerLink, FutureReferenceCount(fb));

  EXPECT_FALSE(flush_future.ready());
  pa.SetResult(absl::OkStatus());
  EXPECT_FALSE(flush_future.ready());
  pb.SetResult(absl::OkStatus());
  EXPECT_TRUE(flush_future.ready());
}

}  // namespace
