// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/batch.h"

#include <stddef.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorstore/batch_impl.h"

namespace {

using ::tensorstore::Batch;
using ::testing::ElementsAre;

using Log = std::vector<std::string>;

template <typename T>
struct Entry : public Batch::Impl::Entry {
  using KeyParam = T;

  Entry(Log& log, size_t nesting_depth, T key)
      : Batch::Impl::Entry(nesting_depth), key_(key), log(log) {}

  T key_;
  T key() const { return key_; }
  virtual void Submit(Batch::View batch) {
    log.push_back(absl::StrCat("begin_submit ", key()));
    for (auto& submit_func : submit_funcs) {
      submit_func(batch);
    }
    log.push_back(absl::StrCat("end_submit ", key()));
    delete this;
  }
  std::vector<std::function<void(Batch::View batch)>> submit_funcs;

  Log& log;
};

template <typename T>
void AddFunc(Log& log, Batch::View batch, size_t nesting_depth, T key,
             std::function<void(Batch::View)> func) {
  auto& entry = Batch::Impl::From(batch)->GetEntry<Entry<T>>(
      key, [&] { return std::make_unique<Entry<T>>(log, nesting_depth, key); });
  entry.submit_funcs.emplace_back(std::move(func));
}

TEST(BatchTest, SingleNestingDepth) {
  Log log;
  auto batch = Batch::New();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      AddFunc<int>(log, batch, /*nesting_depth=*/0, /*key=*/i,
                   [&log, i, j](Batch::View batch) {
                     log.emplace_back(absl::StrFormat("i=%d, j=%d", i, j));
                   });
    }
  }
  EXPECT_THAT(log, ElementsAre());
  batch.Release();
  EXPECT_THAT(log,
              ::testing::UnorderedElementsAre(
                  "begin_submit 0", "i=0, j=0", "i=0, j=1", "end_submit 0",
                  "begin_submit 1", "i=1, j=0", "i=1, j=1", "end_submit 1",
                  "begin_submit 2", "i=2, j=0", "i=2, j=1", "end_submit 2"));
}

TEST(BatchTest, MultipleNestingDepths) {
  Log log;
  auto batch = Batch::New();
  for (int nesting_depth : {2, 3, 0}) {
    AddFunc<int>(log, batch, nesting_depth, /*key=*/nesting_depth,
                 [](Batch::View batch) {});
  }
  EXPECT_THAT(log, ElementsAre());
  batch.Release();
  EXPECT_THAT(log, ::testing::ElementsAre("begin_submit 3", "end_submit 3",
                                          "begin_submit 2", "end_submit 2",
                                          "begin_submit 0", "end_submit 0"));
}

TEST(BatchTest, MultipleTypes) {
  Log log;
  auto batch = Batch::New();
  AddFunc<int>(log, batch, /*nesting_depth=*/0, /*key=*/42,
               [](Batch::View batch) {});
  AddFunc<float>(log, batch, /*nesting_depth=*/0, /*key=*/1.5,
                 [](Batch::View batch) {});
  EXPECT_THAT(log, ElementsAre());
  batch.Release();
  EXPECT_THAT(log,
              ::testing::ElementsAre("begin_submit 42", "end_submit 42",
                                     "begin_submit 1.5", "end_submit 1.5"));
}

TEST(BatchTest, Async) {
  Log log;
  auto batch = Batch::New();

  Batch saved_batch{Batch::no_batch};
  AddFunc<int>(log, batch, /*nesting_depth=*/2, /*key=*/2,
               [&](Batch::View batch) { saved_batch = batch; });
  AddFunc<int>(log, batch, /*nesting_depth=*/1, /*key=*/3,
               [](Batch::View batch) {});
  batch.Release();

  EXPECT_THAT(log, ElementsAre("begin_submit 2", "end_submit 2"));
  log.clear();

  AddFunc<int>(log, saved_batch, /*nesting_depth=*/1, /*key=*/1,
               [](Batch::View batch) {});
  saved_batch.Release();

  EXPECT_THAT(
      log, ::testing::UnorderedElementsAre("begin_submit 1", "end_submit 1",
                                           "begin_submit 3", "end_submit 3"));
}

}  // namespace
