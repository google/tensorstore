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

#include "tensorstore/internal/thread.h"

#include <gtest/gtest.h>

namespace {

TEST(ThreadTest, Basic) {
  tensorstore::internal::Thread my_thread;

  int x = 0;
  tensorstore::internal::Thread::Id id[2];

  my_thread = tensorstore::internal::Thread({}, [&x, &id]() {
    x = 1;
    id[1] = tensorstore::internal::Thread::this_thread_id();
  });
  id[0] = my_thread.get_id();
  my_thread.Join();

  EXPECT_EQ(id[0], id[1]);
  EXPECT_EQ(1, x);
}

}  // namespace
