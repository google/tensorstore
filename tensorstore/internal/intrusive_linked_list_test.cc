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

/// Tests for intrusive_linked_list.h.

#include "tensorstore/internal/intrusive_linked_list.h"
#include <gtest/gtest.h>

namespace {
struct Node {
  Node* prev;
  Node* next;
};

using Accessor =
    tensorstore::internal::intrusive_linked_list::MemberAccessor<Node>;

TEST(IntrusiveLinkedListTest, Initialize) {
  Node head;
  Initialize(Accessor{}, &head);
  EXPECT_EQ(&head, head.next);
  EXPECT_EQ(&head, head.prev);

  EXPECT_TRUE(OnlyContainsNode(Accessor{}, &head));
}

TEST(IntrusiveLinkedListTest, InsertBefore) {
  Node head;
  Node a;
  Node b;
  Initialize(Accessor{}, &head);
  InsertBefore(Accessor{}, &head, &a);
  EXPECT_EQ(&a, head.next);
  EXPECT_EQ(&a, head.prev);
  EXPECT_EQ(&head, a.next);
  EXPECT_EQ(&head, a.prev);
  EXPECT_FALSE(OnlyContainsNode(Accessor{}, &head));
  InsertBefore(Accessor{}, &head, &b);
  EXPECT_EQ(&a, head.next);
  EXPECT_EQ(&b, head.prev);
  EXPECT_EQ(&head, b.next);
  EXPECT_EQ(&a, b.prev);
  EXPECT_EQ(&b, a.next);
  EXPECT_EQ(&head, a.prev);
}

TEST(IntrusiveLinkedListTest, Remove) {
  Node head;
  Node a;
  Node b;
  Initialize(Accessor{}, &head);
  InsertBefore(Accessor{}, &head, &a);
  InsertBefore(Accessor{}, &head, &b);
  Remove(Accessor{}, &b);
  EXPECT_EQ(&a, head.next);
  EXPECT_EQ(&a, head.prev);
  EXPECT_EQ(&head, a.next);
  EXPECT_EQ(&head, a.prev);
}

}  // namespace
