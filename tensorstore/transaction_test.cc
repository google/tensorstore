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

#include "tensorstore/transaction.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {
using tensorstore::MatchesStatus;
using tensorstore::no_transaction;
using tensorstore::Transaction;
using tensorstore::TransactionMode;
using tensorstore::internal::AcquireOpenTransactionPtrOrError;
using tensorstore::internal::OpenTransactionNodePtr;
using tensorstore::internal::TransactionState;
using tensorstore::internal::WeakTransactionNodePtr;

TEST(TransactionTest, NoTransaction) {
  EXPECT_EQ(TransactionMode::no_transaction_mode, no_transaction);
  Transaction txn = no_transaction;
  EXPECT_EQ(TransactionMode::no_transaction_mode, txn.mode());
  ASSERT_TRUE(txn.future().ready());
  TENSORSTORE_EXPECT_OK(txn.future());
  EXPECT_FALSE(txn.aborted());
  EXPECT_FALSE(txn.commit_started());
}

TEST(TransactionTest, Comparison) {
  auto txn1 = Transaction(tensorstore::isolated);
  auto txn2 = Transaction(tensorstore::isolated);
  Transaction no_txn1 = no_transaction;
  Transaction no_txn2(TransactionMode::no_transaction_mode);
  EXPECT_TRUE(no_txn1 == no_txn2);
  EXPECT_TRUE(no_txn1 == no_txn1);
  EXPECT_TRUE(no_txn1 == no_transaction);
  EXPECT_TRUE(no_transaction == no_txn1);
  EXPECT_NE(txn1, no_transaction);
  EXPECT_NE(no_transaction, txn1);
  EXPECT_FALSE(no_txn1 == txn1);
  EXPECT_TRUE(no_txn1 != txn1);
  EXPECT_FALSE(txn1 == no_txn1);
  EXPECT_TRUE(txn1 != no_txn1);
  EXPECT_TRUE(txn1 == txn1);
  EXPECT_FALSE(txn1 != txn1);
  EXPECT_FALSE(txn1 == txn2);
}

TEST(TransactionTest, CommitEmptyTransaction) {
  auto txn = Transaction(tensorstore::isolated);
  EXPECT_EQ(tensorstore::isolated, txn.mode());
  auto future = txn.future();
  EXPECT_FALSE(future.ready());
  txn.CommitAsync().IgnoreFuture();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, AbortEmptyTransaction) {
  auto txn = Transaction(tensorstore::isolated);
  EXPECT_EQ(tensorstore::isolated, txn.mode());
  auto future = txn.future();
  EXPECT_FALSE(future.ready());
  txn.Abort();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              tensorstore::MatchesStatus(absl::StatusCode::kCancelled));
}

TEST(TransactionTest, OpenPtrDefersCommit) {
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    txn.CommitAsync().IgnoreFuture();
    EXPECT_FALSE(future.ready());
  }
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, CommitBlockDefersCommit) {
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  TransactionState::get(txn)->AcquireCommitBlock();
  txn.CommitAsync().IgnoreFuture();
  EXPECT_FALSE(future.ready());
  TransactionState::get(txn)->ReleaseCommitBlock();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, CommitBlockDefersCommitWithNoCommitReferences) {
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  TransactionState::WeakPtr weak_txn(TransactionState::get(txn));
  weak_txn->AcquireCommitBlock();
  txn = no_transaction;
  EXPECT_FALSE(future.ready());
  future.Force();
  EXPECT_FALSE(future.ready());
  weak_txn->ReleaseCommitBlock();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, OpenPtrRetainsFuture) {
  tensorstore::Future<const void> future;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto open_ptr,
        AcquireOpenTransactionPtrOrError(Transaction(tensorstore::isolated)));
    ASSERT_TRUE(open_ptr);
    future = open_ptr->future();
    ASSERT_TRUE(future.valid());
    EXPECT_FALSE(future.ready());
    future.Force();
    EXPECT_FALSE(future.ready());
  }
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

using NodeLog = std::vector<std::string>;

struct TestNode : public TransactionState::Node {
  TestNode(NodeLog* log, std::uintptr_t associated_data)
      : TransactionState::Node(reinterpret_cast<void*>(associated_data)),
        log(log) {}

  void PrepareForCommit() override { log->push_back("prepare:" + Describe()); }
  void Commit() override { log->push_back("commit:" + Describe()); }
  void Abort() override { log->push_back("abort:" + Describe()); }
  std::string Describe() override {
    return tensorstore::StrCat(reinterpret_cast<uintptr_t>(associated_data()));
  }

  NodeLog* log;
  std::string id;
  bool terminal;
};

TEST(TransactionTest, SingleNodeAbort) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node(new TestNode(&log, 1));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node->Register());
  }
  EXPECT_FALSE(future.ready());
  EXPECT_THAT(log, ::testing::ElementsAre());
  txn.Abort();
  EXPECT_THAT(log, ::testing::ElementsAre("abort:1"));
  // Abort isn't done because `node->AbortDone` hasn't been called yet.
  EXPECT_FALSE(future.ready());
  node->AbortDone();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              tensorstore::MatchesStatus(absl::StatusCode::kCancelled));
}

TEST(TransactionTest, SingleNodeCommit) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node(new TestNode(&log, 1));
  OpenTransactionNodePtr<TestNode> open_node;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node->Register());
    // Open reference to node blocks commit.
    open_node.reset(node.get());
  }
  EXPECT_FALSE(future.ready());
  txn.CommitAsync().IgnoreFuture();
  // `open_node` prevents commit from starting.
  EXPECT_FALSE(future.ready());
  EXPECT_THAT(log, ::testing::ElementsAre());
  open_node.reset();

  // Commit starts
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  EXPECT_FALSE(future.ready());

  node->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  EXPECT_FALSE(future.ready());

  node->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "commit:1"));
  EXPECT_FALSE(future.ready());

  // `CommitDone` hasn't been called yet, so commit isn't considered done.
  node->CommitDone();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, CommitViaForcingFuture) {
  NodeLog log;
  tensorstore::Future<const void> future;
  WeakTransactionNodePtr<TestNode> node(new TestNode(&log, 1));
  {
    auto txn = Transaction(tensorstore::isolated);
    future = txn.future();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node->Register());
  }
  EXPECT_FALSE(future.ready());
  future.Force();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "commit:1"));
  node->CommitDone();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, TwoTerminalNodesIsolatedCommit) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node1->SetTransaction(*open_ptr);
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node1->Register());
    TENSORSTORE_EXPECT_OK(node2->Register());
    TENSORSTORE_EXPECT_OK(node1->MarkAsTerminal());
    TENSORSTORE_EXPECT_OK(node2->MarkAsTerminal());
  }
  txn.CommitAsync().IgnoreFuture();
  EXPECT_FALSE(future.ready());
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node1->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node2->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node1->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node2->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2"));
  node2->CommitDone();
  node1->CommitDone();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, TwoTerminalNodesAtomicError) {
  NodeLog log;
  auto txn = Transaction(tensorstore::atomic_isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node1->SetTransaction(*open_ptr);
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node1->Register());
    TENSORSTORE_EXPECT_OK(node2->Register());
    TENSORSTORE_EXPECT_OK(node1->MarkAsTerminal());
    EXPECT_THAT(node2->MarkAsTerminal(),
                tensorstore::MatchesStatus(
                    absl::StatusCode::kInvalidArgument,
                    "Cannot 1 and 2 as single atomic transaction"));
  }
  txn.CommitAsync().IgnoreFuture();
  EXPECT_THAT(log, ::testing::ElementsAre("abort:1", "abort:2"));
  EXPECT_FALSE(future.ready());
  node1->AbortDone();
  node2->AbortDone();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Cannot 1 and 2 as single atomic transaction"));
}

TEST(TransactionTest, OneTerminalNodeOneNonTerminalNodeAtomicSuccess) {
  NodeLog log;
  auto txn = Transaction(tensorstore::atomic_isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node1->SetTransaction(*open_ptr);
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node1->Register());
    TENSORSTORE_EXPECT_OK(node1->MarkAsTerminal());
    TENSORSTORE_EXPECT_OK(node2->Register());
  }
  txn.CommitAsync().IgnoreFuture();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node1->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node1->ReadyForCommit();
  node2->ReadyForCommit();
  node2->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2"));
  EXPECT_FALSE(future.ready());
  node1->CommitDone();
  node2->CommitDone();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, ImplicitTransaction) {
  tensorstore::Future<const void> future;
  TransactionState::WeakPtr weak_ptr;
  {
    auto open_ptr = TransactionState::MakeImplicit();
    ASSERT_TRUE(open_ptr);
    ASSERT_TRUE(open_ptr->implicit_transaction());
    weak_ptr.reset(open_ptr.get());
    future = open_ptr->future();
  }
  EXPECT_FALSE(future.ready());
  EXPECT_FALSE(weak_ptr->future().valid());
  {
    auto open_ptr = weak_ptr->AcquireImplicitOpenPtr();
    EXPECT_TRUE(open_ptr);
    EXPECT_TRUE(weak_ptr->future().valid());
    EXPECT_FALSE(future.ready());
  }
  future.Force();
  EXPECT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, ImplicitTransactionCommitBlock) {
  tensorstore::Future<const void> future;
  TransactionState::WeakPtr weak_ptr;
  {
    auto open_ptr = TransactionState::MakeImplicit();
    ASSERT_TRUE(open_ptr);
    ASSERT_TRUE(open_ptr->implicit_transaction());
    weak_ptr.reset(open_ptr.get());
    future = open_ptr->future();
  }
  EXPECT_FALSE(future.ready());
  EXPECT_FALSE(weak_ptr->future().valid());
  {
    auto open_ptr = weak_ptr->AcquireImplicitOpenPtr();
    EXPECT_TRUE(open_ptr);
    EXPECT_TRUE(weak_ptr->future().valid());
    EXPECT_FALSE(future.ready());
  }
  weak_ptr->AcquireCommitBlock();
  future.Force();
  EXPECT_FALSE(future.ready());
  {
    auto open_ptr = weak_ptr->AcquireImplicitOpenPtr();
    EXPECT_TRUE(open_ptr);
    EXPECT_TRUE(weak_ptr->future().valid());
    EXPECT_FALSE(future.ready());
  }
  weak_ptr->ReleaseCommitBlock();
  EXPECT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, TwoPhases) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  WeakTransactionNodePtr<TestNode> node3(new TestNode(&log, 3));
  WeakTransactionNodePtr<TestNode> node4(new TestNode(&log, 4));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node1->SetTransaction(*open_ptr);
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node2->Register());
    TENSORSTORE_EXPECT_OK(node1->Register());
    open_ptr->Barrier();
    node4->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node4->Register());
    node3->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node3->Register());
  }
  txn.CommitAsync().IgnoreFuture();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node1->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node2->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node1->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node2->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2"));
  EXPECT_FALSE(future.ready());
  node1->CommitDone();
  node2->CommitDone();
  EXPECT_FALSE(future.ready());
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2", "prepare:3"));
  node3->PrepareDone();
  EXPECT_THAT(log,
              ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                     "commit:2", "prepare:3", "prepare:4"));
  node3->ReadyForCommit();
  node4->ReadyForCommit();
  node4->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2", "prepare:3", "prepare:4",
                                          "commit:3", "commit:4"));
  node3->CommitDone();
  node4->CommitDone();
  ASSERT_TRUE(future.ready());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(TransactionTest, TwoPhasesAbort) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  auto future = txn.future();
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  WeakTransactionNodePtr<TestNode> node3(new TestNode(&log, 3));
  WeakTransactionNodePtr<TestNode> node4(new TestNode(&log, 4));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node1->SetTransaction(*open_ptr);
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node2->Register());
    TENSORSTORE_EXPECT_OK(node1->Register());
    open_ptr->Barrier();
    node4->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node4->Register());
    node3->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node3->Register());
  }
  txn.CommitAsync().IgnoreFuture();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node1->PrepareDone();
  node1->SetError(absl::UnknownError("failed"));
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2"));
  node2->PrepareDone();
  node2->ReadyForCommit();
  node1->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2"));
  EXPECT_FALSE(future.ready());
  node1->CommitDone();
  node2->CommitDone();
  EXPECT_FALSE(future.ready());
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:2", "commit:1",
                                          "commit:2", "abort:3", "abort:4"));
  node3->AbortDone();
  node4->AbortDone();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "failed"));
}

TEST(TransactionTest, AutomaticAbort) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  WeakTransactionNodePtr<TestNode> node(new TestNode(&log, 1));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node->Register());
  }
  txn = no_transaction;
  EXPECT_THAT(log, ::testing::ElementsAre("abort:1"));
  node->AbortDone();
}

TEST(TransactionTest, DeferredAbort) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  WeakTransactionNodePtr<TestNode> node(new TestNode(&log, 1));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    node->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node->Register());
    txn.Abort();
    EXPECT_FALSE(txn.future().ready());
    EXPECT_TRUE(txn.aborted());
    EXPECT_THAT(log, ::testing::ElementsAre());
  }
  EXPECT_FALSE(txn.future().ready());
  EXPECT_THAT(log, ::testing::ElementsAre("abort:1"));
  node->AbortDone();
  ASSERT_TRUE(txn.future().ready());
  EXPECT_THAT(txn.future().result(),
              tensorstore::MatchesStatus(absl::StatusCode::kCancelled));
}

TEST(TransactionTest, MultiPhaseNode) {
  NodeLog log;
  auto txn = Transaction(tensorstore::isolated);
  WeakTransactionNodePtr<TestNode> node1(new TestNode(&log, 1));
  WeakTransactionNodePtr<TestNode> node2(new TestNode(&log, 2));
  WeakTransactionNodePtr<TestNode> node3(new TestNode(&log, 3));
  WeakTransactionNodePtr<TestNode> node4(new TestNode(&log, 4));
  WeakTransactionNodePtr<TestNode> node5;
  WeakTransactionNodePtr<TestNode> node6;
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto open_ptr,
                                     AcquireOpenTransactionPtrOrError(txn));
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto open_node, open_ptr->GetOrCreateMultiPhaseNode(
                              /*associated_data=*/reinterpret_cast<void*>(5),
                              [&] { return new TestNode(&log, 5); }));
      node5.reset(static_cast<TestNode*>(open_node.get()));
    }
    node1->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node1->Register());
    open_ptr->Barrier();
    node2->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node2->Register());
    open_ptr->Barrier();
    node3->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node3->Register());
    open_ptr->Barrier();
    node4->SetTransaction(*open_ptr);
    TENSORSTORE_EXPECT_OK(node4->Register());
    open_ptr->Barrier();
    {
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto open_node, open_ptr->GetOrCreateMultiPhaseNode(
                              /*associated_data=*/reinterpret_cast<void*>(6),
                              [&] { return new TestNode(&log, 6); }));
      node6.reset(static_cast<TestNode*>(open_node.get()));
    }
  }
  EXPECT_FALSE(txn.future().ready());
  txn.CommitAsync().IgnoreFuture();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1"));
  node1->PrepareDone();
  node1->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:1", "prepare:5"));
  node5->PrepareDone();
  node5->ReadyForCommit();
  EXPECT_THAT(log,
              ::testing::ElementsAre("prepare:1", "prepare:5", "prepare:6"));
  log.clear();
  node6->PrepareDone();
  node6->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("commit:1", "commit:5", "commit:6"));
  log.clear();
  node1->CommitDone();
  node5->CommitDone(2);
  node6->CommitDone(2);
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:2"));
  log.clear();
  node2->PrepareDone();
  EXPECT_THAT(log, ::testing::ElementsAre());
  node2->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("commit:2"));
  log.clear();
  node2->CommitDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:3"));
  node3->PrepareDone();
  node3->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:3", "prepare:5"));
  node5->PrepareDone();
  node5->ReadyForCommit();
  EXPECT_THAT(log,
              ::testing::ElementsAre("prepare:3", "prepare:5", "prepare:6"));
  log.clear();
  node6->PrepareDone();
  node6->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("commit:3", "commit:5", "commit:6"));
  log.clear();
  node3->CommitDone();
  node5->CommitDone(3);
  EXPECT_THAT(log, ::testing::ElementsAre());
  node6->CommitDone();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:4"));
  node4->PrepareDone();
  node4->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("prepare:4", "prepare:5"));
  log.clear();
  node5->PrepareDone();
  node5->ReadyForCommit();
  EXPECT_THAT(log, ::testing::ElementsAre("commit:4", "commit:5"));
  log.clear();
  node4->CommitDone();
  node5->CommitDone();
  EXPECT_THAT(log, ::testing::ElementsAre());
  ASSERT_TRUE(txn.future().ready());
  TENSORSTORE_EXPECT_OK(txn.future());
}

}  // namespace
