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

#include "tensorstore/internal/intrusive_red_black_tree.h"

#include <algorithm>
#include <set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"

namespace {

namespace rbtree = tensorstore::internal::intrusive_red_black_tree;
namespace ops = tensorstore::internal::intrusive_red_black_tree::ops;

/// Checks that the red-black tree constraints are satisfied.
///
/// \param x The root of the tree to validate.
/// \returns The number of black nodes along all paths rooted at `x`.
int CheckInvariants(ops::NodeData* x) {
  if (!x) return 1;
  ops::NodeData* c1 = ops::Child(x, rbtree::kLeft);
  ops::NodeData* c2 = ops::Child(x, rbtree::kRight);
  if (c1) {
    EXPECT_EQ(x, ops::Parent(c1));
  }
  if (c2) {
    EXPECT_EQ(x, ops::Parent(c2));
  }
  if (ops::GetColor(x) == rbtree::kRed) {
    EXPECT_FALSE(ops::IsRed(c1));
    EXPECT_FALSE(ops::IsRed(c2));
  }
  int lh = CheckInvariants(c1);
  int rh = CheckInvariants(c2);
  EXPECT_EQ(lh, rh);
  if (ops::GetColor(x) == rbtree::kRed) {
    return lh;
  } else {
    return lh + 1;
  }
}

template <typename Node, typename Tag, typename Compare>
void CheckInvariants(rbtree::Tree<Node, Tag>& x, Compare compare) {
  auto* root = static_cast<rbtree::NodeBase<Tag>*>(x.root());
  if (!root) return;
  EXPECT_EQ(rbtree::kBlack, ops::GetColor(root));
  CheckInvariants(root);
  EXPECT_TRUE(std::is_sorted(
      x.begin(), x.end(), [&](Node& a, Node& b) { return compare(a, b) < 0; }));
}

struct Set {
  struct Node : public rbtree::NodeBase<> {
    int value;
  };

  static void FormatNode(std::string& out, const std::string& prefix,
                         Node* node, bool dir) {
    out += prefix;
    out += (dir == rbtree::kLeft) ? "|-  " : " -  ";
    if (!node) {
      out += "null";
    } else {
      out += std::to_string(node->value);
      out += ops::GetColor(node) == rbtree::kBlack ? "(blk)" : "(red)";
    }
    out += '\n';
    if (!node) return;
    std::string child_prefix =
        prefix + ((dir == rbtree::kLeft) ? "|   " : "    ");
    for (int dir = 0; dir < 2; ++dir) {
      FormatNode(out, child_prefix,
                 static_cast<Node*>(
                     ops::Child(node, static_cast<rbtree::Direction>(dir))),
                 static_cast<rbtree::Direction>(dir));
    }
  }

  static std::string FormatTree(rbtree::Tree<Node>& tree) {
    std::string out;
    FormatNode(out, "", tree.root(), rbtree::kRight);
    return out;
  }

  static auto CompareToKey(int key) {
    return [key](Node& node) {
      return rbtree::ThreeWayFromLessThan<>()(key, node.value);
    };
  }

  static auto CompareNodes() {
    return [](Node& a, Node& b) {
      return rbtree::ThreeWayFromLessThan<>()(a.value, b.value);
    };
  }

  static std::vector<int> Elements(rbtree::Tree<Node>& tree) {
    std::vector<int> elements;
    for (auto& node : tree) {
      elements.push_back(node.value);
    }
    return elements;
  }

  using Tree = rbtree::Tree<Node>;
  Tree tree;
  std::set<int> golden_set;

  void CheckTreeInvariants() {
    SCOPED_TRACE("\n" + FormatTree(tree));
    CheckInvariants(tree, CompareNodes());
  }

  bool Contains(int key) {
    bool result = tree.Find(CompareToKey(key)).found;
    EXPECT_EQ(result, golden_set.count(key) == 1);
    return result;
  }

  Node* FindNode(int key) {
    auto* node = tree.Find(CompareToKey(key)).found_node();
    assert(node);
    return node;
  }

  bool Insert(int key) {
    auto [node, inserted] = tree.FindOrInsert(CompareToKey(key), [&] {
      auto* n = new Node;
      n->value = key;
      return n;
    });
    EXPECT_EQ(key, node->value);
    CheckTreeInvariants();
    EXPECT_EQ(inserted, golden_set.insert(key).second);
    return inserted;
  }

  bool Erase(int key) {
    auto node = tree.Find(CompareToKey(key)).found_node();
    bool result;
    if (!node) {
      result = false;
    } else {
      tree.Remove(*node);
      delete node;
      CheckTreeInvariants();
      result = true;
    }
    EXPECT_EQ(static_cast<int>(result), golden_set.erase(key));
    return result;
  }

  void CheckElements() {
    EXPECT_THAT(Elements(), ::testing::ElementsAreArray(golden_set.begin(),
                                                        golden_set.end()));
  }

  void CheckSplitJoin(int key) {
    auto orig_elements = Elements();
    auto split_result = tree.FindSplit([&](Node& node) {
      return rbtree::ThreeWayFromLessThan<>()(key, node.value);
    });
    SCOPED_TRACE("Key=" + std::to_string(key) +  //
                 "\nLeft tree:\n" + FormatTree(split_result.trees[0]) +
                 "\nRight tree:\n" + FormatTree(split_result.trees[1]));
    for (int i = 0; i < 2; ++i) {
      CheckInvariants(split_result.trees[i], CompareNodes());
    }
    std::vector<int> elements_a = Elements(split_result.trees[0]);
    std::vector<int> elements_b = Elements(split_result.trees[1]);
    std::vector<int> combined_elements = elements_a;
    if (split_result.center) {
      EXPECT_EQ(key, split_result.center->value);
      combined_elements.push_back(split_result.center->value);
    }
    combined_elements.insert(combined_elements.end(), elements_b.begin(),
                             elements_b.end());
    EXPECT_THAT(combined_elements, ::testing::ElementsAreArray(orig_elements));
    if (split_result.center) {
      tree = Tree::Join(split_result.trees[0], *split_result.center,
                        split_result.trees[1]);
    } else {
      tree = Tree::Join(split_result.trees[0], split_result.trees[1]);
    }
    CheckTreeInvariants();
    CheckElements();
  }

  void CheckSplitJoin() {
    auto orig_elements = Elements();
    if (orig_elements.empty()) {
      CheckSplitJoin(0);
    } else {
      int min = orig_elements.front() - 1;
      int max = orig_elements.back() + 1;
      for (int x = min; x <= max; ++x) {
        SCOPED_TRACE(x);
        CheckSplitJoin(x);
      }
    }
  }

  std::vector<int> Elements() { return Elements(tree); }

  ~Set() {
    for (auto it = tree.begin(); it != tree.end();) {
      auto next = std::next(it);
      tree.Remove(*it);
      delete &*it;
      it = next;
    }
  }
};

TEST(SetTest, SimpleInsert1) {
  Set rbtree_set;
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(1);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(2);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(3);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
}

TEST(SetTest, SimpleInsert2) {
  Set rbtree_set;
  Set::Tree::Range empty_range = rbtree_set.tree;
  EXPECT_TRUE(empty_range.empty());
  EXPECT_EQ(empty_range, empty_range);

  rbtree_set.Insert(5);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(8);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(1);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(3);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(9);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(7);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();
  rbtree_set.Insert(0);
  rbtree_set.CheckElements();
  rbtree_set.CheckSplitJoin();

  Set::Tree::Range full_range = rbtree_set.tree;
  EXPECT_FALSE(full_range.empty());
  EXPECT_EQ(full_range, full_range);
  EXPECT_NE(full_range, empty_range);
  EXPECT_EQ(full_range.begin(), rbtree_set.tree.begin());
  EXPECT_EQ(full_range.end(), rbtree_set.tree.end());

  Set::Tree::Range partial_range(rbtree_set.FindNode(1),
                                 rbtree_set.FindNode(5));
  EXPECT_NE(partial_range, full_range);
  EXPECT_NE(partial_range, empty_range);
  std::set<int> partial_elements;
  for (auto& node : partial_range) {
    partial_elements.insert(node.value);
  }
  EXPECT_THAT(partial_elements, ::testing::ElementsAre(1, 3));
}

TEST(SetTest, RandomInsert) {
  Set rbtree_set;
  absl::BitGen gen;
  constexpr int kMaxKey = 10;

  for (int i = 0; i < 20; ++i) {
    const int key = absl::Uniform(gen, 0, kMaxKey);
    rbtree_set.Contains(key);
    rbtree_set.Insert(key);
    rbtree_set.CheckElements();
    rbtree_set.CheckSplitJoin();
  }
}

TEST(SetTest, RandomInsertRemove) {
  Set rbtree_set;
  absl::BitGen gen;
  constexpr int kMaxKey = 10;

  for (int i = 0; i < 50; ++i) {
    const int key = absl::Uniform(gen, 0, kMaxKey);
    if (absl::Bernoulli(gen, 0.5)) {
      rbtree_set.Insert(key);
    } else {
      rbtree_set.Erase(key);
    }
  }
}

struct MultiSet {
  using Pair = std::pair<int, int>;

  struct Node : public rbtree::NodeBase<> {
    Pair value;
  };

  struct Compare {
    bool operator()(const Pair& a, const Pair& b) const {
      return a.first < b.first;
    }
  };

  using Tree = rbtree::Tree<Node>;
  Tree tree;

  std::multiset<Pair, Compare> golden_set;

  constexpr static auto ThreeWayCompare = [](Node& a, Node& b) {
    return rbtree::ThreeWayFromLessThan<>()(a.value.first, b.value.first);
  };

  void CheckTreeInvariants() { CheckInvariants(tree, ThreeWayCompare); }

  void Insert(Pair value) {
    tree.FindOrInsert(
        [&](Node& node) {
          // Ensure that if `value.first` is already present, `value` is added
          // at the right side of the existing values, to match `std::multiset`.
          return value.first < node.value.first ? -1 : 1;
        },
        [&] {
          auto* n = new Node;
          n->value = value;
          return n;
        });
    CheckTreeInvariants();
    golden_set.insert(value);
  }

  void CheckElements() {
    EXPECT_THAT(Elements(), ::testing::ElementsAreArray(golden_set.begin(),
                                                        golden_set.end()));
  }

  std::vector<Pair> Elements() {
    std::vector<Pair> elements;
    for (auto& node : tree) {
      elements.push_back(node.value);
    }
    return elements;
  }

  ~MultiSet() {
    for (auto it = tree.begin(); it != tree.end();) {
      auto next = std::next(it);
      tree.Remove(*it);
      delete &*it;
      it = next;
    }
  }
};

TEST(MultiSetTest, SimpleInsert1) {
  MultiSet rbtree_set;
  rbtree_set.Insert({1, 2});
  rbtree_set.CheckElements();
  rbtree_set.Insert({2, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({1, 1});
  rbtree_set.CheckElements();
  rbtree_set.Insert({3, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({3, 1});
  rbtree_set.CheckElements();
  EXPECT_THAT(
      rbtree_set.Elements(),
      ::testing::ElementsAre(::testing::Pair(1, 2), ::testing::Pair(1, 1),
                             ::testing::Pair(2, 0), ::testing::Pair(3, 0),
                             ::testing::Pair(3, 1)));
}

TEST(MultiSetTest, SimpleInsert2) {
  MultiSet rbtree_set;
  rbtree_set.Insert({5, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({8, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({1, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({3, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({9, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({7, 0});
  rbtree_set.CheckElements();
  rbtree_set.Insert({0, 0});
  rbtree_set.CheckElements();
}

TEST(MultiSetTest, RandomInsert) {
  MultiSet rbtree_set;
  absl::BitGen gen;
  constexpr int kMaxKey = 10;
  constexpr int kMaxValue = 100;

  for (int i = 0; i < 20; ++i) {
    rbtree_set.Insert(
        {absl::Uniform(gen, 0, kMaxKey), absl::Uniform(gen, 0, kMaxValue)});
    rbtree_set.CheckElements();
  }
}

}  // namespace
