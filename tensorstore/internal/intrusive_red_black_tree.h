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

#ifndef TENSORSTORE_INTERNAL_INTRUSIVE_RED_BLACK_TREE_H_
#define TENSORSTORE_INTERNAL_INTRUSIVE_RED_BLACK_TREE_H_

/// \file
/// Intrusive red-black tree implementation.
///
/// See https://en.wikipedia.org/wiki/Red%E2%80%93black_tree
///
/// A red-black tree is a binary search tree where:
///
/// 1. Each node is additionally labeled as either red or black.
///
/// 2. The root is black.
///
/// 3. Leaves (null nodes) are black.
///
/// 4. Red nodes must only have black children.
///
/// 5. All paths from the root to any descendant leaf node includes the same
///    number of black nodes.

#include <array>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "tensorstore/internal/tagged_ptr.h"

namespace tensorstore {
namespace internal {
namespace intrusive_red_black_tree {

enum Color : bool { kRed = 0, kBlack = 1 };
enum Direction : bool { kLeft = 0, kRight = 1 };
inline constexpr Direction operator!(Direction d) {
  return static_cast<Direction>(!static_cast<bool>(d));
}

/// Base class for tree nodes.
///
/// Example usage:
///
///     class Node;
///     using Tree = intrusive_red_black_tree::Tree<Node>;
///
///     class Node : public Tree::NodeBase {
///       // ...
///     };
///
/// If a given `Node` type must be in more than one tree at a time, you can
/// specify an optional `Tag` type:
///
///     class Node;
///     struct TagA;
///     struct TagB;
///     using TreeA = intrusive_red_black_tree::Tree<Node, TagA>;
///     using TreeB = intrusive_red_black_tree::Tree<Node, TagB>;
///
///     class Node : public TreeA::NodeBase, public TreeB::NodeBase {
///       // ...
///     };
template <typename Tag = void>
struct NodeBase;

template <>
struct NodeBase<void> {
  NodeBase<>* rbtree_children_[2];
  TaggedPtr<NodeBase<>, 1> rbtree_parent_;
};

template <typename Tag>
struct NodeBase : public NodeBase<void> {};

/// intrusive_linked_list accessor that re-uses the storage of `NodeBase<>` for
/// a linked list node.
template <typename T, typename Tag>
struct LinkedListAccessor {
  using Node = T*;
  static Node Downcast(NodeBase<>* node) {
    return static_cast<Node>(static_cast<NodeBase<Tag>*>(node));
  }
  static NodeBase<>* Upcast(Node node) {
    return static_cast<NodeBase<Tag>*>(node);
  }
  static void SetPrev(Node node, Node prev) {
    Upcast(node)->rbtree_children_[0] = Upcast(prev);
  }
  static void SetNext(Node node, Node next) {
    Upcast(node)->rbtree_children_[1] = Upcast(next);
  }
  static Node GetPrev(Node node) {
    return Downcast(Upcast(node)->rbtree_children_[0]);
  }
  static Node GetNext(Node node) {
    return Downcast(Upcast(node)->rbtree_children_[1]);
  }
};

template <typename LessThan = std::less<>>
struct ThreeWayFromLessThan {
  LessThan compare;
  template <typename T>
  constexpr int operator()(const T& a, const T& b) const {
    return compare(a, b) ? -1 : (compare(b, a) ? 1 : 0);
  }
};

template <typename Node, typename Tag = void>
class Tree;

/// C++ standard library-compatible iterator for iterating over the nodes in a
/// tree.
template <typename Node, typename Tag = void, Direction Dir = kRight>
class Iterator {
 public:
  using Tree = intrusive_red_black_tree::Tree<Node, Tag>;
  using value_type = Node;
  using reference = Node&;
  using pointer = Node*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;

  Iterator(Node* node = nullptr) : node_(node) {}

  operator Node*() const { return node_; }

  Node* operator->() const { return node_; }
  Node& operator*() const { return *node_; }

  Iterator& operator++() {
    node_ = Tree::Traverse(node_, Dir);
    return *this;
  }

  Iterator operator++(int) {
    auto temp = *this;
    ++*this;
    return temp;
  }

  Iterator& operator--() {
    node_ = Tree::Traverse(node_, !Dir);
    return *this;
  }

  Iterator operator--(int) {
    auto temp = *this;
    --*this;
    return temp;
  }

  friend bool operator==(const Iterator& a, const Iterator& b) {
    return a.node_ == b.node_;
  }

  friend bool operator!=(const Iterator& a, const Iterator& b) {
    return a.node_ != b.node_;
  }

 private:
  Node* node_;
};

/// Position at which to insert a node into a tree.
template <typename Node>
struct InsertPosition {
  /// The reference node.
  Node* adjacent;
  /// The direction from `adjacent` at which to insert the node; `kLeft` ->
  /// node will be inserted immediately before `adjacent`, `kRight` -> node
  /// will be inserted immediately after `adjacent`.
  Direction direction;
};

/// Result of a find operation.
template <typename Node>
struct FindResult {
  /// The target node, or the node adjacent to where the target node can be
  /// inserted.
  Node* node;

  /// If `true`, `node` indicates the target node.  If `false`, `node`
  /// indicates the insertion position.
  bool found;

  /// The direction from `node` at which the target node can be inserted.
  Direction insert_direction;

  Node* found_node() const { return found ? node : nullptr; }
  InsertPosition<Node> insert_position() const {
    return {node, insert_direction};
  }
};

/// Represents a red-black tree root.
///
/// Compatible with C++17 range-based for loops.  For example:
///
///     for (auto &node : tree) {
///       // ...
///     }
///
/// \param Node The node type, must inherit publicly from `NodeBase<Tag>`.
/// \param Tag A tag type to distinguish between multiple `NodeBase` bases.
template <typename Node, typename Tag>
class Tree {
 public:
  using NodeBase = intrusive_red_black_tree::NodeBase<Tag>;
  using iterator = Iterator<Node, Tag>;
  using InsertPosition = intrusive_red_black_tree::InsertPosition<Node>;
  using FindResult = intrusive_red_black_tree::FindResult<Node>;

  constexpr static Direction kLeft = intrusive_red_black_tree::kLeft;
  constexpr static Direction kRight = intrusive_red_black_tree::kRight;

  Tree() = default;
  Tree(const Tree&) = delete;
  Tree(Tree&& other) = default;
  Tree& operator=(const Tree&) = delete;
  Tree& operator=(Tree&&) = default;

  /// Returns `true` if the tree is empty.
  bool empty() { return !root_; }

  /// Returns the first/last node in the tree (under in-order traversal order).
  ///
  /// \param dir If `kLeft`, return the first node.  Otherwise, return the last
  ///     node.
  /// \returns The node indicated by `dir`, or `nullptr` if the tree is empty.
  Node* ExtremeNode(Direction dir);

  iterator begin() { return ExtremeNode(kLeft); }

  /// Returns a sentinel iterator that may be used with `begin` in a range-for
  /// loop.
  static iterator end() { return {}; }

  /// Finds a node using a unary three-way comparison function.
  ///
  /// Note that while the C++17 and earlier standard libraries have relied on
  /// boolean "less than" operations for comparison, C++20 adds 3-way comparison
  /// (https://en.cppreference.com/w/cpp/utility/compare/compare_three_way).
  ///
  /// \param compare Function with signature `int (Node&)` that returns a
  ///     negative, zero, or positive value to indicate the target value is less
  ///     than, equal to, or greater than, the specified node, respectively.
  ///     Must be consistent with the existing order of nodes.
  template <typename Compare>
  FindResult Find(Compare compare);

  /// Finds the first/last node satisfying a predicate.
  ///
  /// \tparam BoundDirection If `kLeft`, finds the first node not satisfying
  ///     `predicate`.  If `kRight`, finds the last node satisfying `predicate`.
  /// \param predicate Function with signature `bool (Node&)`, where all `true`
  ///     nodes must occur before any `false` node, i.e. it must be a partition
  ///     function consistent with the existing order of nodes in the tree.
  template <Direction BoundDirection, typename Predicate>
  FindResult FindBound(Predicate predicate);

  /// Inserts a node in the tree.
  ///
  /// The position at which to insert the node is specified by `position`.  To
  /// insert a node adjacent to a specific existing node, `position` may be
  /// specified directly.  Otherwise, the appropriate insert position based on a
  /// comparison function may be determined by calling `Find`.
  ///
  /// \param position The position at which to insert `new_node`.
  /// \param new_node The new node to insert.  Must not be null.
  void Insert(InsertPosition position, Node* new_node);

  /// Inserts a node at the beginning or end of the tree.
  ///
  /// \param dir If equal to `kLeft`, insert at the beginning.  If equal to
  ///     `kRight`, insert at the end.
  /// \param new_node The node to insert.  Must not be null.
  void InsertExtreme(Direction dir, Node* new_node);

  /// Inserts a node in the tree, or returns an existing node.
  ///
  /// \param compare Three-way comparison function with signature `int (Node&)`
  ///     that returns negative, zero, or positive if the node to insert is less
  ///     than, equal to, or greater than the specified node, respectively. Must
  ///     be consistent with the existing order of nodes.
  /// \param make_node Function with signature `Node* ()` that returns the new
  ///     node to insert.  Only will be called if an existing node is not
  ///     already present.
  /// \returns A pair, where `first` is the existing node if present, or the new
  ///     node returned by `make_node()`, and `second` is `true` if
  ///     `make_node()` was called.
  template <typename Compare, typename MakeNode>
  std::pair<Node*, bool> FindOrInsert(Compare compare, MakeNode make_node);

  /// Joins two trees split by a center node.
  ///
  /// This is the inverse of `Split`.
  ///
  /// \param a_tree One of the trees to join.
  /// \param center The node indicating the split point, must be non-null.
  /// \param b_tree The other tree to join.
  /// \param a_dir If equal to `kLeft`, `a_tree` will be ordered before
  ///     `center`, and `b_tree` will be ordered after `center`.  If equal to
  ///     `kRight`, `a_tree` will be ordered after `center`, and `b_tree` will
  ///     be ordered before `center`.
  /// \returns The joined tree, equal to the concatenation of `a_tree`,
  ///     `center`, and `b_tree` if `a_dir == kLeft`, or the concatenation of
  ///     `b_tree`, `center`, and `a_tree` if `a_dir == kRight`.
  /// \post `a_tree.empty() && b_tree.empty()`
  static Tree Join(Tree& a_tree, Node* center, Tree& b_tree,
                   Direction a_dir = kLeft);

  /// Joins/concatenates two trees.
  ///
  /// \param a_tree One of the trees to join.
  /// \param b_tree The other tree to join.
  /// \param a_dir The order of `a_tree` in the joined result.
  /// \returns The joined tree, equal to the concatenation of `a_tree` and
  ///     `b_tree` if `a_dir == kLeft`, or the concatenation of `b_tree` and
  ///     `a_tree` if `a_dir == kRight`.
  /// \post `a_tree.empty() && b_tree.empty()`
  static Tree Join(Tree& a_tree, Tree& b_tree, Direction a_dir = kLeft);

  /// Splits a tree based on a center node.
  ///
  /// This is the inverse of `Join`.
  ///
  /// \param center The split point, must be non-null and contained in this
  ///     tree.
  /// \returns Two trees, the first containing all nodes ordered before
  ///     `center`, and the second containing all nodes ordered after `center`.
  ///     The `center` node itself is not contained in either tree.
  /// \post `this->empty() == true`
  std::array<Tree, 2> Split(Node* center);

  /// Specifies the result of a `FindSplit` operation.
  struct FindSplitResult {
    /// The split trees, containing all nodes before/after the split point.
    std::array<Tree, 2> trees;

    /// The node matching the `compare` function, or `nullptr` if there is no
    /// such node.  Not included in either of the two `trees`.
    Node* center;
  };

  /// Splits a tree based on a three-way comparison function.
  ///
  /// \param compare Three-way comparison function with signature `int (Node&)`,
  ///     must be consistent with the order of nodes.
  /// \returns If there is a node for which `compare` returns `0`, that node
  ///     serves as the split point and is returned as the `center` node.  The
  ///     returned `trees` contain all nodes that occur before/after the
  ///     `center` node.  If there is more than one node for which `compare`
  ///     returns `0`, one is picked arbitrarily (normally the caller will
  ///     ensure that there is at most one such node).  If there is no such
  ///     node, the returned `trees` will be based on the sign of the return
  ///     value of `compare`.
  /// \post `this->empty() == true`
  template <typename Compare>
  FindSplitResult FindSplit(Compare compare);

  /// Replaces an existing node in the tree with a different node.
  ///
  /// \param existing The existing node.
  /// \param replacement The new node to insert in place of `existing`.
  /// \pre `!IsDisconnected(existing) && IsDisconnected(replacement)`
  /// \post `IsDisconnected(existing) && !IsDisconnected(replacement)`
  void Replace(Node* existing, Node* replacement);

  /// Returns `true` if `node` is not contained in a tree.
  ///
  /// \param node The node.
  static bool IsDisconnected(Node* node);

  /// Removes a node from the tree (does not deallocate it).
  ///
  /// \pre `node` is contained in the tree.
  /// \post `IsDisconnected(node)`
  void Remove(Node* node);

  /// Returns the node before/after `x` in the in-order traversal.
  ///
  /// \param x Current traversal node.  Must be non-null.
  /// \param dir If `kLeft`, return the node before `x`.  Otherwise, return the
  ///     node after `x`.
  static Node* Traverse(Node* x, Direction dir);

  Node* root() { return Downcast(root_); }

 private:
  static Node* Downcast(intrusive_red_black_tree::NodeBase<>* node) {
    return static_cast<Node*>(static_cast<NodeBase*>(node));
  }
  static intrusive_red_black_tree::NodeBase<>* Upcast(Node* node) {
    return static_cast<NodeBase*>(node);
  }

  /// Root node of the tree.
  intrusive_red_black_tree::NodeBase<>* root_ = nullptr;
};

/// Internal operations used by the implementation.
namespace ops {

using NodeData = NodeBase<>;

/// Returns an all-zero tagged parent pointer, which indicates a disconnected
/// node.
///
/// A connected node cannot have an all-zero tagged parent pointer, because
/// while the root node has a null parent pointer, it must be black, which is
/// represented by a tag bit of 1.
inline TaggedPtr<NodeData, 1> DisconnectedParentValue() { return {}; }

/// Returns the parent of a node.
inline NodeData* Parent(NodeData* node) { return node->rbtree_parent_; }

/// Returns the color of a node (either `kRed` or `kBlack`).
inline Color GetColor(NodeData* node) {
  return static_cast<Color>(node->rbtree_parent_.tag());
}

/// Returns `true` if `node` is red.  Leaf nodes (`nullptr`) are implicitly
/// black.
inline bool IsRed(NodeData* node) {
  return node && ops::GetColor(node) == kRed;
}

/// Returns the child of `node`.
inline NodeData*& Child(NodeData* node, Direction dir) {
  return node->rbtree_children_[dir];
}

/// Returns `true` if `node` is not contained in a tree.
inline bool IsDisconnected(NodeData* node) {
  return node->rbtree_parent_ == ops::DisconnectedParentValue();
}

/// Returns the first/last node in the tree rooted at `root`, or `nullptr` if
/// `root` is `nullptr`.
///
/// \param dir If equal to `kLeft`, returns the first node.  If equal to
///     `kRight`, returns the last node.
NodeData* TreeExtremeNode(NodeData* root, Direction dir);

/// Traverses one step (using in-order traversal order) in the specified
/// direction from `x`.
///
/// \returns The next node, or `nullptr` if there is none.
NodeData* Traverse(NodeData* x, Direction dir);

/// Inserts `new_node` into the tree rooted at `root` adjacent to `parent` in
/// the specified `direction`.
void Insert(NodeData*& root, NodeData* parent, Direction direction,
            NodeData* new_node);

/// Joins `a_tree`, `center`, and `b_tree` in the order specified by `a_dir`.
NodeData* Join(NodeData* a_tree, NodeData* center, NodeData* b_tree,
               Direction a_dir);

/// Joins `a_tree` and `b_tree` in the order specified by `a_dir`.
NodeData* Join(NodeData* a_tree, NodeData* b_tree, Direction a_dir);

/// Splits the tree rooted at `root` by `center`.
std::array<NodeData*, 2> Split(NodeData* root, NodeData* center);

/// Splits the tree rooted at `root` by `center`.  If `found` is `false`,
/// `center` is additionally inserted into the resultant split tree indicated by
/// `dir`.
std::array<NodeData*, 2> Split(NodeData* root, NodeData*& center, Direction dir,
                               bool found);

/// Inserts `new_node` as the first/last of the tree rooted at `root`.
void InsertExtreme(NodeData*& root, Direction dir, NodeData* new_node);

/// Removes `z` from the tree rooted at `root`.
void Remove(NodeData*& root, NodeData* z);

/// Replaces `existing` with `replacement` in the tree rooted at `root`.
void Replace(NodeData*& root, NodeData* existing, NodeData* replacement);

}  // namespace ops

template <typename Node, typename Tag>
template <typename Compare>
typename Tree<Node, Tag>::FindResult Tree<Node, Tag>::Find(Compare compare) {
  FindResult result;
  result.insert_direction = kLeft;
  ops::NodeData* node = root_;
  ops::NodeData* result_node = nullptr;
  while (node) {
    result_node = node;
    const auto c = compare(*Downcast(node));
    if (c < 0) {
      result.insert_direction = kLeft;
    } else if (c > 0) {
      result.insert_direction = kRight;
    } else {
      result.found = true;
      result.node = Downcast(result_node);
      return result;
    }
    node = ops::Child(node, result.insert_direction);
  }
  result.found = false;
  result.node = Downcast(result_node);
  return result;
}

template <typename Node, typename Tag>
template <Direction BoundDirection, typename Predicate>
typename Tree<Node, Tag>::FindResult Tree<Node, Tag>::FindBound(
    Predicate predicate) {
  FindResult result;
  ops::NodeData* found = nullptr;
  result.insert_direction = kLeft;
  ops::NodeData* node = root_;
  ops::NodeData* result_node = nullptr;
  while (node) {
    result_node = node;
    auto satisfies = static_cast<Direction>(predicate(*Downcast(node)));
    if (satisfies == BoundDirection) found = node;
    result.insert_direction = satisfies;
    node = ops::Child(node, satisfies);
  }
  if (found) {
    result.found = true;
    result.node = Downcast(found);
    result.insert_direction = BoundDirection;
  } else {
    result.node = Downcast(result_node);
    result.found = false;
  }
  return result;
}

template <typename Node, typename Tag>
void Tree<Node, Tag>::Insert(InsertPosition position, Node* new_node) {
  ops::Insert(root_, Upcast(position.adjacent), position.direction,
              Upcast(new_node));
}

template <typename Node, typename Tag>
Tree<Node, Tag> Tree<Node, Tag>::Join(Tree& a_tree, Node* center, Tree& b_tree,
                                      Direction a_dir) {
  Tree<Node, Tag> joined;
  joined.root_ = ops::Join(a_tree.root_, center, b_tree.root_, a_dir);
  a_tree.root_ = nullptr;
  b_tree.root_ = nullptr;
  return joined;
}

template <typename Node, typename Tag>
Tree<Node, Tag> Tree<Node, Tag>::Join(Tree& a_tree, Tree& b_tree,
                                      Direction a_dir) {
  Tree<Node, Tag> joined;
  joined.root_ = ops::Join(a_tree.root_, b_tree.root_, a_dir);
  a_tree.root_ = nullptr;
  b_tree.root_ = nullptr;
  return joined;
}

template <typename Node, typename Tag>
std::array<Tree<Node, Tag>, 2> Tree<Node, Tag>::Split(Node* center) {
  auto split_nodes = ops::Split(root_, center);
  root_ = nullptr;
  std::array<Tree<Node, Tag>, 2> split_trees;
  split_trees[0].root_ = split_nodes[0];
  split_trees[1].root_ = split_nodes[1];
  return split_trees;
}

template <typename Node, typename Tag>
template <typename Compare>
typename Tree<Node, Tag>::FindSplitResult Tree<Node, Tag>::FindSplit(
    Compare compare) {
  FindSplitResult split_result;
  auto find_result = this->Find(std::move(compare));
  auto* center = Upcast(find_result.node);
  auto split_nodes = ops::Split(root_, center, find_result.insert_direction,
                                find_result.found);
  root_ = nullptr;
  split_result.center = Downcast(center);
  split_result.trees[0].root_ = split_nodes[0];
  split_result.trees[1].root_ = split_nodes[1];
  return split_result;
}

template <typename Node, typename Tag>
void Tree<Node, Tag>::InsertExtreme(Direction dir, Node* new_node) {
  ops::InsertExtreme(root_, dir, Upcast(new_node));
}

template <typename Node, typename Tag>
template <typename Compare, typename MakeNode>
std::pair<Node*, bool> Tree<Node, Tag>::FindOrInsert(Compare compare,
                                                     MakeNode make_node) {
  auto find_result = Find(std::move(compare));
  if (find_result.found) return {find_result.node, false};
  auto* new_node = make_node();
  Insert(find_result.insert_position(), new_node);
  return {new_node, true};
}

template <typename Node, typename Tag>
void Tree<Node, Tag>::Remove(Node* node) {
  ops::Remove(root_, Upcast(node));
}

template <typename Node, typename Tag>
void Tree<Node, Tag>::Replace(Node* existing, Node* replacement) {
  ops::Replace(root_, Upcast(existing), Upcast(replacement));
}

template <typename Node, typename Tag>
Node* Tree<Node, Tag>::ExtremeNode(Direction dir) {
  return Downcast(ops::TreeExtremeNode(root_, dir));
}

template <typename Node, typename Tag>
bool Tree<Node, Tag>::IsDisconnected(Node* node) {
  return ops::IsDisconnected(Upcast(node));
}

template <typename Node, typename Tag>
Node* Tree<Node, Tag>::Traverse(Node* x, Direction dir) {
  return Downcast(ops::Traverse(Upcast(x), dir));
}

}  // namespace intrusive_red_black_tree
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTRUSIVE_RED_BLACK_TREE_H_
