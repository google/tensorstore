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

namespace tensorstore {
namespace internal {
namespace intrusive_red_black_tree {
namespace ops {

/// Sets the parent of a node.
inline void SetParent(NodeData* node, NodeData* parent) {
  node->rbtree_parent_ = {parent, node->rbtree_parent_.tag()};
}

/// Sets the color of a node.
inline void SetColor(NodeData* node, Color color) {
  node->rbtree_parent_.set_tag(color);
}

/// Returns the direction (`kLeft` or `kRight`) of `node` relative to its
/// parent.
inline Direction ChildDir(NodeData* node) {
  return static_cast<Direction>(node != ops::Child(ops::Parent(node), kLeft));
}

/// Returns the grandparent of `node`.
inline NodeData* Grandparent(NodeData* node) {
  return ops::Parent(ops::Parent(node));
}

/// Performs a left or right tree rotation.
///
/// See https://en.wikipedia.org/wiki/Tree_rotation
///
///     Q              P         |
///    / \            / \        |
///   P   C  <---->  A   Q       |
///  / \                / \      |
/// A   B              B   C     |
///
/// For `dir == kLeft`:
///
///     y              x         |
///    / \            / \        |
///   x   C  <-----  A   y       |
///  / \                / \      |
/// A   B              B   C     |
///
/// For `dir == kRight`:
///
///     x              y         |
///    / \            / \        |
///   y   C  ----->  A   x       |
///  / \                / \      |
/// A   B              B   C     |
///
/// \param root The tree root.  If `x == root`, `root` will be set to `y`.
///     Otherwise, `root` is unchanged.
/// \param x Parent of the pivot node.
/// \param dir The rotation direction, either `kLeft` or `kRight`.
void Rotate(NodeData*& root, NodeData* x, Direction dir) {
  auto* y = ops::Child(x, !dir);
  // Move B to be a child of `x`.
  ops::Child(x, !dir) = ops::Child(y, dir);
  if (ops::Child(y, dir)) {
    // B is not null, set its parent to x.
    ops::SetParent(ops::Child(y, dir), x);
  }
  // Move y to be the parent of x.
  ops::SetParent(y, ops::Parent(x));
  if (!ops::Parent(x)) {
    // `x` was the root node, now `y` is the root node.
    root = y;
  } else {
    // Move `y` to be the child of the prior parent of `x`.
    ops::Child(ops::Parent(x), ops::ChildDir(x)) = y;
  }
  ops::Child(y, dir) = x;
  ops::SetParent(x, y);
}

/// Repairs the constraint that all children of red nodes are black.
///
/// The tree must already satisfy all properties of a red-black tree except that
/// the node `z`, which must be red, may be the root or may have a red parent.
///
/// \param root The root of the tree to repair.
/// \param z The newly added red node.
/// \returns The amount by which the black height of the tree increased, which
///     is always 0 or 1.
bool InsertFixup(NodeData*& root, NodeData* z) {
  assert(ops::IsRed(z));
  while (ops::IsRed(ops::Parent(z))) {
    // Both `z` and its parent `P` are red.  This violates the constraint that
    // all children of red nodes are black.
    //
    // Note that grandparent `G` of `z` must be black.
    Direction dir = ops::ChildDir(ops::Parent(z));
    // y is the sibling of the parent of `z`.
    if (NodeData* y = ops::Child(ops::Grandparent(z), !dir); ops::IsRed(y)) {
      // Both children of `G` are red, so we can correct the constraint simply
      // by swapping the colors of `G`, `P`, and `y`.
      //
      // Note that in the diagram below, `z` may be the left or right child of
      // `P`, and `P` and `y` may be swapped.
      //
      //       G(blk)                   G(red)         |
      //       /    \                  /     \         |
      //   P(red)   y(red)  --->   P(blk)   y(blk)     |
      //   /                        /                  |
      // z(red)                  z(red)                |
      ops::SetColor(ops::Parent(z), kBlack);
      ops::SetColor(y, kBlack);
      ops::SetColor(ops::Grandparent(z), kRed);
      z = ops::Grandparent(z);
    } else {
      if (ops::ChildDir(z) == !dir) {
        // Note that in the diagram below, `dir == kLeft`.
        //
        //                  G(blk)                        G(blk)         |
        //                  /     \                       /     \        |
        //               P(red)   y(blk)               z(red)   y(blk)   |
        //             /    \                --->      /   \             |
        //      A(blk)    z(red)                   P(red)  C(blk)        |
        //                 /  \                     /  \                 |
        //            B(blk)   C                   A    B                |
        //
        // Then `z` is reassigned to point to `P`.
        z = ops::Parent(z);
        ops::Rotate(root, z, dir);
      }
      // Note that in the diagram below, `dir == kLeft`.
      //
      //        G(blk)                         P(blk)                    |
      //        /     \                       /     \                    |
      //     P(red)   y(blk)     --->      z(red)   G(red)               |
      //     /   \                                   /   \               |
      //  z(red)  B(blk)                        B(blk)   y(blk)          |
      ops::SetColor(ops::Parent(z), kBlack);
      ops::SetColor(ops::Grandparent(z), kRed);
      ops::Rotate(root, ops::Grandparent(z), !dir);
      assert(!ops::IsRed(ops::Parent(z)));
      break;
    }
  }
  const Color existing_color = ops::GetColor(root);
  ops::SetColor(root, kBlack);
  return existing_color == kRed;
}

struct TreeWithBlackHeight {
  NodeData* root = nullptr;
  /// Number of black nodes along any path from the root to a null/leaf node,
  /// excluding the final null/leaf node.
  ///
  /// An empty tree has a black height of 0.
  size_t black_height = 0;
};

/// Returns the black height of the sub-tree rooted at `node`.
size_t BlackHeight(NodeData* node) {
  size_t black_height = 0;
  while (node) {
    if (ops::GetColor(node) == kBlack) ++black_height;
    node = ops::Child(node, kLeft);
  }
  return black_height;
}

/// Internal version of `Join` that maintains the black height.  This is used by
/// `Split` to avoid repeatedly computing the black height.
TreeWithBlackHeight Join(TreeWithBlackHeight a_tree, NodeData* center,
                         TreeWithBlackHeight b_tree, Direction a_dir) {
  assert(a_tree.black_height == ops::BlackHeight(a_tree.root));
  assert(b_tree.black_height == ops::BlackHeight(b_tree.root));
  // Ensure the black height of `a_tree` is >= the black height of `b_tree`.
  if (a_tree.black_height < b_tree.black_height) {
    a_dir = !a_dir;
    std::swap(a_tree, b_tree);
  }
  // Find the point at which we can graft `center` and `b_tree` onto `a_tree`
  // without violating the property that all paths from the root contain the
  // same number of black nodes.  We may violate the constraint that both
  // children of red nodes must be red, but that will be corrected by
  // `InsertFixup`.
  size_t difference = a_tree.black_height - b_tree.black_height;
  // If `difference == 0`, we can use the root as the graft point, making
  // `center` the new root, with `a_tree` and `b_tree` as its children.
  // Otherwise, descend along the `!a_dir` edge of `a_tree` until we reach a
  // black node `a_graft` with black height equal to `b_tree.black_height`.
  // Because all paths from the root are guaranteed to contain the same number
  // of black nodes, we must reach such a node before reaching a `nullptr`
  // (leaf) node.
  NodeData* a_graft = a_tree.root;
  NodeData* a_graft_parent = nullptr;
  while (true) {
    if (!ops::IsRed(a_graft)) {
      if (difference == 0) break;
      --difference;
    }
    a_graft_parent = a_graft;
    a_graft = ops::Child(a_graft, !a_dir);
  }
  assert(!ops::IsRed(a_graft));
  // Graft `center` in place of `a_graft`, making `a_graft` and `b_tree.root`
  // the children of `center`.
  //
  // Note that in the diagram below, `a_dir == kLeft`.
  //
  //         a_tree.root(blk)            a_tree.root(blk)                  |
  //             /     \                     /     \                       |
  //           ...     ...           ->    ...     ...                     |
  //                     \                           \                     |
  //                 a_graft_parent            a_graft_parent              |
  //                       \                          \                    |
  //                   a_graft(blk)               center(red)              |
  //                                               /      \                |
  //                                      a_graft(blk)    b_tree.root(blk) |
  ops::SetColor(center, kRed);
  ops::SetParent(center, a_graft_parent);
  if (a_graft_parent) {
    ops::Child(a_graft_parent, !a_dir) = center;
  } else {
    a_tree.root = center;
  }
  ops::Child(center, a_dir) = a_graft;
  if (a_graft) {
    ops::SetParent(a_graft, center);
  }
  ops::Child(center, !a_dir) = b_tree.root;
  if (b_tree.root) {
    ops::SetParent(b_tree.root, center);
  }
  // Repair red-black tree constraints.
  a_tree.black_height += ops::InsertFixup(a_tree.root, center);
  return a_tree;
}

/// Extracts a sub-tree with the specified black height as a new root.
///
/// If `child` is not `nullptr`, updates its `parent` pointer and ensures it is
/// marked black.
///
/// \param child The sub-tree, may be `nullptr`.
/// \param black_height Must equal `BlackHeight(child)`.
TreeWithBlackHeight ExtractSubtreeWithBlackHeight(NodeData* child,
                                                  size_t black_height) {
  TreeWithBlackHeight tree{child, black_height};
  if (child) {
    ops::SetParent(child, nullptr);
    if (ops::GetColor(child) == kRed) {
      ++tree.black_height;
      ops::SetColor(child, kBlack);
    }
  }
  return tree;
}

NodeData* ExtremeNode(NodeData* x, Direction dir) {
  assert(x);
  while (auto* child = ops::Child(x, dir)) x = child;
  return x;
}

NodeData* TreeExtremeNode(NodeData* root, Direction dir) {
  if (!root) return nullptr;
  return ops::ExtremeNode(root, dir);
}

NodeData* Traverse(NodeData* x, Direction dir) {
  if (auto* child = ops::Child(x, dir)) {
    return ops::ExtremeNode(child, !dir);
  }
  auto* y = ops::Parent(x);
  while (y && x == ops::Child(y, dir)) {
    x = y;
    y = ops::Parent(y);
  }
  return y;
}

void Insert(NodeData*& root, NodeData* parent, Direction direction,
            NodeData* new_node) {
  // Insert as red node.
  if (!parent) {
    assert(!root);
    root = new_node;
  } else {
    if (ops::Child(parent, direction)) {
      // `parent` already has a child in the specified direction.
      parent = ops::Traverse(parent, direction);
      direction = !direction;
    }
    ops::Child(parent, direction) = new_node;
  }
  ops::SetParent(new_node, parent);
  ops::Child(new_node, kLeft) = nullptr;
  ops::Child(new_node, kRight) = nullptr;
  ops::SetColor(new_node, kRed);

  // Repair red-black tree constraints.
  ops::InsertFixup(root, new_node);
}

NodeData* Join(NodeData* a_tree, NodeData* center, NodeData* b_tree,
               Direction a_dir) {
  return ops::Join({a_tree, ops::BlackHeight(a_tree)}, center,
                   {b_tree, ops::BlackHeight(b_tree)}, a_dir)
      .root;
}

NodeData* Join(NodeData* a_tree, NodeData* b_tree, Direction a_dir) {
  if (!a_tree) return b_tree;
  if (!b_tree) return a_tree;
  auto* center = ops::ExtremeNode(a_tree, !a_dir);
  ops::Remove(a_tree, center);
  return ops::Join(a_tree, center, b_tree, a_dir);
}

std::array<NodeData*, 2> Split(NodeData* root, NodeData* center) {
  std::array<TreeWithBlackHeight, 2> split_trees;
  size_t center_black_height = ops::BlackHeight(center);
  size_t child_black_height =
      center_black_height - (ops::GetColor(center) == kBlack);
  for (int dir = 0; dir < 2; ++dir) {
    split_trees[dir] = ops::ExtractSubtreeWithBlackHeight(
        ops::Child(center, static_cast<Direction>(dir)), child_black_height);
  }
  NodeData* parent = ops::Parent(center);
  while (parent) {
    Direction dir =
        static_cast<Direction>(ops::Child(parent, kRight) == center);
    NodeData* grandparent = ops::Parent(parent);
    auto parent_color = ops::GetColor(parent);
    // Note: In the diagram below, `dir == kLeft`.  In the `tree` diagram on the
    // bottom left, the edge from `E` to `F` has already implicitly been cut;
    // likewise, the edge from `C` to `E` in the `tree` diagram on the bottom
    // right has already been implicitly cut.
    //
    // split_trees[0]:    ->   split_trees[0]:    |
    //      A                       A             |
    //                                            |
    // split_trees[1]:    ->   split_trees[1]:    |
    //      B                       E             |
    //                            /  \            |
    //                           B    G           |
    //                                            |
    //  tree:             ->   tree:              |
    //     ...                    ...             |
    //                                            |
    //      C                   C=parent          |
    //    /  \                   /   \-\          |
    //   D   E=parent           D  E=center       |
    //     /-/    \                               |
    //  F=center   G                              |
    split_trees[!dir] =
        ops::Join(split_trees[!dir], parent,
                  ops::ExtractSubtreeWithBlackHeight(ops::Child(parent, !dir),
                                                     center_black_height),
                  dir);
    center = parent;
    parent = grandparent;
    center_black_height += (parent_color == kBlack);
  }
  assert(center == root);
  return {{split_trees[0].root, split_trees[1].root}};
}

std::array<NodeData*, 2> Split(NodeData* root, NodeData*& center, Direction dir,
                               bool found) {
  if (!center) return {{nullptr, nullptr}};
  auto split_trees = ops::Split(root, center);
  if (!found) {
    // The `center` node is just an insertion point, not the target node.
    // Insert it into the appropriate split tree.
    ops::InsertExtreme(split_trees[!dir], dir, center);
    center = nullptr;
  }
  return split_trees;
}

void InsertExtreme(NodeData*& root, Direction dir, NodeData* new_node) {
  ops::Insert(root, ops::TreeExtremeNode(root, dir), dir, new_node);
}

void Remove(NodeData*& root, NodeData* z) {
  NodeData* y;
  // Remove `z`.
  if (!ops::Child(z, kLeft) || !ops::Child(z, kRight)) {
    // `z` has a leaf node as a child, can remove it directly.
    y = z;
  } else {
    // Since `z` doesn't have any leaf node children, its successor node `y`
    // must have a leaf node child.
    y = ops::Traverse(z, kRight);
  }
  // `x` is the only child of `y` that may not be a leaf node.
  NodeData* x =
      ops::Child(y, static_cast<Direction>(ops::Child(y, kLeft) == nullptr));
  NodeData* px = ops::Parent(y);
  if (x) {
    ops::SetParent(x, px);
  }
  if (!px) {
    root = x;
  } else {
    ops::Child(px, ops::ChildDir(y)) = x;
  }
  const Color color_removed = ops::GetColor(y);
  if (y != z) {
    // `z` could not be removed directly.  Swap its location and color in the
    // tree with that of its successor node `y`.
    if (px == z) px = y;
    Replace(root, z, y);
  } else {
    z->rbtree_parent_ = ops::DisconnectedParentValue();
  }
  if (color_removed == kRed) {
    // Constraints already satisfied.
    return;
  }
  // Repair the tree, starting at `x` (whose parent is `px`).  The path from the
  // root to `x` contains one less black node than it should.  Note that `x` may
  // be `nullptr`.
  while (px && !ops::IsRed(x)) {
    const Direction dir = static_cast<Direction>(x == ops::Child(px, kRight));
    NodeData* w = ops::Child(px, !dir);
    // `x` must have a sibling `w`, as otherwise property 5 defined above
    // could not be satisfied.
    //
    // Currently paths rooted at `x` has one fewer black node than paths
    // rooted at `w`.  We need to correct this.
    assert(w != nullptr);
    if (ops::GetColor(w) == kRed) {
      // Rotate the tree so that both `x` and its sibling are black.  This
      // does not by itself correct the constraint.
      //
      // In the diagram below, `dir == kLeft`.
      //
      //     px(blk)                   w(blk)             |
      //      /   \        --->       /      \            |
      //  x(blk)  w(red)           px(red)   C(blk)       |
      //          /   \            /     \                |
      //      B(blk)  C(blk)    x(blk)  B(blk)            |
      ops::SetColor(w, kBlack);
      ops::SetColor(px, kRed);
      ops::Rotate(root, px, dir);
      // Re-assign `w` to point to `B`, the new sibling of `x`.
      w = ops::Child(px, !dir);
    }
    assert(ops::GetColor(w) == kBlack);
    if (!ops::IsRed(ops::Child(w, kLeft)) &&
        !ops::IsRed(ops::Child(w, kRight))) {
      // `w` has no red children, we can simply make it red, which corrects
      // the constraint by decreasing by 1 the number of black nodes in paths
      // rooted at `w`.
      ops::SetColor(w, kRed);
      x = px;
      px = ops::Parent(x);
    } else {
      // Increase by 1 the number of black nodes in paths rooted at `x`.  This
      // terminates the repair procedure.
      if (!ops::IsRed(ops::Child(w, !dir))) {
        // In the diagram below, `dir == kLeft`.
        //
        //        w(blk)                 B(blk)           |
        //       /     \                /      \          |
        //    B(red)  D(blk)    --->  A(blk)  w(red)      |
        //   /    \                           /    \      |
        // A(blk) C(blk)                  C(blk)  D(blk)  |
        ops::SetColor(ops::Child(w, dir), kBlack);
        ops::SetColor(w, kRed);
        ops::Rotate(root, w, !dir);
        // Re-assign `w` to point to `B`, the new sibling of `x`.
        w = ops::Child(px, !dir);
      }
      // In the diagram below, `dir == kLeft`:
      //
      //     px(col)                      w(col)         |
      //     /    \                       /    \         |
      //  x(blk)  w(blk)      ---->    px(blk)  C(blk)   |
      //          /    \               /    \            |
      //         B    C(red)        x(blk)   B           |
      ops::SetColor(w, ops::GetColor(px));
      ops::SetColor(px, kBlack);
      ops::SetColor(ops::Child(w, !dir), kBlack);
      ops::Rotate(root, px, dir);
      x = root;
      px = nullptr;
    }
  }
  // Ensure the root is black.
  if (x) ops::SetColor(x, kBlack);
}

void Replace(NodeData*& root, NodeData* existing, NodeData* replacement) {
  *replacement = *existing;
  // Fix parent pointers of children.
  for (int dir = 0; dir < 2; ++dir) {
    if (ops::Child(replacement, static_cast<Direction>(dir))) {
      ops::SetParent(ops::Child(replacement, static_cast<Direction>(dir)),
                     replacement);
    }
  }
  // Fix child/root pointer referencing `existing`.
  if (!ops::Parent(existing)) {
    root = replacement;
  } else {
    ops::Child(ops::Parent(existing), ops::ChildDir(existing)) = replacement;
  }
  existing->rbtree_parent_ = ops::DisconnectedParentValue();
}

}  // namespace ops
}  // namespace intrusive_red_black_tree
}  // namespace internal
}  // namespace tensorstore
