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

#ifndef TENSORSTORE_INTERNAL_INTRUSIVE_LINKED_LIST_H_
#define TENSORSTORE_INTERNAL_INTRUSIVE_LINKED_LIST_H_

/// \file
/// Simple intrusive circular doubly-linked list functionality.
///
/// Operations on a doubly-linked list are performed based on an Accessor object
/// of a type that satisfies the List Accessor concept, defined below:
///
/// List Accessor Concept:
///
/// The Accessor should be copy constructible, should define a nested `Node`
/// type specifying the type of a node reference, and should also define the
/// following member functions:
///
///     Node GetPrev(Node node);
///     Node GetNext(Node node);
///     void SetPrev(Node node, Node prev);
///     void SetNext(Node node, Node next);

namespace tensorstore {
namespace internal {
namespace intrusive_linked_list {

/// Stateless List Accessor implementation where nodes are referenced as `T *`
/// and the `next` and `prev` node pointers are specified as members of `T`.
template <typename T, T* T::*PrevMember = &T::prev,
          T* T::*NextMember = &T::next>
struct MemberAccessor {
  using Node = T*;
  static void SetPrev(T* node, T* prev) { node->*PrevMember = prev; }
  static void SetNext(T* node, T* next) { node->*NextMember = next; }
  static T* GetPrev(T* node) { return node->*PrevMember; }
  static T* GetNext(T* node) { return node->*NextMember; }
};

/// Initializes a singleton list containing `node` (commonly the dummy head
/// node).
template <typename Accessor>
void Initialize(Accessor accessor, typename Accessor::Node node) {
  accessor.SetPrev(node, node);
  accessor.SetNext(node, node);
}

/// Inserts `new_node` immediately before `existing_node` in a list.  The
/// existing `prev` and `next` pointers of `new_node` are ignored.
template <typename Accessor>
void InsertBefore(Accessor accessor, typename Accessor::Node existing_node,
                  typename Accessor::Node new_node) {
  accessor.SetPrev(new_node, accessor.GetPrev(existing_node));
  accessor.SetNext(new_node, existing_node);
  accessor.SetNext(accessor.GetPrev(existing_node), new_node);
  accessor.SetPrev(existing_node, new_node);
}

/// Removes `node` from the list.  Does not update the `prev` and `next`
/// pointers of `node`.
template <typename Accessor>
void Remove(Accessor accessor, typename Accessor::Node node) {
  accessor.SetPrev(accessor.GetNext(node), accessor.GetPrev(node));
  accessor.SetNext(accessor.GetPrev(node), accessor.GetNext(node));
}

}  // namespace intrusive_linked_list
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTRUSIVE_LINKED_LIST_H_
