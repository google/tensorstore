// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HETEROGENEOUS_CONTAINER_H_
#define TENSORSTORE_INTERNAL_HETEROGENEOUS_CONTAINER_H_

#include <functional>

#include "absl/container/flat_hash_set.h"

namespace tensorstore {
namespace internal {

/// Wrapper that marks a function object type as supporting heterogeneous
/// lookup.
template <typename T>
struct SupportsHeterogeneous : public T {
  using is_transparent = void;
};

/// Type that may be used to define heterogeneous comparison/hash functions for
/// container types.
///
/// Example usage:
///
///     struct Entry {
///       std::string id;
///       // Other data members
///     };
///
///     using Key = KeyAdapter<const Entry*, std::string_view, &Entry::id>;
///     using Map = absl::flat_hash_set<
///                     const Entry*,
///                     SupportsHeterogeneous<absl::Hash<Key>>,
///                     SupportsHeterogeneous<std::equal_to<Key>>>;
template <typename EntryPointer, typename T, auto Getter>
struct KeyAdapter {
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  KeyAdapter(U&& key) : value(std::forward<U>(key)) {}
  KeyAdapter(const EntryPointer& e) : value(std::invoke(Getter, *e)) {}

  template <typename H>
  friend H AbslHashValue(H h, const KeyAdapter& key) {
    return H::combine(std::move(h), key.value);
  }

  friend bool operator==(const KeyAdapter& a, const KeyAdapter& b) {
    return a.value == b.value;
  }

  T value;
};

/// `absl::flat_hash_set` of `EntryPointer` that supports heterogeneous lookup
/// based on a member of the entry.
template <typename EntryPointer, typename T, auto Getter>
using HeterogeneousHashSet = absl::flat_hash_set<
    EntryPointer,
    SupportsHeterogeneous<absl::Hash<KeyAdapter<EntryPointer, T, Getter>>>,
    SupportsHeterogeneous<std::equal_to<KeyAdapter<EntryPointer, T, Getter>>>>;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HETEROGENEOUS_CONTAINER_H_
