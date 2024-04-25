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

#ifndef TENSORSTORE_INTERNAL_CONTAINER_HASH_SET_OF_ANY_H_
#define TENSORSTORE_INTERNAL_CONTAINER_HASH_SET_OF_ANY_H_

#include <stddef.h>

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/container/flat_hash_set.h"

namespace tensorstore {
namespace internal {

/// Hash set that contains arbitrary objects that inherit from `Entry` and
/// define a `key()` method that returns an `absl::Hash`-compatible key.
///
/// The derived entry type itself is also part of the key, and must be supplied
/// when inserting/querying entries.
///
/// For example usage, see `hash_set_of_any_test.cc`.
class HashSetOfAny {
 public:
  /// Base class of entries in the hash set.
  ///
  /// To use `HashSetOfAny`, you must define a subclass of `Entry` that defines
  /// the following members:
  ///
  ///   using KeyParam = ...;
  ///
  ///   auto key();
  ///
  /// where `key()` must return a type that is hash-compatible and
  /// equality-compatible with `KeyParam`.  In most cases, `key()` would return
  /// exactly `KeyParam`, but `key()` could also return for example
  /// `std::string` while `KeyParam` is `std::string_view`.
  class Entry {
   public:
    virtual ~Entry() = default;

   private:
    friend class HashSetOfAny;
    // Precomputed hash code of the `DerivedEntry::key()` value.  This avoids
    // the need for virtual dispatch to compute the hash code.
    size_t hash_;
  };

  /// Returns the entry of type `DerivedEntry` for `key`, creating it if
  /// necessary.
  ///
  /// \tparam DerivedEntry The derived entry type that defines a `key` member.
  /// \param key The key to lookup for `DerivedEntry`.
  /// \param make_entry Function with signature
  ///     `std::unique_ptr<DerivedEntry>()` to be called if `key` is not
  ///     present.  The returned pointer must have typeid equal to
  ///     `derived_type` and `key()` equal to the specified `key`.
  /// \param derived_type Actual typeid of entry returned by `make_entry`.  The
  ///     corresponding type must be `DerivedEntry` or a subclass.
  /// \returns Pair of the pointer to the entry, and a `bool` to indicate if the
  ///     entry was just created.
  template <typename DerivedEntry, typename MakeEntry>
  std::pair<DerivedEntry*, bool> FindOrInsert(
      typename DerivedEntry::KeyParam key, MakeEntry make_entry,
      const std::type_info& derived_type = typeid(DerivedEntry)) {
    static_assert(std::is_base_of_v<Entry, DerivedEntry>);
    KeyFor<DerivedEntry> key_wrapper{derived_type, key};
    size_t hash = Hash{}(key_wrapper);
    auto it = entries_.find(key_wrapper);
    if (it != entries_.end()) {
      return {static_cast<DerivedEntry*>(*it), false};
    }
    std::unique_ptr<DerivedEntry> derived_entry = make_entry();
    auto* entry = static_cast<Entry*>(derived_entry.get());
    assert(derived_type == typeid(*entry));
    entry->hash_ = hash;
    [[maybe_unused]] auto inserted = entries_.insert(entry).second;
    assert(inserted);
    return {derived_entry.release(), true};
  }

  /// Removes the specified entry from the hash set, but does not destroy it.
  void erase(Entry* entry) { entries_.erase(entry); }

  /// Removes all entries from the hash table without destroying them.
  void clear() { entries_.clear(); }

  auto begin() { return entries_.begin(); }
  auto begin() const { return entries_.begin(); }

  auto end() { return entries_.end(); }
  auto end() const { return entries_.end(); }

  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

 private:
  template <typename DerivedEntry>
  struct KeyFor {
    const std::type_info& derived_type;
    typename DerivedEntry::KeyParam key;

    friend bool operator==(Entry* entry, const KeyFor<DerivedEntry>& other) {
      return typeid(*entry) == other.derived_type &&
             static_cast<DerivedEntry*>(entry)->key() == other.key;
    }
  };

  struct Hash {
    using is_transparent = void;

    template <typename DerivedEntry>
    size_t operator()(KeyFor<DerivedEntry> key) const {
      return absl::HashOf(std::type_index(key.derived_type), key.key);
    }

    size_t operator()(Entry* entry) const { return entry->hash_; }
  };

  struct Eq : public std::equal_to<void> {
    using is_transparent = void;
  };

  absl::flat_hash_set<Entry*, Hash, Eq> entries_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONTAINER_HASH_SET_OF_ANY_H_
