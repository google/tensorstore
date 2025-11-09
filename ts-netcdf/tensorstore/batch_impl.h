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

#ifndef TENSORSTORE_BATCH_IMPL_H_
#define TENSORSTORE_BATCH_IMPL_H_

#include <stddef.h>

#include <type_traits>
#include <typeinfo>

#include "absl/synchronization/mutex.h"
#include "tensorstore/batch.h"
#include "tensorstore/internal/container/hash_set_of_any.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"

namespace tensorstore {

// The nesting depth of an entry must be greater than the nesting depth of any
// batch entry directly or indirectly accessed via `Batch::Impl::GetEntry` when
// `Entry::Submit` is called.
//
// The nesting depth may be 0 if `Entry::Submit` does not access any other
// entries.  For example, the file kvstore read implementation does not need to
// access any other entries.

class Batch::Impl : public Batch::ImplBase {
 public:
  class Entry;
  friend class Batch;

 private:
  // Tree of depths for which an entry is present.
  using DepthTree = internal::intrusive_red_black_tree::Tree<Entry>;

 public:
  class Entry : public internal::HashSetOfAny::Entry,
                public DepthTree::NodeBase {
   public:
    Entry(size_t nesting_depth) : nesting_depth_(nesting_depth) {}

    // Submits the batch, and is responsible for destroying the entry when done.
    virtual void Submit(Batch::View batch) = 0;

   private:
    friend class Batch;
    size_t nesting_depth_;
    Entry* next_at_same_depth_;
  };

  /// Returns the batch entry of type `derived_type` with the specified `key`,
  /// creating it if necessary.
  ///
  /// \tparam DerivedEntry The derived entry type that defines a `key` member.
  /// \param key The key to lookup for `DerivedEntry`.
  /// \param make_entry Function with signature
  ///     `std::unique_ptr<DerivedEntry>()` to be called if `key` is not
  ///     present.  The returned pointer must have typeid equal to
  ///     `derived_type`.
  /// \param derived_type Actual typeid of entry returned by `make_entry`.  The
  ///     corresponding type must be `DerivedEntry` or a subclass.
  template <typename DerivedEntry, typename MakeEntry>
  DerivedEntry& GetEntry(
      typename DerivedEntry::KeyParam key, MakeEntry make_entry,
      const std::type_info& derived_type = typeid(DerivedEntry)) {
    static_assert(std::is_base_of_v<Entry, DerivedEntry>);
    absl::MutexLock lock(&mutex_);
    auto [entry, inserted] =
        entries_.FindOrInsert<DerivedEntry>(key, make_entry, derived_type);
    if (inserted) {
      InsertIntoDepthTree(*entry);
    }
    return *entry;
  }

  static Impl* From(View batch) { return static_cast<Impl*>(batch.impl_); }
  constexpr static Batch::View ToBatch(Impl* impl) {
    Batch::View batch;
    batch.impl_ = impl;
    return batch;
  }

  ~Impl();

 private:
  void InsertIntoDepthTree(Entry& entry);

  absl::Mutex mutex_;
  internal::HashSetOfAny entries_;
  DepthTree nesting_depths_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_READ_BATCH_IMPL_H_
