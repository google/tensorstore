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

#ifndef TENSORSTORE_KVSTORE_GENERATION_H_
#define TENSORSTORE_KVSTORE_GENERATION_H_

#include <cstring>
#include <iosfwd>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tensorstore {

/// Represents a generation identifier associated with a stored object.
///
/// The generation identifier must change each time an object is updated.
///
/// A `StorageGeneration` should be treated as an opaque identifier, with its
/// length and content specific to the particular storage system, except that
/// there are three special values:
///
///   `StorageGeneration::Unknown()`, equal to the empty string, which can be
///   used to indicate an unspecified generation.
///
///   `StorageGeneration::NoValue()`, equal to `{0}`, which can be used to
///   specify a condition that the object not exist.
///
///   `StorageGeneration::Invalid()`, equal to `{1}`, which must not match any
///   valid generation.
///
/// A storage implementation must ensure that the encoding of generations does
/// not conflict with these two special generation values.
///
/// For example:
///
/// t=0: Object does not exist, implicitly has generation G0, which may equal
///      `StorageGeneration::NoValue()`.
///
/// t=1: Value V1 is written, storage system assigns generation G1.
///
/// t=2: Value V2 is written, storage system assigns generation G2, which must
///      not equal G1.
///
/// t=3: Value V1 is written again, storage system assigns generation G3, which
///      must not equal G1 or G2.
///
/// t=4: Object is deleted, implicitly has generation G4, which may equal G0
///      and/or `StorageGeneration::NoValue()`.
///
/// Note: Some storage implementations always use a generation of
/// `StorageGeneration::NoValue()` for not-present objects.  Other storage
/// implementations may use other generations to indicate not-present objects.
/// For example, if some sharding scheme is used pack multiple objects together
/// into a "shard", the generation associated with a not-present object may be
/// the generation of the shard in which the object is missing.
struct StorageGeneration {
  /// The generation is indicated by a variable-length byte string (the
  /// "generation identifier") followed by a sequence of bytes containing a
  /// bitwise-OR combination of the flags defined below:
  ///
  /// The flag bytes are parsed from the end of `value` in reverse.  The start
  /// of the flag bytes is indicated by a flag byte with `kBaseGeneration` set.
  /// For example, clean generation with identifier `gen_id`:
  ///
  ///     gen_id + kBaseGeneration
  ///
  /// Dirty generation with identifier `gen_id`:
  ///
  ///     gen_id + (kBaseGeneration|kDirty)
  ///
  /// Clean "no value" generation indicating a key not present in the store:
  ///
  ///     gen_id + (kBaseGeneration|kNoValue)
  ///
  /// Dirty generation derived from "no value" generation:
  ///
  ///     gen_id + (kBaseGeneration|kNoValue|kDirty)
  ///
  /// In the case of multiple layers of modifications, e.g. the
  /// neuroglancer_precomputed uint64_sharded_key_value_store, where we need to
  /// separately track whether an individual chunk within a shard is dirty, but
  /// also whether the full shard on which the chunk is based was also dirty
  /// from a prior `ReadModifyWriteSource`.  This is done by appending
  /// additional flag bytes with `kBaseGeneration` not set.  Each flag byte is
  /// referred to as a layer and corresponds to a sequence of potential
  /// modifications by `ReadModifyWriteSource` objects. The last flag byte
  /// corresponds to the innermost layer.  In the case of
  /// uint64_sharded_key_value_store, modifications to an individual chunk
  /// correspond to the inner layer, while prior modifications to the entire
  /// shard correspond to the outer layer.
  ///
  /// Generation that is dirty in inner layer, but not in outer layer:
  ///
  ///     gen_id + kBaseGeneration + kDirty
  ///
  /// Generation that is dirty in outer layer, but not in inner layer:
  ///
  ///     gen_id + (kBaseGeneration|kDirty) + '\x00'
  ///
  /// The `kNewlyDirty` flag must only be used as a temporary flag when calling
  /// `ReadModifyWriteSource::WritebackReceiver`, and must only be set on the
  /// last flag byte.  A read operation must not return a generation with
  /// `kNewlyDirty` set.
  std::string value;

  constexpr static char kBaseGeneration = 1;
  constexpr static char kDirty = 2;
  constexpr static char kNewlyDirty = 16;
  constexpr static char kNoValue = 4;
  constexpr static char kInvalid = 8;

  /// Returns the special generation value that indicates the StorageGeneration
  /// is unspecified.
  static StorageGeneration Unknown() { return {}; }

  /// Returns the special generation value that corresponds to an object not
  /// being present.
  static StorageGeneration NoValue() {
    return StorageGeneration{std::string(1, kBaseGeneration | kNoValue)};
  }

  /// Returns the invalid generation value guaranteed not to equal any valid
  /// generation.
  static StorageGeneration Invalid() {
    return StorageGeneration{std::string(1, kInvalid)};
  }

  /// Returns a base generation that encodes the specified 64-bit number.
  static StorageGeneration FromUint64(uint64_t n);

  /// Validates that `generation` may have been constructed from `FromUint64`.
  static bool IsUint64(const StorageGeneration& generation) {
    return generation.value.size() == 9 &&
           generation.value.back() == kBaseGeneration;
  }

  static uint64_t ToUint64(const StorageGeneration& generation) {
    uint64_t n = 0;
    if (IsUint64(generation)) {
      std::memcpy(&n, generation.value.data(), 8);
    }
    return n;
  }

  /// Modifies this generation to set `kDirty` and `kNewlyDirty` in the last
  /// (innermost) flag byte.
  void MarkDirty();

  static bool IsNewlyDirty(const StorageGeneration& generation) {
    return !generation.value.empty() && (generation.value.back() & kNewlyDirty);
  }

  bool ClearNewlyDirty() {
    bool is_newly_dirty = IsNewlyDirty(*this);
    if (is_newly_dirty) {
      value.back() &= ~kNewlyDirty;
    }
    return is_newly_dirty;
  }

  /// Returns a base generation that encodes the specified string.
  static StorageGeneration FromString(std::string_view s);

  /// Returns a base generation that encodes one or more trivial values via
  /// memcpy.
  template <typename... T>
  static StorageGeneration FromValues(const T&... value) {
    const size_t n = (sizeof(T) + ...);
    StorageGeneration gen;
    gen.value.resize(n + 1);
    size_t offset = 0;
    ((std::memcpy(&gen.value[offset], &value, sizeof(T)), offset += sizeof(T)),
     ...);
    gen.value[n] = kBaseGeneration;
    return gen;
  }

  static StorageGeneration Dirty(StorageGeneration generation);

  static StorageGeneration Clean(StorageGeneration generation);

  /// Determines if two generations are equivalent by comparing their canonical
  /// generations.
  static bool Equivalent(std::string_view a, std::string_view b);

  static bool IsDirty(const StorageGeneration& generation);

  static bool IsInnerLayerDirty(const StorageGeneration& generation);

  /// Returns a `new_generation` for which
  /// `IsModified(new_generation) == IsModified(generation)` and
  /// `BaseGeneration(new_generation) == BaseGeneration(condition)`.
  static StorageGeneration Condition(const StorageGeneration& generation,
                                     StorageGeneration condition);

  static StorageGeneration AddLayer(StorageGeneration generation);

  static bool IsConditional(const StorageGeneration& generation);

  static bool EqualOrUnspecified(const StorageGeneration& generation,
                                 const StorageGeneration& if_equal);

  static bool NotEqualOrUnspecified(const StorageGeneration& generation,
                                    const StorageGeneration& if_not_equal);

  static bool IsUnknown(const StorageGeneration& generation) {
    return generation.value.empty();
  }

  static bool IsClean(const StorageGeneration& generation) {
    return !generation.value.empty() &&
           (generation.value.back() & (kBaseGeneration | kDirty)) ==
               kBaseGeneration;
  }

  static bool IsNoValue(const StorageGeneration& generation) {
    return generation.value.size() == 1 &&
           generation.value[0] == (kNoValue | kBaseGeneration);
  }

  friend inline bool operator==(const StorageGeneration& a,
                                const StorageGeneration& b) {
    return Equivalent(a.value, b.value);
  }
  friend inline bool operator==(absl::string_view a,
                                const StorageGeneration& b) {
    return Equivalent(a, b.value);
  }
  friend inline bool operator==(const StorageGeneration& a,
                                absl::string_view b) {
    return Equivalent(a.value, b);
  }

  friend inline bool operator!=(const StorageGeneration& a,
                                const StorageGeneration& b) {
    return !(a == b);
  }
  friend inline bool operator!=(const StorageGeneration& a,
                                absl::string_view b) {
    return !(a == b);
  }
  friend inline bool operator!=(absl::string_view a,
                                const StorageGeneration& b) {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os, const StorageGeneration& g);
};

/// Combines a local timestamp with a StorageGeneration indicating the local
/// time for which the generation is known to be current.
struct TimestampedStorageGeneration {
  TimestampedStorageGeneration() = default;

  TimestampedStorageGeneration(StorageGeneration generation, absl::Time time)
      : generation(std::move(generation)), time(std::move(time)) {}
  StorageGeneration generation;
  absl::Time time = absl::InfinitePast();

  bool unconditional() const { return time == absl::InfiniteFuture(); }

  static TimestampedStorageGeneration Unconditional() {
    return {StorageGeneration::Unknown(), absl::InfiniteFuture()};
  }

  friend bool operator==(const TimestampedStorageGeneration& a,
                         const TimestampedStorageGeneration& b) {
    return a.generation == b.generation && a.time == b.time;
  }
  friend bool operator!=(const TimestampedStorageGeneration& a,
                         const TimestampedStorageGeneration& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const TimestampedStorageGeneration& x);
};

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GENERATION_H_
