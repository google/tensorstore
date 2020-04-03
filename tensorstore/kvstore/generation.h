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
  std::string value;

  /// Returns the special generation value that indicates the StorageGeneration
  /// is unspecified.
  static StorageGeneration Unknown() { return {}; }

  /// Returns the special generation value that corresponds to an object not
  /// being present.
  static StorageGeneration NoValue() {
    return StorageGeneration{std::string(1, '\0')};
  }

  /// Returns the invalid generation value guaranteed not to equal any valid
  /// generation.
  static StorageGeneration Invalid() {
    return StorageGeneration{std::string(1, '\1')};
  }

  static bool IsUnknown(const StorageGeneration& generation) {
    return generation.value.empty();
  }

  static bool IsNoValue(const StorageGeneration& generation) {
    return generation.value.size() == 1 && generation.value[0] == 0;
  }

  friend inline bool operator==(const StorageGeneration& a,
                                const StorageGeneration& b) {
    return a.value == b.value;
  }
  friend inline bool operator==(absl::string_view a,
                                const StorageGeneration& b) {
    return a == b.value;
  }
  friend inline bool operator==(const StorageGeneration& a,
                                absl::string_view b) {
    return a.value == b;
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
  absl::Time time;

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
