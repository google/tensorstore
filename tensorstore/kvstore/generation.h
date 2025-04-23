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

#include <stdint.h>

#include <cstring>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/time/time.h"
#include "tensorstore/serialization/fwd.h"

namespace tensorstore {

/// Represents a generation identifier associated with a stored object.
///
/// The generation identifier must change each time an object with a given key
/// is updated. Generation identifiers are *not* comparable across keys: for two
/// different keys, the same generation identifier *may* correspond to different
/// values.
///
/// A `StorageGeneration` should be treated as an opaque identifier, with its
/// length and content specific to the particular storage system.
///
/// For example:
///
/// - t=0: Object does not exist, implicitly has generation G0, which may equal
///   `StorageGeneration::NoValue()`.
///
/// - t=1: Value V1 is written, storage system assigns generation G1.
///
/// - t=2: Value V2 is written, storage system assigns generation G2, which must
///   not equal G1.
///
/// - t=3: Value V1 is written again, storage system assigns generation G3,
///   which must not equal G1 or G2.
///
/// - t=4: Object is deleted, implicitly has generation G4, which may equal G0
///   and/or `StorageGeneration::NoValue()`.
///
/// Note: Some storage implementations always use a generation of
/// `StorageGeneration::NoValue()` for not-present objects.  Other storage
/// implementations may use other generations to indicate not-present objects.
/// For example, if some sharding scheme is used pack multiple objects together
/// into a "shard", the generation associated with a not-present object may be
/// the generation of the shard in which the object is missing.
///
/// \ingroup kvstore
struct StorageGeneration {
  /// A storage generation is encoded as a variable-length that specifies an
  /// optional "base generation" and information about uncommitted modifications
  /// made within a transaction.
  //
  // Logically it specifies:
  //
  // - A sequence of uncommitted mutations (in reverse order of application),
  //   each specifying:
  //
  //   - A 64-bit mutation identifier, guaranteed to be unique within the
  //     process.
  //
  //   - A flag indicating if this mutation is the start of a new "layer" (in
  //     order of inner to outer). This flag is used by key-value store adapters
  //     like zarr3_sharding_indexed to distinguish between mutations made
  //     within a given transaction node.
  //
  // - A base generation, specifying information about the commited state of a
  //   key, which may be:
  //
  //   - A regular base generation
  //
  //   - The special "no value" generation
  //
  //   - An empty string, indicating an unspecified base generation.
  //
  // The encoding is:
  //
  // - Empty string, to indicate an unspecified base generation; or
  //
  // - A byte sequence consisting of:
  //
  //   - One or more "tags":
  //
  //     - 1 indicator byte where:
  //
  //       - bits 0-1 => indicate the final tag status
  //
  //           0b01 -> non-final tag.
  //
  //           0b00 -> final tag, and the base generation is:
  //                   - unknown (if 0 remaining bytes); or
  //                   - a normal base generation (>0 remaining bytes)
  //
  //           0b10 -> final tag, base generation is "no value".
  //                   The number of remaining bytes must be 0.
  //
  //           0b11 -> invalid.
  //
  //       - bit 2 => mutation bit
  //
  //           1 -> mutation id is included in the tag.
  //           0 -> no mutation, must be the final tag.
  //
  //       - bit 3 => layer bit
  //
  //           1 -> start of new layer, mutation bit must be set.
  //
  //       - remaining bits are currently unused and must be 0.
  //
  //     - If the mutation bit specified that a mutation id stored, the
  //       indicator byte is followed by the 8 byte little-endian mutation id.
  //
  //   - After the final tag, the remaining bytes specify the base generation.
  //
  // For example, suppose we have a zarr3_sharding_indexed kvstore where:
  //
  // - Base generation is "base".
  // - Mutation 1 modifies the shard file directly
  // - Mutation 2 also modifies the shard file directly
  // - Mutation 3 modifies an individual entry.
  //
  // Then the generation would be represented as:
  //
  // Debug string: M3+|M2+M1+"base"
  //
  // - mutation_id=3, new_layer=false
  // - mutation_id=2, new_layer=true
  // - mutation_id=1, new_layer=false
  // - base_generation="base"
  //
  // In the case that the zarr3_sharding_indexed is nested, and each entry
  // is itself in zarr3_sharding_indexed format, then we could have:
  //
  // - Base generation is "base".
  // - Mutation 1 modifies the shard file directly
  // - Mutation 2 also modifies the shard file directly
  // - Mutation 3 modifies an individual entry.
  // - Mutation 4 modifies an individual nested entry within the individual
  //   entry.
  //
  // Then the generation would be represented as:
  //
  // Debug string: M4+|M3+|M2+M1+"base"
  //
  // - mutation_id=4, new_layer=false
  // - mutation_id=3, new_layer=true
  // - mutation_id=2, new_layer=true
  // - mutation_id=1, new_layer=false
  // - base_generation="base"
  std::string value;

  /// Indicates whether a generation is specified.
  explicit operator bool() const { return !value.empty(); }

  /// Returns a debug representation of the storage generation.
  //
  // The debug representation has the following grammer:
  //
  // generation :=
  //     valid_generation
  //   | "invalid:" quoted_string
  //
  // valid_generation :=
  //     ( layer_end? mutation "+" )* base_generation
  //
  // base_generation :=
  //     quoted_string | "Unknown" | "NoValue"
  //
  // layer_end :=
  //     "|"
  //
  // mutation :=
  //     "M" [0-9]+
  std::string DebugString() const;

  /// Check if the generation is valid, i.e. not derived from `Invalid()`.
  bool IsValid() const;

  /// Returns the special generation value that indicates the StorageGeneration
  /// is unspecified.
  static StorageGeneration Unknown() { return {}; }

  /// Returns the special generation value that corresponds to an object not
  /// being present.
  static StorageGeneration NoValue() {
    return StorageGeneration{std::string(1, kNoValue)};
  }

  /// Returns the invalid generation value guaranteed not to equal any valid
  /// generation.
  static StorageGeneration Invalid() {
    return StorageGeneration{std::string(1, kInvalid)};
  }

  // Inverse of `FromString`.
  //
  // Returns an empty string if `generation` could not have been created by
  // `FromString`.
  static std::string_view DecodeString(const StorageGeneration& generation);

  /// Checks if `if_equal` is unspecified, or equal to `generation`.
  static bool EqualOrUnspecified(const StorageGeneration& generation,
                                 const StorageGeneration& if_equal) {
    return StorageGeneration::IsUnknown(if_equal) ||
           generation.value == if_equal.value;
  }

  /// Checks if `if_not_equal` is unspecified, or not equal to `generation`.
  static bool NotEqualOrUnspecified(const StorageGeneration& generation,
                                    const StorageGeneration& if_not_equal) {
    return StorageGeneration::IsUnknown(if_not_equal) ||
           generation.value != if_not_equal.value;
  }

  /// Returns `true` if `generation` is equal to the special
  /// `StorageGeneration::Unknown()` value, i.e. an empty string.
  ///
  /// This usually indicates an unspecified generation; in
  /// `kvstore::ReadGenerationConditions::if_equal` and
  /// `kvstore::ReadGenerationConditions::if_not_equal`, it indicates
  /// that the condition does not apply.
  static bool IsUnknown(const StorageGeneration& generation) {
    return generation.value.empty();
  }

  /// Returns `true` if `generation` is equal to the special `NoValue()`
  /// generation.
  ///
  /// .. warning::
  ///
  ///    While all kvstore drivers support `StorageGeneration::NoValue()` in
  ///    store::ReadGenerationConditions::if_equal` and
  ///    `kvstore::ReadGenerationConditions::if_not_equal`, some kvstore drivers
  ///    may return a different generation for missing values.
  static bool IsNoValue(const StorageGeneration& generation) {
    return generation.value.size() == 1 && generation.value[0] == kNoValue;
  }

  /// Returns `true` if `generation` represents a "clean" state without any
  /// uncommitted modifications within a transaction.
  ///
  /// Note that this returns `true` for `StorageGeneration::NoValue()` and
  /// `StorageGeneration::Invalid()`.
  static bool IsClean(const StorageGeneration& generation);

  /// Checks if two generations are equivalent.
  friend inline bool operator==(const StorageGeneration& a,
                                const StorageGeneration& b) {
    return Equivalent(a.value, b.value);
  }
  friend inline bool operator==(std::string_view a,
                                const StorageGeneration& b) {
    return Equivalent(a, b.value);
  }
  friend inline bool operator==(const StorageGeneration& a,
                                std::string_view b) {
    return Equivalent(a.value, b);
  }
  friend inline bool operator!=(const StorageGeneration& a,
                                const StorageGeneration& b) {
    return !(a == b);
  }
  friend inline bool operator!=(const StorageGeneration& a,
                                std::string_view b) {
    return !(a == b);
  }
  friend inline bool operator!=(std::string_view a,
                                const StorageGeneration& b) {
    return !(a == b);
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const StorageGeneration& g);

  // ---------------------------------------------------
  // Tensorstore-internal details follow

  // Returns `true` if this is a clean generation, not locally-modified,
  // not `StorageGeneration::NoValue()` and not `StorageGeneration::Invalid()`.
  static bool IsCleanValidValue(const StorageGeneration& generation) {
    return !generation.value.empty() && generation.value[0] == 0;
  }

  // Returns a base generation that encodes the specified 64-bit number.
  static StorageGeneration FromUint64(uint64_t n);

  // Validates that `generation` may have been constructed from `FromUint64`.
  static bool IsUint64(const StorageGeneration& generation) {
    return generation.value.size() == 9 && generation.value[0] == 0;
  }

  // Inverse of `FromUint64`.
  //
  // If `!IsUint64(generation)`, returns `0`.
  static uint64_t ToUint64(const StorageGeneration& generation) {
    uint64_t n = 0;
    if (IsUint64(generation)) {
      std::memcpy(&n, &generation.value[1], 8);
    }
    return n;
  }

  // Returns a base generation that encodes the specified string.
  static StorageGeneration FromString(std::string_view s);

  // Uniquely identifies an operation within a transaction that mutates a value.
  using MutationId = uint64_t;

  // Allocates a `MutationId` that is unique for the duration of the process.
  //
  // A mutation id is generally allocated for each `ReadModifyWriteSource`.
  static MutationId AllocateMutationId();

  // Special mutation id that may be used only to indicate an unconditional
  // deletion of the value. In all other cases a fresh mutation id must be
  // allocated.
  constexpr static MutationId kDeletionMutationId = 0;

  // Mask for tag indicator byte that indicates an additional tag follows this
  // tag.
  constexpr static char kAdditionalTags = 1;

  // Mask for tag indicator byte that indicates the base generation is the
  // special "no value" generation.
  constexpr static char kNoValue = 2;

  // Mask for tag indicator byte that indicates a mutation id is stored.
  constexpr static char kMutation = 4;

  // Mask for tag indicator byte that indicates this is the first tag within its
  // layer.
  constexpr static char kNewLayer = 8;

  // Included in the tag indicator byte to indicate an invalid generation.
  constexpr static char kInvalid = 16;

  // Returns the base generation (without a tag), or `std::nullopt` if the base
  // generation is NoValue, or an empty string if the base generation is
  // unspecified.
  std::optional<std::string_view> BaseGeneration() const;

  // Returns a base generation that encodes one or more trivial values via
  // memcpy.
  //
  // \param value Value to encode.  If the type is `std::string_view` or
  //     `std::string`, the contents will be encoded directly (without any
  //     length indicator).  Otherwise, the value is assume to be a trivial
  //     type and will be encoded via `std::memcpy`.
  template <typename... T>
  static StorageGeneration FromValues(const T&... value) {
    constexpr auto as_string_view = [](const auto& value) -> std::string_view {
      using value_type = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<std::string_view, value_type> ||
                    std::is_same_v<std::string, value_type>) {
        return value;
      } else {
        static_assert(std::is_trivial_v<value_type>);
        return std::string_view(reinterpret_cast<const char*>(&value),
                                sizeof(value));
      }
    };
    const size_t n = (as_string_view(value).size() + ...);
    StorageGeneration gen;
    gen.value.resize(n + 1);
    size_t offset = 1;
    const auto copy_value = [&](const auto& value) {
      auto s = as_string_view(value);
      std::memcpy(&gen.value[offset], s.data(), s.size());
      offset += s.size();
    };
    (copy_value(value), ...);
    return gen;
  }

  // Checks if the latest mutation id is equal to `id`.
  bool LastMutatedBy(MutationId id) const;

  // Adds the specified mutation id as the latest mutation.
  void MarkDirty(MutationId mutation_id);
  static StorageGeneration Dirty(StorageGeneration generation,
                                 MutationId mutation_id);

  // Checks if `generation` is equal to `Dirty(base, mutation_id)`.
  static bool IsDirtyOf(const StorageGeneration& generation,
                        const StorageGeneration& base, MutationId mutation_id);

  // Strips off the latest mutation tag.
  //
  // Returns `generation` unchanged if there are no mutations.
  static StorageGeneration StripTag(const StorageGeneration& generation);

  // Checks if `generation` represents a locally-modified value.
  static bool IsDirty(const StorageGeneration& generation);

  // Returns the "clean" base generation on which `generation` is based.
  //
  // If `generation` is already a "clean" state, it is returned as is.  If
  // `generation` indicates local modifications, returns the generation on
  // which those local modifications are conditioned.
  //
  // This returns the fixed point of `StripTag`.
  static StorageGeneration Clean(StorageGeneration generation);

  // Determines if two generations are equal, ignoring `kNewLayer` indicators.
  static bool Equivalent(std::string_view a, std::string_view b);

  // Propagates writeback conditions from `condition` to `generation`.
  //
  // If `generation` is already conditioned on a base generation, i.e.
  // `IsConditional(generation)`, it is returned unchanged.
  //
  // Otherwise:
  //
  // 1. Any current-layer mutations in `condition` are removed, i.e.
  //    `condition = AddLayer(StripLayer(condition))`.
  //
  // 2. Applies all of the mutation tags in `generation` (including any
  //    `kNewLayer` markers) to `condition`. Thus, the latest mutation of
  //    `generation`, if any, becomes the latest mutation of `condition`.
  //
  // The new `condition` generation is then returned.
  //
  // Note:
  //
  //     If an individual read-modify-write operation produces an unconditional
  //     writeback result, but there is a prior operation in the same
  //     transaction that imposes constraints on the existing value (e.g.
  //     read_repeatable or certain constraints on array metadata), then this
  //     function is used to ensure that the overall writeback result for that
  //     key is appropriately marked as conditionl, to ensure that the necessary
  //     validation is performed.
  //
  //     This is used internally by the kvstore transaction machinery and by
  //     kvs_backed_cache.
  static StorageGeneration Condition(const StorageGeneration& generation,
                                     StorageGeneration condition);

  // Adds another layer of local modification to `generation`.
  //
  // This is used for kvstore adapters like zarr3_sharding_indexed and
  // neuroglancer_uint64_sharded.
  static StorageGeneration AddLayer(StorageGeneration generation);

  // Strips off any mutations from the inner (latest) layer.
  static StorageGeneration StripLayer(StorageGeneration generation);

  // Checks if the innermost layer of `generation` is locally-modified.
  static bool IsInnerLayerDirty(const StorageGeneration& generation);

  // Checks if `Clean(generation) != Unknown()`.
  static bool IsConditional(const StorageGeneration& generation);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.value);
  };

  // Abseil hash support.
  template <typename H>
  friend H AbslHashValue(H h, const StorageGeneration& x) {
    return H::combine(std::move(h), x.value);
  }
};

/// Combines a local timestamp with a StorageGeneration indicating the local
/// time for which the generation is known to be current.
///
/// \ingroup kvstore
struct TimestampedStorageGeneration {
  /// Constructs with an unspecified generation and infinite past timestamp.
  ///
  /// \id default
  TimestampedStorageGeneration() = default;

  /// Constructs from the specified generation and time.
  ///
  /// \id generation, time
  TimestampedStorageGeneration(StorageGeneration generation, absl::Time time)
      : generation(std::move(generation)), time(std::move(time)) {}

  /// Storage generation.
  StorageGeneration generation;

  /// Timestamp associated with `generation`.
  absl::Time time = absl::InfinitePast();

  /// Checks if `time` is equal to the infinite future value.
  bool unconditional() const { return time == absl::InfiniteFuture(); }

  /// Returns a timestamped generation with unspecified generation and infinite
  /// future timestamp.
  static TimestampedStorageGeneration Unconditional() {
    return {StorageGeneration::Unknown(), absl::InfiniteFuture()};
  }

  /// Compares two timestamped generations for equality.
  friend bool operator==(const TimestampedStorageGeneration& a,
                         const TimestampedStorageGeneration& b) {
    return a.generation == b.generation && a.time == b.time;
  }
  friend bool operator!=(const TimestampedStorageGeneration& a,
                         const TimestampedStorageGeneration& b) {
    return !(a == b);
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const TimestampedStorageGeneration& x);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.generation, x.time);
  };
};

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::StorageGeneration)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::TimestampedStorageGeneration)

#endif  // TENSORSTORE_KVSTORE_GENERATION_H_
