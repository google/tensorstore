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

#ifndef TENSORSTORE_OPEN_MODE_H_
#define TENSORSTORE_OPEN_MODE_H_

#include <iosfwd>
#include <optional>

#include "tensorstore/staleness_bound.h"

namespace tensorstore {

enum class OpenMode {
  /// Open an existing TensorStore.  Unless `create` is also specified, a
  /// non-existent TensorStore will result in an error.
  open = 1,

  /// Create a new TensorStore.  Unless `open` is also specified, an existing
  /// TensorStore will result in an error.
  create = 2,

  /// If the TensorStore already exists, delete it.  This is only valid in
  /// conjunction with `create`.
  delete_existing = 4,

  /// Open succeeds even if the specified creation properties do not match those
  /// of the existing TensorStore.  This is only valid in conjunction with
  /// `open`.
  allow_option_mismatch = 8,

  /// Open an existing TensorStore or create a new TensorStore if it does not
  /// exist.
  open_or_create = open + create,
};

constexpr inline OpenMode operator&(OpenMode a, OpenMode b) {
  return static_cast<OpenMode>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr inline OpenMode operator|(OpenMode a, OpenMode b) {
  return static_cast<OpenMode>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr inline bool operator!(OpenMode a) { return !static_cast<int>(a); }

std::ostream& operator<<(std::ostream& os, OpenMode mode);

enum class ReadWriteMode {
  /// Indicates that the mode is unspecified (only used at compile time).
  dynamic = 0,
  /// Indicates that reading is supported.
  read = 1,
  /// Indicates that writing is supported.
  write = 2,
  /// Indicates that reading and writing are supported.
  read_write = 3,
};

constexpr inline ReadWriteMode operator&(ReadWriteMode a, ReadWriteMode b) {
  return static_cast<ReadWriteMode>(static_cast<int>(a) & static_cast<int>(b));
}

constexpr inline ReadWriteMode& operator&=(ReadWriteMode& a, ReadWriteMode b) {
  return a = (a & b);
}

constexpr inline ReadWriteMode operator|(ReadWriteMode a, ReadWriteMode b) {
  return static_cast<ReadWriteMode>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr inline ReadWriteMode& operator|=(ReadWriteMode& a, ReadWriteMode b) {
  return a = (a | b);
}

constexpr inline bool operator!(ReadWriteMode a) {
  return !static_cast<int>(a);
}

constexpr inline ReadWriteMode operator~(ReadWriteMode a) {
  return static_cast<ReadWriteMode>(
      ~static_cast<std::underlying_type_t<ReadWriteMode>>(a));
}

constexpr inline bool IsModeExplicitlyConvertible(ReadWriteMode source,
                                                  ReadWriteMode target) {
  return (target == ReadWriteMode::dynamic ||
          source == ReadWriteMode::dynamic || (target & source) == target);
}

absl::string_view to_string(ReadWriteMode mode);
std::ostream& operator<<(std::ostream& os, ReadWriteMode mode);

/// Specifies options for opening a TensorStore.
struct OpenOptions {
  OpenOptions() = default;

  OpenOptions(std::optional<OpenMode> open_mode,
              ReadWriteMode read_write_mode = ReadWriteMode::dynamic,
              std::optional<StalenessBounds> staleness = {})
      : open_mode(open_mode),
        read_write_mode(read_write_mode),
        staleness(staleness) {}

  OpenOptions(OpenMode open_mode,
              ReadWriteMode read_write_mode = ReadWriteMode::dynamic,
              std::optional<StalenessBounds> staleness = {})
      : open_mode(open_mode),
        read_write_mode(read_write_mode),
        staleness(staleness) {}

  OpenOptions(ReadWriteMode read_write_mode,
              std::optional<StalenessBounds> staleness = {})
      : read_write_mode(read_write_mode), staleness(staleness) {}

  OpenOptions(std::optional<StalenessBounds> staleness)
      : staleness(staleness) {}

  std::optional<OpenMode> open_mode;
  ReadWriteMode read_write_mode = ReadWriteMode::dynamic;
  std::optional<StalenessBounds> staleness;
};

namespace internal {

/// Returns the mask of potentially supported read/write modes given a
/// compile-time `mode` value.
///
/// If `mode` is `dynamic`, then no information is known at compile time about
/// the supported modes, and this returns `read_write`.  Otherwise, this returns
/// `mode`.
constexpr ReadWriteMode StaticReadWriteMask(ReadWriteMode mode) {
  return mode == ReadWriteMode::dynamic ? ReadWriteMode::read_write : mode;
}

/// Returns `true` if, and only if, `mode` is compatible with the mode
/// constraint `constraint`.
constexpr bool IsModePossible(ReadWriteMode mode, ReadWriteMode constraint) {
  return constraint == ReadWriteMode::dynamic ? mode != ReadWriteMode::dynamic
                                              : mode == constraint;
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_OPEN_MODE_H_
