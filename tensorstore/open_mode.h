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

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/schema.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/option.h"

namespace tensorstore {

/// Specifies the mode to use when opening a `TensorStore`.
///
/// \relates Spec
enum class OpenMode {
  /// Open an existing `TensorStore`.  Unless `create` is also specified, a
  /// non-existent TensorStore will result in an error.
  open = 1,

  /// Create a new TensorStore.  Unless `open` is also specified, an existing
  /// TensorStore will result in an error.
  create = 2,

  /// If the TensorStore already exists, delete it.  This is only valid in
  /// conjunction with `create`.
  delete_existing = 4,

  /// Open an existing TensorStore or create a new TensorStore if it does not
  /// exist.
  open_or_create = open + create,

  /// When opening a TensorStore, skip reading the metadata if possible.
  /// Instead, just assume any necessary metadata based on constraints in the
  /// spec and any defaults used by TensorStore.  This option requires care as
  /// it can lead to data corruption if the assumed metadata does not match the
  /// stored metadata, or multiple concurrent writers use different assumed
  /// metadata.
  assume_metadata = 8,
};

/// Returns the intersection of two open modes.
///
/// \relates OpenMode
/// \id OpenMode
constexpr inline OpenMode operator&(OpenMode a, OpenMode b) {
  return static_cast<OpenMode>(static_cast<int>(a) & static_cast<int>(b));
}

/// Returns the union of two open modes.
///
/// \relates OpenMode
/// \id OpenMode
constexpr inline OpenMode operator|(OpenMode a, OpenMode b) {
  return static_cast<OpenMode>(static_cast<int>(a) | static_cast<int>(b));
}

/// Checks if any open mode has been set.
///
/// \relates OpenMode
/// \id OpenMode
constexpr inline bool operator!(OpenMode a) { return !static_cast<int>(a); }

/// Prints a string representation the mode to an `std::ostream`.
///
/// \relates OpenMode
/// \id OpenMode
std::ostream& operator<<(std::ostream& os, OpenMode mode);

/// Specifies whether reading and/or writing is permitted.
///
/// \relates TensorStore
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

/// Computes the intersection of two modes.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
constexpr inline ReadWriteMode operator&(ReadWriteMode a, ReadWriteMode b) {
  return static_cast<ReadWriteMode>(static_cast<int>(a) & static_cast<int>(b));
}
constexpr inline ReadWriteMode& operator&=(ReadWriteMode& a, ReadWriteMode b) {
  return a = (a & b);
}

/// Computes the union of two modes.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
constexpr inline ReadWriteMode operator|(ReadWriteMode a, ReadWriteMode b) {
  return static_cast<ReadWriteMode>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr inline ReadWriteMode& operator|=(ReadWriteMode& a, ReadWriteMode b) {
  return a = (a | b);
}

/// Checks if the mode is not equal to `ReadWriteMode::dynamic`.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
constexpr inline bool operator!(ReadWriteMode a) {
  return !static_cast<int>(a);
}

/// Returns the complement of a mode.
////
/// \relates ReadWriteMode
/// \id ReadWriteMode
constexpr inline ReadWriteMode operator~(ReadWriteMode a) {
  return static_cast<ReadWriteMode>(
      ~static_cast<std::underlying_type_t<ReadWriteMode>>(a));
}

/// Checks if `source` is potentially compatible with `target`.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
constexpr inline bool IsModeExplicitlyConvertible(ReadWriteMode source,
                                                  ReadWriteMode target) {
  return (target == ReadWriteMode::dynamic ||
          source == ReadWriteMode::dynamic || (target & source) == target);
}

/// Returns a string representation of the mode.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
std::string_view to_string(ReadWriteMode mode);

/// Prints a string representation of the mode to an `std::ostream`.
///
/// \relates ReadWriteMode
/// \id ReadWriteMode
std::ostream& operator<<(std::ostream& os, ReadWriteMode mode);

/// Indicates a minimal spec, i.e. missing information necessary to recreate.
///
/// This is an option for use with interfaces that accept `SpecRequestOptions`.
///
/// \relates Spec
class MinimalSpec {
 public:
  constexpr explicit MinimalSpec(bool minimal_spec = true)
      : minimal_spec_(minimal_spec) {}
  bool minimal_spec() const { return minimal_spec_; }

 private:
  bool minimal_spec_;
};

/// Options for mutating `Spec` objects.
///
/// \relates Spec
struct SpecOptions : public Schema {
  OpenMode open_mode = {};
  RecheckCachedData recheck_cached_data;
  RecheckCachedMetadata recheck_cached_metadata;
  bool minimal_spec = false;
  kvstore::Spec kvstore;

  /// Excludes `Schema`, `MinimalSpec`, and `kvstore::Spec`.
  template <typename T>
  constexpr static inline bool IsCommonOption = false;

  absl::Status Set(OpenMode mode) {
    open_mode = open_mode | mode;
    return absl::OkStatus();
  }

  absl::Status Set(RecheckCachedData value) {
    if (value.specified()) {
      recheck_cached_data = value;
    }
    return absl::OkStatus();
  }

  absl::Status Set(RecheckCachedMetadata value) {
    if (value.specified()) {
      recheck_cached_metadata = value;
    }
    return absl::OkStatus();
  }

  absl::Status Set(RecheckCached value) {
    if (value.specified()) {
      static_cast<RecheckCacheOption&>(recheck_cached_data) = value;
      static_cast<RecheckCacheOption&>(recheck_cached_metadata) = value;
    }
    return absl::OkStatus();
  }

  absl::Status Set(kvstore::Spec value) {
    if (value.valid()) {
      kvstore = std::move(value);
    }
    return absl::OkStatus();
  }
};

// While C++17 allows these explicit specialization to be defined at class
// scope, GCC does not support that:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
template <>
constexpr inline bool SpecOptions::IsCommonOption<OpenMode> = true;

template <>
constexpr inline bool SpecOptions::IsCommonOption<RecheckCachedData> = true;

template <>
constexpr inline bool SpecOptions::IsCommonOption<RecheckCachedMetadata> = true;

template <>
constexpr inline bool SpecOptions::IsCommonOption<RecheckCached> = true;

/// Options for requesting a `Spec` from an open `TensorStore`.
///
/// \relates Spec
struct SpecRequestOptions : public SpecOptions {
  ContextBindingMode context_binding_mode = ContextBindingMode::unspecified;

  template <typename T>
  constexpr static inline bool IsOption = SpecOptions::IsCommonOption<T>;

  using SpecOptions::Set;

  absl::Status Set(MinimalSpec value) {
    minimal_spec = value.minimal_spec();
    return absl::OkStatus();
  }

  absl::Status Set(ContextBindingMode value) {
    if (value > context_binding_mode) context_binding_mode = value;
    return absl::OkStatus();
  }
};

template <>
constexpr inline bool SpecRequestOptions::IsOption<MinimalSpec> = true;

template <>
constexpr inline bool SpecRequestOptions::IsOption<ContextBindingMode> = true;

/// Options for converting an existing `Spec`.
///
/// \relates Spec
struct SpecConvertOptions : public SpecRequestOptions {
  Context context;

  template <typename T>
  constexpr static inline bool IsOption =
      SpecRequestOptions::IsOption<T> || Schema::IsOption<T>;

  using Schema::Set;
  using SpecRequestOptions::Set;

  // Additionally supports `Context`.

  absl::Status Set(Context value) {
    context = std::move(value);
    return absl::OkStatus();
  }
};

template <>
constexpr inline bool SpecConvertOptions::IsOption<Context> = true;

template <>
constexpr inline bool SpecConvertOptions::IsOption<kvstore::Spec> = true;

/// Options for opening a `Spec`.
///
/// \relates Spec
struct OpenOptions : public SpecOptions {
  Context context;
  ReadWriteMode read_write_mode = ReadWriteMode::dynamic;

  // Supports all common options of `SpecOptions`, and all options of
  // `Schema`.
  template <typename T>
  constexpr static inline bool IsOption =
      SpecOptions::IsCommonOption<T> || Schema::IsOption<T>;
  using Schema::Set;
  using SpecOptions::Set;

  // Additionally supports `ReadWriteMode`.

  absl::Status Set(ReadWriteMode value) {
    read_write_mode = read_write_mode | value;
    return absl::OkStatus();
  }

  // Additionally supports `Context`.

  absl::Status Set(Context value) {
    context = std::move(value);
    return absl::OkStatus();
  }
};

template <>
constexpr inline bool OpenOptions::IsOption<ReadWriteMode> = true;

template <>
constexpr inline bool OpenOptions::IsOption<Context> = true;

template <>
constexpr inline bool OpenOptions::IsOption<kvstore::Spec> = true;

/// Options for opening a `Spec` with optional transaction.
///
/// \relates Spec
struct TransactionalOpenOptions : public OpenOptions {
  Transaction transaction{no_transaction};

  // Supports all options of `OpenOptions`.
  template <typename T>
  constexpr static inline bool IsOption = OpenOptions::IsOption<T>;
  using OpenOptions::Set;

  // Additionally supports `Transaction`.
  absl::Status Set(Transaction value) {
    transaction = std::move(value);
    return absl::OkStatus();
  }
};

template <>
constexpr inline bool TransactionalOpenOptions::IsOption<Transaction> = true;

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

/// Verifies that `mode` includes `ReadWriteMode::read`.
///
/// \error `absl::StatusCode::kInvalidArgument` if condition is not satisfied.
absl::Status ValidateSupportsRead(ReadWriteMode mode);

/// Verifies that `mode` includes `ReadWriteMode::write`.
///
/// \error `absl::StatusCode::kInvalidArgument` if condition is not satisfied.
absl::Status ValidateSupportsWrite(ReadWriteMode mode);

absl::Status ValidateSupportsModes(ReadWriteMode mode,
                                   ReadWriteMode required_modes);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_OPEN_MODE_H_
