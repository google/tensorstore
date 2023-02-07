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

#ifndef TENSORSTORE_OPEN_OPTIONS_H_
#define TENSORSTORE_OPEN_OPTIONS_H_

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/option.h"

namespace tensorstore {

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

}  // namespace tensorstore

#endif  // TENSORSTORE_OPEN_OPTIONS_H_
