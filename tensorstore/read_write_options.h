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

#ifndef TENSORSTORE_READ_WRITE_OPTIONS_H_
#define TENSORSTORE_READ_WRITE_OPTIONS_H_

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/batch.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/progress.h"

namespace tensorstore {

/// Options for `tensorstore::Read` into an existing target array.
///
/// \relates Read[TensorStore, Array]
struct ReadOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  absl::Status Set(DomainAlignmentOptions value) {
    this->alignment_options = value;
    return absl::OkStatus();
  }

  absl::Status Set(ReadProgressFunction value) {
    this->progress_function = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(Batch value) {
    this->batch = std::move(value);
    return absl::OkStatus();
  }

  /// Constrains how the source TensorStore may be aligned to the target array.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  ReadProgressFunction progress_function;

  /// Optional batch.
  Batch batch{no_batch};
};

template <>
constexpr inline bool ReadOptions::IsOption<DomainAlignmentOptions> = true;

template <>
constexpr inline bool ReadOptions::IsOption<ReadProgressFunction> = true;

template <>
constexpr inline bool ReadOptions::IsOption<Batch> = true;

template <>
constexpr inline bool ReadOptions::IsOption<Batch::View> = true;

/// Options for `tensorstore::Read` into new array.
///
/// \relates Read[TensorStore]
struct ReadIntoNewArrayOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  absl::Status Set(ContiguousLayoutOrder value) {
    this->layout_order = value;
    return absl::OkStatus();
  }

  absl::Status Set(ReadProgressFunction value) {
    this->progress_function = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(Batch value) {
    this->batch = std::move(value);
    return absl::OkStatus();
  }

  /// Specifies the layout order of the newly-allocated array.  Defaults to
  /// `c_order`.
  ContiguousLayoutOrder layout_order = c_order;

  /// Optional progress callback.
  ReadProgressFunction progress_function;

  /// Optional batch.
  Batch batch{no_batch};
};

template <>
constexpr inline bool ReadIntoNewArrayOptions::IsOption<ContiguousLayoutOrder> =
    true;

template <>
constexpr inline bool ReadIntoNewArrayOptions::IsOption<ReadProgressFunction> =
    true;

template <>
constexpr inline bool ReadIntoNewArrayOptions::IsOption<Batch> = true;

template <>
constexpr inline bool ReadIntoNewArrayOptions::IsOption<Batch::View> = true;

/// Specifies restrictions on how references to the source array/source
/// TensorStore may be used by write operations.
///
/// \relates TensorStore
enum SourceDataReferenceRestriction {
  /// References to the source data must be released as soon as possible, which
  /// may result in additional copies.
  cannot_reference_source_data = 0,

#if 0
  // TODO(jbms): Add this option once it is supported.

  /// References to the source data may be retained until the write is
  /// committed.  The source data must not be modified until the write is
  /// committed.
  can_reference_source_data_until_commit = 1,
#endif

  /// References to the source data may be retained indefinitely, even after the
  /// write is committed.  The source data must not be modified until all
  /// references are released.
  can_reference_source_data_indefinitely = 2,
};

/// Options for `tensorstore::Write`.
///
/// \relates Write[Array, TensorStore]
struct WriteOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  absl::Status Set(DomainAlignmentOptions value) {
    this->alignment_options = value;
    return absl::OkStatus();
  }

  absl::Status Set(WriteProgressFunction value) {
    this->progress_function = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(SourceDataReferenceRestriction value) {
    this->source_data_reference_restriction = value;
    return absl::OkStatus();
  }

  /// Constrains how the source array may be aligned to the target TensorStore.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  WriteProgressFunction progress_function;

  /// Specifies restrictions on how the source data may be referenced (as
  /// opposed to copied).
  SourceDataReferenceRestriction source_data_reference_restriction =
      cannot_reference_source_data;
};

template <>
constexpr inline bool WriteOptions::IsOption<DomainAlignmentOptions> = true;

template <>
constexpr inline bool WriteOptions::IsOption<WriteProgressFunction> = true;

template <>
constexpr inline bool WriteOptions::IsOption<SourceDataReferenceRestriction> =
    true;

/// Options for `tensorstore::Copy`.
///
/// \relates Copy[TensorStore, TensorStore]
struct CopyOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  absl::Status Set(DomainAlignmentOptions value) {
    this->alignment_options = value;
    return absl::OkStatus();
  }

  absl::Status Set(CopyProgressFunction value) {
    this->progress_function = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(SourceDataReferenceRestriction value) {
    this->source_data_reference_restriction = value;
    return absl::OkStatus();
  }

  absl::Status Set(Batch value) {
    this->batch = std::move(value);
    return absl::OkStatus();
  }

  /// Constrains how the source TensorStore may be aligned to the target
  /// TensorStore.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  CopyProgressFunction progress_function;

  /// Specifies restrictions on how the source data may be referenced (as
  /// opposed to copied).
  SourceDataReferenceRestriction source_data_reference_restriction =
      cannot_reference_source_data;

  /// Optional batch for reading.
  Batch batch{no_batch};
};

template <>
constexpr inline bool CopyOptions::IsOption<DomainAlignmentOptions> = true;

template <>
constexpr inline bool CopyOptions::IsOption<CopyProgressFunction> = true;

template <>
constexpr inline bool CopyOptions::IsOption<SourceDataReferenceRestriction> =
    true;

template <>
constexpr inline bool CopyOptions::IsOption<Batch> = true;

template <>
constexpr inline bool CopyOptions::IsOption<Batch::View> = true;

}  // namespace tensorstore

#endif  // TENSORSTORE_READ_WRITE_OPTIONS_H_
