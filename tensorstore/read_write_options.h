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

#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/progress.h"

namespace tensorstore {

/// Options for `tensorstore::Read` into an existing target array.
///
/// \relates Read[TensorStore, Array]
struct ReadOptions {
  /// Constructs the options.
  ReadOptions(
      DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all,
      ReadProgressFunction progress_function = {})
      : alignment_options(alignment_options),
        progress_function(std::move(progress_function)) {}
  ReadOptions(ReadProgressFunction progress_function)
      : progress_function(std::move(progress_function)) {}

  /// Constrains how the source TensorStore may be aligned to the target array.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  ReadProgressFunction progress_function;
};

/// Options for `tensorstore::Read` into new array.
///
/// \relates Read[TensorStore]
struct ReadIntoNewArrayOptions {
  /// Constructs the options.
  ReadIntoNewArrayOptions(ContiguousLayoutOrder layout_order = {},
                          ReadProgressFunction progress_function = {})
      : layout_order(layout_order),
        progress_function(std::move(progress_function)) {}
  ReadIntoNewArrayOptions(ReadProgressFunction progress_function)
      : progress_function(std::move(progress_function)) {}

  /// Specifies the layout order of the newly-allocated array.  Defaults to
  /// `c_order`.
  ContiguousLayoutOrder layout_order = c_order;

  /// Optional progress callback.
  ReadProgressFunction progress_function;
};

/// Options for `tensorstore::Write`.
///
/// \relates Write[Array, TensorStore]
struct WriteOptions {
  /// Constructs the options.
  WriteOptions(
      DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all,
      WriteProgressFunction progress_function = {})
      : alignment_options(alignment_options),
        progress_function(std::move(progress_function)) {}
  WriteOptions(WriteProgressFunction progress_function)
      : progress_function(std::move(progress_function)) {}

  /// Constrains how the source array may be aligned to the target TensorStore.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  WriteProgressFunction progress_function;
};

/// Options for `tensorstore::Copy`.
///
/// \relates Copy[TensorStore, TensorStore]
struct CopyOptions {
  /// Constructs the options.
  CopyOptions(
      DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all,
      CopyProgressFunction progress_function = {})
      : alignment_options(alignment_options),
        progress_function(std::move(progress_function)) {}
  CopyOptions(CopyProgressFunction progress_function)
      : progress_function(std::move(progress_function)) {}

  /// Constrains how the source TensorStore may be aligned to the target
  /// TensorStore.
  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  /// Optional progress callback.
  CopyProgressFunction progress_function;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_READ_WRITE_OPTIONS_H_
