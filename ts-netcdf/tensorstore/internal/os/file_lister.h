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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_LISTER_H_
#define TENSORSTORE_INTERNAL_OS_FILE_LISTER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"

namespace tensorstore {
namespace internal_os {

/// Represents an entry within a directory that is being iterated.
class ListerEntry {
 public:
  struct Impl;
  ListerEntry(Impl* impl) : impl_(impl) {}

  /// Returns whether this entry is a directory.
  bool IsDirectory();

  /// Returns the full path the the current entry.
  const std::string& GetFullPath();

  /// Returns the last path component, or the base filename.
  std::string_view GetPathComponent();

  /// Returns the size or -1 if unavailable.
  int64_t GetSize();

  /// Deletes the file/directory represented by the entry.
  ///
  /// \returns `absl::OkStatus` on success, or some failure (
  ///    such as absl::NotFound) in case of an error.
  absl::Status Delete();

 private:
  Impl* impl_;
};

/// Recursively iterate over the directory starting at `root_directory`.
/// `recurse_into` will be called for each directory entry to allow control
/// of recursion.
///
/// `on_item` will be called for each directory entry.
absl::Status RecursiveFileList(
    std::string root_directory,
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item);

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_LISTER_H_
