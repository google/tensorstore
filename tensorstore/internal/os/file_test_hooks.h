// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_TEST_HOOKS_H_
#define TENSORSTORE_INTERNAL_OS_FILE_TEST_HOOKS_H_

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/open_flags.h"

namespace tensorstore {
namespace internal_os {

/// Tag for CloseFileDescriptor operation.
/// Triggered by: CloseFileDescriptor
struct CloseOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor);
};

/// Tag for ReadFromFile and PReadFromFile operations.
/// Triggered by: ReadFromFile, PReadFromFile
struct ReadOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor);
};

/// Tag for WriteToFile and WriteCordToFile operations.
/// Triggered by: WriteToFile, WriteCordToFile
struct WriteOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor);
};

/// Tag for FsyncFile operation.
/// Triggered by: FsyncFile
struct FsyncOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor);
};

/// Tag for RenameOpenFile operation.
/// Triggered by: RenameOpenFile
struct RenameOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor,
                                               const std::string&,
                                               const std::string&);
};

/// Tag for OpenFileWrapper operation.
/// Triggered by: OpenFileWrapper
struct OpenOpTag {
  using HookFunc = std::optional<absl::Status>(const std::string&, OpenFlags);
};

/// Tag for DeleteOpenFile and DeleteFile operations.
/// Triggered by: DeleteOpenFile, DeleteFile
struct DeleteOpTag {
  using HookFunc = std::optional<absl::Status>(FileDescriptor,
                                               const std::string&);
};

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_TEST_HOOKS_H_
