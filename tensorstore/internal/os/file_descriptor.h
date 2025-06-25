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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_
#define TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_

#include "tensorstore/internal/os/unique_handle.h"

namespace tensorstore {
namespace internal_os {

#ifdef _WIN32
// Specializations for Windows.

/// Representation of open file/directory.
using FileDescriptor = void*;  // HANDLE
#else
// Specializations for Posix.

/// Representation of open file/directory.
using FileDescriptor = int;
#endif

/// File descriptor traits for use with `UniqueHandle`.
struct FileDescriptorTraits {
  static FileDescriptor Invalid() { return ((FileDescriptor)-1); }
  static void Close(FileDescriptor fd);
};

/// Unique handle to an open file descriptor.
///
/// The file descriptor is closed automatically by the destructor.
using UniqueFileDescriptor = UniqueHandle<FileDescriptor, FileDescriptorTraits>;

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_
