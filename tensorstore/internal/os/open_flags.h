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

#ifndef TENSORSTORE_INTERNAL_OS_OPEN_FLAGS_H_
#define TENSORSTORE_INTERNAL_OS_OPEN_FLAGS_H_

#include <stddef.h>
#include <stdint.h>

// Include system headers last to reduce impact of macros.
#ifndef _WIN32
#include <fcntl.h>
#endif

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {

// Restricted subset of POSIX open flags.
enum class OpenFlags : int {
  OpenReadOnly = O_RDONLY,
  OpenWriteOnly = O_WRONLY,
  OpenReadWrite = O_RDWR,
  Create = O_CREAT,
  Append = O_APPEND,
  Truncate = O_TRUNC,
  Exclusive = O_EXCL,
#if defined(O_DIRECT)
  Direct = O_DIRECT,
#else
  Direct = 0x4000,
#endif
  CloseOnExec = O_CLOEXEC,
  ReadWriteMask = O_RDONLY | O_WRONLY | O_RDWR,

  DefaultRead = O_RDONLY | O_CLOEXEC,
  DefaultWrite = O_CREAT | O_WRONLY | O_CLOEXEC,
};

constexpr OpenFlags operator|(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr OpenFlags operator&(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) & static_cast<int>(b));
}

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_OPEN_FLAGS_H_
