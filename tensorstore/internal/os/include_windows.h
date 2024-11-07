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

#ifndef TENSORSTORE_INTERNAL_OS_INCLUDE_WINDOWS_H_
#define TENSORSTORE_INTERNAL_OS_INCLUDE_WINDOWS_H_

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // IWYU pragma: export

namespace tensorstore {
namespace internal_os {

// FileDispositionInfoEx is a new API in Windows 10 and this structure
// does not seem to be defined in the Windows SDK
//
// https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information_ex
struct FileDispositionInfoExData {
  ULONG Flags;
};

// The ::open flag constants are used in the OpenFlags enum.
#if !defined(O_RDONLY)
#define O_RDONLY 0x0
#endif
#if !defined(O_WRONLY)
#define O_WRONLY 0x1
#endif
#if !defined(O_RDWR)
#define O_RDWR 0x2
#endif
#if !defined(O_CREAT)
#define O_CREAT 0x40
#endif
#if !defined(O_EXCL)
#define O_EXCL 0x80
#endif
#if !defined(O_APPEND)
#define O_APPEND 0x400
#endif

}  // namespace internal_os
}  // namespace tensorstore
#endif  // _WIN32

#endif  // TENSORSTORE_INTERNAL_OS_INCLUDE_WINDOWS_H_
