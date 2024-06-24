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

#ifndef _WIN32
#error "Use file_util_posix.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Defined before including <stdio.h> to ensure `::rand_s` is defined.
#define _CRT_RAND_S

// Windows 10 1607 ("Redstone 1") is required for FileDispositionInfoEx,
// FileRenameInfoEx.
//
// In `minwinbase.h`, those definitions are conditioned on:
//
//     #if _WIN32_WINNT >= 0x0A000002
//
// Therefore, we ensure that `_WIN32_WINNT` is at least that value.
//
// This condition appears to be a mistake in `minwinbase.h`:
//
// According to the documentation here
// https://learn.microsoft.com/en-us/windows/win32/winprog/using-the-windows-headers,
// `_WIN32_WINNT` seems to be intended to have coarse granularity and mostly
// only specifies major versions of Windows.  The highest documented value (as
// of 2023-03-09) is `_WIN32_WINNT_WIN10 = 0x0A00`.
//
// The `NTDDI_VERSION` macro seems to be intended to specify a version
// requirement at a finer granularity, and `0x0A000002` is the value of
// `NTDDI_WIN10_RS1`.  It appears that the condition should have been
// `#if NTDDI_VERSION >= 0x0A000002` instead.
//
// Here we ensure both `_WIN32_WINNT` and `NTDDI_VERSION` are at least
// `0x0A000002`, in case `minwinbase.h` is ever fixed to condition on
// `NTDDI_VERSION` instead.
#if _WIN32_WINNT < 0x0A000002
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0A000002  // NTDDI_WIN10_RS1
#endif

#if NTDDI_VERSION < 0x0A000002
#ifdef NTDDI_VERSION
#undef NTDDI_VERSION
#endif
#define NTDDI_VERSION 0x0A000002
#endif

// Maintain include ordering here:

#include <stddef.h>
#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/internal/os/wstring.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

// Not your standard include order.
#include "tensorstore/internal/os/file_util.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal::ConvertUTF8ToWindowsWide;
using ::tensorstore::internal::ConvertWindowsWideToUTF8;
using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {
namespace {

/// Maximum length of Windows path, including terminating NUL.
constexpr size_t kMaxWindowsPathSize = 32768;

inline ::OVERLAPPED GetOverlappedWithOffset(uint64_t offset) {
  ::OVERLAPPED overlapped = {};
  overlapped.Offset = static_cast<DWORD>(offset & 0xffffffff);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  return overlapped;
}

/// Returns an OVERLAPPED with the lock offset used for the lock files.
inline ::OVERLAPPED GetLockOverlapped() {
  // Use a very high lock offset to ensure it does not conflict with any valid
  // byte range in the file.  This is important because we rename the lock file
  // to the real key before releasing the lock, and during the time between the
  // rename and the lock release, we don't want any read attempts to fail.
  return GetOverlappedWithOffset(0xffffffff'fffffffe);
}

bool RenameFilePosix(FileDescriptor fd, const std::wstring& new_name) {
  alignas(::FILE_RENAME_INFO) char
      file_rename_info_buffer[sizeof(::FILE_RENAME_INFO) + kMaxWindowsPathSize -
                              1];
  auto* rename_info = new (file_rename_info_buffer)::FILE_RENAME_INFO{};
  rename_info->FileNameLength = 2 * (new_name.size() + 1);
  std::memcpy(rename_info->FileName, new_name.c_str(),
              rename_info->FileNameLength);
  // https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
  rename_info->Flags = 0x00000001 /*FILE_RENAME_POSIX_SEMANTICS*/ |
                       0x00000002 /*FILE_RENAME_REPLACE_IF_EXISTS*/;
  return static_cast<bool>(::SetFileInformationByHandle(
      fd, FileRenameInfoEx, rename_info, std::size(file_rename_info_buffer)));
}

bool DeleteFilePosix(FileDescriptor fd) {
  FileDispositionInfoExData disposition_info;
  disposition_info.Flags = 0x00000001 /*FILE_DISPOSITION_DELETE*/ |
                           0x00000002 /*FILE_DISPOSITION_POSIX_SEMANTICS*/;
  return static_cast<bool>(::SetFileInformationByHandle(
      fd, FileDispositionInfoEx, &disposition_info, sizeof(disposition_info)));
}

std::string_view GetDirName(std::string_view path) {
  size_t i = path.size();
  while (i > 0 && !IsDirSeparator(path[i - 1])) --i;
  return path.substr(0, i);
}

#if 0
Result<DWORD> GetFileAttributes(const std::wstring& filename) {
  if (const DWORD attrs = ::GetFileAttributesW(filename.c_str());
      attrs != INVALID_FILE_ATTRIBUTES) {
    return attrs;
  }
  return StatusFromOsError(::GetLastError(), "GetFileAttributesW failed");
}
#endif

}  // namespace

void FileDescriptorTraits::Close(FileDescriptor handle) {
  ::CloseHandle(handle);
}

absl::Status FileLockTraits::Acquire(FileDescriptor fd) {
  auto lock_offset = GetLockOverlapped();
  if (::LockFileEx(fd, /*dwFlags=*/LOCKFILE_EXCLUSIVE_LOCK,
                   /*dwReserved=*/0,
                   /*nNumberOfBytesToLockLow=*/1,
                   /*nNumberOfBytesToLockHigh=*/0,
                   /*lpOverlapped=*/&lock_offset)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError(), "Failed to lock file");
}

void FileLockTraits::Close(FileDescriptor fd) {
  auto lock_offset = GetLockOverlapped();
  // Ignore any errors.
  ::UnlockFileEx(fd, /*dwReserved=*/0, /*nNumberOfBytesToUnlockLow=*/1,
                 /*nNumberOfBytesToUnlockHigh=*/0,
                 /*lpOverlapped=*/&lock_offset);
}

Result<UniqueFileDescriptor> OpenExistingFileForReading(
    const std::string& path) {
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));

  FileDescriptor fd = ::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr);

  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(::GetLastError(),
                             "Failed to open: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

Result<UniqueFileDescriptor> OpenFileForWriting(const std::string& path) {
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));

  FileDescriptor fd = ::CreateFileW(
      wpath.c_str(),
      // Even though we only need write access, the `CreateFile`
      // documentation recommends specifying GENERIC_READ as well for better
      // performance if the file is on a network share.
      /*dwDesiredAccess=*/GENERIC_READ | GENERIC_WRITE | DELETE,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_ALWAYS,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr);

  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(::GetLastError(),
                             "Failed to create: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset) {
  auto overlapped = GetOverlappedWithOffset(static_cast<uint64_t>(offset));
  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD bytes_read;
  if (::ReadFile(fd, buf, static_cast<DWORD>(count), &bytes_read,
                 &overlapped)) {
    return static_cast<ptrdiff_t>(bytes_read);
  }
  return StatusFromOsError(::GetLastError(), "Failed to read from file");
}

Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf,
                              size_t count) {
  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD num_written;
  if (::WriteFile(fd, buf, static_cast<DWORD>(count), &num_written,
                  /*lpOverlapped=*/nullptr)) {
    return static_cast<size_t>(num_written);
  }
  return StatusFromOsError(::GetLastError(), "Failed to write to file");
}

Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  // If we switched to OVERLAPPED io on Windows, then using WriteFileGather
  // would be similar to the unix ::writev call.
  size_t value_remaining = value.size();
  for (absl::Cord::CharIterator char_it = value.char_begin();
       value_remaining;) {
    auto chunk = absl::Cord::ChunkRemaining(char_it);
    auto result = WriteToFile(fd, chunk.data(), chunk.size());
    if (!result.ok()) return result;
    value_remaining -= result.value();
    absl::Cord::Advance(&char_it, result.value());
  }
  return value.size();
}

absl::Status TruncateFile(FileDescriptor fd) {
  if (::SetEndOfFile(fd)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError(), "Failed to truncate file");
}

absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name) {
  std::wstring wpath_new;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(new_name, wpath_new));

  // Try using Posix semantics.
  if (RenameFilePosix(fd, wpath_new)) {
    return absl::OkStatus();
  }

#ifndef NDEBUG
  ABSL_LOG_FIRST_N(INFO, 1) << StatusFromOsError(
      ::GetLastError(), "Failed to rename: ", QuoteString(old_name),
      " using posix; using move file.");
#endif

  std::wstring wpath_old;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(old_name, wpath_old));

  // Try using MoveFileEx, which may not be atomic.
  if (::MoveFileExW(wpath_old.c_str(), wpath_new.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return absl::OkStatus();
  }

  return StatusFromOsError(::GetLastError(),
                           "Failed to rename: ", QuoteString(old_name),
                           " to: ", QuoteString(new_name));
}

absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path) {
  // This relies on the "POSIX Semantics" flag supported by Windows 10 in
  // order to remove the file from its containing directory as soon as the
  // handle is closed.  However, after the call to
  // `SetFileInformationByHandle` but before the file handle is closed, the
  // file still exists in the directory but cannot be opened, which would
  // result in the normal read/write paths failing with an error.  To avoid
  // that problem, we first rename the file to a random name, with a suffix of
  // `kLockSuffix` to prevent it from being included in List results.

  unsigned int buf[5];
  for (int i = 0; i < 5; ++i) {
    ::rand_s(&buf[i]);
  }
  char temp_name[64];
  size_t temp_name_size = ::snprintf(
      temp_name, std::size(temp_name), "_deleted_%08x%08x%08x%08x%08x%.*s",
      buf[0], buf[1], buf[2], buf[3], buf[4],
      static_cast<int>(kLockSuffix.size()), kLockSuffix.data());

  std::wstring wpath_temp;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(
      absl::StrCat(GetDirName(path),
                   std::string_view(temp_name, temp_name_size)),
      wpath_temp));

  if (!RenameFilePosix(fd, wpath_temp)) {
#ifndef NDEBUG
    ABSL_LOG_FIRST_N(INFO, 1) << StatusFromOsError(
        ::GetLastError(), "Failed to rename for delete: ", QuoteString(path),
        " using posix; using fallback delete.");
#endif
    // Fallback to original path.
    TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath_temp));
  }
  // Attempt to delete the open handle using posix semantics?
  if (DeleteFilePosix(fd)) {
    return absl::OkStatus();
  }
#ifndef NDEBUG
  ABSL_LOG_FIRST_N(INFO, 1) << StatusFromOsError(
      ::GetLastError(), "Failed to delete: ", QuoteString(path),
      " using posix; using fallback delete.");
#endif
  // The file has been renamed, so delete the renamed file.
  if (::DeleteFileW(wpath_temp.c_str())) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError(),
                           "Failed to delete: ", QuoteString(path));
}

absl::Status DeleteFile(const std::string& path) {
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  UniqueFileDescriptor delete_fd(::CreateFileW(
      wpath.c_str(),
      // Even though we only need write access, the `CreateFile`
      // documentation recommends specifying GENERIC_READ as well for better
      // performance if the file is on a network share.
      /*dwDesiredAccess=*/GENERIC_READ | GENERIC_WRITE | DELETE,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr));
  if (delete_fd.valid()) {
    return DeleteOpenFile(delete_fd.get(), path);
  }
  return StatusFromOsError(::GetLastError(),
                           "Failed to delete: ", QuoteString(path));
}

absl::Status FsyncFile(FileDescriptor fd) {
  if (::FlushFileBuffers(fd)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError());
}

absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info) {
  if (::GetFileInformationByHandle(fd, info)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError());
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  // The typedef uses BY_HANDLE_FILE_INFO, which includes device and index
  // metadata, and requires an open handle.
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  FileDescriptor fd = ::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_FLAG_BACKUP_SEMANTICS,
      /*hTemplateFile=*/nullptr);
  if (fd != FileDescriptorTraits::Invalid()) {
    auto status = GetFileInfo(fd, info);
    ::CloseHandle(fd);
    return status;
  }
  return StatusFromOsError(::GetLastError());
}

Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path) {
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  FileDescriptor fd = ::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_FLAG_BACKUP_SEMANTICS,
      /*hTemplateFile=*/nullptr);
  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(::GetLastError(),
                             "Failed to open directory: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

absl::Status MakeDirectory(const std::string& path) {
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  if (::CreateDirectoryW(wpath.c_str(),
                         /*lpSecurityAttributes=*/nullptr) ||
      ::GetLastError() == ERROR_ALREADY_EXISTS) {
    return absl::OkStatus();
  }
  return StatusFromOsError(::GetLastError(),
                           "Failed to create directory: ", QuoteString(path));
}

absl::Status FsyncDirectory(FileDescriptor fd) {
  // Windows does not support Fsync on directories.
  return absl::OkStatus();
}

Result<std::string> GetWindowsTempDir() {
  wchar_t buf[MAX_PATH + 1];
  DWORD retval = GetTempPathW(MAX_PATH + 1, buf);
  if (retval == 0) {
    return StatusFromOsError(::GetLastError(), "Failed to get temp directory");
  }
  assert(retval <= MAX_PATH);
  if (buf[retval - 1] == L'\\') {
    buf[retval - 1] = 0;
    retval--;
  }
  std::string path;
  TENSORSTORE_RETURN_IF_ERROR(
      ConvertWindowsWideToUTF8(std::wstring_view(buf, retval), path));
  return path;
}

}  // namespace internal_os
}  // namespace tensorstore
