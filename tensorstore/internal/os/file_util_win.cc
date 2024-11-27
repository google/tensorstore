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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/internal/os/wstring.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// Not your standard include order.
#include "tensorstore/internal/os/file_util.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal::ConvertUTF8ToWindowsWide;
using ::tensorstore::internal::ConvertWindowsWideToUTF8;
using ::tensorstore::internal::StatusFromOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

namespace tensorstore {
namespace internal_os {
namespace {

auto& mmap_count = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/file/mmap_count",
    internal_metrics::MetricMetadata("Count of total mmap files"));

auto& mmap_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/file/mmap_bytes",
    internal_metrics::MetricMetadata("Count of total mmap bytes"));

auto& mmap_active = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/file/mmap_active",
    internal_metrics::MetricMetadata("Count of active mmap files"));

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");

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
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1));
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
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

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

void UnlockWin32Lock(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  auto lock_offset = GetLockOverlapped();
  // Ignore any errors.
  ::UnlockFileEx(fd, /*dwReserved=*/0, /*nNumberOfBytesToUnlockLow=*/1,
                 /*nNumberOfBytesToUnlockHigh=*/0,
                 /*lpOverlapped=*/&lock_offset);
}

FileDescriptor OpenFileImpl(const std::wstring& wpath, OpenFlags flags) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1));
  // Setup Win32 flags to somewhat mimic the POSIX flags.
  DWORD access = 0;
  if ((flags & OpenFlags::OpenReadOnly) == OpenFlags::OpenReadOnly) {
    access = GENERIC_READ;
  } else if ((flags & OpenFlags::OpenWriteOnly) == OpenFlags::OpenWriteOnly) {
    access = GENERIC_WRITE | DELETE;
  } else if ((flags & OpenFlags::OpenReadWrite) == OpenFlags::OpenReadWrite) {
    access = GENERIC_READ | GENERIC_WRITE | DELETE;
  }
  if ((flags & OpenFlags::Create) == OpenFlags::Create) {
    access |= GENERIC_WRITE | DELETE;
  }
  if ((flags & OpenFlags::Append) == OpenFlags::Append) {
    access ^= GENERIC_WRITE;
    access |= FILE_APPEND_DATA;
  }

  DWORD createmode = 0;
  if ((flags & OpenFlags::Create) == OpenFlags::Create) {
    if ((flags & OpenFlags::Exclusive) == OpenFlags::Exclusive) {
      createmode = CREATE_NEW;
    } else {
      createmode = OPEN_ALWAYS;
    }
  } else {
    createmode = OPEN_EXISTING;
  }

  return ::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/access,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/createmode,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr);
}

}  // namespace

void FileDescriptorTraits::Close(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  ::CloseHandle(fd);
}

Result<UnlockFn> AcquireFdLock(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  auto lock_offset = GetLockOverlapped();
  if (::LockFileEx(fd, /*dwFlags=*/LOCKFILE_EXCLUSIVE_LOCK,
                   /*dwReserved=*/0,
                   /*nNumberOfBytesToLockLow=*/1,
                   /*nNumberOfBytesToLockHigh=*/0,
                   /*lpOverlapped=*/&lock_offset)) {
    return UnlockWin32Lock;
  }
  auto status = StatusFromOsError(::GetLastError(), "Failed to lock file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<UniqueFileDescriptor> OpenFileWrapper(const std::string& path,
                                             OpenFlags flags) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));

  FileDescriptor fd = OpenFileImpl(wpath, flags);

  if (fd == FileDescriptorTraits::Invalid()) {
    auto status = StatusFromOsError(::GetLastError(),
                                    "Failed to open: ", QuoteString(path));
    return std::move(tspan).EndWithStatus(std::move(status));
  }
  tspan.Log("fd", fd);
  return UniqueFileDescriptor(fd);
}

Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"handle", fd}, {"size", count}});

  auto overlapped = GetOverlappedWithOffset(static_cast<uint64_t>(offset));
  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD bytes_read;
  if (::ReadFile(fd, buf, static_cast<DWORD>(count), &bytes_read,
                 &overlapped)) {
    return static_cast<ptrdiff_t>(bytes_read);
  }
  auto status = StatusFromOsError(::GetLastError(), "Failed to read from file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf,
                              size_t count) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"handle", fd}, {"size", count}});

  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD num_written;
  if (::WriteFile(fd, buf, static_cast<DWORD>(count), &num_written,
                  /*lpOverlapped=*/nullptr)) {
    return static_cast<size_t>(num_written);
  }
  auto status = StatusFromOsError(::GetLastError(), "Failed to write to file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"handle", fd}, {"size", value.size()}});

  // If we switched to OVERLAPPED io on Windows, then using WriteFileGather
  // would be similar to the unix ::writev call.
  size_t value_remaining = value.size();
  for (absl::Cord::CharIterator char_it = value.char_begin();
       value_remaining;) {
    auto chunk = absl::Cord::ChunkRemaining(char_it);
    auto result = WriteToFile(fd, chunk.data(), chunk.size());
    if (!result.ok()) {
      return result;
    }
    value_remaining -= result.value();
    absl::Cord::Advance(&char_it, result.value());
  }
  return value.size();
}

absl::Status TruncateFile(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (::SetEndOfFile(fd)) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(::GetLastError(), "Failed to truncate file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name) {
  LoggedTraceSpan tspan(
      __func__, detail_logging.Level(1),
      {{"handle", fd}, {"old_name", old_name}, {"new_name", new_name}});

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

  auto status = StatusFromOsError(::GetLastError(),
                                  "Failed to rename: ", QuoteString(old_name),
                                  " to: ", QuoteString(new_name));
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"handle", fd}, {"path", path}});

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
  auto status = StatusFromOsError(::GetLastError(),
                                  "Failed to delete: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status DeleteFile(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

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
  return std::move(tspan).EndWithStatus(StatusFromOsError(
      ::GetLastError(), "Failed to delete: ", QuoteString(path)));
}

uint32_t GetDefaultPageSize() {
  static const uint32_t kDefaultPageSize = []() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwAllocationGranularity;
  }();
  return kDefaultPageSize;
}

Result<MappedRegion> MemmapFileReadOnly(FileDescriptor fd, size_t offset,
                                        size_t size) {
  if (offset > 0 && offset % GetDefaultPageSize() != 0) {
    return absl::InvalidArgumentError(
        "Offset must be a multiple of the default page size.");
  }
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"handle", fd}, {"offset", offset}, {"size", size}});

  if (size == 0) {
    ::BY_HANDLE_FILE_INFORMATION info;
    if (!::GetFileInformationByHandle(fd, &info)) {
      return std::move(tspan).EndWithStatus(StatusFromOsError(
          ::GetLastError(), "Failed in GetFileInformationByHandle"));
    }

    uint64_t file_size = GetSize(info);
    if (offset + size > file_size) {
      return std::move(tspan).EndWithStatus(absl::OutOfRangeError(
          absl::StrCat("Requested offset ", offset, " + size ", size,
                       " exceeds file size ", file_size)));
    } else if (size == 0) {
      size = file_size - offset;
    }
  }

  UniqueFileDescriptor map_fd(
      ::CreateFileMappingA(fd, NULL, PAGE_READONLY, 0, 0, nullptr));
  if (!map_fd.valid()) {
    return std::move(tspan).EndWithStatus(
        StatusFromOsError(::GetLastError(), "Failed in CreateFileMappingA"));
  }

  void* address = ::MapViewOfFile(
      map_fd.get(), FILE_MAP_READ, static_cast<DWORD>(offset >> 32),
      static_cast<DWORD>(offset & 0xffffffff), size);
  if (!address) {
    return std::move(tspan).EndWithStatus(
        StatusFromOsError(::GetLastError(), "Failed in MapViewOfFile"));
  }

  {
    WIN32_MEMORY_RANGE_ENTRY entry{address, size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
  }

  mmap_count.Increment();
  mmap_bytes.IncrementBy(size);
  mmap_active.Increment();
  return MappedRegion(static_cast<const char*>(address), size);
}

MappedRegion::~MappedRegion() {
  if (data_) {
    if (!::UnmapViewOfFile(const_cast<char*>(data_))) {
      ABSL_LOG(FATAL) << StatusFromOsError(::GetLastError(),
                                           "Failed in UnmapViewOfFile");
    }
    mmap_active.Decrement();
  }
}

absl::Status FsyncFile(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (::FlushFileBuffers(fd)) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(::GetLastError());
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (::GetFileInformationByHandle(fd, info)) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(::GetLastError());
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  // The typedef uses BY_HANDLE_FILE_INFO, which includes device and index
  // metadata, and requires an open handle.
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  UniqueFileDescriptor stat_fd(::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_FLAG_BACKUP_SEMANTICS,
      /*hTemplateFile=*/nullptr));
  if (stat_fd.valid()) {
    if (::GetFileInformationByHandle(stat_fd.get(), info)) {
      return absl::OkStatus();
    }
  }
  auto status = StatusFromOsError(::GetLastError(),
                                  "Failed to stat file: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});
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
    auto status = StatusFromOsError(
        ::GetLastError(), "Failed to open directory: ", QuoteString(path));
    return std::move(tspan).EndWithStatus(std::move(status));
  }
  tspan.Log("fd", fd);
  return UniqueFileDescriptor(fd);
}

absl::Status MakeDirectory(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  if (::CreateDirectoryW(wpath.c_str(),
                         /*lpSecurityAttributes=*/nullptr) ||
      ::GetLastError() == ERROR_ALREADY_EXISTS) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(
      ::GetLastError(), "Failed to create directory: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
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
