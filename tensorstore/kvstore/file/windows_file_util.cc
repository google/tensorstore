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

#ifdef _WIN32

// Defined before including <stdio.h> to ensure `::rand_s` is defined.
#define _CRT_RAND_S

#include "tensorstore/kvstore/file/windows_file_util.h"

#include <stdio.h>

#include "absl/log/absl_check.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/kvstore/file/file_util.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

// Windows 10 1607 required for FileDispositionInfoEx, FileRenameInfoEx
#if defined(NTDDI_VERSION) && (NTDDI_VERSION < NTDDI_WIN10_RS1)
// NTDDI_VERSION should be >= NTDDI_WIN10_RS1
// _WIN32_WINNT  should be >= _WIN32_WINNT_WIN10
#error "NTDDI_VERSION must be WIN10 "
#endif

namespace tensorstore {
namespace internal_file_util {

inline ::OVERLAPPED GetOverlappedWithOffset(std::uint64_t offset) {
  ::OVERLAPPED overlapped = {};
  overlapped.Offset = static_cast<DWORD>(offset & 0xffffffff);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  return overlapped;
}

std::ptrdiff_t ReadFromFile(FileDescriptor fd, void* buf, std::size_t count,
                            std::int64_t offset) {
  auto overlapped = GetOverlappedWithOffset(static_cast<std::uint64_t>(offset));
  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD bytes_read;
  if (!::ReadFile(fd, buf, static_cast<DWORD>(count), &bytes_read, &overlapped))
    return -1;
  return static_cast<std::ptrdiff_t>(bytes_read);
}
std::ptrdiff_t WriteToFile(FileDescriptor fd, const void* buf,
                           std::size_t count) {
  if (count > std::numeric_limits<DWORD>::max()) {
    count = std::numeric_limits<DWORD>::max();
  }
  DWORD num_written;
  if (!::WriteFile(fd, buf, static_cast<DWORD>(count), &num_written,
                   /*lpOverlapped=*/nullptr)) {
    return -1;
  }
  return static_cast<std::size_t>(num_written);
}

std::ptrdiff_t WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  // If we switched to OVERLAPPED io on Windows, then using WriteFileGather
  // would be similar to the unix ::writev call.
  size_t value_remaining = value.size();
  for (absl::Cord::CharIterator char_it = value.char_begin();
       value_remaining;) {
    auto chunk = absl::Cord::ChunkRemaining(char_it);
    std::ptrdiff_t n = WriteToFile(fd, chunk.data(), chunk.size());
    if (n > 0) {
      value_remaining -= n;
      absl::Cord::Advance(&char_it, n);
      continue;
    }
    return n;
  }
  return value.size();
}

namespace {
/// Maximum length of Windows path, including terminating NUL.
constexpr size_t kMaxWindowsPathSize = 32768;
class WindowsPathConverter {
 public:
  explicit WindowsPathConverter(std::string_view path) {
    if (path.size() > std::numeric_limits<int>::max()) {
      ::SetLastError(ERROR_BUFFER_OVERFLOW);
      failed_ = true;
      return;
    }
    if (path.size() == 0) {
      failed_ = false;
      wpath_[0] = 0;
      wpath_size_ = 0;
      return;
    }
    int n = ::MultiByteToWideChar(
        /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS, path.data(),
        static_cast<int>(path.size()), wpath_, kMaxWindowsPathSize - 1);
    if (n == 0) {
      failed_ = true;
      return;
    }
    failed_ = false;
    wpath_[n] = 0;
    wpath_size_ = n;
  }

  bool failed() const { return failed_; }
  const wchar_t* wc_str() const { return wpath_; }
  size_t size() const { return wpath_size_; }

 private:
  bool failed_;
  size_t wpath_size_;
  wchar_t wpath_[kMaxWindowsPathSize];
};
}  // namespace

FileDescriptor OpenDirectoryDescriptor(const char* path) {
  WindowsPathConverter converter(path);
  if (converter.failed()) return INVALID_HANDLE_VALUE;
  return ::CreateFileW(
      converter.wc_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_FLAG_BACKUP_SEMANTICS,
      /*hTemplateFile=*/nullptr);
}

bool MakeDirectory(const char* path) {
  WindowsPathConverter converter(path);
  if (converter.failed()) return false;
  return static_cast<bool>(::CreateDirectoryW(
             converter.wc_str(), /*lpSecurityAttributes=*/nullptr)) ||
         ::GetLastError() == ERROR_ALREADY_EXISTS;
}
bool FsyncFile(FileDescriptor fd) {
  return static_cast<bool>(::FlushFileBuffers(fd));
}
bool RenameFile(FileDescriptor fd, std::string_view new_name) {
  WindowsPathConverter converter(new_name);
  if (converter.failed()) return false;
  alignas(::FILE_RENAME_INFO) char
      file_rename_info_buffer[sizeof(::FILE_RENAME_INFO) + kMaxWindowsPathSize -
                              1];
  auto* rename_info = new (file_rename_info_buffer)::FILE_RENAME_INFO{};
  rename_info->FileNameLength = 2 * (converter.size() + 1);
  std::memcpy(rename_info->FileName, converter.wc_str(),
              rename_info->FileNameLength);
  // https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
  rename_info->Flags = 0x00000001 /*FILE_RENAME_POSIX_SEMANTICS*/ |
                       0x00000002 /*FILE_RENAME_REPLACE_IF_EXISTS*/;
  return static_cast<bool>(::SetFileInformationByHandle(
      fd, FileRenameInfoEx, rename_info, std::size(file_rename_info_buffer)));
}

bool RenameOpenFile(FileDescriptor fd, std::string_view old_name,
                    std::string_view new_name) {
  return internal_file_util::RenameFile(fd, new_name);
}

std::string_view GetDirName(std::string_view path) {
  size_t i = path.size();
  while (i > 0 && !internal_file_util::IsDirSeparator(path[i - 1])) --i;
  return path.substr(0, i);
}

bool DeleteOpenFile(FileDescriptor fd, std::string_view path) {
  // This relies on the "POSIX Semantics" flag supported by Windows 10 in order
  // to remove the file from its containing directory as soon as the handle is
  // closed.  However, after the call to `SetFileInformationByHandle` but before
  // the file handle is closed, the file still exists in the directory but
  // cannot be opened, which would result in the normal read/write paths failing
  // with an error.  To avoid that problem, we first rename the file to a random
  // name, with a suffix of `kLockSuffix` to prevent it from being included in
  // List results.

  unsigned int buf[5];
  for (int i = 0; i < 5; ++i) {
    ::rand_s(&buf[i]);
  }
  char temp_name[64];
  size_t temp_name_size = ::snprintf(
      temp_name, std::size(temp_name), "_deleted_%08x%08x%08x%08x%08x%.*s",
      buf[0], buf[1], buf[2], buf[3], buf[4],
      static_cast<int>(kLockSuffix.size()), kLockSuffix.data());
  if (!internal_file_util::RenameFile(
          fd,
          tensorstore::StrCat(GetDirName(path),
                              std::string_view(temp_name, temp_name_size)))) {
    return false;
  }

  // FileDispositionInfoEx is a new API in Windows 10 and this structure does
  // not seem to be defined in the Windows SDK
  //
  // https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information_ex
  struct FileDispositionInfoExData {
    ULONG Flags;
  };
  FileDispositionInfoExData disposition_info;
  disposition_info.Flags = 0x00000001 /*FILE_DISPOSITION_DELETE*/ |
                           0x00000002 /*FILE_DISPOSITION_POSIX_SEMANTICS*/;
  return static_cast<bool>(::SetFileInformationByHandle(
      fd, FileDispositionInfoEx, &disposition_info, sizeof(disposition_info)));
}

bool DeleteFile(std::string_view path) {
  WindowsPathConverter converter(path);
  if (converter.failed()) return false;
  UniqueFileDescriptor delete_fd(::CreateFileW(
      converter.wc_str(),
      // Even though we only need write access, the `CreateFile`
      // documentation recommends specifying GENERIC_READ as well for better
      // performance if the file is on a network share.
      /*dwDesiredAccess=*/GENERIC_READ | GENERIC_WRITE | DELETE,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr));
  if (!delete_fd.valid()) return false;
  return DeleteOpenFile(delete_fd.get(), path);
}

UniqueFileDescriptor OpenExistingFileForReading(std::string_view path) {
  UniqueFileDescriptor fd;
  WindowsPathConverter converter(path);
  if (!converter.failed()) {
    fd.reset(::CreateFileW(
        converter.wc_str(), /*dwDesiredAccess=*/GENERIC_READ,
        /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        /*lpSecurityAttributes=*/nullptr,
        /*dwCreationDisposition=*/OPEN_EXISTING,
        /*dwFlagsAndAttributes=*/0,
        /*hTemplateFile=*/nullptr));
  }
  return fd;
}

UniqueFileDescriptor OpenFileForWriting(std::string_view path) {
  WindowsPathConverter converter(path);
  if (converter.failed()) return {};
  return UniqueFileDescriptor(::CreateFileW(
      converter.wc_str(),
      // Even though we only need write access, the `CreateFile`
      // documentation recommends specifying GENERIC_READ as well for better
      // performance if the file is on a network share.
      /*dwDesiredAccess=*/GENERIC_READ | GENERIC_WRITE | DELETE,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_ALWAYS,
      /*dwFlagsAndAttributes=*/0,
      /*hTemplateFile=*/nullptr));
}

/// Returns an OVERLAPPED with the lock offset used for the lock files.
inline ::OVERLAPPED GetLockOverlapped() {
  // Use a very high lock offset to ensure it does not conflict with any valid
  // byte range in the file.  This is important because we rename the lock file
  // to the real key before releasing the lock, and during the time between the
  // rename and the lock release, we don't want any read attempts to fail.
  return GetOverlappedWithOffset(0xffffffff'fffffffe);
}

void FileLockTraits::Close(HANDLE handle) {
  auto lock_offset = GetLockOverlapped();
  // Ignore any errors.
  ::UnlockFileEx(handle, /*dwReserved=*/0, /*nNumberOfBytesToUnlockLow=*/1,
                 /*nNumberOfBytesToUnlockHigh=*/0,
                 /*lpOverlapped=*/&lock_offset);
}

bool FileLockTraits::Acquire(HANDLE handle) {
  auto lock_offset = GetLockOverlapped();
  return ::LockFileEx(handle, /*dwFlags=*/LOCKFILE_EXCLUSIVE_LOCK,
                      /*dwReserved=*/0,
                      /*nNumberOfBytesToLockLow=*/1,
                      /*nNumberOfBytesToLockHigh=*/0,
                      /*lpOverlapped=*/&lock_offset);
}

bool DirectoryIterator::Entry::Delete(bool is_directory) const {
  if (is_directory) {
    WindowsPathConverter converter(path);
    return !converter.failed() && ::RemoveDirectoryW(converter.wc_str());
  } else {
    return internal_file_util::DeleteFile(path);
  }
}

void DirectoryIterator::Update(const ::WIN32_FIND_DATAW& find_data) {
  is_directory_ =
      static_cast<bool>(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  int utf8_size = ::WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, find_data.cFileName, -1,
      path_component_utf8, std::size(path_component_utf8), nullptr, nullptr);
  if (utf8_size == 0) {
    last_error = ::GetLastError();
  } else {
    last_error = 0;
  }
  path_component_size = utf8_size - 1;
}
bool DirectoryIterator::Next() {
  if (initial) {
    initial = false;
  } else {
    ::WIN32_FIND_DATAW find_data;
    if (::FindNextFileW(find_handle.get(), &find_data)) {
      Update(find_data);
    } else {
      last_error = ::GetLastError();
    }
  }
  return last_error == 0;
}

DirectoryIterator::Entry DirectoryIterator::GetEntry() const {
  if (directory_path_.empty() || absl::EndsWith(directory_path_, "/")) {
    return {tensorstore::StrCat(directory_path_, path_component())};
  } else {
    return {tensorstore::StrCat(directory_path_, "/", path_component())};
  }
}

bool DirectoryIterator::Make(Entry entry,
                             std::unique_ptr<DirectoryIterator>* new_iterator) {
  auto glob_pattern = entry.path + "/*";
  WindowsPathConverter converter(glob_pattern);
  if (converter.failed()) return false;
  UniqueFileDescriptor find_handle;
  ::WIN32_FIND_DATAW find_data;
  auto* it = new DirectoryIterator;
  new_iterator->reset(it);
  it->directory_path_ = std::move(entry.path);
  it->find_handle.reset(::FindFirstFileExW(converter.wc_str(), FindExInfoBasic,
                                           &find_data, FindExSearchNameMatch,
                                           /*lpSearchFilter=*/nullptr,
                                           /*dwAdditionalFlags=*/0));
  if (!it->find_handle.valid()) {
    it->last_error = ::GetLastError();
  } else {
    it->Update(find_data);
  }
  return true;
}

Result<std::string> GetCwd() {
  // Determine required buffer size.
  DWORD size = ::GetCurrentDirectoryW(0, nullptr);
  // size is equal to the required length, INCLUDING the terminating NUL.
  while (true) {
    // size of 0 indicates an error
    if (size == 0) break;

    std::vector<wchar_t> buf(size);

    // If `size` was sufficient, `new_size` is equal to the path length,
    // EXCLUDING the terminating NUL.
    //
    // If `size` was insufficient, `new_size` is equal to the path length,
    // INCLUDING the terminating NUL.
    DWORD new_size = ::GetCurrentDirectoryW(size, buf.data());
    if (new_size != size - 1) {
      // Another thread changed the current working directory between the two
      // calls to `GetCurrentDirectoryW`.

      // It is not valid for `size` to exactly equal `new_size`, since that
      // would simultaneously mean `size` was insufficient but also the correct
      // size.
      ABSL_CHECK_NE(new_size, size);

      if (new_size > size) {
        size = new_size;
        continue;
      }
    }
    std::string utf8_buf;
    utf8_buf.resize(new_size * 3);
    int utf8_size = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, buf.data(), static_cast<int>(new_size),
        utf8_buf.data(), utf8_buf.size(), /*lpDefaultChar=*/nullptr,
        /*lpUsedDefaultChar=*/nullptr);
    if (utf8_size == 0) {
      return internal::StatusFromOsError(
          ::GetLastError(),
          "Failed to convert current working directory to UTF-8");
    }
    utf8_buf.resize(utf8_size);
    return utf8_buf;
  }

  return internal::StatusFromOsError(::GetLastError(),
                                     "Failed to get current working directory");
}

absl::Status SetCwd(const std::string& path) {
  WindowsPathConverter converter(path);
  if (converter.failed()) {
    return internal::StatusFromOsError(
        ::GetLastError(),
        "Failed to convert path to UTF-16: ", tensorstore::QuoteString(path));
  }
  if (!::SetCurrentDirectoryW(converter.wc_str())) {
    return internal::StatusFromOsError(
        ::GetLastError(), "Failed to set current working directory to: ",
        tensorstore::QuoteString(path));
  }
  return absl::OkStatus();
}

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // defined(_WIN32)
