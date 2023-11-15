// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/os/subprocess.h"

#ifndef _WIN32
#error "Use subprocess_posix.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

// keep below windows.h
#include <processthreadsapi.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/wstring.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {
namespace {

/// Append an argument to the command_line
void AppendToCommandLine(std::string* command_line,
                         const std::string& argument) {
  if (argument.empty()) {
    return;
  }
  if (argument.find_first_of(" \t\n\v\"") == std::string::npos) {
    absl::StrAppend(command_line, argument);
    return;
  }
  command_line->append(1, '"');
  for (auto it = argument.begin(); it != argument.end(); ++it) {
    switch (*it) {
      case '"':
        // escape "
        command_line->append(1, '\\');
        break;
      case '\\':
        // escape backslash
        command_line->append(1, '\\');
        break;
      default:
        break;
    }
    command_line->append(1, *it);
  }
  command_line->append(1, '"');
}

std::wstring BuildEnvironmentBlock(
    const absl::flat_hash_map<std::string, std::string>& env) {
  std::wstring result;
  for (const auto& [key, value] : env) {
    auto env_str = absl::StrCat(key, "=", value);
    std::wstring env_wstr;
    if (ConvertUTF8ToWindowsWide(env_str, env_wstr).ok()) {
      result.append(env_wstr);
      result.push_back(0);
    }
  }
  result.push_back(0);
  result.push_back(0);
  return result;
}

absl::Status SetupHandles(const SubprocessOptions& options, STARTUPINFOEXW& ex,
                          std::vector<HANDLE>& handles_to_close,
                          std::vector<HANDLE>& handles_to_inherit) {
  ex.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  ex.StartupInfo.hStdInput = nullptr;
  ex.StartupInfo.hStdOutput = nullptr;
  ex.StartupInfo.hStdError = nullptr;

  // Only add a handle once, and only add file handles which are
  // DISK or PIPE to the child process.
  auto maybe_inherit_handle = [&](HANDLE h) {
    for (HANDLE x : handles_to_inherit) {
      if (x == h) {
        return;
      }
    }
    auto handle_type = GetFileType(h);
    if (handle_type == FILE_TYPE_DISK || handle_type == FILE_TYPE_PIPE) {
      handles_to_inherit.push_back(h);
    }
  };

  SECURITY_ATTRIBUTES securityAttributes;
  ZeroMemory(&securityAttributes, sizeof(securityAttributes));
  securityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  securityAttributes.bInheritHandle = TRUE;
  securityAttributes.lpSecurityDescriptor = NULL;

  auto open_for_subprocess = [&](std::string_view filename, HANDLE& location,
                                 bool read) -> absl::Status {
    std::wstring wfilename;
    TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(filename, wfilename));

    location = CreateFileW(
        wfilename.c_str(),
        /*dwDesiredAccess=*/GENERIC_READ | (read ? 0 : GENERIC_WRITE),
        /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        /*lpSecurityAttributes=*/&securityAttributes,
        /*dwCreationDisposition=*/(read ? OPEN_EXISTING : OPEN_ALWAYS),
        /*dwFlagsAndAttributes=*/FILE_ATTRIBUTE_NORMAL,
        /*hTemplateFile=*/nullptr);
    if (location == INVALID_HANDLE_VALUE) {
      location = nullptr;
      return StatusFromOsError(GetLastErrorCode(),
                               "SpawnSubprocess: CreateFileW ", filename,
                               " failed");
    }
    handles_to_close.push_back(location);
    maybe_inherit_handle(location);
    return absl::OkStatus();
  };

  // stdin
  if (auto* redir =
          std::get_if<SubprocessOptions::Redirect>(&options.stdin_action);
      redir != nullptr) {
    TENSORSTORE_RETURN_IF_ERROR(
        open_for_subprocess(redir->filename, ex.StartupInfo.hStdInput, true));
  }

  // stdout
  const std::string* stdout_filename = nullptr;
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stdout_action)) {
    ex.StartupInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    maybe_inherit_handle(ex.StartupInfo.hStdOutput);
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stdout_action);
             redir != nullptr) {
    stdout_filename = &redir->filename;
    TENSORSTORE_RETURN_IF_ERROR(
        open_for_subprocess(redir->filename, ex.StartupInfo.hStdOutput, false));
  }

  // stderr
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stderr_action)) {
    ex.StartupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    maybe_inherit_handle(ex.StartupInfo.hStdOutput);
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stderr_action);
             redir != nullptr) {
    if (stdout_filename && redir->filename == *stdout_filename) {
      ex.StartupInfo.hStdError = ex.StartupInfo.hStdOutput;
    } else {
      TENSORSTORE_RETURN_IF_ERROR(open_for_subprocess(
          redir->filename, ex.StartupInfo.hStdError, false));
    }
  }

  return absl::OkStatus();
}

}  // namespace

struct Subprocess::Impl {
  Impl();
  ~Impl();

  absl::Status Kill(int signal);
  Result<int> Join(bool block);

  PROCESS_INFORMATION pi_;
};

Subprocess::Impl::Impl() { ZeroMemory(&pi_, sizeof(pi_)); }

Subprocess::Impl::~Impl() {
  // If stdin were a pipe, close it here.
  ::CloseHandle(pi_.hProcess);
  ::CloseHandle(pi_.hThread);
}

absl::Status Subprocess::Impl::Kill(int signal) {
  unsigned int exit_code = 99;
  if (0 != TerminateProcess(pi_.hProcess, exit_code)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(GetLastErrorCode(), "On Subprocess::Kill");
}

Result<int> Subprocess::Impl::Join(bool block) {
  if (!block) {
    DWORD process_exit_code = 0;
    if (0 != GetExitCodeProcess(pi_.hProcess, &process_exit_code)) {
      if (process_exit_code == STILL_ACTIVE) {
        return absl::UnavailableError("");
      }
      return static_cast<int>(process_exit_code);
    }
  } else {
    // If stdin were a pipe, close it here.
    DWORD wait_status = WaitForSingleObject(pi_.hProcess, INFINITE);
    if (wait_status == WAIT_OBJECT_0) {
      DWORD process_exit_code = 0;
      if (0 != GetExitCodeProcess(pi_.hProcess, &process_exit_code)) {
        return static_cast<int>(process_exit_code);
      }
    }
  }
  return StatusFromOsError(GetLastErrorCode(), "Subprocess::Join failed");
}

Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options) {
  if (options.executable.empty()) {
    return absl::InvalidArgumentError(
        "SpawnSubprocess: executable not specified.");
  }
  if (options.executable.find_first_of('"') != std::string::npos) {
    return absl::InvalidArgumentError(
        "SpawnSubprocess: executable string includes \" character.");
  }

  // Unlike posix, Windows composes a commandline as one large string,
  // where "executable" (arg[0]) has different quoting conventions from
  // the remaining args.
  std::string command_line = absl::StrCat("\"", options.executable, "\"");
  for (const auto& arg : options.args) {
    command_line.append(1, ' ');
    AppendToCommandLine(&command_line, arg);
  }

  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(command_line, wpath));
  constexpr size_t kMaxWindowsPathSize = 32768;
  if (wpath.size() >= kMaxWindowsPathSize) {
    return absl::InvalidArgumentError("SpawnSubprocess: path too large.");
  }

  // Build the environment block.
  std::wstring env;
  if (options.env.has_value()) {
    if (!options.env->count("PATH")) {
      // Windows uses the PATH when searching for DLLs:
      // https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
      ABSL_LOG(INFO) << "SpawnSubprocess: environment missing PATH; Some DLLs "
                        "may fail to load.";
    }
    env = BuildEnvironmentBlock(*options.env);
  }

  // Setup STARTUPINFO to redirect handles.
  // Without STARTF_USESTDHANDLES, the subprocess inherit the current process
  // handles, but (1) we don't want to inherit stdin, and we sometimes want to
  // override inheritance of stdout/stderr, so set them explicitly.
  STARTUPINFOEXW ex;
  ZeroMemory(&ex, sizeof(ex));
  ex.StartupInfo.cb = sizeof(ex);

  LPVOID lpEnvironment = nullptr;
  DWORD dwCreationFlags = EXTENDED_STARTUPINFO_PRESENT;  // DETACHED_PROCESS?
  if (!env.empty()) {
    lpEnvironment = env.data();
    dwCreationFlags |= CREATE_UNICODE_ENVIRONMENT;
  }

  std::vector<HANDLE> handles_to_close;
  std::vector<HANDLE> handles_to_inherit;

  auto status = SetupHandles(options, ex, handles_to_close, handles_to_inherit);

  std::unique_ptr<uint8_t[]> ex_storage;

  // Initialize the ProcThreadAttributeList to pass on inheritable handles.
  // This avoids potential pitfalls caused by inheriting all handles.
  status.Update([&]() {
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    ex_storage = std::make_unique<uint8_t[]>(size);
    ex.lpAttributeList =
        reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(ex_storage.get());
    if (0 ==
        InitializeProcThreadAttributeList(ex.lpAttributeList, 1, 0, &size)) {
      return StatusFromOsError(
          GetLastErrorCode(),
          "SpawnSubprocess: InitializeProcThreadAttributeList failed");
    }
    if (HANDLE self =
            OpenProcess(STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFF, 1,
                        GetCurrentProcessId());
        self != nullptr) {
      handles_to_inherit.push_back(self);
      handles_to_close.push_back(self);
    }
    if (0 == UpdateProcThreadAttribute(
                 ex.lpAttributeList,
                 /*dwFlags=*/0,
                 /*Attribute=*/PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                 /*lpValue=*/&handles_to_inherit[0],
                 /*cbSize=*/handles_to_inherit.size() *
                     sizeof(handles_to_inherit[0]),
                 /*lpPreviousValue=*/nullptr,
                 /*lpReturnSize=*/nullptr)) {
      return StatusFromOsError(
          GetLastErrorCode(),
          "SpawnSubprocess: UpdateProcThreadAttribute failed");
    }
    return absl::OkStatus();
  }());

  std::shared_ptr<Subprocess::Impl> impl;

  if (status.ok()) {
    impl = std::make_shared<Subprocess::Impl>();
    if (0 == CreateProcessW(
                 /*lpApplicationName=*/nullptr,
                 /*lpCommandLine=*/wpath.data(),
                 /*lpProcessAttributes=*/nullptr,
                 /*lpThreadAttributes=*/nullptr,
                 /*bInheritHandles=*/TRUE,
                 /*dwCreationFlags=*/dwCreationFlags,
                 /*lpEnvironment=*/lpEnvironment,
                 /*lpCurrentDirectory=*/nullptr,
                 /*lpStartupInfo,=*/&ex.StartupInfo,
                 /*lpProcessInformation=*/&impl->pi_)) {
      // Create failed.
      status = StatusFromOsError(GetLastErrorCode(),
                                 "SpawnSubprocess: CreateProcessW ",
                                 options.executable, " failed");
    }
  }

  // Cleanup handles opened for the process.
  for (HANDLE h : handles_to_close) {
    CloseHandle(h);
  }
  if (status.ok()) {
    assert(impl);
    return Subprocess(std::move(impl));
  }

  return status;
}

// ===================================================================

Subprocess::~Subprocess() = default;

absl::Status Subprocess::Kill(int signal) const {
  assert(impl_);
  return impl_->Kill(signal);
}

Result<int> Subprocess::Join(bool block) const {
  assert(impl_);
  return impl_->Join(block);
}

}  // namespace internal
}  // namespace tensorstore
