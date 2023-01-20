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

#include "tensorstore/internal/subprocess.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else  // !_WIN32
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

#ifdef _WIN32
namespace {

/// Append an argument to the command_line
void AppendToCommandLine(std::string* command_line,
                         const std::string& argument) {
  if (argument.empty()) {
    return;
  }
  if (!absl::StrContains(argument, " \t\n\v")) {
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

}  // namespace

struct Subprocess::Impl {
  Impl();
  ~Impl();

  absl::Status Kill(int signal);
  Result<int> Join();

  PROCESS_INFORMATION pi_;
};

Subprocess::Impl::Impl() { ZeroMemory(&pi_, sizeof(pi_)); }

Subprocess::Impl::~Impl() {
  // If stdin were a pipe, close it here.
  CloseHandle(pi_.hProcess);
  CloseHandle(pi_.hThread);
}

absl::Status Subprocess::Impl::Kill(int signal) {
  unsigned int exit_code = 99;
  if (0 != TerminateProcess(pi_.hProcess, exit_code)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(GetLastErrorCode(), "On Subprocess::Kill");
}

Result<int> Subprocess::Impl::Join() {
  // If stdin were a pipe, close it here.
  DWORD wait_status = WaitForSingleObject(pi_.hProcess, INFINITE);
  if (wait_status == WAIT_OBJECT_0) {
    DWORD process_exit_code = 0;
    if (0 != GetExitCodeProcess(pi_.hProcess, &process_exit_code)) {
      return static_cast<int>(process_exit_code);
    }
  }
  return StatusFromOsError(GetLastErrorCode(), "Subprocess::Join failed");
}

Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options) {
  if (options.executable.empty()) {
    return absl::InvalidArgumentError(
        "SpawnSubprocess: executable not specified.");
  }

  /// Quote path and convert from utf-8 to wchar_t.
  std::string command_line;
  AppendToCommandLine(&command_line, options.executable);
  for (const auto& arg : options.args) {
    command_line.append(1, ' ');
    AppendToCommandLine(&command_line, arg);
  }
  constexpr size_t kMaxWindowsPathSize = 32768;
  std::unique_ptr<wchar_t[]> wpath(new wchar_t[kMaxWindowsPathSize]);
  int n = ::MultiByteToWideChar(
      /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS,
      command_line.data(), static_cast<int>(command_line.size()), wpath.get(),
      kMaxWindowsPathSize - 1);
  if (n == 0) {
    return StatusFromOsError(GetLastErrorCode(), "MultiByteToWideChar failed");
  }
  wpath[n] = 0;

  // Setup STARTUPINFO to redirect handles.
  // Without STARTF_USESTDHANDLES, the subprocess inherit the current process
  // handles, but (1) we don't want to inherit stdin, and we sometimes want to
  // override inheritance of stdout/stderr, so set them explicitly.
  STARTUPINFOW startup_info;
  ZeroMemory(&startup_info, sizeof(startup_info));
  startup_info.cb = sizeof(startup_info);
  startup_info.dwFlags = STARTF_USESTDHANDLES;
  startup_info.hStdInput = nullptr;
  startup_info.hStdOutput = nullptr;
  startup_info.hStdError = nullptr;
  if (options.inherit_stdout) {
    startup_info.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  }
  if (options.inherit_stderr) {
    startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  }

  auto impl = std::make_shared<Subprocess::Impl>();
  if (0 == CreateProcessW(
               /*lpApplicationName=*/nullptr,
               /*lpCommandLine=*/wpath.get(),
               /*lpProcessAttributes=*/nullptr,
               /*lpThreadAttributes=*/nullptr,
               /*bInheritHandles=*/TRUE,
               /*dwCreationFlags=*/0,  // DETACHED_PROCESS?
               /*lpEnvironment=*/nullptr,
               /*lpCurrentDirectory=*/nullptr,
               /*lpStartupInfo,=*/&startup_info,
               /*lpProcessInformation=*/&impl->pi_)) {
    // Create failed.
    return StatusFromOsError(GetLastErrorCode(), "CreateProcessW ",
                             options.executable, " failed");
  }
  return Subprocess(std::move(impl));
}

// ===================================================================

#else  // _WIN32

namespace {

bool retry(int e) {
  return ((e == EINTR) || (e == EAGAIN) || (e == EWOULDBLOCK));
}

}  // namespace

struct Subprocess::Impl {
  ~Impl();

  absl::Status Kill(int signal);
  Result<int> Join();

  std::atomic<pid_t> child_pid_{-1};
  std::atomic<int> exit_code_{-1};
};

Subprocess::Impl::~Impl() {
  // If stdin were a pipe, close it here.
  if (pid_t pid = child_pid_; pid != -1) {
    ABSL_LOG(INFO) << "Subprocess may have zombie pid: " << pid;
  }
}

absl::Status Subprocess::Impl::Kill(int signal) {
  auto pid = child_pid_.load();
  if (pid == -1) {
    // Already joined.
    return absl::InternalError("Subprocess already exited.");
  }
  if (0 == kill(pid, signal)) {
    return absl::OkStatus();
  }
  return StatusFromOsError(GetLastErrorCode(), "On Subprocess::Kill");
}

Result<int> Subprocess::Impl::Join() {
  // If stdin were a pipe, close it here.
  int status;
  for (;;) {
    auto pid = child_pid_.load();
    if (pid == -1) {
      // Already joined. NOTE: This races with setting the pid to -1, so it
      // may not be correct when Join is called concurrently.
      return exit_code_.load();
    }

    int result = waitpid(pid, &status, 0);
    if ((result < 0) && !retry(errno)) {
      return StatusFromOsError(GetLastErrorCode(), "Subprocess::Join failed");
    }
    if (result != pid) continue;
    if (WIFEXITED(status)) {
      int exit_code = WEXITSTATUS(status);
      if (child_pid_.exchange(-1) == pid) {
        exit_code_ = exit_code;
      }
      return exit_code;
    }
    if (WIFSIGNALED(status)) {
      int exit_code = -1;
      if (child_pid_.exchange(-1) == pid) {
        exit_code_ = exit_code;
        ABSL_LOG(INFO) << "Child terminated by signal " << WTERMSIG(status);
      }
      return exit_code;
    }
  }
}

Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options) {
  if (options.executable.empty()) {
    return absl::InvalidArgumentError(
        "SpawnSubprocess: executable not specified.");
  }

  // Create `argv` and `envp` parameters for exec().
  std::vector<const char*> argv;
  argv.reserve(options.args.size() + 2);
  argv.push_back(options.executable.c_str());
  for (auto& arg : options.args) {
    argv.push_back(arg.c_str());
  }
  argv.push_back(nullptr);

  // TODO: Add environment variables.
  std::vector<char*> envp({nullptr});

  /// Add posix file actions; specifically, redirect stdin/stdout/sterr to
  /// /dev/null to ensure that the file descriptors are not reused.
  posix_spawn_file_actions_t file_actions;
  if (0 != posix_spawn_file_actions_init(&file_actions)) {
    return absl::InternalError("posix_spawn failure");
  }
  int action_result = [&]() -> int {
    int ret = posix_spawn_file_actions_addopen(&file_actions, 0, "/dev/null",
                                               O_RDWR, 0);
    if (ret != 0) {
      return ret;
    }
    if (!options.inherit_stdout) {
      ret = posix_spawn_file_actions_adddup2(&file_actions, 0, 1);
      if (ret != 0) {
        return ret;
      }
    }
    if (!options.inherit_stderr) {
      return posix_spawn_file_actions_adddup2(&file_actions, 0, 2);
    }
    return 0;
  }();
  if (action_result != 0) {
    auto status = StatusFromOsError(GetLastErrorCode(),
                                    "posix_spawn_file_actions failed");
    posix_spawn_file_actions_destroy(&file_actions);
    return status;
  }

  pid_t child_pid = 0;
  int err = posix_spawn(&child_pid, argv[0], &file_actions, nullptr,
                        const_cast<char* const*>(argv.data()), envp.data());
  ABSL_LOG(INFO) << "posix_spawn " << argv[0] << " err:" << err
                 << " pid: " << child_pid;

  posix_spawn_file_actions_destroy(&file_actions);
  if (err != 0) {
    return StatusFromOsError(GetLastErrorCode(), "posix_spawn ",
                             options.executable, " failed");
  }
  ABSL_CHECK_GT(child_pid, 0);

  auto impl = std::make_shared<Subprocess::Impl>();
  impl->child_pid_ = child_pid;
  return Subprocess(std::move(impl));
}

#endif  // !_WIN32

Subprocess::~Subprocess() = default;

absl::Status Subprocess::Kill(int signal) const { return impl_->Kill(signal); }

Result<int> Subprocess::Join() const { return impl_->Join(); }

}  // namespace internal
}  // namespace tensorstore
