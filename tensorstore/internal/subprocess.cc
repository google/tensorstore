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
// keep below windows.h
#include <processthreadsapi.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <string.h>  // IWYU pragma: keep

#include <atomic>  // IWYU pragma: keep
#include <cassert>
#include <limits>  // IWYU pragma: keep
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>  // IWYU pragma: keep

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

#ifndef _WIN32
extern char** environ;
#endif

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

/// Converts a UTF-8 string to a windows Multibyte string.
/// TODO: Consider consolidating with kvstore/file/windows_file_util.cc
absl::Status ConvertStringToMultibyte(std::string_view in, std::wstring& out) {
  if (in.size() > std::numeric_limits<int>::max()) {
    return StatusFromOsError(ERROR_BUFFER_OVERFLOW,
                             "ConvertStringToMultibyte buffer overflow");
  }
  if (in.empty()) {
    out.clear();
    return absl::OkStatus();
  }
  int n = ::MultiByteToWideChar(
      /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS, in.data(),
      static_cast<int>(in.size()), nullptr, 0);
  if (n <= 0) {
    return StatusFromOsError(GetLastErrorCode(), "MultiByteToWideChar failed");
  }
  out.resize(n);
  int m = ::MultiByteToWideChar(
      /*CodePage=*/CP_UTF8, /*dwFlags=*/MB_ERR_INVALID_CHARS, in.data(),
      static_cast<int>(in.size()), out.data(), n);
  if (n <= 0) {
    return StatusFromOsError(GetLastErrorCode(), "MultiByteToWideChar failed");
  }
  return absl::OkStatus();
}

std::wstring BuildEnvironmentBlock(
    const absl::flat_hash_map<std::string, std::string>& env) {
  std::wstring result;
  for (const auto& [key, value] : env) {
    auto env_str = absl::StrCat(key, "=", value);
    std::wstring env_wstr;
    if (ConvertStringToMultibyte(env_str, env_wstr).ok()) {
      result.append(env_wstr);
      result.push_back(0);
    }
  }
  result.push_back(0);
  result.push_back(0);
  return result;
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
  TENSORSTORE_RETURN_IF_ERROR(ConvertStringToMultibyte(command_line, wpath));
  constexpr size_t kMaxWindowsPathSize = 32768;
  if (wpath.size() >= kMaxWindowsPathSize) {
    return absl::InvalidArgumentError("SpawnSubprocess: path too large.");
  }

  std::wstring env;
  if (options.env.has_value()) {
    env = BuildEnvironmentBlock(*options.env);
  }

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

  LPVOID lpEnvironment = nullptr;
  DWORD dwCreationFlags = 0;  // DETACHED_PROCESS?
  if (!env.empty()) {
    lpEnvironment = env.data();
    dwCreationFlags |= CREATE_UNICODE_ENVIRONMENT;
  }

  auto impl = std::make_shared<Subprocess::Impl>();
  if (0 == CreateProcessW(
               /*lpApplicationName=*/nullptr,
               /*lpCommandLine=*/wpath.data(),
               /*lpProcessAttributes=*/nullptr,
               /*lpThreadAttributes=*/nullptr,
               /*bInheritHandles=*/TRUE,
               /*dwCreationFlags=*/dwCreationFlags,
               /*lpEnvironment=*/lpEnvironment,
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
  Impl(pid_t pid) : child_pid_(pid), exit_code_(-1) {}
  ~Impl();

  absl::Status Kill(int signal);
  Result<int> Join(bool block);

  std::atomic<pid_t> child_pid_;
  std::atomic<int> exit_code_;
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

Result<int> Subprocess::Impl::Join(bool block) {
  // If stdin were a pipe, close it here.
  int status;
  for (;;) {
    auto pid = child_pid_.load();
    if (pid == -1) {
      // Already joined. NOTE: This races with setting the pid to -1, so it
      // may not be correct when Join is called concurrently.
      return exit_code_.load();
    }

    int result = waitpid(pid, &status, block ? 0 : WNOHANG);
    if ((result < 0) && !retry(errno)) {
      return StatusFromOsError(GetLastErrorCode(), "Subprocess::Join failed");
    }
    if (!block && result == 0) {
      return absl::UnavailableError("");
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

  std::vector<std::string> envp_strings;
  std::vector<char*> envp;
  if (options.env.has_value()) {
    envp_strings.reserve(options.env->size());
    envp.reserve(options.env->size() + 1);
    for (const auto& [key, value] : *options.env) {
      envp_strings.push_back(absl::StrCat(key, "=", value));
      envp.push_back(envp_strings.back().data());
    }
    envp.push_back(nullptr);
  }

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
                        const_cast<char* const*>(argv.data()),
                        options.env.has_value()
                            ? const_cast<char* const*>(envp.data())
                            : environ);

  ABSL_LOG(INFO) << "posix_spawn " << argv[0] << " err:" << err
                 << " pid: " << child_pid;

  posix_spawn_file_actions_destroy(&file_actions);
  if (err != 0) {
    return StatusFromOsError(GetLastErrorCode(), "posix_spawn ",
                             options.executable, " failed");
  }
  ABSL_CHECK_GT(child_pid, 0);
  return Subprocess(std::make_shared<Subprocess::Impl>(child_pid));
}

#endif  // !_WIN32

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
