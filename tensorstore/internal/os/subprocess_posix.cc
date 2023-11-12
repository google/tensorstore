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

#include <cerrno>

#include "absl/container/inlined_vector.h"
#include "tensorstore/internal/os/subprocess.h"

#ifdef _WIN32
#error "Use subprocess_win.cc instead."
#endif

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/wstring.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"  // IWYU pragma: keep

extern char** environ;

namespace tensorstore {
namespace internal {
namespace {

bool retry(int e) {
  return ((e == EINTR) || (e == EAGAIN) || (e == EWOULDBLOCK));
}

Result<int> open_with_retry(const char* path, int mode) {
  int ret;
  do {
    ret = ::open(path, mode, 0666);
  } while (ret < 0 && errno == EINTR);
  if (ret < 0) {
    return StatusFromOsError(errno, "While opening ", path);
  }
  return ret;
}

// Open all the files used in spawn-process stdin/stdout/stderr
// and add them to the file actions. While posix_spawn_file_actions_
// can open files, the precise cause on failure is harder to determine.
absl::Status AddPosixFileActions(const SubprocessOptions& options,
                                 posix_spawn_file_actions_t* file_actions,
                                 absl::InlinedVector<int, 3>& fds_to_close) {
  int fds[3] = {-1, -1, -1};
  {
    const char* stdin_filename = "/dev/null";
    if (auto* redir =
            std::get_if<SubprocessOptions::Redirect>(&options.stdin_action);
        redir != nullptr) {
      stdin_filename = redir->filename.c_str();
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        int fd, open_with_retry(stdin_filename, O_RDONLY | O_CLOEXEC));
    fds_to_close.push_back(fd);
    fds[STDIN_FILENO] = fd;
  }

  const char* stdout_filename = nullptr;
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stdout_action)) {
    fds[STDOUT_FILENO] = STDOUT_FILENO;
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stdout_action);
             redir != nullptr) {
    stdout_filename = redir->filename.c_str();
  } else {  // Ignore
    stdout_filename = "/dev/null";
  }

  const char* stderr_filename = nullptr;
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stderr_action)) {
    fds[STDERR_FILENO] = STDERR_FILENO;
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stderr_action);
             redir != nullptr) {
    stderr_filename = redir->filename.c_str();
  } else {  // Ignore
    stderr_filename = "/dev/null";
  }

  if (stdout_filename) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        int fd,
        open_with_retry(stdout_filename, O_WRONLY | O_CREAT | O_CLOEXEC));
    fds_to_close.push_back(fd);
    fds[STDOUT_FILENO] = fd;
  }
  if (stderr_filename) {
    if (std::string_view(stdout_filename) ==
        std::string_view(stderr_filename)) {
      fds[STDERR_FILENO] = fds[STDOUT_FILENO];
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(
          int fd,
          open_with_retry(stderr_filename, O_WRONLY | O_CREAT | O_CLOEXEC));
      fds_to_close.push_back(fd);
      fds[STDOUT_FILENO] = fd;
    }
  }

  // Now dup the handles.
  // An error here is really a precondition or out-of-memory error.
  int r = 0;
  r = posix_spawn_file_actions_adddup2(file_actions, fds[STDIN_FILENO],
                                       STDIN_FILENO);
  if (r != 0) {
    return StatusFromOsError(r, "posix_spawn_file_actions_adddup2");
  }
  r = posix_spawn_file_actions_adddup2(file_actions, fds[STDOUT_FILENO],
                                       STDOUT_FILENO);
  if (r != 0) {
    return StatusFromOsError(r, "posix_spawn_file_actions_adddup2");
  }
  r = posix_spawn_file_actions_adddup2(file_actions, fds[STDERR_FILENO],
                                       STDERR_FILENO);
  if (r != 0) {
    return StatusFromOsError(r, "posix_spawn_file_actions_adddup2");
  }
  return absl::OkStatus();
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

  pid_t child_pid = 0;
  absl::InlinedVector<int, 3> fds_to_close;
  auto status = AddPosixFileActions(options, &file_actions, fds_to_close);
  if (status.ok()) {
    // File
    int err = posix_spawn(&child_pid, argv[0], &file_actions, nullptr,
                          const_cast<char* const*>(argv.data()),
                          options.env.has_value()
                              ? const_cast<char* const*>(envp.data())
                              : environ);

    ABSL_LOG(INFO) << "posix_spawn " << argv[0] << " err:" << err
                   << " pid:" << child_pid;
    if (err != 0) {
      status = StatusFromOsError(GetLastErrorCode(), "posix_spawn ",
                                 options.executable, " failed");
    }
  }

  for (int fd : fds_to_close) {
    ::close(fd);
  }
  posix_spawn_file_actions_destroy(&file_actions);

  if (!status.ok()) {
    return status;
  }

  ABSL_CHECK_GT(child_pid, 0);
  return Subprocess(std::make_shared<Subprocess::Impl>(child_pid));
}

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
