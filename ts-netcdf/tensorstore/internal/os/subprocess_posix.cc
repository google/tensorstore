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

#ifdef _WIN32
#error "Use subprocess_win.cc instead."
#endif

#include "tensorstore/internal/os/subprocess.h"
// Normal include order here.

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
#include <cerrno>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// IWYU pragma keep: Used by the windows version.
#include "absl/container/flat_hash_map.h"
#include "tensorstore/internal/os/wstring.h"

extern char** environ;

using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::OpenFileWrapper;
using ::tensorstore::internal_os::OpenFlags;
using ::tensorstore::internal_os::UniqueFileDescriptor;

namespace tensorstore {
namespace internal {
namespace {

bool retry(int e) {
  return ((e == EINTR) || (e == EAGAIN) || (e == EWOULDBLOCK));
}

absl::Status SetCloseOnExec(FileDescriptor fd, bool close_on_exec) {
  int flags = fcntl(fd, F_GETFD, 0);
  if (flags != -1) {
    if (static_cast<bool>(flags & FD_CLOEXEC) == close_on_exec) {
      return absl::OkStatus();
    }
    if (close_on_exec) {
      flags |= FD_CLOEXEC;
    } else {
      flags &= ~FD_CLOEXEC;
    }
    if (fcntl(fd, F_SETFD, flags) != -1) {
      return absl::OkStatus();
    }
  }
  return StatusFromOsError(errno, "fcntl");
}

// Open all the files used in spawn-process stdin/stdout/stderr
// and add them to the file actions. While posix_spawn_file_actions_
// can open files, the precise cause on failure is harder to determine.
absl::Status AddPosixFileActions(const SubprocessOptions& options,
                                 posix_spawn_file_actions_t* file_actions,
                                 absl::flat_hash_set<int>& fds_to_close,
                                 UniqueFileDescriptor& stdout_pipe_read,
                                 UniqueFileDescriptor& stderr_pipe_read) {
  absl::flat_hash_set<int> close_added;

  auto add_close_action = [&](int fd) {
    if (fd == -1 || close_added.contains(fd)) return;
    if (fd == STDIN_FILENO || fd == STDOUT_FILENO || fd == STDERR_FILENO)
      return;
    close_added.insert(fd);
    posix_spawn_file_actions_addclose(file_actions, fd);
  };

  // Set up a pipe for output.
  auto output_via_pipe = [&](int& write_fd,
                             UniqueFileDescriptor& read_fd) -> absl::Status {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
      return StatusFromOsError(errno, "pipe");
    }
    // TENSORSTORE_RETURN_IF_ERROR(SetNonblock(pipefd[0]));
    // TENSORSTORE_RETURN_IF_ERROR(SetNonblock(pipefd[1]));
    fds_to_close.insert(pipefd[1]);
    add_close_action(pipefd[0]);
    read_fd = UniqueFileDescriptor(pipefd[0]);
    write_fd = pipefd[1];
    return absl::OkStatus();
  };

  const std::string dev_null = "/dev/null";

  int fds[3] = {-1, -1, -1};

  // stdin
  const std::string* stdin_filename = nullptr;
  if (auto* fd =
          std::get_if<SubprocessOptions::RedirectFd>(&options.stdin_action)) {
    fds[STDERR_FILENO] = fd->fd;
    TENSORSTORE_RETURN_IF_ERROR(SetCloseOnExec(fd->fd, false));
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stdin_action);
             redir != nullptr) {
    stdin_filename = &redir->filename;
  } else {
    stdin_filename = &dev_null;
  }
  if (stdin_filename) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto unique_fd,
        OpenFileWrapper(*stdin_filename, OpenFlags::DefaultRead));
    fds_to_close.insert(unique_fd.get());
    fds[STDIN_FILENO] = unique_fd.release();
  }

  // stdout
  const std::string* stdout_filename = nullptr;
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stdout_action)) {
    fds[STDOUT_FILENO] = STDOUT_FILENO;
  } else if (std::holds_alternative<SubprocessOptions::Pipe>(
                 options.stdout_action)) {
    TENSORSTORE_RETURN_IF_ERROR(
        output_via_pipe(fds[STDOUT_FILENO], stdout_pipe_read));
  } else if (auto* fd = std::get_if<SubprocessOptions::RedirectFd>(
                 &options.stdout_action)) {
    fds[STDOUT_FILENO] = fd->fd;
    TENSORSTORE_RETURN_IF_ERROR(SetCloseOnExec(fd->fd, false));
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stdout_action);
             redir != nullptr) {
    stdout_filename = &redir->filename;
  } else {  // Ignore
    stdout_filename = &dev_null;
  }

  // stderr
  const std::string* stderr_filename = nullptr;
  if (std::holds_alternative<SubprocessOptions::Inherit>(
          options.stderr_action)) {
    fds[STDERR_FILENO] = STDERR_FILENO;
  } else if (std::holds_alternative<SubprocessOptions::Pipe>(
                 options.stderr_action)) {
    TENSORSTORE_RETURN_IF_ERROR(
        output_via_pipe(fds[STDERR_FILENO], stderr_pipe_read));
  } else if (auto* fd = std::get_if<SubprocessOptions::RedirectFd>(
                 &options.stderr_action)) {
    fds[STDERR_FILENO] = fd->fd;
    TENSORSTORE_RETURN_IF_ERROR(SetCloseOnExec(fd->fd, false));
  } else if (auto* redir = std::get_if<SubprocessOptions::Redirect>(
                 &options.stderr_action);
             redir != nullptr) {
    stderr_filename = &redir->filename;
  } else {  // Ignore
    stderr_filename = &dev_null;
  }

  if (stdout_filename) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto unique_fd,
        OpenFileWrapper(*stdout_filename,
                        OpenFlags::Create | OpenFlags::OpenWriteOnly));
    fds[STDOUT_FILENO] = unique_fd.get();
    fds_to_close.insert(unique_fd.release());
  }
  if (stderr_filename) {
    if (stdout_filename && *stdout_filename == *stderr_filename) {
      fds[STDERR_FILENO] = fds[STDOUT_FILENO];
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto unique_fd,
          OpenFileWrapper(*stderr_filename,
                          OpenFlags::Create | OpenFlags::OpenWriteOnly));
      fds[STDERR_FILENO] = unique_fd.get();
      fds_to_close.insert(unique_fd.release());
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

  // Add close actions.
  add_close_action(fds[STDIN_FILENO]);
  add_close_action(fds[STDOUT_FILENO]);
  add_close_action(fds[STDERR_FILENO]);
  return absl::OkStatus();
}

}  // namespace

struct Subprocess::Impl {
  Impl(pid_t pid, UniqueFileDescriptor stdout_pipe_read,
       UniqueFileDescriptor stderr_pipe_read)
      : child_pid_(pid),
        exit_code_(-1),
        stdout_pipe_read_(std::move(stdout_pipe_read)),
        stderr_pipe_read_(std::move(stderr_pipe_read)) {}

  ~Impl();

  absl::Status Kill(int signal);
  Result<int> Join(bool block);

  std::atomic<pid_t> child_pid_;
  std::atomic<int> exit_code_;

  UniqueFileDescriptor stdout_pipe_read_;
  UniqueFileDescriptor stderr_pipe_read_;
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
  while (true) {
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

  // Add posix file actions; specifically, redirect stdin/stdout/sterr to
  // /dev/null to ensure that the file descriptors are not reused.
  posix_spawn_file_actions_t file_actions;
  if (0 != posix_spawn_file_actions_init(&file_actions)) {
    return absl::InternalError("posix_spawn failure");
  }
  UniqueFileDescriptor stdout_pipe_read;
  UniqueFileDescriptor stderr_pipe_read;

  pid_t child_pid = 0;
  absl::flat_hash_set<int> fds_to_close;
  auto status = AddPosixFileActions(options, &file_actions, fds_to_close,
                                    stdout_pipe_read, stderr_pipe_read);
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
  return Subprocess(std::make_shared<Subprocess::Impl>(
      child_pid, std::move(stdout_pipe_read), std::move(stderr_pipe_read)));
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

internal_os::FileDescriptor Subprocess::stdout_pipe() const {
  return impl_->stdout_pipe_read_.get();
}
internal_os::FileDescriptor Subprocess::stderr_pipe() const {
  return impl_->stderr_pipe_read_.get();
}

}  // namespace internal
}  // namespace tensorstore
