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

#ifndef TENSORSTORE_INTERNAL_SUBPROCESS_H_
#define TENSORSTORE_INTERNAL_SUBPROCESS_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

class Subprocess;

struct SubprocessOptions {
  std::string executable;         // Path to executable.
  std::vector<std::string> args;  // Arguments to executable.

  // Environment; when unset, inherits from the current process.
  std::optional<absl::flat_hash_map<std::string, std::string>> env;

  // Child handle is inherited from the parent process.
  struct Inherit {};

  // Child handle is set to the equivalent of /dev/null.
  struct Ignore {};

  // Child handle is redirected to the specified file.
  struct Redirect {
    std::string filename;
  };

  std::variant<Ignore, Redirect> stdin_action = Ignore{};
  std::variant<Inherit, Ignore, Redirect> stdout_action = Inherit{};
  std::variant<Inherit, Ignore, Redirect> stderr_action = Inherit{};
};

/// Spawn a subprocess. On success, a Subprocess object is returned.
Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options);

// Utility class representing a subprocess.
class Subprocess {
 public:
  Subprocess(const Subprocess&) = default;
  Subprocess& operator=(const Subprocess&) = default;
  Subprocess(Subprocess&&) = default;
  Subprocess& operator=(Subprocess&&) = default;

  /// NOTE: Join() is not invoked by the destructor when the last Subprocess
  /// reference is destroyed, which may result in a zombie process.
  ~Subprocess();

  /// Kill the subprocess; signal may be ignored on some platforms.
  absl::Status Kill(int signal = 9) const;

  /// Block until the process exits, returning the exit status. If necessary
  /// file handles to the process may be closed.
  ///
  /// If `block` is `false`, return immediately with
  /// `absl::StatusCode::kUnavailable` if the process has not already exited.
  Result<int> Join(bool block = true) const;

 private:
  friend Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options);

  struct Impl;

  Subprocess(std::shared_ptr<Subprocess::Impl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<Subprocess::Impl> impl_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SUBPROCESS_H_
