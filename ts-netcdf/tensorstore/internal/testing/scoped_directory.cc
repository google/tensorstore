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

#include "tensorstore/internal/testing/scoped_directory.h"

#include <iterator>
#include <string>
#include <string_view>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/cwd.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/os/filesystem.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_testing {

ScopedTemporaryDirectory::ScopedTemporaryDirectory() {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";

  absl::BitGen gen;
  char data[24];
  for (auto& x : data) {
    x = kAlphabet[absl::Uniform(gen, 0u, std::size(kAlphabet) - 1)];
  }

  std::string basename = absl::StrCat("tmp_tensorstore_test_",
                                      std::string_view(data, std::size(data)));
  path_ = internal::JoinPath(internal_os::TemporaryDirectoryPath(), basename);
  TENSORSTORE_CHECK_OK(internal_os::MakeDirectory(path_));
  path_ = internal::LexicalNormalizePath(path_);
}

ScopedTemporaryDirectory::~ScopedTemporaryDirectory() {
  auto status = internal_os::RemoveAll(path_);
  if (absl::IsNotFound(status) /* already removed */
      || absl::IsFailedPrecondition(status) /* WIN32: not empty. */) {
    status = absl::OkStatus();
  }
  TENSORSTORE_CHECK_OK(status);
}

ScopedCurrentWorkingDirectory::ScopedCurrentWorkingDirectory(
    const std::string& new_cwd) {
  TENSORSTORE_CHECK_OK_AND_ASSIGN(old_cwd_, internal_os::GetCwd());
  TENSORSTORE_CHECK_OK(internal_os::SetCwd(new_cwd));
}

ScopedCurrentWorkingDirectory::~ScopedCurrentWorkingDirectory() {
  TENSORSTORE_CHECK_OK(internal_os::SetCwd(old_cwd_));
}

}  // namespace internal_testing
}  // namespace tensorstore
