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
#include "tensorstore/internal/os/filesystem.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/file/file_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_testing {

ScopedTemporaryDirectory::ScopedTemporaryDirectory() {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";

  absl::BitGen gen;
  char data[24];
  for (auto& x : data) {
    x = kAlphabet[absl::Uniform(gen, 0u, std::size(kAlphabet) - 1)];
  }

  std::string basename = tensorstore::StrCat(
      "tmp_tensorstore_test_", std::string_view(data, std::size(data)));
  path_ = internal::JoinPath(internal_os::TemporaryDirectoryPath(), basename);

  TENSORSTORE_CHECK_OK(internal_os::MakeDirectory(path_));
}

ScopedTemporaryDirectory::~ScopedTemporaryDirectory() {
  TENSORSTORE_CHECK_OK(internal_os::RemoveAll(path_));
}

ScopedCurrentWorkingDirectory::ScopedCurrentWorkingDirectory(
    const std::string& new_cwd) {
  TENSORSTORE_CHECK_OK_AND_ASSIGN(old_cwd_, internal_file_util::GetCwd());
  TENSORSTORE_CHECK_OK(internal_file_util::SetCwd(new_cwd));
}

ScopedCurrentWorkingDirectory::~ScopedCurrentWorkingDirectory() {
  TENSORSTORE_CHECK_OK(internal_file_util::SetCwd(old_cwd_));
}

}  // namespace internal_testing
}  // namespace tensorstore
