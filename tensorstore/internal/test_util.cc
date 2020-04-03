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

#include "tensorstore/internal/test_util.h"

#include <filesystem>
#include <iterator>
#include <string>
#include <system_error>  // NOLINT

#include "absl/random/random.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {
ScopedTemporaryDirectory::ScopedTemporaryDirectory() {
  std::error_code ec;
  auto base_dir = std::filesystem::temp_directory_path(ec);
  TENSORSTORE_CHECK(!ec);
  absl::BitGen gen;
  char data[16];
  for (size_t i = 0; i < std::size(data); ++i) {
    data[i] = absl::Uniform<unsigned char>(gen);
  }
  auto path =
      base_dir /
      ("tmp_tensorstore_test_" +
       absl::BytesToHexString(absl::string_view(data, std::size(data))));
  std::filesystem::create_directory(path, ec);
  TENSORSTORE_CHECK(!ec);
  path_ = path.string();
}

ScopedTemporaryDirectory::~ScopedTemporaryDirectory() {
  std::filesystem::remove_all(path_);
}

}  // namespace internal
}  // namespace tensorstore
