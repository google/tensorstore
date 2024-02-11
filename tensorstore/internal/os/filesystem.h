// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_OS_FILESYSTEM_H_
#define TENSORSTORE_INTERNAL_OS_FILESYSTEM_H_

#include <functional>
#include <string>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal_os {

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath();

// similar to std::filesystem::create_directory
absl::Status MakeDirectory(const std::string& dirname);

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& dirname);

// Recursively enumerate all the paths starting at the given directory.
absl::Status EnumeratePaths(
    const std::string& directory,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_directory_entry);

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILESYSTEM_H_
