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

#ifndef TENSORSTORE_KVSTORE_FILE_FILE_UTIL_H_
#define TENSORSTORE_KVSTORE_FILE_FILE_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_file_util {

/// Returns the path to the current working directory.
Result<std::string> GetCwd();

/// Sets the current working directory.
absl::Status SetCwd(const std::string& path);

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_FILE_UTIL_H_
