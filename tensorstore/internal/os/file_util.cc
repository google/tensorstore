// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/os/file_util.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {

Result<std::string> ReadAllToString(const std::string& path) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto fd, OpenFileWrapper(path, internal_os::OpenFlags::OpenReadOnly));

  internal_os::FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd.get(), &info));
  if (internal_os::GetSize(info) == 0) {
    return absl::InternalError(
        absl::StrCat("File ", QuoteString(path), " is empty"));
  }
  std::string result;
  result.resize(internal_os::GetSize(info));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto read,
      internal_os::ReadFromFile(fd.get(), result.data(), result.size(), 0));
  if (read != result.size()) {
    return absl::InternalError("Failed to read entire file");
  }
  return result;
}

}  // namespace internal_os
}  // namespace tensorstore
