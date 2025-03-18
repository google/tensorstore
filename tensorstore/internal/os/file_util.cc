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

#include <stddef.h>

#include <cassert>
#include <cstring>
#include <string>

#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {

Result<std::string> ReadAllToString(const std::string& path) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto fd,
                               OpenFileWrapper(path, OpenFlags::OpenReadOnly));

  FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(GetFileInfo(fd.get(), &info));

  // Handle the case where the file is empty.
  std::string result(internal_os::GetSize(info), 0);
  if (result.empty()) {
    result.resize(4096);
  }

  size_t offset = 0;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto read,
      PReadFromFile(fd.get(), tensorstore::span(&result[0], result.size()),
                    offset));
  offset += read;

  while (true) {
    assert(offset <= result.size());
    if ((result.size() - offset) < 4096) {
      // In most cases the buffer will be at least as large as necessary and
      // this read will return 0, avoiding the copy+resize.
      char buffer[4096];
      TENSORSTORE_ASSIGN_OR_RETURN(
          read,
          PReadFromFile(fd.get(), tensorstore::span(buffer, sizeof(buffer)),
                        offset));
      if (read > 0) {
        // Amortized resize; double the size of the buffer.
        if (read > result.size()) {
          result.resize(read + result.size() * 2);
        }
        memcpy(&result[offset], buffer, read);
      }
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(
          read, PReadFromFile(
                    fd.get(),
                    tensorstore::span(&result[offset], result.size() - offset),
                    offset));
    }
    if (read == 0) {
      result.resize(offset);
      return result;
    }
    offset += read;
  }
}

}  // namespace internal_os
}  // namespace tensorstore
