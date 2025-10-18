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

#ifndef TENSORSTORE_KVSTORE_FILE_UTIL_H_
#define TENSORSTORE_KVSTORE_FILE_UTIL_H_

#include <stdint.h>

#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_file_util {

/// A key is valid if is either an absolute path or a relative path, and it
/// consists of one or more path components separated by '/'. Each path
/// component must not equal "." or "..", and must not contain '\0'.
bool IsKeyValid(std::string_view key, std::string_view lock_suffix);

/// Returns the longest directory prefix of a key range.
std::string_view LongestDirectoryPrefix(const KeyRange& range);

/// Creates any directory ancestors of `path` that do not exist, and returns an
/// open file descriptor to the parent directory of `path`.
Result<internal_os::UniqueFileDescriptor> OpenParentDirectory(std::string path);

/// Reads a range of bytes from a file descriptor.
///
/// If `block_alignment` is non-zero, then the read range and underlying buffer
/// allocation will be aligned to the block size, which is required for DirectIO
/// reads.
Result<absl::Cord> ReadFromFileDescriptor(internal_os::FileDescriptor fd,
                                          ByteRange byte_range,
                                          int64_t block_alignment);

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_UTIL_H_
