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

#ifdef _WIN32
#error "Use get_bios_info_win.cc instead."
#endif

#include "tensorstore/internal/os/get_bios_info.h"
//

#include <sys/stat.h>

#include <cerrno>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {

Result<std::string> GetGcpProductName() {
#ifndef __linux__
  return absl::UnimplementedError("Not implemented on this platform.");
#else
  std::string path = "/sys/class/dmi/id/product_name";

  struct ::stat info;
  PotentiallyBlockingRegion region;
  if (::stat(path.c_str(), &info) != 0) {
    return StatusFromOsError(errno, "error reading bios info in ",
                             QuoteString(path));
  }
  if (!S_ISREG(info.st_mode)) {
    return absl::UnknownError(absl::StrCat("error reading bios info in ",
                                           QuoteString(path),
                                           " is not a regular file"));
  }

  std::ifstream product_name_file(path);
  std::string contents;
  if (!product_name_file.is_open()) {
    return absl::UnknownError(
        absl::StrCat("unable to open file ", QuoteString(path)));
  }

  std::getline(product_name_file, contents);
  product_name_file.close();

  return contents;
#endif
}

}  // namespace internal_os
}  // namespace tensorstore
