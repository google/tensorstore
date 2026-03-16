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

#ifndef TENSORSTORE_INTERNAL_URI_PATH_H_
#define TENSORSTORE_INTERNAL_URI_PATH_H_

#include <stdint.h>

#include <string>
#include <string_view>

#include "tensorstore/internal/uri/parse.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_uri {

/// Returns a file uri for an os-style path.
Result<std::string> OsPathToFileUri(std::string_view path);

/// Returns an os-style path for a file uri.
Result<std::string> FileUriToOsPath(ParsedGenericUri parsed);

}  // namespace internal_uri
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_URI_PATH_H_
