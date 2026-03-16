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

#include "tensorstore/internal/uri/path.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/uri/parse.h"
#include "tensorstore/internal/uri/percent_coder.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_uri {
namespace {

using ::tensorstore::internal::LexicalNormalizePath;

// Returns whether `path` represents a windows drive letter, e.g. "C:/tmp".
bool IsWindowsDriveLetter(std::string_view path) {
  return path.length() >= 2 && path[1] == ':' && absl::ascii_isalpha(path[0]);
}

}  // namespace

Result<std::string> OsPathToFileUri(std::string_view path) {
  if (!internal::IsAbsolutePath(path)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "file: URIs do not support relative paths: %v", QuoteString(path)));
  }

  std::string_view authority_part;
  if (std::string_view root_name = internal::PathRootName(path);
      !root_name.empty() && !IsWindowsDriveLetter(root_name)) {
    path.remove_prefix(root_name.size());
    authority_part = root_name.substr(2);
  }

#ifdef _WIN32
  constexpr const char kDirSeparator[] = "/\\";
#else
  constexpr const char kDirSeparator[] = "/";
#endif

  auto splitter = absl::StrSplit(path, absl::ByAnyChar(kDirSeparator));
  auto it = splitter.begin();
  while (it != splitter.end() && it->empty()) it++;
  return absl::StrCat(
      "file://", authority_part, "/",
      PercentEncodeUriPath(absl::StrJoin(it, splitter.end(), "/")));
}

Result<std::string> FileUriToOsPath(ParsedGenericUri parsed) {
  TENSORSTORE_RETURN_IF_ERROR(
      EnsureSchemaWithAuthorityDelimiter(parsed, "file"));
  TENSORSTORE_RETURN_IF_ERROR(EnsureNoQueryOrFragment(parsed));

  if (parsed.path.empty() || parsed.path[0] != '/') {
    return absl::InvalidArgumentError(
        absl::StrFormat("file: URIs do not support relative paths: %v",
                        QuoteString(parsed.path)));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(std::string decoded_path,
                               PercentDecode(parsed.path));

#ifdef _WIN32
  if (std::string uri_path = decoded_path.substr(1);
      IsWindowsDriveLetter(uri_path)) {
    // Translate windows drive letter paths.
    if (uri_path.size() > 2 && uri_path[2] != '/') {
      return absl::InvalidArgumentError(
          absl::StrFormat("file: URIs do not support relative paths: %v",
                          QuoteString(uri_path)));
    }
    return LexicalNormalizePath(uri_path);
  } else if (!parsed.authority.empty()) {
    // Allow windows network shares to be used via file uris.
    TENSORSTORE_ASSIGN_OR_RETURN(std::string decoded_authority,
                                 PercentDecode(parsed.authority));
    return absl::StrCat("//", decoded_authority,
                        LexicalNormalizePath(decoded_path));
  }
#endif

  if (!parsed.authority.empty() && parsed.authority != "localhost") {
    return absl::InvalidArgumentError(
        absl::StrCat("file: URIs do not support authority: ",
                     QuoteString(parsed.authority)));
  }
  return LexicalNormalizePath(std::move(decoded_path));
}

}  // namespace internal_uri
}  // namespace tensorstore
