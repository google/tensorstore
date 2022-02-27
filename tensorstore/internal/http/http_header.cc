// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/http/http_header.h"

#include <string_view>

#include "absl/status/status.h"
#include "re2/re2.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_http {

absl::Status ValidateHttpHeader(std::string_view header) {
  static LazyRE2 kHeaderPattern = {// field-name
                                   "[!#\\$%&'*+\\-\\.\\^_`|~0-9a-zA-Z]+"
                                   ":"
                                   // OWS field-content OWS
                                   "[\t\x20-\x7e\x80-\xff]*",
                                   // Use Latin1 because the pattern applies to
                                   // raw bytes, not UTF-8.
                                   RE2::Latin1};

  if (!RE2::FullMatch(header, *kHeaderPattern)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid HTTP header: ", tensorstore::QuoteString(header)));
  }
  return absl::OkStatus();
}

}  // namespace internal_http
}  // namespace tensorstore
