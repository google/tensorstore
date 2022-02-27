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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_

#include <string_view>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal_http {

/// Validates that `header` is a valid HTTP header line (excluding the final
/// "\r\n").
///
/// The syntax is defined by:
/// https://datatracker.ietf.org/doc/html/rfc7230#section-3.2
///
///   header-field   = field-name ":" OWS field-value OWS
///   field-name     = token
///   field-value    = *( field-content / obs-fold )
///   field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
///   field-vchar    = VCHAR / obs-text
///   OWS            = *( SP / HTAB )
///   VCHAR          = any visible ascii character
///   obs-text       = %x80-FF
///   obs-fold = CRLF 1*( SP / HTAB )
///
/// This function differs from the spec in that it does not accept `obs-fold`
/// (obsolete line folding) syntax.
absl::Status ValidateHttpHeader(std::string_view header);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
