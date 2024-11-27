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

#include "tensorstore/internal/tracing/logged_trace_span.h"

#include <ostream>
#include <string_view>

#include "absl/strings/str_format.h"

namespace tensorstore {
namespace internal_tracing {

void LoggedTraceSpan::BeginLog(std::ostream& stream) {
  stream << absl::StreamFormat("%x: Start %s", id_, method());
}

std::ostream& LoggedTraceSpan::EndLog(std::ostream& stream) {
  stream << absl::StreamFormat("%x: End %s", id_, method());
  return stream;
}

void LoggedTraceSpan::LogImpl(std::string_view name, const void* val,
                              std::ostream& stream) {
  stream << absl::StreamFormat("%x: %s=%p", id_, name, val);
}

void LoggedTraceSpan::LogImpl(std::string_view name, const char* val,
                              std::ostream& stream) {
  stream << absl::StreamFormat("%x: %s=%s", id_, name, val);
}

}  // namespace internal_tracing
}  // namespace tensorstore
