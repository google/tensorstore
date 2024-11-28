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

#ifndef TENSORSTORE_INTERNAL_TRACING_LOGGED_TRACE_SPAN_H_
#define TENSORSTORE_INTERNAL_TRACING_LOGGED_TRACE_SPAN_H_

#include <stdint.h>

#include <initializer_list>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <variant>

#include "absl/log/log_streamer.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/tracing/span_attribute.h"
#include "tensorstore/internal/tracing/trace_span.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_tracing {

/// A TraceSpan which optionally includes scoped logging to ABSL INFO.
class LoggedTraceSpan : public TraceSpan {
  // Generates a random ID for logging the Begin/End log messages.
  static uint64_t random_id();

 public:
  LoggedTraceSpan(std::string_view method, bool log,
                  const SourceLocation& location = SourceLocation::current())
      : TraceSpan(method, location),
        location_(location),
        id_(log ? random_id() : 0) {
    if (id_)
      BeginLog(absl::LogInfoStreamer(location_.file_name(), location_.line())
                   .stream());
  }

  LoggedTraceSpan(std::string_view method, bool log,
                  tensorstore::span<const SpanAttribute> attributes,
                  const SourceLocation& location = SourceLocation::current())
      : TraceSpan(method, attributes, location),
        location_(location),
        id_(log ? random_id() : 0) {
    if (id_)
      BeginLog(absl::LogInfoStreamer(location_.file_name(), location_.line())
                   .stream(),
               attributes);
  }

  LoggedTraceSpan(std::string_view method, bool log,
                  std::initializer_list<SpanAttribute> attributes,
                  const SourceLocation& location = SourceLocation::current())
      : LoggedTraceSpan(method, log,
                        tensorstore::span<const SpanAttribute>(
                            attributes.begin(), attributes.end()),
                        location) {}

  ~LoggedTraceSpan() {
    if (id_)
      EndLog(absl::LogInfoStreamer(location_.file_name(), location_.line())
                 .stream());
  }

  /// Log an key=value pair with the current LoggedTraceSpan Id to the INFO log.
  template <typename T>
  void Log(std::string_view name, T val,
           const SourceLocation& location = SourceLocation::current()) {
    if (id_)
      LogImpl(name, val,
              absl::LogInfoStreamer(location.file_name(), location.line())
                  .stream());
  }

  /// Finish the span with a logged status.
  absl::Status EndWithStatus(
      absl::Status&& status,
      const SourceLocation& location = SourceLocation::current()) && {
    if (id_) {
      EndLog(
          absl::LogInfoStreamer(location.file_name(), location.line()).stream())
          << status;
      id_ = 0;
    }
    return status;
  }

  using TraceSpan::method;

 private:
  void BeginLog(std::ostream& stream);

  void BeginLog(std::ostream& stream,
                tensorstore::span<const SpanAttribute> attributes) {
    BeginLog(stream);
    for (const auto& attr : attributes) {
      stream << absl::StreamFormat(", %s=", attr.name);
      std::visit([&stream](auto v) { stream << v; }, attr.value);
    }
  }

  std::ostream& EndLog(std::ostream& stream);

  void LogImpl(std::string_view name, const void* val, std::ostream& stream);
  void LogImpl(std::string_view name, const char* val, std::ostream& stream);

  template <typename T>
  std::enable_if_t<!std::is_pointer_v<T>, void>  //
  LogImpl(std::string_view name, T val, std::ostream& stream) {
    stream << absl::StreamFormat("%x: %s=", id_, name) << val;
  }

  SourceLocation location_;
  uint64_t id_ = 0;
};

}  // namespace internal_tracing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TRACING_LOGGED_TRACE_SPAN_H_
