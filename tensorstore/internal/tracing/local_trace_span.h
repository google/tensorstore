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

#ifndef TENSORSTORE_INTERNAL_TRACING_LOCAL_TRACE_SPAN_H_
#define TENSORSTORE_INTERNAL_TRACING_LOCAL_TRACE_SPAN_H_

#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <string_view>

#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/tracing/span_attribute.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_tracing {

/// A LocalTraceSpan is a function scope trace annotation used to tag
/// potentially expensive functions such as io, etc.
///
/// A LocalTraceSpan should be allocated on the stack within a single thread,
/// and not on the heap. Use `OperationTraceSpan` for asynchronous operations.
class LocalTraceSpan {
 public:
  LocalTraceSpan(std::string_view method,
                 const SourceLocation& location = SourceLocation::current())
  {}

  LocalTraceSpan(std::string_view method,
                 tensorstore::span<const SpanAttribute> attributes,
                 const SourceLocation& location = SourceLocation::current())
  {}

  LocalTraceSpan(std::string_view method,
                 std::initializer_list<SpanAttribute> attributes,
                 const SourceLocation& location = SourceLocation::current())
      : LocalTraceSpan(method,
                       tensorstore::span<const SpanAttribute>(
                           attributes.begin(), attributes.end()),
                       location) {}

  ~LocalTraceSpan() = default;

 private:
};

}  // namespace internal_tracing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TRACING_LOCAL_TRACE_SPAN_H_
