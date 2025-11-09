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

#ifndef TENSORSTORE_INTERNAL_TRACING_SPAN_ATTRIBUTE_H_
#define TENSORSTORE_INTERNAL_TRACING_SPAN_ATTRIBUTE_H_

#include <stdint.h>

#include <string_view>
#include <variant>

namespace tensorstore {
namespace internal_tracing {

/// Spans can have additional attributes added to them.
struct SpanAttribute {
  std::string_view name;
  std::variant<bool, int64_t, uint64_t, double, std::string_view, void*> value;

  SpanAttribute(std::string_view name, bool value) : name(name), value(value) {}

  SpanAttribute(std::string_view name, int value)
      : name(name), value(static_cast<int64_t>(value)) {}
  SpanAttribute(std::string_view name, long value)  // NOLINT
      : name(name), value(static_cast<int64_t>(value)) {}
  SpanAttribute(std::string_view name, long long value)  // NOLINT
      : name(name), value(static_cast<int64_t>(value)) {}

  SpanAttribute(std::string_view name, unsigned int value)
      : name(name), value(static_cast<uint64_t>(value)) {}
  SpanAttribute(std::string_view name, unsigned long value)  // NOLINT
      : name(name), value(static_cast<uint64_t>(value)) {}
  SpanAttribute(std::string_view name, unsigned long long value)  // NOLINT
      : name(name), value(static_cast<uint64_t>(value)) {}

  SpanAttribute(std::string_view name, float value)
      : name(name), value(static_cast<double>(value)) {}
  SpanAttribute(std::string_view name, double value)
      : name(name), value(value) {}

  SpanAttribute(std::string_view name, std::string_view value)
      : name(name), value(value) {}
  SpanAttribute(std::string_view name, const char* value)
      : name(name), value(std::string_view(value)) {}

  SpanAttribute(std::string_view name, void* value)
      : name(name), value(value) {}
};

}  // namespace internal_tracing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TRACING_SPAN_ATTRIBUTE_H_
