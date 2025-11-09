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

#include <stdint.h>

#include <atomic>
#include <ostream>
#include <string_view>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tensorstore {
namespace internal_tracing {

/* static */
uint64_t LoggedTraceSpan::random_id() {
  static std::atomic<int64_t> base{absl::ToUnixNanos(absl::Now())};

  thread_local uint64_t id =
      static_cast<uint64_t>(base.fetch_add(1, std::memory_order_relaxed));

  // Apply xorshift64, which has a period of 2^64-1, to the per-thread id
  // to generate the next id.
  uint64_t x = id;
  do {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
  } while (x == 0);
  return id = x;
}

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
