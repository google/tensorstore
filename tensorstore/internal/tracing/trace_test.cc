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

#include <stdint.h>

#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/internal/tracing/span_attribute.h"
#include "tensorstore/internal/tracing/trace_span.h"

namespace {

using ::tensorstore::internal_tracing::LoggedTraceSpan;
using ::tensorstore::internal_tracing::SpanAttribute;
using ::tensorstore::internal_tracing::TraceSpan;
using ::testing::_;
using ::testing::HasSubstr;

TEST(TraceTest, Span) {
  TraceSpan span("TraceSpan",
                 {
                     SpanAttribute{"int", 1},
                     SpanAttribute{"string", "hello"},
                     SpanAttribute{"uint", 1ull},
                     SpanAttribute{"bool", true},
                     SpanAttribute{"double", 1.0},
                     SpanAttribute{"string_view", std::string_view("hello")},
                     SpanAttribute{"void*", (void*)0},
                 });

  EXPECT_EQ(span.method(), "TraceSpan");
}

TEST(TraceTest, LoggedSpan) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);

  EXPECT_CALL(
      log, Log(absl::LogSeverity::kInfo, _,
               HasSubstr("Start LoggedTraceSpan, int=1, string=hello, uint=1, "
                         "bool=true, double=1, string_view=hello, void*=")));

  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, HasSubstr("hello=world")));

  EXPECT_CALL(
      log, Log(absl::LogSeverity::kInfo, _, HasSubstr("End LoggedTraceSpan")));

  log.StartCapturingLogs();

  {
    LoggedTraceSpan span("LoggedTraceSpan", true,
                         {
                             {"int", 1},
                             {"string", "hello"},
                             {"uint", 1ull},
                             {"bool", true},
                             {"double", 1.0},
                             {"string_view", std::string_view("hello")},
                             {"void*", (void*)0},
                         });

    EXPECT_EQ(span.method(), "LoggedTraceSpan");
    span.Log("hello", "world");

    std::move(span).EndWithStatus(absl::OkStatus()).IgnoreError();
  }
}

}  // namespace
