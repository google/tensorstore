// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_METRICS_REGISTRATION_H_
#define TENSORSTORE_INTERNAL_METRICS_REGISTRATION_H_

#include "absl/base/attributes.h"              // IWYU pragma: keep
#include "tensorstore/internal/metrics/fwd.h"  // IWYU pragma: keep
#include "tensorstore/internal/metrics/metadata.h"  // IWYU pragma: export
#include "tensorstore/internal/metrics/registry.h"  // IWYU pragma: keep
#include "tensorstore/internal/preprocessor/strip_parens.h"  // IWYU pragma: keep

// Declares the TENSORSTORE_DECLARE_AND_REGISTER_METRIC macro and
// provides a manual registration example.
//
// Manual registration example (without using the macro):
//
// 1. Define a struct containing the metrics. Since all metric types
//    are constexpr-constructible, the struct itself can be
//    constexpr-constructible and constant-initialized.
//
//    using ::tensorstore::internal_metrics::Counter;
//    using ::tensorstore::internal_metrics::MetricMetadata;
//
//    struct MyMetrics {
//      Counter<int64_t> read_count;
//      Counter<int64_t, std::string> read_bytes;
//    };
//    static MyMetrics g_metrics;  // Constant-initialized.
//
// 2. Register the metrics explicitly, typically in an initialization
//    function. Registration is idempotent.
//
//    void RegisterMyMetrics() {
//      auto& r = tensorstore::internal_metrics::GetMetricRegistry();
//      r.Register(
//          &g_metrics.read_count,
//          MetricMetadata("/my/read_count", "Read count"));
//      r.Register(
//          &g_metrics.read_bytes,
//          MetricMetadata("/my/read_bytes", "Read bytes")
//              .WithFieldNames({"user"}));
//    }
//
// 3. Use the metrics directly:
//
//    g_metrics.read_count.Increment();
//    g_metrics.read_bytes.IncrementBy(100, "alice");

// Declares and registers a metric, wrapping the declaration in an
// anonymous namespace for internal linkage.
//
// The macro automatically qualifies the `type` with
// `::tensorstore::internal_metrics::`, so type names should be
// unqualified (e.g., `Counter<int64_t>` not
// `internal_metrics::Counter<int64_t>`).  `DefaultBucketer`,
// `MetricMetadata`, and `Units` are brought into scope via
// using-declarations and can be used unqualified in the `type` and
// `metadata` arguments.
//
// \param name     Variable name of the declared metric.
// \param type     Metric type (e.g., `Counter<int64_t>`,
//     `Gauge<double>`). If the type contains commas (template
//     arguments), it must be enclosed in parentheses:
//     `(Counter<int64_t, std::string>)`.
// \param metadata `MetricMetadata` object specifying path and
//     description.
// \param ...      Optional field names for the metric's dimensions.
//
// Example:
//   TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
//       gfile_read_by_user, (Counter<int64_t, std::string>),
//       MetricMetadata("/tensorstore/kvstore/gfile/read_by_user",
//                      "Gfile read count by user"),
//       "username");
//
//   gfile_read_by_user.Increment("alice");
//
// NOTE:
// This macro wraps the metric counter in an anonymous namespace.
// It MUST ONLY be used at file scope in `.cc` files.
// Using it in a header will create distinct instances per translation
// unit, and cause a failure at startup.
//
#define TENSORSTORE_DECLARE_AND_REGISTER_METRIC(name, type, metadata, ...)  \
  namespace {                                                               \
  using ::tensorstore::internal_metrics::DefaultBucketer;                   \
  using ::tensorstore::internal_metrics::MetricMetadata;                    \
  using ::tensorstore::internal_metrics::Units;                             \
  using ::tensorstore::internal_metrics::DomainField;                       \
  ABSL_CONST_INIT /**/                                                      \
      ::tensorstore::internal_metrics::TENSORSTORE_STRIP_PARENS(type) name; \
  const struct name##_registered_type {                                     \
    name##_registered_type() {                                              \
      ::tensorstore::internal_metrics::GetMetricRegistry().Register(        \
          &name, metadata.WithFieldNames({__VA_ARGS__}));                   \
    }                                                                       \
  } name##_registered;                                                      \
  }  // namespace

#endif  // TENSORSTORE_INTERNAL_METRICS_REGISTRATION_H_
