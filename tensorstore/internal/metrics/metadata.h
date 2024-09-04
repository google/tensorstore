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

#ifndef TENSORSTORE_INTERNAL_METRICS_METADATA_H_
#define TENSORSTORE_INTERNAL_METRICS_METADATA_H_

#include <string_view>

namespace tensorstore {
namespace internal_metrics {

// Describes the metric units in MetricMetadata.
enum class Units : int {
  kUnknown = 0,
  kSeconds,
  kMilliseconds,
  kMicroseconds,
  kNanoseconds,
  kBits,
  kBytes,
  kKilobytes,
  kMegabytes,
};

/// Metadata for a metric, such as description.
/// Consider adding more here as necessary.
struct MetricMetadata {
  MetricMetadata() = default;
  explicit MetricMetadata(const char* description)
      : description(description), units(Units::kUnknown) {}

  MetricMetadata(const char* description, Units u)
      : description(description), units(u) {}

  std::string_view description;
  Units units = Units::kUnknown;
};

/// Returns whether name is a valid metric name.
/// Names must be path-like, each path component matches the RE: [a-zA-Z0-9_]+.
/// Metric names *should* begin with "/tensorstore/"
///
/// Example:
///   /tensorstore/foo/bar
///
bool IsValidMetricName(std::string_view name);

/// Returns whether name is a valid metric label.
/// Valid labels match the RE: [a-zA-Z][a-zA-Z0-9_]*
bool IsValidMetricLabel(std::string_view name);

/// Returns a unit-type string for the given units.
std::string_view UnitsToString(Units units);

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_METADATA_H_
