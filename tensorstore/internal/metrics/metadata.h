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

#include <stddef.h>

#include <array>
#include <string>
#include <string_view>

#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_metrics {

// A wrapper around std::string_view that prevents construction from std::string
// to avoid dangling pointers when stored in MetricMetadata.
struct StaticStringView {
  std::string_view value;
  constexpr StaticStringView(const char* s) : value(s) {}
  constexpr StaticStringView(std::string_view s) : value(s) {}
  template <size_t N>
  constexpr StaticStringView(const char (&s)[N]) : value(s, N - 1) {}

  StaticStringView(const std::string& s) = delete;
  StaticStringView(std::string&& s) = delete;

  constexpr operator std::string_view() const { return value; }
};

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
  constexpr MetricMetadata() = default;

  constexpr MetricMetadata(StaticStringView metric_name,
                           StaticStringView description,
                           Units units = Units::kUnknown)
      : metric_name(metric_name), description(description), units(units) {}

  template <size_t N>
  constexpr MetricMetadata(StaticStringView metric_name,
                           StaticStringView description,
                           const std::string_view (&fields)[N],
                           Units units = Units::kUnknown)
      : metric_name(metric_name),
        description(description),
        num_fields(N),
        units(units) {
    static_assert(N <= 8, "Max 8 fields supported");
    for (size_t i = 0; i < N; ++i) {
      field_names[i] = fields[i];
    }
  }

  constexpr MetricMetadata(StaticStringView metric_name,
                           StaticStringView description,
                           tensorstore::span<const std::string_view> fields,
                           Units units = Units::kUnknown)
      : metric_name(metric_name),
        description(description),
        num_fields(fields.size()),
        units(units) {
    if (fields.size() > 8) {
      // Cause compile error in constexpr context, or abort at runtime.
      tensorstore::span<const std::string_view>
          compile_time_error_max_8_fields_supported;
      (void)compile_time_error_max_8_fields_supported[fields.size()];
    }
    for (size_t i = 0; i < fields.size() && i < 8; ++i) {
      field_names[i] = fields[i];
    }
  }

  constexpr MetricMetadata WithDescription(StaticStringView desc) const {
    MetricMetadata copy = *this;
    copy.description = desc;
    return copy;
  }

  constexpr MetricMetadata WithUnits(Units u) const {
    MetricMetadata copy = *this;
    copy.units = u;
    return copy;
  }

  template <size_t N>
  constexpr MetricMetadata WithFieldNames(
      const std::string_view (&names)[N]) const {
    static_assert(N <= 8, "Max 8 fields supported");
    MetricMetadata copy = *this;
    copy.num_fields = N;
    for (size_t i = 0; i < N; ++i) {
      copy.field_names[i] = names[i];
    }
    return copy;
  }

  constexpr MetricMetadata WithFieldNames(
      tensorstore::span<const std::string_view> names) const {
    MetricMetadata copy = *this;
    copy.num_fields = names.size();
    if (names.size() > 8) {
      tensorstore::span<const std::string_view>
          compile_time_error_max_8_fields_supported;
      (void)compile_time_error_max_8_fields_supported[names.size()];
    }
    for (size_t i = 0; i < names.size() && i < 8; ++i) {
      copy.field_names[i] = names[i];
    }
    return copy;
  }

  constexpr tensorstore::span<const std::string_view> fields() const {
    return {field_names.data(), static_cast<ptrdiff_t>(num_fields)};
  }

  std::string_view metric_name;
  std::string_view description;
  std::array<std::string_view, 8> field_names = {};
  size_t num_fields = 0;
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
