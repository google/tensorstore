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

#include "tensorstore/internal/metrics/collect.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/compare.h"
#include <nlohmann/json.hpp>

namespace tensorstore {
namespace internal_metrics {
namespace {
struct IsNonZero {
  bool operator()(int64_t x) { return x != 0; }
  bool operator()(double x) { return x != 0; }
  bool operator()(const std::string& x) { return !x.empty(); }
  bool operator()(std::monostate) { return false; }
};

struct VisitStrAppend {
  std::string* line;
  const char* before;
  const char* after;

  void operator()(int64_t x) { absl::StrAppend(line, before, x, after); }
  void operator()(double x) { absl::StrAppend(line, before, x, after); }
  void operator()(const std::string& x) {
    absl::StrAppend(line, before, x, after);
  }
  void operator()(std::monostate) {}
};

struct VisitJsonDictify {
  ::nlohmann::json::object_t& dest;
  const char* key;

  void operator()(int64_t x) { dest[key] = x; }
  void operator()(double x) { dest[key] = x; }
  void operator()(const std::string& x) { dest[key] = x; }
  void operator()(std::monostate) {}
};

// Helper to subtract variants for Value::value
template <typename... Ts>
std::variant<Ts...> SubtractVariants(const std::variant<Ts...>& before,
                                     const std::variant<Ts...>& after) {
  if (std::holds_alternative<std::monostate>(before)) {
    return after;
  }
  if (std::holds_alternative<int64_t>(before)) {
    if (std::holds_alternative<int64_t>(after)) {
      return std::get<int64_t>(after) - std::get<int64_t>(before);
    }
    return -std::get<int64_t>(before);
  }
  if (std::holds_alternative<double>(before)) {
    if (std::holds_alternative<double>(after)) {
      return std::get<double>(after) - std::get<double>(before);
    }
    return -std::get<double>(before);
  }
  return std::monostate{};
}

template <typename T>
inline absl::weak_ordering DoThreeWayCompare(const T& a, const T& b) {
  if (a < b) return absl::weak_ordering::less;
  if (b < a) return absl::weak_ordering::greater;
  return absl::weak_ordering::equivalent;
}

}  // namespace

/// Compares two collected metrics, a, and b.
absl::weak_ordering CompareByName(const CollectedMetric& a,
                                  const CollectedMetric& b) {
  if (auto c = DoThreeWayCompare(a.metric_name, b.metric_name);
      c != absl::weak_ordering::equivalent) {
    return c;
  }
  if (auto c = DoThreeWayCompare(a.tag, b.tag);
      c != absl::weak_ordering::equivalent) {
    return c;
  }
  for (size_t i = 0; i < a.field_names.size() && i < b.field_names.size();
       ++i) {
    if (auto c = DoThreeWayCompare(a.field_names[i], b.field_names[i]);
        c != absl::weak_ordering::equivalent) {
      return c;
    }
  }
  return DoThreeWayCompare(a.field_names.size(), b.field_names.size());
}

bool IsCollectedMetricNonZero(const CollectedMetric& metric) {
  if (!metric.values.empty()) {
    for (const auto& v : metric.values) {
      if (std::visit(IsNonZero{}, v.value)) return true;
      if (std::visit(IsNonZero{}, v.max_value)) return true;
    }
  }
  if (!metric.histograms.empty()) {
    for (const auto& v : metric.histograms) {
      if (v.count != 0) return true;
    }
  }
  return false;
}

CollectedMetric CollectedMetricDelta(const CollectedMetric& before,
                                     const CollectedMetric& after) {
  assert(CompareByName(before, after) == 0);
  // Return before - after for each value / histogram.
  CollectedMetric result;
  result.metric_name = before.metric_name;
  result.field_names = before.field_names;
  result.metadata = before.metadata;
  result.tag = before.tag;
  result.histogram_labels = before.histogram_labels;

  // Construct ordered value deltas.
  if (!before.values.empty() || !after.values.empty()) {
    result.values.reserve(std::max(before.values.size(), after.values.size()));
    std::vector<CollectedMetric::Value> before_values = before.values;
    std::vector<CollectedMetric::Value> after_values = after.values;

    // fields is only a partial order, so we need stable_sort.
    std::stable_sort(
        before_values.begin(), before_values.end(),
        [](const auto& a, const auto& b) { return a.fields < b.fields; });
    std::stable_sort(
        after_values.begin(), after_values.end(),
        [](const auto& a, const auto& b) { return a.fields < b.fields; });

    const std::variant<std::monostate, int64_t, double, std::string> zero_value;
    const std::variant<std::monostate, int64_t, double> zero_max_value;

    auto before_it = before_values.begin();
    auto after_it = after_values.begin();
    while (before_it != before_values.end() && after_it != after_values.end()) {
      absl::weak_ordering c =
          DoThreeWayCompare(before_it->fields, after_it->fields);
      if (c == absl::weak_ordering::greater) {
        result.values.push_back(std::move(*after_it));
        ++after_it;
      } else if (c == absl::weak_ordering::less) {
        // Negate the "before" value as it has been removed.
        before_it->value = SubtractVariants(before_it->value, zero_value);
        before_it->max_value =
            SubtractVariants(before_it->max_value, zero_max_value);
        result.values.push_back(std::move(*before_it));
        ++before_it;
      } else {
        // combine.
        CollectedMetric::Value v;
        v.fields = std::move(before_it->fields);
        v.value = SubtractVariants(before_it->value, after_it->value);
        v.max_value =
            SubtractVariants(before_it->max_value, after_it->max_value);
        result.values.push_back(std::move(v));
        ++before_it;
        ++after_it;
      }
    }
    for (; before_it != before_values.end(); ++before_it) {
      before_it->value = SubtractVariants(before_it->value, zero_value);
      before_it->max_value =
          SubtractVariants(before_it->max_value, zero_max_value);
      result.values.push_back(std::move(*before_it));
    }
    for (; after_it != after_values.end(); ++after_it) {
      result.values.push_back(std::move(*after_it));
    }
  }

  // Construct ordered histogram deltas.
  if (!before.histograms.empty() || !after.histograms.empty()) {
    result.histograms.reserve(
        std::max(before.histograms.size(), after.histograms.size()));

    std::vector<CollectedMetric::Histogram> before_values = before.histograms;
    std::vector<CollectedMetric::Histogram> after_values = after.histograms;

    // fields is only a partial order, so we need stable_sort.
    std::stable_sort(
        before_values.begin(), before_values.end(),
        [](const auto& a, const auto& b) { return a.fields < b.fields; });
    std::stable_sort(
        after_values.begin(), after_values.end(),
        [](const auto& a, const auto& b) { return a.fields < b.fields; });

    auto before_it = before_values.begin();
    auto after_it = after_values.begin();
    while (before_it != before_values.end() && after_it != after_values.end()) {
      absl::weak_ordering c =
          DoThreeWayCompare(before_it->fields, after_it->fields);
      if (c == absl::weak_ordering::less) {
        // Negate the "before" histogram as it has been removed.
        before_it->count = -before_it->count;
        before_it->mean = -before_it->mean;
        before_it->sum_of_squared_deviation =
            -before_it->sum_of_squared_deviation;
        for (size_t j = 0; j < before_it->buckets.size(); ++j) {
          before_it->buckets[j] = -before_it->buckets[j];
        }
        result.histograms.push_back(std::move(*before_it));
        ++before_it;
      } else if (c == absl::weak_ordering::greater) {
        result.histograms.push_back(std::move(*after_it));
        ++after_it;
      } else {
        // combine.
        CollectedMetric::Histogram h;
        h.fields = std::move(before_it->fields);
        h.count = after_it->count - before_it->count;
        h.mean = after_it->mean - before_it->mean;
        h.sum_of_squared_deviation = after_it->sum_of_squared_deviation -
                                     before_it->sum_of_squared_deviation;
        size_t end =
            std::max(before_it->buckets.size(), after_it->buckets.size());
        h.buckets.resize(end);
        for (size_t j = 0; j < end; ++j) {
          h.buckets[j] =
              (j < after_it->buckets.size() ? after_it->buckets[j] : 0) -
              (j < before_it->buckets.size() ? before_it->buckets[j] : 0);
        }
        result.histograms.push_back(std::move(h));
        ++before_it;
        ++after_it;
      }
    }
    for (; before_it != before_values.end(); ++before_it) {
      // Negate the "before" histogram as it has been removed.
      before_it->count = -before_it->count;
      before_it->mean = -before_it->mean;
      before_it->sum_of_squared_deviation =
          -before_it->sum_of_squared_deviation;
      for (size_t j = 0; j < before_it->buckets.size(); ++j) {
        before_it->buckets[j] = -before_it->buckets[j];
      }
      result.histograms.push_back(std::move(*before_it));
    }
    for (; after_it != after_values.end(); ++after_it) {
      result.histograms.push_back(std::move(*after_it));
    }
  }
  return result;
}

std::vector<CollectedMetric> CollectedMetricsDelta(
    std::vector<CollectedMetric>& before, std::vector<CollectedMetric>& after) {
  std::sort(before.begin(), before.end(), [](const auto& a, const auto& b) {
    return CompareByName(a, b) == absl::weak_ordering::less;
  });
  std::sort(after.begin(), after.end(), [](const auto& a, const auto& b) {
    return CompareByName(a, b) == absl::weak_ordering::less;
  });

  std::vector<CollectedMetric> result;

  auto before_it = before.begin();
  for (auto after_it = after.begin(); after_it != after.end(); ++after_it) {
    if (before_it == before.end()) {
      result.push_back(*after_it);
      continue;
    }

    while (CompareByName(*before_it, *after_it) == absl::weak_ordering::less) {
      ++before_it;
    }
    if (CompareByName(*before_it, *after_it) == absl::weak_ordering::greater) {
      result.push_back(*after_it);
      continue;
    }
    result.push_back(CollectedMetricDelta(*before_it, *after_it));
  }
  return result;
}

void FormatCollectedMetric(
    const CollectedMetric& metric,
    absl::FunctionRef<void(bool has_value, std::string formatted_line)>
        handle_line) {
  std::string field_names;
  if (!metric.field_names.empty()) {
    field_names = absl::StrJoin(metric.field_names, ", ");
  }
  auto metric_name_with_fields = [&](auto& v) -> std::string {
    if (v.fields.empty()) return std::string(metric.metric_name);
    return absl::StrCat(metric.metric_name, "<", field_names, ">[",
                        absl::StrJoin(v.fields, ", "), "]");
  };

  if (!metric.values.empty()) {
    for (auto& v : metric.values) {
      bool has_value = false;
      std::string line = metric_name_with_fields(v);
      if (std::holds_alternative<std::monostate>(v.max_value) &&
          std::holds_alternative<std::monostate>(v.value)) {
      } else {
        has_value |= std::visit(IsNonZero{}, v.value);
        has_value |= std::visit(IsNonZero{}, v.max_value);
        if (std::holds_alternative<std::monostate>(v.max_value)) {
          // A value.
          std::visit(VisitStrAppend{&line, "=", ""}, v.value);
        } else if (std::holds_alternative<std::monostate>(v.value)) {
          // A max_value.
          std::visit(VisitStrAppend{&line, "=", ""}, v.max_value);
        } else {
          // A composite value + max_value
          std::visit(VisitStrAppend{&line, "={value=", ""}, v.value);
          std::visit(VisitStrAppend{&line, ", max=", "}"}, v.max_value);
        }
      }
      handle_line(has_value, std::move(line));
    }
  }
  if (!metric.histograms.empty()) {
    for (auto& v : metric.histograms) {
      std::string line = metric_name_with_fields(v);
      absl::StrAppend(&line, "={count=", v.count, " mean=", v.mean,
                      " buckets=[");

      // find the last bucket with data.
      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) end--;

      // element 0 is typically the underflow bucket.
      auto it = v.buckets.begin();
      if (end > 0) {
        absl::StrAppend(&line, *it);
      }
      // every 10 elements insert an extra space.
      for (size_t i = 1; i < end;) {
        size_t j = std::min(i + 10, end);
        absl::StrAppend(&line, ",  ");
        absl::StrAppend(&line, absl::StrJoin(it + i, it + j, ","));
        i = j;
      }
      absl::StrAppend(&line, "]}");
      handle_line(/*has_value=*/v.count, std::move(line));
    }
  }
}

/// Converts a CollectedMetric to json.
::nlohmann::json CollectedMetricToJson(const CollectedMetric& metric) {
  ::nlohmann::json::object_t result;
  result["name"] = metric.metric_name;

  auto set_field_keys = [&](auto& v, ::nlohmann::json::object_t& h) {
    assert(metric.field_names.size() == v.fields.size());
    for (size_t i = 0; i < metric.field_names.size(); ++i) {
      if (metric.field_names[i] == "value" ||
          metric.field_names[i] == "count" ||
          metric.field_names[i] == "max_value" ||
          metric.field_names[i] == "sum") {
        h[absl::StrCat("_", metric.field_names[i])] = v.fields[i];
      } else {
        h[std::string(metric.field_names[i])] = v.fields[i];
      }
    }
  };

  std::vector<::nlohmann::json> values;
  if (!metric.values.empty()) {
    for (const auto& v : metric.values) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      std::visit(VisitJsonDictify{tmp, "value"}, v.value);
      std::visit(VisitJsonDictify{tmp, "max_value"}, v.max_value);
      values.push_back(std::move(tmp));
    }
  }
  if (!metric.histograms.empty()) {
    for (const auto& v : metric.histograms) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      tmp["count"] = v.count;
      tmp["mean"] = v.mean;
      tmp["sum_of_squared_deviation"] = v.sum_of_squared_deviation;

      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) end--;

      for (size_t i = 0; i < end; ++i) {
        assert(i < metric.histogram_labels.size());
        tmp[absl::StrCat(metric.histogram_labels[i])] = v.buckets[i];
      }
      values.push_back(std::move(tmp));
    }
  }
  result["values"] = std::move(values);
  return result;
}

}  // namespace internal_metrics
}  // namespace tensorstore
