// Copyright 2023 The TensorStore Authors
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {
namespace {

/// Converts a CollectedMetric to json.
::nlohmann::json CollectedMetricToJson(
    const internal_metrics::CollectedMetric& metric) {
  ::nlohmann::json::object_t result;
  result["name"] = metric.metric_name;

  auto set_field_keys = [&](auto& v, ::nlohmann::json::object_t& h) {
    assert(metric.field_names.size() == v.fields.size());
    for (size_t i = 0; i < metric.field_names.size(); ++i) {
      if (metric.field_names[i] == "value" ||
          metric.field_names[i] == "count" ||
          metric.field_names[i] == "max_value" ||
          metric.field_names[i] == "sum") {
        h[tensorstore::StrCat("_", metric.field_names[i])] = v.fields[i];
      } else {
        h[std::string(metric.field_names[i])] = v.fields[i];
      }
    }
  };

  std::vector<::nlohmann::json> values;
  if (!metric.gauges.empty()) {
    for (const auto& v : metric.gauges) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      std::visit([&](auto x) { tmp["value"] = x; }, v.value);
      std::visit([&](auto x) { tmp["max_value"] = x; }, v.max_value);
      values.push_back(std::move(tmp));
    }
  } else if (!metric.values.empty()) {
    for (const auto& v : metric.values) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      std::visit([&](auto x) { tmp["value"] = x; }, v.value);
      values.push_back(std::move(tmp));
    }
  } else if (!metric.counters.empty()) {
    for (const auto& v : metric.counters) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      std::visit([&](auto x) { tmp["count"] = x; }, v.value);
      values.push_back(std::move(tmp));
    }
  } else if (!metric.histograms.empty()) {
    for (const auto& v : metric.histograms) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      tmp["count"] = v.count;
      tmp["sum"] = v.sum;

      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) end--;

      auto it = v.buckets.begin();
      for (size_t i = 0; i < end; ++i) {
        tmp[tensorstore::StrCat(i)] = *it++;
      }
      values.push_back(std::move(tmp));
    }
  }
  result["values"] = std::move(values);
  return result;
}

std::vector<::nlohmann::json> CollectMatchingMetrics(
    std::string metric_prefix) {
  std::vector<::nlohmann::json> lines;

  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix)) {
    lines.push_back(CollectedMetricToJson(metric));
  }

  std::sort(std::begin(lines), std::end(lines));
  return lines;
}

}  // namespace

void RegisterMetricBindings(pybind11::module_ m, Executor defer) {
  m.def("experimental_collect_matching_metrics", &CollectMatchingMetrics,
        pybind11::arg("metric_prefix"), R"(
Collects metrics with a matching prefix.

Returns a :py:obj:`list` of a :py:obj:`dict` of metrics.

Group:
  Experimental
)");
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterMetricBindings, /*priority=*/-1);
}

}  // namespace internal_python
}  // namespace tensorstore
