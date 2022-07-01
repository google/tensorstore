#include "tensorstore/internal/metrics/collect.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_metrics {

void FormatCollectedMetric(
    const CollectedMetric& metric,
    absl::FunctionRef<void(bool has_value, std::string formatted_line)>
        handle_line) {
  std::string field_names;
  if (!metric.field_names.empty()) {
    field_names = StrCat("<", absl::StrJoin(metric.field_names, ", "), ">");
  }
  if (!metric.counters.empty()) {
    for (const auto& v : metric.counters) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      std::visit(
          [&](auto x) {
            handle_line(
                /*has_value=*/x != 0,
                StrCat(metric.metric_name, field_names, fields, "=", x));
          },
          v.value);
    }
  }
  if (!metric.gauges.empty()) {
    for (const auto& v : metric.gauges) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      bool has_value = false;
      std::string value;
      std::visit(
          [&](auto x) {
            has_value = (x != 0);
            value = StrCat("={value=", x);
          },
          v.value);
      std::visit(
          [&](auto x) {
            has_value |= (x != 0);
            absl::StrAppend(&value, ", max=", x, "}");
          },
          v.max_value);
      handle_line(has_value,
                  StrCat(metric.metric_name, field_names, fields, value));
    }
  }
  if (!metric.histograms.empty()) {
    for (auto& v : metric.histograms) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      handle_line(/*has_value=*/v.count || v.sum,
                  StrCat(metric.metric_name, field_names, fields,
                         "={count=", v.count, " sum=", v.sum, " buckets=[",
                         absl::StrJoin(v.buckets, ","), "]}"));
    }
  }
  if (!metric.values.empty()) {
    for (auto& v : metric.values) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      std::visit(
          [&](auto x) {
            decltype(x) d{};
            handle_line(
                /*has_value=*/x != d,
                StrCat(metric.metric_name, field_names, fields, "=", x));
          },
          v.value);
    }
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
