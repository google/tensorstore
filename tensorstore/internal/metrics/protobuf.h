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

#ifndef TENSORSTORE_INTERNAL_METRICS_PROTOBUF_H_
#define TENSORSTORE_INTERNAL_METRICS_PROTOBUF_H_

#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metrics.pb.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_metrics {

/// Convert a `CollectedMetric` to a `metrics_proto::Metric`.
void CollectedMetricToProto(const CollectedMetric& metric,
                            metrics_proto::Metric& proto);

/// Convert a collection of `CollectedMetric` to a
/// `metrics_proto::MetricCollection`.
void CollectedMetricToProtoCollection(span<const CollectedMetric> metrics,
                                      metrics_proto::MetricCollection& proto);

/// Sorts the underlying MetricCollection.
void SortProtoCollection(metrics_proto::MetricCollection& proto);

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_PROTOBUF_H_
