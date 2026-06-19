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

#ifndef TENSORSTORE_INTERNAL_METRICS_FWD_H_
#define TENSORSTORE_INTERNAL_METRICS_FWD_H_

// Forward declarations for use in TENSORSTORE_DECLARE_AND_REGISTER_METRIC.
namespace tensorstore {
namespace internal_metrics {
struct DefaultBucketer;

// Domains
template <typename Spec, bool CaseSensitive>
class DomainField;

// Metrics
template <typename T, typename... Fields>
class Counter;
template <typename T, typename... Fields>
class Gauge;
template <typename T, typename... Fields>
class MaxGauge;
template <typename Bucketer, typename... Fields>
class Histogram;
template <typename Bucketer, typename... Fields>
class Value;

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_FWD_H_
