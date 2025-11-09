// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_COMMON_METRICS_H_
#define TENSORSTORE_KVSTORE_COMMON_METRICS_H_

#include <stdint.h>

#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_kvstore {

// Holds references to the common read metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/read
//   /tensorstore/kvstore/driver/list
struct CommonReadMetrics {
  internal_metrics::Counter<int64_t>& read;
  internal_metrics::Counter<int64_t>& list;
};

// Holds references to the common write metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/write
//   /tensorstore/kvstore/driver/delete_range
struct CommonWriteMetrics {
  internal_metrics::Counter<int64_t>& write;
  internal_metrics::Counter<int64_t>& delete_range;
};

// Holds references to the common read metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/batch_read
//   /tensorstore/kvstore/driver/bytes_read
//   /tensorstore/kvstore/driver/read_latency_ms
struct DetailedReadMetrics {
  internal_metrics::Counter<int64_t>& batch_read;
  internal_metrics::Counter<int64_t>& bytes_read;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>&
      read_latency_ms;
};

// Holds references to the common write metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/bytes_written
//   /tensorstore/kvstore/driver/write_latency_ms
struct DetailedWriteMetrics {
  internal_metrics::Counter<int64_t>& bytes_written;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>&
      write_latency_ms;
};

// Holds references to the common read and write metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/read
//   /tensorstore/kvstore/driver/list
//   /tensorstore/kvstore/driver/write
//   /tensorstore/kvstore/driver/delete_range
//   /tensorstore/kvstore/driver/batch_read
//   /tensorstore/kvstore/driver/bytes_read
//   /tensorstore/kvstore/driver/read_latency_ms
//   /tensorstore/kvstore/driver/bytes_written
//   /tensorstore/kvstore/driver/write_latency_ms
//
// Example:
//   namespace {
//     auto my_metrics = TENSORSTORE_KVSTORE_COMMON_METRICS(driver);
//   }  // namespace
//
//   my_metrics.read.Increment();
//   my_metrics.bytes_read.IncrementBy(100);
//
struct CommonMetrics : public CommonReadMetrics,
                       public CommonWriteMetrics,
                       public DetailedReadMetrics,
                       public DetailedWriteMetrics {
  // no additional members
};

#define TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, NAME, DESC, ...) \
  internal_metrics::Counter<int64_t>::New(                         \
      "/tensorstore/kvstore/" #KVSTORE "/" #NAME,                  \
      internal_metrics::MetricMetadata(#KVSTORE " " DESC, ##__VA_ARGS__))

#define TENSORSTORE_KVSTORE_LATENCY_IMPL(KVSTORE, NAME, METRIC_FN)     \
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New( \
      "/tensorstore/kvstore/" #KVSTORE "/" #NAME,                      \
      internal_metrics::MetricMetadata(                                \
          #KVSTORE " kvstore::" #METRIC_FN " latency (ms)",            \
          internal_metrics::Units::kMilliseconds))

#define TENSORSTORE_KVSTORE_COMMON_READ_METRICS(KVSTORE)              \
  []() -> ::tensorstore::internal_kvstore::CommonReadMetrics {        \
    return {TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, read,           \
                                             "kvstore::Read calls"),  \
            TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, list,           \
                                             "kvstore::List calls")}; \
  }()

#define TENSORSTORE_KVSTORE_COMMON_WRITE_METRICS(KVSTORE)                    \
  []() -> ::tensorstore::internal_kvstore::CommonWriteMetrics {              \
    return {TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, write,                 \
                                             "kvstore::Write calls"),        \
            TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, delete_range,          \
                                             "kvstore::DeleteRange calls")}; \
  }()

#define TENSORSTORE_KVSTORE_DETAILED_READ_METRICS(KVSTORE)                  \
  []() -> ::tensorstore::internal_kvstore::DetailedReadMetrics {            \
    return {                                                                \
        TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, batch_read,               \
                                         "kvstore::Read after batching"),   \
        TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, bytes_read, "bytes read", \
                                         internal_metrics::Units::kBytes),  \
        TENSORSTORE_KVSTORE_LATENCY_IMPL(KVSTORE, read_latency_ms, Read)};  \
  }()

#define TENSORSTORE_KVSTORE_DETAILED_WRITE_METRICS(KVSTORE)                  \
  []() -> ::tensorstore::internal_kvstore::DetailedWriteMetrics {            \
    return {                                                                 \
        TENSORSTORE_KVSTORE_COUNTER_IMPL(KVSTORE, bytes_written,             \
                                         "bytes written",                    \
                                         internal_metrics::Units::kBytes),   \
        TENSORSTORE_KVSTORE_LATENCY_IMPL(KVSTORE, write_latency_ms, Write)}; \
  }()

#define TENSORSTORE_KVSTORE_COMMON_METRICS(KVSTORE)               \
  []() -> ::tensorstore::internal_kvstore::CommonMetrics {        \
    return {TENSORSTORE_KVSTORE_COMMON_READ_METRICS(KVSTORE),     \
            TENSORSTORE_KVSTORE_COMMON_WRITE_METRICS(KVSTORE),    \
            TENSORSTORE_KVSTORE_DETAILED_READ_METRICS(KVSTORE),   \
            TENSORSTORE_KVSTORE_DETAILED_WRITE_METRICS(KVSTORE)}; \
  }()

}  // namespace internal_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_COMMON_METRICS_H_
